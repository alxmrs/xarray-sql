// Licensed under the Apache License, Version 2.0
//!
//! Lazy Arrow stream table provider for xarray-sql.
//!
//! This module provides `LazyArrowStreamTable`, which wraps a Python object
//! implementing `__arrow_c_stream__` and exposes it as a DataFusion TableProvider
//! via the `__datafusion_table_provider__` protocol.
//!
//! ## Key Features
//!
//! - **Lazy evaluation**: Data is not read from the Python stream until query
//!   execution time (during `collect()`), not at registration time.
//!
//! - **Error propagation**: Errors during batch reading are properly propagated
//!   to DataFusion and surfaced to the user.
//!
//! - **Reusable tables**: The factory pattern allows the same table to be queried
//!   multiple times, with fresh data streams created for each query.
//!
//! ## Streaming Behavior
//!
//! Batches are read lazily one at a time during query execution. The GIL is acquired
//! for each batch read, allowing DataFusion to process and potentially filter batches
//! incrementally. This enables processing of larger-than-memory datasets when combined
//! with DataFusion's streaming execution.
//!
//! ## Parallel Execution Note
//!
//! When using DataFusion's parallel execution (multiple partitions), aggregation queries
//! without ORDER BY may return partial results due to how our stream interacts with
//! DataFusion's async runtime. To ensure complete results:
//! - Add ORDER BY to aggregation queries, or
//! - Use `SessionConfig().with_target_partitions(1)` for single-threaded execution
//! TODO(#106): Implenet proper parallelism and partition handling.

use std::ffi::CString;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::pyarrow::FromPyArrow;
use datafusion::catalog::streaming::StreamingTable;
use datafusion::common::DataFusionError;
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_ffi::table_provider::FFI_TableProvider;
use futures::stream::unfold;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use tokio::runtime::Handle;

/// A partition stream that wraps a Python factory function that creates streams.
///
/// The factory is called lazily on each `execute()` invocation, allowing
/// the same table to be queried multiple times.
struct PyArrowStreamPartition {
    schema: SchemaRef,
    /// A Python callable (factory) that returns a fresh stream implementing `__arrow_c_stream__`.
    /// Called on each execute() to create a new stream.
    stream_factory: Py<PyAny>,
}

impl PyArrowStreamPartition {
    fn new(stream_factory: Py<PyAny>, schema: SchemaRef) -> Self {
        Self {
            schema,
            stream_factory,
        }
    }
}

impl Debug for PyArrowStreamPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyArrowStreamPartition")
            .field("schema", &self.schema)
            .finish()
    }
}

/// Shared state for the lazy stream, protected by Mutex for thread safety.
struct SharedStreamState {
    /// The PyArrow RecordBatchReader (None until first batch is requested)
    reader: Option<Py<PyAny>>,
    /// The factory to create the reader (consumed on first use)
    factory: Option<Py<PyAny>>,
    /// Whether the stream has ended
    done: bool,
}

impl PartitionStream for PyArrowStreamPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.schema);

        // Clone the factory with the GIL held
        let factory = Python::attach(|py| self.stream_factory.clone_ref(py));

        // TODO(alxmrs/CC): I think we need to do something datafusion-native here;
        //  I suspect that adding a mutex will significantly impact performance.
        //  This is OK for now.
        // Create shared state protected by Mutex
        let shared_state = Arc::new(Mutex::new(SharedStreamState {
            reader: None,
            factory: Some(factory),
            done: false,
        }));

        // Create a lazy stream using unfold.
        // The Arc<Mutex<...>> is cloned for each iteration, ensuring thread-safe access.
        let batch_stream = unfold(shared_state, |state| async move {
            // Clone Arc for potential return
            let state_clone = Arc::clone(&state);

            // Lock the mutex to access state
            let mut guard = state.lock().unwrap();

            if guard.done {
                return None;
            }

            // Acquire GIL and process
            let result = Python::attach(|py| {
                // Initialize reader on first poll
                if guard.reader.is_none() {
                    if let Some(factory) = guard.factory.take() {
                        match factory.call0(py) {
                            Ok(reader) => {
                                guard.reader = Some(reader);
                            }
                            Err(e) => {
                                guard.done = true;
                                return Some(Err(DataFusionError::Execution(format!(
                                    "Failed to call stream factory: {e}"
                                ))));
                            }
                        }
                    }
                }

                // Read next batch from reader
                if let Some(ref reader) = guard.reader {
                    let bound_reader = reader.bind(py);
                    match bound_reader.call_method0("read_next_batch") {
                        Ok(batch_obj) => match RecordBatch::from_pyarrow_bound(&batch_obj) {
                            Ok(batch) => Some(Ok(batch)),
                            Err(e) => {
                                guard.done = true;
                                Some(Err(DataFusionError::Execution(format!(
                                    "Failed to convert batch from PyArrow: {e}"
                                ))))
                            }
                        },
                        Err(e) => {
                            if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                guard.done = true;
                                None // Stream exhausted normally
                            } else {
                                guard.done = true;
                                Some(Err(DataFusionError::Execution(format!(
                                    "Error reading batch from stream: {e}"
                                ))))
                            }
                        }
                    }
                } else {
                    guard.done = true;
                    None
                }
            });

            // Release lock before returning
            drop(guard);

            // Map result to include state for next iteration
            result.map(|batch_result| (batch_result, state_clone))
        });

        Box::pin(RecordBatchStreamAdapter::new(schema, batch_stream))
    }
}

/// A lazy table provider that wraps a Python stream factory.
///
/// This class implements the `__datafusion_table_provider__` protocol, allowing
/// it to be registered with DataFusion's `SessionContext.register_table()`.
///
/// Data is NOT read until query execution time - this enables true lazy evaluation.
/// The factory function is called on each query execution to create a fresh stream,
/// allowing the same table to be queried multiple times.
///
/// # Example
///
/// ```python
/// from datafusion import SessionContext
/// from xarray_sql import LazyArrowStreamTable, XarrayRecordBatchReader
///
/// # Create a factory that produces lazy readers
/// def make_reader():
///     return XarrayRecordBatchReader(ds, chunks={'time': 240})
///
/// # Get schema from a sample reader
/// sample = make_reader()
/// schema = sample.schema
///
/// # Wrap factory in lazy table - NO DATA LOADED
/// table = LazyArrowStreamTable(make_reader, schema)
///
/// # Register with DataFusion - STILL NO DATA LOADED
/// ctx = SessionContext()
/// ctx.register_table("air", table)
///
/// # Data only loaded HERE during collect()
/// # Each query creates a fresh stream via the factory
/// result = ctx.sql("SELECT AVG(air) FROM air").collect()
/// result2 = ctx.sql("SELECT * FROM air LIMIT 10").collect()  # Works!
/// ```
#[pyclass(name = "LazyArrowStreamTable")]
struct LazyArrowStreamTable {
    /// The underlying StreamingTable
    table: Arc<StreamingTable>,
}

#[pymethods]
impl LazyArrowStreamTable {
    /// Create a new LazyArrowStreamTable from a stream factory function.
    ///
    /// Args:
    ///     stream_factory: A callable that returns a Python object implementing
    ///             the Arrow PyCapsule interface (`__arrow_c_stream__`).
    ///             Called on each query execution to create a fresh stream.
    ///     schema: A PyArrow Schema for the table. Required since the factory
    ///             hasn't been called yet.
    ///
    /// Raises:
    ///     TypeError: If the schema is not a valid PyArrow Schema.
    #[new]
    fn new(stream_factory: &Bound<'_, PyAny>, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert the PyArrow schema to Arrow schema
        use arrow::datatypes::Schema;
        use arrow::pyarrow::FromPyArrow;

        let arrow_schema = Schema::from_pyarrow_bound(schema).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
        })?;
        let schema_ref = Arc::new(arrow_schema);

        // Create the partition stream with the factory
        let partition =
            PyArrowStreamPartition::new(stream_factory.clone().unbind(), schema_ref.clone());

        // Create the StreamingTable
        let table =
            StreamingTable::try_new(schema_ref, vec![Arc::new(partition)]).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create StreamingTable: {e}"
                ))
            })?;

        Ok(Self {
            table: Arc::new(table),
        })
    }

    /// Returns a PyCapsule implementing the DataFusion TableProvider FFI.
    ///
    /// This method is called by DataFusion's `register_table()` to get a
    /// foreign table provider that can be used in queries.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        // Create the FFI table provider
        let provider: Arc<dyn TableProvider + Send> = self.table.clone();

        // Try to get the current tokio runtime handle (available when called from DataFusion context)
        let runtime = Handle::try_current().ok();

        // Create FFI wrapper (v49 API takes 3 arguments)
        let ffi_provider = FFI_TableProvider::new(
            provider, false, // can_support_pushdown_filters
            runtime,
        );

        // Create the capsule name
        let name = CString::new("datafusion_table_provider").unwrap();

        // Create the PyCapsule without a destructor closure
        // The PyCapsule takes ownership of the FFI_TableProvider
        PyCapsule::new(py, ffi_provider, Some(name))
    }

    /// Get the schema of the table as a PyArrow Schema.
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use arrow::pyarrow::ToPyArrow;
        self.table
            .schema()
            .to_pyarrow(py)
            .map(|bound| bound.unbind())
    }

    fn __repr__(&self) -> String {
        format!("LazyArrowStreamTable(schema={:?})", self.table.schema())
    }
}

/// Python module initialization
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LazyArrowStreamTable>()?;
    Ok(())
}
