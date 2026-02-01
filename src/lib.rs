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
//! ## Parallel Execution
//!
//! Each xarray chunk becomes a separate partition, enabling parallel execution across
//! multiple cores. Due to a bug in DataFusion v51.0.0's `collect()` method, aggregation
//! queries should use `to_arrow_table()` instead to ensure complete results.
//! TODO(#107): Upgrading to the latest datafusion-python (52+) should fix this.

use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::pyarrow::FromPyArrow;
use async_stream::try_stream;
use datafusion::catalog::streaming::StreamingTable;
use datafusion::common::DataFusionError;
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_ffi::table_provider::FFI_TableProvider;
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

impl PartitionStream for PyArrowStreamPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.schema);

        // Clone the factory with the GIL held
        let factory = Python::attach(|py| self.stream_factory.clone_ref(py));

        // Create a lazy stream using try_stream! macro.
        // This is cleaner than manual state management with unfold.
        // Each iteration acquires the GIL and reads one batch.
        let batch_stream = try_stream! {
            // Call factory to get the PyArrow RecordBatchReader
            let reader: Py<PyAny> = Python::attach(|py| {
                factory.call0(py).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to call stream factory: {e}"))
                })
            })?;

            // Read batches until StopIteration
            loop {
                let batch_result = Python::attach(|py| {
                    let bound_reader = reader.bind(py);
                    match bound_reader.call_method0("read_next_batch") {
                        Ok(batch_obj) => {
                            match RecordBatch::from_pyarrow_bound(&batch_obj) {
                                Ok(batch) => Ok(Some(batch)),
                                Err(e) => Err(DataFusionError::Execution(format!(
                                    "Failed to convert batch from PyArrow: {e}"
                                ))),
                            }
                        }
                        Err(e) => {
                            if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                Ok(None) // Stream exhausted normally
                            } else {
                                Err(DataFusionError::Execution(format!(
                                    "Error reading batch from stream: {e}"
                                )))
                            }
                        }
                    }
                });

                match batch_result {
                    Ok(Some(batch)) => yield batch,
                    Ok(None) => break,
                    Err(e) => Err(e)?,
                }
            }
        };

        Box::pin(RecordBatchStreamAdapter::new(schema, batch_stream))
    }
}

/// A lazy table provider that wraps Python stream factory functions.
///
/// This class implements the `__datafusion_table_provider__` protocol, allowing
/// it to be registered with DataFusion's `SessionContext.register_table()`.
///
/// Data is NOT read until query execution time - this enables true lazy evaluation.
/// Each partition has its own factory function that is called on query execution
/// to create a fresh stream, enabling true parallelism in DataFusion.
///
/// # Note
///
/// Due to a bug in DataFusion v51.0.0's `collect()` method, use `to_arrow_table()`
/// instead for aggregation queries to ensure complete results.
///
/// # Example
///
/// ```python
/// from datafusion import SessionContext
/// from xarray_sql import LazyArrowStreamTable
/// import pyarrow as pa
///
/// # Create factories for each partition (chunk)
/// factories = [
///     lambda: pa.RecordBatchReader.from_batches(schema, batches_chunk_0),
///     lambda: pa.RecordBatchReader.from_batches(schema, batches_chunk_1),
/// ]
///
/// # Wrap factories in lazy table - NO DATA LOADED
/// table = LazyArrowStreamTable(factories, schema)
///
/// # Register with DataFusion - STILL NO DATA LOADED
/// ctx = SessionContext()
/// ctx.register_table("air", table)
///
/// # Data only loaded HERE during query execution
/// # Each partition runs in parallel with its own factory
/// # Use to_arrow_table() for aggregation queries
/// result = ctx.sql("SELECT AVG(air) FROM air").to_arrow_table()
/// ```
#[pyclass(name = "LazyArrowStreamTable")]
struct LazyArrowStreamTable {
    /// The underlying StreamingTable
    table: Arc<StreamingTable>,
}

#[pymethods]
impl LazyArrowStreamTable {
    /// Create a new LazyArrowStreamTable from stream factory functions.
    ///
    /// Args:
    ///     stream_factories: A list of callables, each returning a Python object
    ///             implementing the Arrow PyCapsule interface (`__arrow_c_stream__`).
    ///             Each factory represents one partition, enabling parallel execution.
    ///             Called on each query execution to create fresh streams.
    ///     schema: A PyArrow Schema for the table. Required since the factories
    ///             haven't been called yet.
    ///
    /// Raises:
    ///     TypeError: If the schema is not a valid PyArrow Schema.
    ///     ValueError: If stream_factories is empty.
    #[new]
    fn new(stream_factories: &Bound<'_, PyAny>, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert the PyArrow schema to Arrow schema
        use arrow::datatypes::Schema;
        use arrow::pyarrow::FromPyArrow;

        let arrow_schema = Schema::from_pyarrow_bound(schema).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
        })?;
        let schema_ref = Arc::new(arrow_schema);

        // Extract factories from the Python list
        let factories: Vec<Py<PyAny>> = stream_factories.extract().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "stream_factories must be a list of callables: {e}"
            ))
        })?;

        if factories.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stream_factories must not be empty",
            ));
        }

        // Create one partition per factory
        let partitions: Vec<Arc<dyn PartitionStream>> = factories
            .into_iter()
            .map(|factory| {
                Arc::new(PyArrowStreamPartition::new(factory, schema_ref.clone()))
                    as Arc<dyn PartitionStream>
            })
            .collect();

        // Create the StreamingTable with multiple partitions
        let table = StreamingTable::try_new(schema_ref, partitions).map_err(|e| {
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

        // Create FFI wrapper
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
