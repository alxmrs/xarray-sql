// Licensed under the Apache License, Version 2.0
//!
//! Lazy Arrow stream table provider for xarray-sql.
//!
//! This module provides `LazyArrowStreamTable`, which wraps a Python object
//! implementing `__arrow_c_stream__` and exposes it as a DataFusion TableProvider
//! via the `__datafusion_table_provider__` protocol.
//!
//! The key feature is **lazy evaluation**: data is not read from the Python stream
//! until query execution time (during `collect()`), not at registration time.

use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use datafusion::catalog::streaming::StreamingTable;
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::memory::MemoryStream;
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
        // Call the factory to get a fresh stream for this execution
        let batches: Vec<RecordBatch> = Python::attach(|py| {
            // Call the factory to get a fresh stream
            let stream_result = self.stream_factory.call0(py);

            match stream_result {
                Ok(stream_obj) => {
                    let bound = stream_obj.bind(py);

                    match ArrowArrayStreamReader::from_pyarrow_bound(bound) {
                        Ok(reader) => {
                            // Collect batches, propagating errors as warnings
                            // In streaming context, we can't easily return errors,
                            // so we log and skip failed batches
                            reader
                                .filter_map(|result| match result {
                                    Ok(batch) => Some(batch),
                                    Err(e) => {
                                        eprintln!("Warning: Failed to read batch: {e}");
                                        None
                                    }
                                })
                                .collect()
                        }
                        Err(e) => {
                            eprintln!("Warning: Failed to create stream reader: {e}");
                            vec![]
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to call stream factory: {e}");
                    vec![]
                }
            }
        });

        Box::pin(
            MemoryStream::try_new(batches, Arc::clone(&self.schema), None)
                .expect("MemoryStream creation should not fail with valid schema"),
        )
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
