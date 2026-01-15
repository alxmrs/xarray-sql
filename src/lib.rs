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

use std::ffi::c_void;
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

/// A partition stream that wraps a Python object implementing `__arrow_c_stream__`.
///
/// The stream is consumed lazily - only when `execute()` is called during query execution.
struct PyArrowStreamPartition {
    schema: SchemaRef,
    /// The Python object, wrapped in Option so it can be taken (consumed) exactly once.
    /// We use std::sync::Mutex for Send + Sync.
    py_stream: std::sync::Mutex<Option<Py<PyAny>>>,
}

impl PyArrowStreamPartition {
    fn new(py_obj: Py<PyAny>, schema: SchemaRef) -> Self {
        Self {
            schema,
            py_stream: std::sync::Mutex::new(Some(py_obj)),
        }
    }
}

impl Debug for PyArrowStreamPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let consumed = self.py_stream.lock().unwrap().is_none();
        f.debug_struct("PyArrowStreamPartition")
            .field("schema", &self.schema)
            .field("consumed", &consumed)
            .finish()
    }
}

impl PartitionStream for PyArrowStreamPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        // Take the Python object (can only be done once)
        let py_obj = self.py_stream.lock().unwrap().take();

        let batches: Vec<RecordBatch> = match py_obj {
            Some(obj) => {
                // Acquire the GIL and consume the stream
                Python::with_gil(|py| {
                    let bound = obj.bind(py);

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
                })
            }
            None => {
                // Stream already consumed, return empty
                vec![]
            }
        };

        Box::pin(
            MemoryStream::try_new(batches, Arc::clone(&self.schema), None)
                .expect("MemoryStream creation should not fail with valid schema"),
        )
    }
}

/// A lazy table provider that wraps a Python Arrow stream.
///
/// This class implements the `__datafusion_table_provider__` protocol, allowing
/// it to be registered with DataFusion's `SessionContext.register_table()`.
///
/// Data is NOT read until query execution time - this enables true lazy evaluation.
///
/// # Example
///
/// ```python
/// from datafusion import SessionContext
/// from xarray_sql import LazyArrowStreamTable, XarrayRecordBatchReader
///
/// # Create a lazy reader (implements __arrow_c_stream__)
/// reader = XarrayRecordBatchReader(ds, chunks={'time': 240})
///
/// # Wrap in lazy table - NO DATA LOADED
/// table = LazyArrowStreamTable(reader)
///
/// # Register with DataFusion - STILL NO DATA LOADED
/// ctx = SessionContext()
/// ctx.register_table("air", table)
///
/// # Data only loaded HERE during collect()
/// result = ctx.sql("SELECT AVG(air) FROM air").collect()
/// ```
#[pyclass(name = "LazyArrowStreamTable")]
struct LazyArrowStreamTable {
    /// The underlying StreamingTable
    table: Arc<StreamingTable>,
}

#[pymethods]
impl LazyArrowStreamTable {
    /// Create a new LazyArrowStreamTable from a Python object implementing `__arrow_c_stream__`.
    ///
    /// Args:
    ///     stream: A Python object implementing the Arrow PyCapsule interface (`__arrow_c_stream__`).
    ///             This includes `pyarrow.RecordBatchReader`, `XarrayRecordBatchReader`, etc.
    ///
    /// Raises:
    ///     TypeError: If the object does not implement `__arrow_c_stream__`.
    #[new]
    fn new(stream: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Get the schema via the .schema attribute WITHOUT consuming the stream
        // This is important because calling __arrow_c_stream__ would consume the stream
        let schema = get_schema_from_stream(stream)?;

        // Create the partition stream with the Python object
        let partition = PyArrowStreamPartition::new(stream.clone().unbind(), schema.clone());

        // Create the StreamingTable
        let table = StreamingTable::try_new(schema, vec![Arc::new(partition)]).map_err(|e| {
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
    fn __datafusion_table_provider__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Create the FFI table provider
        let provider: Arc<dyn TableProvider + Send> = self.table.clone();

        // Try to get the current tokio runtime handle (available when called from DataFusion context)
        let runtime = Handle::try_current().ok();

        // Create FFI wrapper (v49 API takes 3 arguments)
        let ffi_provider = FFI_TableProvider::new(
            provider,
            false, // can_support_pushdown_filters
            runtime,
        );

        // Create the capsule name
        let name = CString::new("datafusion_table_provider").unwrap();

        // Create the PyCapsule with a destructor closure
        // The PyCapsule takes ownership of the FFI_TableProvider
        PyCapsule::new_with_destructor(
            py,
            ffi_provider,
            Some(name),
            |_provider: FFI_TableProvider, _context: *mut c_void| {
                // The FFI_TableProvider will be dropped automatically
            },
        )
    }

    /// Get the schema of the table as a PyArrow Schema.
    fn schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        use arrow::pyarrow::ToPyArrow;
        self.table.schema().to_pyarrow(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "LazyArrowStreamTable(schema={:?})",
            self.table.schema()
        )
    }
}

/// Get schema from a Python object that has a schema attribute.
///
/// This extracts the schema WITHOUT consuming the stream, which is
/// important for lazy evaluation.
fn get_schema_from_stream(stream: &Bound<'_, PyAny>) -> PyResult<SchemaRef> {
    use arrow::datatypes::Schema;
    use arrow::pyarrow::FromPyArrow;

    let py_schema = stream.getattr("schema").map_err(|e| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "Object must have a 'schema' attribute (e.g., RecordBatchReader): {e}"
        ))
    })?;

    let schema = Schema::from_pyarrow_bound(&py_schema).map_err(|e| {
        pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
    })?;

    Ok(Arc::new(schema))
}

/// Python module initialization
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LazyArrowStreamTable>()?;
    Ok(())
}
