// Licensed under the Apache License, Version 2.0
//!
//! Lazy Arrow stream table provider for xarray-sql with projection pushdown.
//!
//! This module provides `LazyArrowStreamTable`, which wraps a Python factory
//! function and exposes it as a DataFusion TableProvider with support for
//! column projection pushdown.
//!
//! Key features:
//! - **Lazy evaluation**: Data is not read until query execution time
//! - **Projection pushdown**: Only columns needed by the query are read
//! - **Multiple queries**: Same table can be queried multiple times

use std::any::Any;
use std::ffi::c_void;
use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use datafusion::catalog::Session;
use datafusion::common::Result as DFResult;
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList};
use tokio::runtime::Handle;

/// A partition stream that wraps a Python factory function with projection support.
///
/// The factory is called lazily with a projection (list of column indices)
/// on each `execute()` invocation.
struct ProjectedPyArrowStreamPartition {
    /// The projected schema (only columns that were requested)
    projected_schema: SchemaRef,
    /// Python callable that accepts (projection: Optional[List[int]]) and returns a stream
    stream_factory: Py<PyAny>,
    /// The projection to apply (column indices), or None for all columns
    projection: Option<Vec<usize>>,
}

impl ProjectedPyArrowStreamPartition {
    fn new(
        stream_factory: Py<PyAny>,
        projected_schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> Self {
        Self {
            projected_schema,
            stream_factory,
            projection,
        }
    }
}

impl Debug for ProjectedPyArrowStreamPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectedPyArrowStreamPartition")
            .field("schema", &self.projected_schema)
            .field("projection", &self.projection)
            .finish()
    }
}

impl PartitionStream for ProjectedPyArrowStreamPartition {
    fn schema(&self) -> &SchemaRef {
        &self.projected_schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);

        // Call the factory with the projection to get a fresh stream
        let batches: Vec<RecordBatch> = Python::with_gil(|py| {
            // Convert projection to Python list or None
            let py_projection = match &projection {
                Some(indices) => {
                    let list = PyList::new(py, indices.iter().map(|&i| i as i64)).unwrap();
                    list.into_any().unbind()
                }
                None => py.None(),
            };

            // Call factory with projection argument
            let stream_result = self.stream_factory.call1(py, (py_projection,));

            match stream_result {
                Ok(stream_obj) => {
                    let bound = stream_obj.bind(py);

                    match ArrowArrayStreamReader::from_pyarrow_bound(bound) {
                        Ok(reader) => reader
                            .filter_map(|result| match result {
                                Ok(batch) => Some(batch),
                                Err(e) => {
                                    eprintln!("Warning: Failed to read batch: {e}");
                                    None
                                }
                            })
                            .collect(),
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
            MemoryStream::try_new(batches, projected_schema, None)
                .expect("MemoryStream creation should not fail with valid schema"),
        )
    }
}

/// A streaming execution plan that supports projection pushdown.
#[derive(Debug)]
struct ProjectedStreamingExec {
    projected_schema: SchemaRef,
    partitions: Vec<Arc<dyn PartitionStream>>,
    properties: PlanProperties,
}

impl ProjectedStreamingExec {
    fn new(projected_schema: SchemaRef, partitions: Vec<Arc<dyn PartitionStream>>) -> Self {
        // Create properties for this execution plan
        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&projected_schema)),
            Partitioning::UnknownPartitioning(partitions.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            projected_schema,
            partitions,
            properties,
        }
    }
}

impl DisplayAs for ProjectedStreamingExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose | DisplayFormatType::TreeRender => {
                write!(f, "ProjectedStreamingExec: partitions={}", self.partitions.len())
            }
        }
    }
}

impl ExecutionPlan for ProjectedStreamingExec {
    fn name(&self) -> &str {
        "ProjectedStreamingExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.projected_schema)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if partition >= self.partitions.len() {
            return Err(datafusion::error::DataFusionError::Internal(format!(
                "Partition {partition} out of bounds (have {})",
                self.partitions.len()
            )));
        }
        Ok(self.partitions[partition].execute(context))
    }
}

/// Internal table provider with projection pushdown support.
///
/// This wraps a Python factory and implements DataFusion's TableProvider trait.
struct ProjectedTableProvider {
    /// Full schema of the table (all columns)
    full_schema: SchemaRef,
    /// Python callable: (projection: Optional[List[int]]) -> ArrowStream
    stream_factory: Py<PyAny>,
}

impl Debug for ProjectedTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjectedTableProvider")
            .field("schema", &self.full_schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for ProjectedTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.full_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let projection = projection.cloned();
        let full_schema = Arc::clone(&self.full_schema);
        // Clone the Python object reference (requires GIL)
        let stream_factory = Python::with_gil(|py| self.stream_factory.clone_ref(py));

        // Compute the projected schema
        let projected_schema = match &projection {
            Some(indices) => {
                let fields: Vec<_> = indices
                    .iter()
                    .map(|&i| full_schema.field(i).clone())
                    .collect();
                Arc::new(arrow::datatypes::Schema::new(fields))
            }
            None => full_schema,
        };

        // Create a partition stream with the projection
        let partition = Arc::new(ProjectedPyArrowStreamPartition::new(
            stream_factory,
            Arc::clone(&projected_schema),
            projection,
        ));

        Ok(Arc::new(ProjectedStreamingExec::new(
            projected_schema,
            vec![partition],
        )) as Arc<dyn ExecutionPlan>)
    }
}

/// A lazy table provider that wraps a Python stream factory with projection pushdown.
///
/// This class implements the `__datafusion_table_provider__` protocol, allowing
/// it to be registered with DataFusion's `SessionContext.register_table_provider()`.
///
/// Key features:
/// - Data is NOT read until query execution time (lazy evaluation)
/// - Only columns needed by the query are read (projection pushdown)
/// - The table can be queried multiple times with different column needs
///
/// # Example
///
/// ```python
/// from datafusion import SessionContext
/// from xarray_sql import LazyArrowStreamTable
///
/// # Create a factory that accepts projection and returns a stream
/// def make_reader(projection=None):
///     # projection is a list of column indices, or None for all columns
///     return XarrayRecordBatchReader(ds, chunks={'time': 240}, projection=projection)
///
/// # Get schema (all columns)
/// schema = make_reader().schema
///
/// # Create table - NO DATA LOADED
/// table = LazyArrowStreamTable(make_reader, schema)
///
/// # Register with DataFusion
/// ctx = SessionContext()
/// ctx.register_table_provider("weather", table)
///
/// # Query only temperature - only temperature column is read!
/// result = ctx.sql("SELECT AVG(temperature) FROM weather").collect()
///
/// # Query humidity - only humidity column is read!
/// result2 = ctx.sql("SELECT AVG(humidity) FROM weather").collect()
/// ```
#[pyclass(name = "LazyArrowStreamTable")]
struct LazyArrowStreamTable {
    /// The underlying table provider with projection support
    provider: Arc<ProjectedTableProvider>,
}

#[pymethods]
impl LazyArrowStreamTable {
    /// Create a new LazyArrowStreamTable from a stream factory function.
    ///
    /// Args:
    ///     stream_factory: A callable that accepts an optional projection
    ///             (list of column indices) and returns a Python object
    ///             implementing the Arrow PyCapsule interface (`__arrow_c_stream__`).
    ///             Signature: (projection: Optional[List[int]]) -> ArrowStream
    ///     schema: A PyArrow Schema for the table (all columns).
    ///
    /// The factory will be called at query execution time with the projection
    /// that DataFusion determines is needed for the query.
    #[new]
    fn new(stream_factory: &Bound<'_, PyAny>, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        use arrow::datatypes::Schema;
        use arrow::pyarrow::FromPyArrow;

        let arrow_schema = Schema::from_pyarrow_bound(schema).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
        })?;
        let schema_ref = Arc::new(arrow_schema);

        let provider = Arc::new(ProjectedTableProvider {
            full_schema: schema_ref,
            stream_factory: stream_factory.clone().unbind(),
        });

        Ok(Self { provider })
    }

    /// Returns a PyCapsule implementing the DataFusion TableProvider FFI.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let provider: Arc<dyn TableProvider + Send> = self.provider.clone();
        let runtime = Handle::try_current().ok();

        let ffi_provider = FFI_TableProvider::new(provider, false, runtime);

        let name = CString::new("datafusion_table_provider").unwrap();

        PyCapsule::new_with_destructor(
            py,
            ffi_provider,
            Some(name),
            |_provider: FFI_TableProvider, _context: *mut c_void| {},
        )
    }

    /// Get the schema of the table as a PyArrow Schema.
    fn schema(&self, py: Python<'_>) -> PyResult<PyObject> {
        use arrow::pyarrow::ToPyArrow;
        self.provider.full_schema.to_pyarrow(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "LazyArrowStreamTable(schema={:?})",
            self.provider.full_schema
        )
    }
}

/// Python module initialization
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LazyArrowStreamTable>()?;
    Ok(())
}
