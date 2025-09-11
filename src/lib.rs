use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyCapsule;
use arrow::pyarrow::FromPyArrow;
use std::sync::Arc;
use std::ffi::CString;
use arrow_zarr::table::ZarrTable;
use zarrs_object_store::AsyncObjectStore;
use object_store::local::LocalFileSystem;
use datafusion_ffi::table_provider::FFI_TableProvider;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use arrow_schema::SchemaRef;

/// A simple adapter that wraps arrow-zarr's ZarrTable for Python bindings
#[pyclass(name = "ZarrTableProvider")]
#[derive(Clone, Debug)]
pub struct ZarrTableProvider {
    #[allow(dead_code)]
    store_path: String, // Kept for debugging/logging purposes
    inner: Arc<ZarrTable>,
}

impl ZarrTableProvider {
    /// Create a new ZarrTableProvider from a store path and schema
    pub fn new_with_schema(store_path: String, schema: SchemaRef) -> Result<Self, DataFusionError> {
        // Create file store following the pattern from arrow-zarr tests
        let filesystem = LocalFileSystem::new_with_prefix(&store_path)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let store = Arc::new(AsyncObjectStore::new(filesystem));
        
        // Create ZarrTable with schema and storage
        let zarr_table = ZarrTable::new(schema, store);
        
        Ok(Self { 
            store_path,
            inner: Arc::new(zarr_table),
        })
    }
}

// Implement TableProvider by delegating to the inner ZarrTable
#[async_trait]
impl TableProvider for ZarrTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        // Delegate to the inner ZarrTable
        self.inner.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>, DataFusionError> {
        self.inner.supports_filters_pushdown(filters)
    }
}

#[pymethods]
impl ZarrTableProvider {
    #[new]
    pub fn new(store_path: String, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert PyArrow schema to Arrow schema
        let schema_ref: SchemaRef = Arc::new(arrow_schema::Schema::from_pyarrow_bound(schema)?);
        Self::new_with_schema(store_path, schema_ref)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn __datafusion_table_provider__<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyCapsule>> {
        let name = CString::new("datafusion_table_provider").unwrap();
        let provider = FFI_TableProvider::new(Arc::new(self.clone()), false, None);
        PyCapsule::new(py, provider, Some(name))
    }
}

#[pymodule]
fn zarrquet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZarrTableProvider>()?;
    Ok(())
}