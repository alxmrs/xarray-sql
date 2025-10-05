use arrow_schema::SchemaRef;
use arrow_zarr::table::ZarrTable;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::CString;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// A simple adapter that wraps arrow-zarr's ZarrTable for Python bindings
#[pyclass(name = "ZarrTableProvider")]
#[derive(Clone, Debug)]
pub struct ZarrTableProvider {
    inner: Arc<ZarrTable>,
}

impl ZarrTableProvider {
    /// Create a new ZarrTableProvider from a store path using arrow-zarr's ZarrTable
    pub fn from_path(store_path: String) -> Result<Self, DataFusionError> {
        // Create a tokio runtime to handle the async table creation
        let rt = Runtime::new().map_err(|e| DataFusionError::External(Box::new(e)))?;

        rt.block_on(async {
            println!("Creating ZarrTable from path: {}", store_path);
            let zarr_table = ZarrTable::from_path(store_path.clone()).await;
            println!("Schema: {:?}", zarr_table.schema());
            Ok(Self {
                inner: Arc::new(zarr_table),
            })
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
    pub fn new(store_path: String) -> PyResult<Self> {
        Self::from_path(store_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn __datafusion_table_provider__<'a>(
        &self,
        py: Python<'a>,
    ) -> PyResult<Bound<'a, PyCapsule>> {
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
