use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyCapsule;
use std::sync::Arc;
use std::ffi::CString;
use arrow_zarr::table::ZarrTable;
use zarrs_object_store::AsyncObjectStore;
use object_store::local::LocalFileSystem;
use datafusion_ffi::table_provider::FFI_TableProvider;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use arrow_schema::{SchemaRef, Schema, Field, Fields, DataType};
use std::path::PathBuf;
use tokio::runtime::Runtime;
use zarrs::array::Array;
use zarrs_metadata::ArrayMetadata;
use zarrs_metadata::v3::array::data_type::DataTypeMetadataV3;
use zarrs_storage::{AsyncReadableListableStorageTraits, StorePrefix};

// Taken from arrow-zarr (private)
/// helpers to infer the schema from a zarr store, which involves reading
/// directory names and reading some metadata, so it's a bit trickier than
/// e.g. get a schema from a parquet file.
fn get_schema_type(value: &DataTypeMetadataV3) -> DfResult<DataType> {
    match value {
        DataTypeMetadataV3::Bool => Ok(DataType::Boolean),
        DataTypeMetadataV3::UInt8 => Ok(DataType::UInt8),
        DataTypeMetadataV3::UInt16 => Ok(DataType::UInt16),
        DataTypeMetadataV3::UInt32 => Ok(DataType::UInt32),
        DataTypeMetadataV3::UInt64 => Ok(DataType::UInt64),
        DataTypeMetadataV3::Int8 => Ok(DataType::Int8),
        DataTypeMetadataV3::Int16 => Ok(DataType::Int16),
        DataTypeMetadataV3::Int32 => Ok(DataType::Int32),
        DataTypeMetadataV3::Int64 => Ok(DataType::Int64),
        DataTypeMetadataV3::Float32 => Ok(DataType::Float32),
        DataTypeMetadataV3::Float64 => Ok(DataType::Float64),
        DataTypeMetadataV3::String => Ok(DataType::Utf8),
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported type {value} from zarr metadata"
        ))),
    }
}


// Taken from arrow-zarr (private)
async fn infer_schema(store: Arc<dyn AsyncReadableListableStorageTraits>) -> DfResult<Schema> {
    let dirs = store
        .list_dir(&StorePrefix::new("").map_err(|e| DataFusionError::External(Box::new(e)))?)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let prefixes = dirs.prefixes();
    let mut fields = Vec::with_capacity(prefixes.len());

    for prefix in prefixes {
        let field_name = prefix
            .as_str()
            .strip_suffix("/")
            .ok_or(DataFusionError::Execution(
                "Invalid directory name in zarr store".into(),
            ))?;

        let arr = Array::async_open(store.clone(), &("/".to_owned() + field_name))
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let meta = match arr.metadata() {
            ArrayMetadata::V3(meta) => Ok(meta),
            _ => Err(DataFusionError::Execution(
                "Only Zarr v3 metadata is supported".into(),
            )),
        }?;

        fields.push(Field::new(
            field_name,
            get_schema_type(&meta.data_type)?,
            true,
        ));
    }

    Ok(Schema::new(Fields::from(fields)))
}

/// A simple adapter that wraps arrow-zarr's ZarrTable for Python bindings
#[pyclass(name = "ZarrTableProvider")]
#[derive(Clone, Debug)]
pub struct ZarrTableProvider {
    #[allow(dead_code)]
    store_path: String, // Kept for debugging/logging purposes
    inner: Arc<ZarrTable>,
}

impl ZarrTableProvider {
    // TODO(alxmrs): Consider contributing somethign like this upstream.
    /// Create a new ZarrTableProvider from a store path using arrow-zarr's schema inference
    /// This mimics ZarrTableFactory::create but avoids the command execution overhead
    pub fn from_path(store_path: String) -> Result<Self, DataFusionError> {
        // Create a tokio runtime to handle the async schema inference
        let rt = Runtime::new().map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        rt.block_on(async {
            // 1. Create store (same as ZarrTableFactory)
            let p = PathBuf::from(&store_path);
            let f = LocalFileSystem::new_with_prefix(p)
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let store = Arc::new(AsyncObjectStore::new(f));

            let inferred_schema = infer_schema(store.clone()).await?;

            // 3. Create ZarrTable with minimal schema 
            let zarr_table = ZarrTable::new(Arc::new(inferred_schema), store);
            
            Ok(Self {
                store_path,
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
        // Use arrow-zarr's schema inference instead of requiring schema from Python
        Self::from_path(store_path)
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