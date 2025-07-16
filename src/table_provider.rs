use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::MemTable;
use datafusion::error::DataFusionError;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::catalog::Session;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyCapsule;
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use std::path::Path;
use std::sync::Arc;
use async_trait::async_trait;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::array::Array;
use zarrs::array::data_type::DataType as ZarrDataType;
use zarrs::array_subset::ArraySubset;
use zarrs::array::chunk_grid::ChunkGrid;


/// A DataFusion TableProvider that reads from Zarr stores
#[pyclass(name = "ZarrTableProvider", module = "zarrquet", subclass)]
#[derive(Clone, Debug)]
pub(crate) struct ZarrTableProvider {
    store_path: String,
    store: Option<Arc<FilesystemStore>>,
}

impl ZarrTableProvider {
    /// Create a new ZarrTableProvider from a store path
    pub fn from_path(store_path: String) -> Result<Self, DataFusionError> {
        let store = Self::create_store(&store_path)?;
        Ok(Self {
            store_path,
            store: Some(Arc::new(store)),
        })
    }
    
    /// Create a readable storage from a path
    fn create_store(store_path: &str) -> Result<FilesystemStore, DataFusionError> {
        // For now, assume filesystem store
        // TODO: Add support for other storage backends (S3, GCS, etc.)
        let path = Path::new(store_path);
        if path.exists() {
            FilesystemStore::new(path)
                .map_err(|e| DataFusionError::External(Box::new(e)))
        } else {
            Err(DataFusionError::External(
                format!("Zarr store path does not exist: {}", store_path).into(),
            ))
        }
    }
    
    /// Get the underlying zarr store
    pub fn store(&self) -> Option<&Arc<FilesystemStore>> {
        self.store.as_ref()
    }
    
    /// Infer Arrow schema from Zarr metadata
    pub fn infer_schema(&self) -> Result<Arc<Schema>, DataFusionError> {
        let store = self.store.as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;
        
        // Read the zarr group metadata
        let group = Group::open(store.clone(), "/")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        let mut fields = Vec::new();
        
        // First, collect all child arrays
        let children = group.children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        for child in &children {
            // Try to open as an array
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                // Add coordinate dimensions as fields
                let shape = array.shape();
                for (dim_idx, &dim_size) in shape.iter().enumerate() {
                    let dim_name = format!("dim_{}", dim_idx);
                    // For now, assume dimensions are Int64 coordinates
                    fields.push(Field::new(dim_name, DataType::Int64, false));
                }
                
                // Add the data variable itself
                let arrow_type = self.zarr_type_to_arrow(array.data_type())
                    .unwrap_or(DataType::Float64); // Default fallback
                fields.push(Field::new(path_str, arrow_type, true));
                
                // For now, just handle the first array
                break;
            }
        }
        
        if fields.is_empty() {
            return Err(DataFusionError::External("No arrays found in Zarr store".into()));
        }
        
        Ok(Arc::new(Schema::new(fields)))
    }
    
    /// Convert Zarr data type to Arrow data type
    fn zarr_type_to_arrow(&self, zarr_type: &ZarrDataType) -> Result<DataType, DataFusionError> {
        match zarr_type {
            ZarrDataType::Bool => Ok(DataType::Boolean),
            ZarrDataType::Int8 => Ok(DataType::Int8),
            ZarrDataType::Int16 => Ok(DataType::Int16),
            ZarrDataType::Int32 => Ok(DataType::Int32),
            ZarrDataType::Int64 => Ok(DataType::Int64),
            ZarrDataType::UInt8 => Ok(DataType::UInt8),
            ZarrDataType::UInt16 => Ok(DataType::UInt16),
            ZarrDataType::UInt32 => Ok(DataType::UInt32),
            ZarrDataType::UInt64 => Ok(DataType::UInt64),
            ZarrDataType::Float16 => Ok(DataType::Float16),
            ZarrDataType::Float32 => Ok(DataType::Float32),
            ZarrDataType::Float64 => Ok(DataType::Float64),
            ZarrDataType::Complex64 => {
                // Complex types don't have direct Arrow equivalents
                // For now, we could represent as struct with real/imag fields
                Err(DataFusionError::External("Complex64 not yet supported".into()))
            }
            ZarrDataType::Complex128 => {
                Err(DataFusionError::External("Complex128 not yet supported".into()))
            }
            ZarrDataType::RawBits(_) => {
                Err(DataFusionError::External("RawBits not yet supported".into()))
            }
            _ => {
                Err(DataFusionError::External(format!("Unsupported zarr data type: {:?}", zarr_type).into()))
            }
        }
    }
    
    /// Get chunk grid information for the first array in the store
    pub fn get_chunk_grid(&self) -> Result<ChunkGrid, DataFusionError> {
        let store = self.store.as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;
        
        // Read the zarr group metadata
        let group = Group::open(store.clone(), "/")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        // Get the first array to work with
        let children = group.children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                return Ok(array.chunk_grid().clone());
            }
        }
        
        Err(DataFusionError::External("No arrays found in Zarr store".into()))
    }
    
    /// Get chunk indices for iterating over chunks
    pub fn get_chunk_indices(&self) -> Result<Vec<Vec<u64>>, DataFusionError> {
        let store = self.store.as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;
        
        // Read the zarr group metadata
        let group = Group::open(store.clone(), "/")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        // Get the first array to work with
        let children = group.children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                // Get the chunk grid shape
                let chunk_grid_shape = array.chunk_grid_shape()
                    .ok_or_else(|| DataFusionError::External("Failed to get chunk grid shape".into()))?;
                
                // Create an ArraySubset covering all chunks
                let chunks_subset = ArraySubset::new_with_shape(chunk_grid_shape);
                
                // Get chunk indices iterator and collect into Vec
                let chunk_indices: Vec<Vec<u64>> = chunks_subset.indices().iter().collect();
                return Ok(chunk_indices);
            }
        }
        
        Err(DataFusionError::External("No arrays found in Zarr store".into()))
    }
    
    /// Get chunk subset for a specific chunk index
    pub fn get_chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, DataFusionError> {
        let store = self.store.as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;
        
        // Read the zarr group metadata
        let group = Group::open(store.clone(), "/")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        // Get the first array to work with
        let children = group.children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                // Get chunk subset for the given chunk indices
                let chunk_subset = array.chunk_subset(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                
                return Ok(chunk_subset);
            }
        }
        
        Err(DataFusionError::External("No arrays found in Zarr store".into()))
    }
}

/// Placeholder for future ZarrExecutionPlan implementation
/// For now, we'll use MemTable in the scan method
pub struct ZarrExecutionPlan {
    schema: SchemaRef,
    zarr_provider: ZarrTableProvider,
}

#[async_trait]
impl TableProvider for ZarrTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        // Return the inferred schema, fallback to empty schema on error
        self.infer_schema()
            .unwrap_or_else(|_| Arc::new(Schema::new(vec![] as Vec<Field>)))
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>, DataFusionError> {
        // For now, indicate that we cannot push down any filters
        // TODO: Implement predicate pushdown for chunk filtering
        Ok(vec![TableProviderFilterPushDown::Unsupported; filters.len()])
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        // Get the schema for this table
        let schema = self.schema();
        
        // Apply projection if specified
        let projected_schema = if let Some(projection) = projection {
            let projected_fields: Vec<Field> = projection
                .iter()
                .map(|&i| schema.field(i).clone())
                .collect();
            Arc::new(Schema::new(projected_fields))
        } else {
            schema
        };

        // For now, use MemTable as placeholder
        // TODO: Implement proper ZarrExecutionPlan for streaming
        let empty_batches: Vec<RecordBatch> = vec![];
        let mem_table = MemTable::try_new(projected_schema, vec![empty_batches])?;
        
        // Return the MemTable's execution plan
        mem_table.scan(_state, projection, _filters, _limit).await
    }
}


#[pymethods]
impl ZarrTableProvider {
    #[new]
    pub fn new(store_path: String) -> PyResult<Self> {
        Self::from_path(store_path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let name = cr"datafusion_table_provider".into();

        // Use the ZarrTableProvider itself as the TableProvider
        let provider = FFI_TableProvider::new(Arc::new(self.clone()), false, None);

        PyCapsule::new(py, provider, Some(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_zarr_table_provider_creation() {
        // Test with a nonexistent path - this should fail
        let nonexistent_path = "/nonexistent/path/to/zarr";
        let result = ZarrTableProvider::from_path(nonexistent_path.to_string());
        assert!(result.is_err());
        
        // Test with a valid directory but empty - this should succeed
        let temp_dir = std::env::temp_dir().join("test_zarr_store");
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        let store_path = temp_dir.to_string_lossy().to_string();
        let result = ZarrTableProvider::from_path(store_path);
        assert!(result.is_ok());
        
        // Test schema inference on empty store should fail
        let provider = result.unwrap();
        let schema_result = provider.infer_schema();
        assert!(schema_result.is_err());
        
        // Clean up
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
    
    #[test]
    fn test_zarr_type_to_arrow_conversion() {
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        // Test basic type conversions
        assert_eq!(
            provider.zarr_type_to_arrow(&ZarrDataType::Float64).unwrap(),
            DataType::Float64
        );
        assert_eq!(
            provider.zarr_type_to_arrow(&ZarrDataType::Int32).unwrap(),
            DataType::Int32
        );
        assert_eq!(
            provider.zarr_type_to_arrow(&ZarrDataType::Bool).unwrap(),
            DataType::Boolean
        );
        
        // Test unsupported types
        assert!(provider.zarr_type_to_arrow(&ZarrDataType::Complex64).is_err());
    }
    
    #[test]
    fn test_chunk_discovery_methods() {
        // Create a temporary test directory
        let temp_dir = std::env::temp_dir().join("test_zarr_chunk_store");
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        let store_path = temp_dir.to_string_lossy().to_string();
        let provider = ZarrTableProvider::from_path(store_path).unwrap();
        
        // Test with empty store - should return error
        let chunk_grid_result = provider.get_chunk_grid();
        assert!(chunk_grid_result.is_err());
        
        let chunk_indices_result = provider.get_chunk_indices();
        assert!(chunk_indices_result.is_err());
        
        let chunk_subset_result = provider.get_chunk_subset(&[0, 0]);
        assert!(chunk_subset_result.is_err());
        
        // Clean up
        std::fs::remove_dir_all(&temp_dir).unwrap();
    }
    
    #[test]
    fn test_chunk_methods_with_valid_provider() {
        // This test verifies that the chunk methods don't panic and have correct signatures
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        // All methods should return errors when no store is available
        assert!(provider.get_chunk_grid().is_err());
        assert!(provider.get_chunk_indices().is_err());
        assert!(provider.get_chunk_subset(&[0, 0]).is_err());
    }
    
    #[test]
    fn test_table_provider_trait() {
        // Test that ZarrTableProvider implements TableProvider correctly
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        // Test basic TableProvider methods
        assert_eq!(provider.table_type(), TableType::Base);
        
        // Schema should return empty schema when no store available
        let schema = provider.schema();
        assert_eq!(schema.fields().len(), 0);
        
        // Filter pushdown should return unsupported for all filters
        let filters = vec![];
        let pushdown_result = provider.supports_filters_pushdown(&filters);
        assert!(pushdown_result.is_ok());
        assert_eq!(pushdown_result.unwrap().len(), 0);
    }
    
    #[tokio::test]
    async fn test_table_provider_scan() {
        // Test scan method with empty store
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        // Create a mock session state
        use datafusion::execution::context::SessionConfig;
        use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeConfig};
        use datafusion::execution::context::SessionState;
        
        let config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime = Arc::new(RuntimeEnv::new(runtime_config).unwrap());
        let state = SessionState::new_with_config_rt(config, runtime);
        
        // Test scan with empty filters
        let filters = vec![];
        let scan_result = provider.scan(&state, None, &filters, None).await;
        assert!(scan_result.is_ok());
        
        // The execution plan should be valid
        let execution_plan = scan_result.unwrap();
        assert_eq!(execution_plan.schema().fields().len(), 0);
    }
    
    #[tokio::test]
    async fn test_table_scanning_with_predicates() {
        // Test that the table provider can handle basic scanning with predicates
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        // Create a mock session state
        use datafusion::execution::context::SessionConfig;
        use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeConfig};
        use datafusion::execution::context::SessionState;
        use datafusion::logical_expr::lit;
        
        let config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime = Arc::new(RuntimeEnv::new(runtime_config).unwrap());
        let state = SessionState::new_with_config_rt(config, runtime);
        
        // Create a simple filter expression
        let filter_expr = lit(true); // Simple boolean literal
        let filters = vec![filter_expr];
        
        // Test scan with filters
        let scan_result = provider.scan(&state, None, &filters, None).await;
        assert!(scan_result.is_ok());
        
        // Verify that filter pushdown returns unsupported for all filters
        let filter_refs: Vec<&Expr> = filters.iter().collect();
        let pushdown_result = provider.supports_filters_pushdown(&filter_refs);
        assert!(pushdown_result.is_ok());
        let pushdown_decisions = pushdown_result.unwrap();
        assert_eq!(pushdown_decisions.len(), 1);
        assert_eq!(pushdown_decisions[0], TableProviderFilterPushDown::Unsupported);
    }
    
    #[tokio::test]
    async fn test_table_scanning_with_projection() {
        // Test that the table provider can handle projection
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        // Create a mock session state
        use datafusion::execution::context::SessionConfig;
        use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeConfig};
        use datafusion::execution::context::SessionState;
        
        let config = SessionConfig::new();
        let runtime_config = RuntimeConfig::new();
        let runtime = Arc::new(RuntimeEnv::new(runtime_config).unwrap());
        let state = SessionState::new_with_config_rt(config, runtime);
        
        // Test scan with projection - should work even with empty schema
        let projection = vec![]; // Empty projection
        let filters = vec![];
        let scan_result = provider.scan(&state, Some(&projection), &filters, None).await;
        assert!(scan_result.is_ok());
        
        // The execution plan should have empty schema for empty projection
        let execution_plan = scan_result.unwrap();
        assert_eq!(execution_plan.schema().fields().len(), 0);
    }
}