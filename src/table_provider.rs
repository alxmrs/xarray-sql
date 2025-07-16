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
use arrow_array::{Int64Array, Float64Array, BooleanArray, StringArray};
use arrow_array::builder::{Int64Builder, Float64Builder, BooleanBuilder, StringBuilder};
use std::collections::HashMap;

/// Trait for types that can be converted to Arrow arrays with minimal copying
trait ToArrowArray: Clone + Sized {
    type ArrowArray: arrow_array::Array + 'static;
    
    /// Convert a flattened ndarray to an Arrow array with zero-copy when possible
    fn to_arrow_array(
        flat_data: &ndarray::ArrayBase<ndarray::CowRepr<'_, Self>, ndarray::Dim<[usize; 1]>>,
    ) -> Vec<Self>;
    
    /// Get the Arrow DataType for this type
    fn arrow_data_type() -> DataType;
    
    /// Create an Arrow array from a vector
    fn from_vec(data: Vec<Self>) -> Arc<Self::ArrowArray>;
}

impl ToArrowArray for f64 {
    type ArrowArray = Float64Array;
    
    fn to_arrow_array(
        flat_data: &ndarray::ArrayBase<ndarray::CowRepr<'_, Self>, ndarray::Dim<[usize; 1]>>,
    ) -> Vec<Self> {
        if flat_data.is_standard_layout() {
            flat_data.as_slice().unwrap().to_vec()
        } else {
            flat_data.iter().cloned().collect()
        }
    }
    
    fn arrow_data_type() -> DataType {
        DataType::Float64
    }
    
    fn from_vec(data: Vec<Self>) -> Arc<Self::ArrowArray> {
        Arc::new(Float64Array::from(data))
    }
}

impl ToArrowArray for f32 {
    type ArrowArray = Float64Array;
    
    fn to_arrow_array(
        flat_data: &ndarray::ArrayBase<ndarray::CowRepr<'_, Self>, ndarray::Dim<[usize; 1]>>,
    ) -> Vec<Self> {
        if flat_data.is_standard_layout() {
            flat_data.as_slice().unwrap().to_vec()
        } else {
            flat_data.iter().cloned().collect()
        }
    }
    
    fn arrow_data_type() -> DataType {
        DataType::Float64  // Convert f32 to f64 for consistency
    }
    
    fn from_vec(data: Vec<Self>) -> Arc<Self::ArrowArray> {
        let f64_data: Vec<f64> = data.iter().map(|&x| x as f64).collect();
        Arc::new(Float64Array::from(f64_data))
    }
}

impl ToArrowArray for i64 {
    type ArrowArray = Int64Array;
    
    fn to_arrow_array(
        flat_data: &ndarray::ArrayBase<ndarray::CowRepr<'_, Self>, ndarray::Dim<[usize; 1]>>,
    ) -> Vec<Self> {
        if flat_data.is_standard_layout() {
            flat_data.as_slice().unwrap().to_vec()
        } else {
            flat_data.iter().cloned().collect()
        }
    }
    
    fn arrow_data_type() -> DataType {
        DataType::Int64
    }
    
    fn from_vec(data: Vec<Self>) -> Arc<Self::ArrowArray> {
        Arc::new(Int64Array::from(data))
    }
}

impl ToArrowArray for i32 {
    type ArrowArray = arrow_array::Int32Array;
    
    fn to_arrow_array(
        flat_data: &ndarray::ArrayBase<ndarray::CowRepr<'_, Self>, ndarray::Dim<[usize; 1]>>,
    ) -> Vec<Self> {
        if flat_data.is_standard_layout() {
            flat_data.as_slice().unwrap().to_vec()
        } else {
            flat_data.iter().cloned().collect()
        }
    }
    
    fn arrow_data_type() -> DataType {
        DataType::Int32
    }
    
    fn from_vec(data: Vec<Self>) -> Arc<Self::ArrowArray> {
        Arc::new(arrow_array::Int32Array::from(data))
    }
}


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
    
    /// Transform a Zarr chunk into a RecordBatch
    pub fn chunk_to_record_batch(
        &self,
        chunk_indices: &[u64],
    ) -> Result<RecordBatch, DataFusionError> {
        let store = self.store.as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;
        
        // Get the zarr group
        let group = Group::open(store.clone(), "/")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        // Get the first array (for now, we'll support single arrays)
        let children = group.children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                return self.array_chunk_to_record_batch(&array, chunk_indices, &path_str);
            }
        }
        
        Err(DataFusionError::External("No arrays found in Zarr store".into()))
    }
    
    /// Convert a specific array chunk to RecordBatch
    fn array_chunk_to_record_batch(
        &self,
        array: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        array_name: &str,
    ) -> Result<RecordBatch, DataFusionError> {
        // Get the chunk subset
        let chunk_subset = array.chunk_subset(chunk_indices)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        // Handle different data types using the generic implementation
        match array.data_type() {
            ZarrDataType::Float64 => {
                let chunk_data = array.retrieve_chunk_ndarray::<f64>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            ZarrDataType::Float32 => {
                let chunk_data = array.retrieve_chunk_ndarray::<f32>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            ZarrDataType::Int64 => {
                let chunk_data = array.retrieve_chunk_ndarray::<i64>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            ZarrDataType::Int32 => {
                let chunk_data = array.retrieve_chunk_ndarray::<i32>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            other => Err(DataFusionError::External(
                format!("Unsupported zarr data type for chunk reading: {:?}", other).into()
            ))
        }
    }
    
    /// Generate coordinate arrays for n-dimensional data
    fn generate_coordinates(
        &self,
        shape: &[usize],
        chunk_start: &[u64],
        total_elements: usize,
    ) -> Vec<Vec<i64>> {
        let ndim = shape.len();
        let mut coord_arrays: Vec<Vec<i64>> = (0..ndim)
            .map(|_| Vec::with_capacity(total_elements))
            .collect();
        
        // Generate coordinates efficiently using mathematical approach
        for flat_idx in 0..total_elements {
            let mut coords = vec![0u64; ndim];
            let mut remainder = flat_idx;
            
            // Convert flat index to multi-dimensional coordinates
            for dim in (0..ndim).rev() {
                coords[dim] = (remainder % shape[dim]) as u64;
                remainder /= shape[dim];
            }
            
            // Add global coordinates (chunk start + local coordinates)
            for (dim, &coord) in coords.iter().enumerate() {
                let global_coord = chunk_start[dim] + coord;
                coord_arrays[dim].push(global_coord as i64);
            }
        }
        
        coord_arrays
    }
    
    /// Generic method to convert ndarray to Arrow RecordBatch in tabular format
    fn ndarray_to_record_batch<T>(
        &self,
        data: ndarray::ArrayD<T>,
        chunk_subset: &ArraySubset,
        array_name: &str,
    ) -> Result<RecordBatch, DataFusionError>
    where
        T: ToArrowArray + Clone,
    {
        let original_shape = data.shape();
        let ndim = original_shape.len();
        let total_elements = data.len();
        
        // Reshape to 1D for efficient processing (zero-copy when possible)
        let flat_data = data.to_shape(total_elements)
            .map_err(|e| DataFusionError::External(format!("Failed to reshape array: {}", e).into()))?;
        
        // Get the chunk start indices from the subset
        let chunk_start = chunk_subset.start();
        
        // Generate coordinate arrays efficiently
        let coord_arrays = self.generate_coordinates(original_shape, chunk_start, total_elements);
        
        // Create Arrow arrays efficiently
        let mut arrays: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
        
        // Add coordinate columns from pre-allocated vectors
        for coord_array in coord_arrays {
            let array = Arc::new(Int64Array::from(coord_array));
            arrays.push(array);
        }
        
        // Create data column from ndarray slice (zero-copy when possible)
        let data_vec = T::to_arrow_array(&flat_data);
        let data_array = T::from_vec(data_vec);
        arrays.push(data_array as Arc<dyn arrow_array::Array>);
        
        // Create the schema
        let mut fields = Vec::new();
        for dim_idx in 0..ndim {
            fields.push(Field::new(format!("dim_{}", dim_idx), DataType::Int64, false));
        }
        fields.push(Field::new(array_name, T::arrow_data_type(), true));
        
        let schema = Arc::new(Schema::new(fields));
        
        // Create the RecordBatch
        RecordBatch::try_new(schema, arrays)
            .map_err(|e| DataFusionError::External(Box::new(e)))
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
    
    #[test]
    fn test_ndarray_to_record_batch_transformation() {
        use ndarray::ArrayD;
        
        // Create a test provider
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        // Create a simple 2D ndarray for testing
        let data = ArrayD::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        // Create a mock chunk subset
        let chunk_start = vec![0u64, 0u64];
        let chunk_shape = vec![2u64, 3u64];
        let chunk_subset = ArraySubset::new_with_start_shape(chunk_start, chunk_shape)
            .expect("Failed to create chunk subset");
        
        // Transform to RecordBatch
        let result = provider.ndarray_to_record_batch(data, &chunk_subset, "test_array");
        assert!(result.is_ok());
        
        let batch = result.unwrap();
        
        // Verify schema
        let schema = batch.schema();
        assert_eq!(schema.fields().len(), 3); // 2 coordinate dimensions + 1 data column
        assert_eq!(schema.field(0).name(), "dim_0");
        assert_eq!(schema.field(1).name(), "dim_1");
        assert_eq!(schema.field(2).name(), "test_array");
        
        // Verify data types
        assert_eq!(schema.field(0).data_type(), &DataType::Int64);
        assert_eq!(schema.field(1).data_type(), &DataType::Int64);
        assert_eq!(schema.field(2).data_type(), &DataType::Float64);
        
        // Verify number of rows (should be 2 * 3 = 6)
        assert_eq!(batch.num_rows(), 6);
        
        // Verify some coordinate values
        let dim_0_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let dim_1_array = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let data_array = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        
        // Check first row: coordinates (0, 0) with value 1.0
        assert_eq!(dim_0_array.value(0), 0);
        assert_eq!(dim_1_array.value(0), 0);
        assert_eq!(data_array.value(0), 1.0);
        
        // Check last row: coordinates (1, 2) with value 6.0
        assert_eq!(dim_0_array.value(5), 1);
        assert_eq!(dim_1_array.value(5), 2);
        assert_eq!(data_array.value(5), 6.0);
    }
    
    #[test]
    fn test_chunk_to_record_batch_with_no_store() {
        // Test error handling when no store is available
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        let chunk_indices = vec![0u64, 0u64];
        let result = provider.chunk_to_record_batch(&chunk_indices);
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(error.to_string().contains("No store available"));
    }
}