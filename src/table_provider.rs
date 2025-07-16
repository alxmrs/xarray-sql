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
use arrow_array::{Int64Array, Float64Array};

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
pub struct ZarrTableProvider {
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
    
    /// Infer Arrow schema from Zarr metadata with multi-variable support
    pub fn infer_schema(&self) -> Result<Arc<Schema>, DataFusionError> {
        let store = self.store.as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;
        
        // Read the zarr group metadata
        let group = Group::open(store.clone(), "/")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        // Collect all arrays and separate coordinates from data variables
        let children = group.children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        let mut all_arrays = Vec::new();
        
        // First pass: collect all arrays with their metadata
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                let shape = array.shape().to_vec();
                let data_type = array.data_type().clone();
                all_arrays.push((path_str, shape, data_type));
            }
        }
        
        if all_arrays.is_empty() {
            return Err(DataFusionError::External("No arrays found in Zarr store".into()));
        }
        
        // Identify data variables vs coordinates
        // Data variables typically have the highest dimensionality
        // Coordinates are typically 1D arrays
        let max_dims = all_arrays.iter()
            .map(|(_, shape, _)| shape.len())
            .max()
            .unwrap_or(0);
        
        let mut data_variables = Vec::new();
        let mut coordinate_arrays = Vec::new();
        
        for (name, shape, data_type) in all_arrays {
            if shape.len() == max_dims && shape.len() > 1 {
                // This is likely a data variable (multi-dimensional)
                data_variables.push((name, shape, data_type));
            } else {
                // This is likely a coordinate array (1D or lower dimensionality)
                coordinate_arrays.push((name, shape, data_type));
            }
        }
        
        // Validate that data variables have consistent dimensions
        let mut reference_shape: Option<Vec<u64>> = None;
        for (name, shape, _) in &data_variables {
            if let Some(ref ref_shape) = reference_shape {
                if shape != ref_shape {
                    return Err(DataFusionError::External(
                        format!(
                            "Inconsistent dimensions across data variables. Variable '{}' has shape {:?}, but expected {:?}. All data variables must have the same dimensional structure.",
                            name, shape, ref_shape
                        ).into()
                    ));
                }
            } else {
                reference_shape = Some(shape.clone());
            }
        }
        
        if data_variables.is_empty() {
            return Err(DataFusionError::External("No arrays found in Zarr store".into()));
        }
        
        // Build unified schema: dimensions first, then data variables
        let mut fields = Vec::new();
        
        // Add coordinate/dimension fields
        if let Some(ref shape) = reference_shape {
            for (dim_idx, &_dim_size) in shape.iter().enumerate() {
                // TODO: Extract actual coordinate names from metadata
                let dim_name = format!("dim_{}", dim_idx);
                fields.push(Field::new(dim_name, DataType::Int64, false));
            }
        }
        
        // Add data variable fields
        for (var_name, _shape, data_type) in &data_variables {
            let arrow_type = self.zarr_type_to_arrow(data_type)
                .unwrap_or(DataType::Float64); // Default fallback
            fields.push(Field::new(var_name.clone(), arrow_type, true));
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
    
    /// Transform a Zarr chunk into a RecordBatch with multi-variable support
    pub fn chunk_to_record_batch(
        &self,
        chunk_indices: &[u64],
    ) -> Result<RecordBatch, DataFusionError> {
        let store = self.store.as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;
        
        // Get the zarr group
        let group = Group::open(store.clone(), "/")
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        // Collect all arrays and separate coordinates from data variables
        let children = group.children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        
        let mut all_arrays = Vec::new();
        
        // First pass: collect all arrays with their metadata
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                let shape = array.shape().to_vec();
                all_arrays.push((path_str, array, shape));
            }
        }
        
        if all_arrays.is_empty() {
            return Err(DataFusionError::External("No arrays found in Zarr store".into()));
        }
        
        // Identify data variables (highest dimensionality, >1D)
        let max_dims = all_arrays.iter()
            .map(|(_, _, shape)| shape.len())
            .max()
            .unwrap_or(0);
        
        let mut arrays_data = Vec::new();
        let mut reference_shape: Option<Vec<u64>> = None;
        let mut reference_chunk_subset: Option<ArraySubset> = None;
        
        // Second pass: process only data variables
        for (path_str, array, shape) in all_arrays {
            // Only process data variables (multi-dimensional arrays)
            if shape.len() == max_dims && shape.len() > 1 {
                // Validate chunk alignment
                let chunk_subset = array.chunk_subset(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                
                // Check shape and chunk consistency
                if let Some(ref ref_shape) = reference_shape {
                    if shape != *ref_shape {
                        return Err(DataFusionError::External(
                            format!(
                                "Inconsistent array shapes. Variable '{}' has shape {:?}, expected {:?}",
                                path_str, shape, ref_shape
                            ).into()
                        ));
                    }
                } else {
                    reference_shape = Some(shape);
                }
                
                // Check chunk subset consistency  
                if let Some(ref ref_subset) = reference_chunk_subset {
                    if chunk_subset.shape() != ref_subset.shape() {
                        return Err(DataFusionError::External(
                            format!(
                                "Inconsistent chunk shapes. Variable '{}' chunk has shape {:?}, expected {:?}. All variables must have aligned chunks.",
                                path_str, chunk_subset.shape(), ref_subset.shape()
                            ).into()
                        ));
                    }
                } else {
                    reference_chunk_subset = Some(chunk_subset.clone());
                }
                
                arrays_data.push((path_str, array, chunk_subset));
            }
        }
        
        if arrays_data.is_empty() {
            return Err(DataFusionError::External("No arrays found in Zarr store".into()));
        }
        
        // Now create the multi-variable RecordBatch
        self.create_multi_variable_record_batch(arrays_data, chunk_indices)
    }
    
    /// Create a RecordBatch from multiple variables with proper cartesian product
    fn create_multi_variable_record_batch(
        &self,
        arrays_data: Vec<(String, Array<FilesystemStore>, ArraySubset)>,
        chunk_indices: &[u64],
    ) -> Result<RecordBatch, DataFusionError> {
        if arrays_data.is_empty() {
            return Err(DataFusionError::External("No arrays provided".into()));
        }
        
        // Get reference dimensions from the first array
        let (_, _ref_array, ref_chunk_subset) = &arrays_data[0];
        let chunk_shape = ref_chunk_subset.shape();
        let ndim = chunk_shape.len();
        let total_elements = chunk_shape.iter().product::<u64>() as usize;
        
        // Generate coordinate arrays (same for all variables)
        let chunk_start = ref_chunk_subset.start();
        let coord_arrays = self.generate_coordinates_from_shape(&chunk_shape, chunk_start, total_elements);
        
        // Collect data from all variables
        let mut all_data_arrays = Vec::new();
        
        for (var_name, array, _chunk_subset) in &arrays_data {
            // Retrieve data based on the array's data type
            let data_array = match array.data_type() {
                ZarrDataType::Float64 => {
                    let chunk_data = array.retrieve_chunk_ndarray::<f64>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_f64(chunk_data)?
                }
                ZarrDataType::Float32 => {
                    let chunk_data = array.retrieve_chunk_ndarray::<f32>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_f32(chunk_data)?
                }
                ZarrDataType::Int64 => {
                    let chunk_data = array.retrieve_chunk_ndarray::<i64>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i64(chunk_data)?
                }
                ZarrDataType::Int32 => {
                    let chunk_data = array.retrieve_chunk_ndarray::<i32>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i32(chunk_data)?
                }
                other => {
                    return Err(DataFusionError::External(
                        format!("Unsupported zarr data type for variable '{}': {:?}", var_name, other).into()
                    ));
                }
            };
            
            all_data_arrays.push((var_name.clone(), data_array));
        }
        
        // Build the complete Arrow arrays list: coordinates first, then data variables
        let mut arrows: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
        
        // Add coordinate columns
        for coord_array in coord_arrays {
            arrows.push(Arc::new(Int64Array::from(coord_array)));
        }
        
        // Add data variable columns
        for (_var_name, data_array) in all_data_arrays {
            arrows.push(data_array);
        }
        
        // Create the schema
        let mut fields = Vec::new();
        
        // Add dimension fields
        for dim_idx in 0..ndim {
            fields.push(Field::new(format!("dim_{}", dim_idx), DataType::Int64, false));
        }
        
        // Add data variable fields
        for (var_name, array, _) in &arrays_data {
            let arrow_type = self.zarr_type_to_arrow(array.data_type())
                .unwrap_or(DataType::Float64);
            fields.push(Field::new(var_name.clone(), arrow_type, true));
        }
        
        let schema = Arc::new(Schema::new(fields));
        
        // Create the RecordBatch
        RecordBatch::try_new(schema, arrows)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }
    
    /// Helper method to generate coordinates from chunk shape
    fn generate_coordinates_from_shape(
        &self,
        chunk_shape: &[u64],
        chunk_start: &[u64],
        total_elements: usize,
    ) -> Vec<Vec<i64>> {
        let shape: Vec<usize> = chunk_shape.iter().map(|&x| x as usize).collect();
        self.generate_coordinates(&shape, chunk_start, total_elements)
    }
    
    /// Create Arrow array from f64 ndarray
    fn create_data_array_f64(&self, data: ndarray::ArrayD<f64>) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements)
            .map_err(|e| DataFusionError::External(format!("Failed to reshape f64 array: {}", e).into()))?;
        
        let data_vec = f64::to_arrow_array(&flat_data);
        Ok(f64::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }
    
    /// Create Arrow array from f32 ndarray  
    fn create_data_array_f32(&self, data: ndarray::ArrayD<f32>) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements)
            .map_err(|e| DataFusionError::External(format!("Failed to reshape f32 array: {}", e).into()))?;
        
        let data_vec = f32::to_arrow_array(&flat_data);
        Ok(f32::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }
    
    /// Create Arrow array from i64 ndarray
    fn create_data_array_i64(&self, data: ndarray::ArrayD<i64>) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements)
            .map_err(|e| DataFusionError::External(format!("Failed to reshape i64 array: {}", e).into()))?;
        
        let data_vec = i64::to_arrow_array(&flat_data);
        Ok(i64::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }
    
    /// Create Arrow array from i32 ndarray
    fn create_data_array_i32(&self, data: ndarray::ArrayD<i32>) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements)
            .map_err(|e| DataFusionError::External(format!("Failed to reshape i32 array: {}", e).into()))?;
        
        let data_vec = i32::to_arrow_array(&flat_data);
        Ok(i32::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
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

    // ===== MULTI-VARIABLE TESTS =====
    // These tests define the expected behavior for multi-variable Zarr datasets
    
    #[test]
    fn test_multi_variable_schema_inference_no_store() {
        // Test that schema inference fails gracefully when no store is available
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };
        
        let result = provider.infer_schema();
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert!(error.to_string().contains("No store available"));
    }

    #[test]
    fn test_multi_variable_dimension_consistency_check() {
        // Test that we properly validate dimension consistency across variables
        
        // Expected behavior:
        // Given variables with inconsistent dimensions:
        // - temperature(time, lat, lon) - shape [10, 5, 8]
        // - pressure(time, lat) - shape [10, 5]
        // Should return an error explaining the inconsistency
        
        // TODO: Implement this test
        // let provider = create_test_inconsistent_dimensions_provider();
        // let result = provider.infer_schema();
        // 
        // assert!(result.is_err());
        // let error = result.unwrap_err();
        // assert!(error.to_string().contains("inconsistent dimensions"));
    }

    #[test]
    fn test_multi_variable_chunk_alignment() {
        // Test that chunks are properly aligned across multiple variables
        
        // Expected behavior:
        // Given variables with different chunk shapes:
        // - temperature: chunks [5, 3, 4]
        // - pressure: chunks [5, 3, 2]  
        // Should return an error explaining chunk misalignment
        
        // TODO: Implement this test
        // let provider = create_test_misaligned_chunks_provider();
        // let result = provider.chunk_to_record_batch(&[0, 0, 0]);
        // 
        // assert!(result.is_err());
        // let error = result.unwrap_err();
        // assert!(error.to_string().contains("chunk alignment"));
    }

    #[test]
    fn test_multi_variable_record_batch_creation() {
        // Test that RecordBatch creation works correctly for multiple variables
        
        // Expected behavior:
        // Given variables: temperature(time, lat, lon), pressure(time, lat, lon)
        // With shapes [2, 2, 2] and data:
        // temperature = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        // pressure = [[[100.0, 101.0], [102.0, 103.0]], [[104.0, 105.0], [106.0, 107.0]]]
        //
        // Expected table rows:
        // (time=0, lat=0, lon=0, temp=1.0, pressure=100.0)
        // (time=0, lat=0, lon=1, temp=2.0, pressure=101.0)
        // (time=0, lat=1, lon=0, temp=3.0, pressure=102.0)
        // (time=0, lat=1, lon=1, temp=4.0, pressure=103.0)
        // (time=1, lat=0, lon=0, temp=5.0, pressure=104.0)
        // (time=1, lat=0, lon=1, temp=6.0, pressure=105.0)
        // (time=1, lat=1, lon=0, temp=7.0, pressure=106.0)
        // (time=1, lat=1, lon=1, temp=8.0, pressure=107.0)
        
        // TODO: Implement this test
        // let provider = create_test_multi_variable_provider();
        // let batch = provider.chunk_to_record_batch(&[0, 0, 0]).unwrap();
        // 
        // assert_eq!(batch.num_rows(), 8); // 2 * 2 * 2 = 8 rows
        // assert_eq!(batch.schema().fields().len(), 5); // time, lat, lon, temp, pressure
        // 
        // // Test first row
        // let time_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        // let lat_array = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        // let lon_array = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        // let temp_array = batch.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        // let pressure_array = batch.column(4).as_any().downcast_ref::<Float64Array>().unwrap();
        // 
        // assert_eq!(time_array.value(0), 0);
        // assert_eq!(lat_array.value(0), 0);
        // assert_eq!(lon_array.value(0), 0);
        // assert_eq!(temp_array.value(0), 1.0);
        // assert_eq!(pressure_array.value(0), 100.0);
        // 
        // // Test last row
        // assert_eq!(time_array.value(7), 1);
        // assert_eq!(lat_array.value(7), 1);
        // assert_eq!(lon_array.value(7), 1);
        // assert_eq!(temp_array.value(7), 8.0);
        // assert_eq!(pressure_array.value(7), 107.0);
    }

    #[test]
    fn test_multi_variable_with_mixed_data_types() {
        // Test handling of multiple variables with different data types
        
        // Expected behavior:
        // Given variables:
        // - temperature: Float64
        // - pressure: Float32
        // - humidity: Int32
        // - is_raining: Bool
        //
        // Should create a schema with appropriate data types for each column
        
        // TODO: Implement this test
        // let provider = create_test_mixed_types_provider();
        // let schema = provider.infer_schema().unwrap();
        // 
        // assert_eq!(schema.field_with_name("temperature").unwrap().data_type(), &DataType::Float64);
        // assert_eq!(schema.field_with_name("pressure").unwrap().data_type(), &DataType::Float64); // f32 converted to f64
        // assert_eq!(schema.field_with_name("humidity").unwrap().data_type(), &DataType::Int32);
        // assert_eq!(schema.field_with_name("is_raining").unwrap().data_type(), &DataType::Boolean);
    }

    #[test]
    fn test_coordinate_names_from_zarr_metadata() {
        // Test that we can extract proper coordinate names from Zarr metadata
        // instead of using generic dim_0, dim_1, etc.
        
        // Expected behavior:
        // Given Zarr metadata with coordinate names: ["time", "latitude", "longitude"]
        // Schema should use these names instead of dim_0, dim_1, dim_2
        
        // TODO: Implement this test
        // let provider = create_test_named_coordinates_provider();
        // let schema = provider.infer_schema().unwrap();
        // 
        // assert_eq!(schema.field(0).name(), "time");
        // assert_eq!(schema.field(1).name(), "latitude");
        // assert_eq!(schema.field(2).name(), "longitude");
    }

    #[test]
    fn test_multi_variable_chunked_reading() {
        // Test that we can read multiple variables chunk-by-chunk correctly
        
        // Expected behavior:
        // Given a dataset with 4 chunks, reading chunk [1, 0] should return
        // data from the correct spatial/temporal region for ALL variables
        
        // TODO: Implement this test
        // let provider = create_test_chunked_multi_variable_provider();
        // let batch = provider.chunk_to_record_batch(&[1, 0]).unwrap();
        // 
        // // Verify that coordinates reflect the correct chunk offset
        // let time_array = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        // assert!(time_array.value(0) >= 10); // Assuming chunk 1 starts at time=10
        // 
        // // Verify that all variables have data for the same coordinates
        // assert_eq!(batch.num_rows(), expected_chunk_size);
        // for row in 0..batch.num_rows() {
        //     // All variables should have non-null values for the same coordinates
        //     assert!(!batch.column(3).is_null(row)); // temperature
        //     assert!(!batch.column(4).is_null(row)); // pressure
        // }
    }

    // Helper functions for creating test providers (to be implemented)
    
    // fn create_test_multi_variable_provider() -> ZarrTableProvider {
    //     // Create a test provider with multiple variables having consistent dimensions
    //     todo!("Implement test provider creation")
    // }
    
    // fn create_test_inconsistent_dimensions_provider() -> ZarrTableProvider {
    //     // Create a test provider with variables having inconsistent dimensions
    //     todo!("Implement test provider creation")
    // }
    
    // fn create_test_misaligned_chunks_provider() -> ZarrTableProvider {
    //     // Create a test provider with misaligned chunks
    //     todo!("Implement test provider creation")
    // }
}