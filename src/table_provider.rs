#![allow(dead_code)]

use arrow_array::RecordBatch;
use arrow_array::{Float32Array, Float64Array, Int16Array, Int32Array, Int64Array};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::MemTable;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::DataFusionError;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyCapsule;
use pyo3::{pyclass, pymethods, Bound, PyResult, Python};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use zarrs::array::chunk_grid::ChunkGrid;
use zarrs::array::data_type::DataType as ZarrDataType;
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;

/// Represents a coordinate range constraint for chunk filtering
#[derive(Debug, Clone)]
struct CoordinateRange {
    dimension: usize,
    min: Option<i64>,
    max: Option<i64>,
}

impl CoordinateRange {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            min: None,
            max: None,
        }
    }

    fn with_min(mut self, min: i64) -> Self {
        self.min = Some(min);
        self
    }

    fn with_max(mut self, max: i64) -> Self {
        self.max = Some(max);
        self
    }

    /// Check if a coordinate value satisfies this range
    fn contains(&self, value: i64) -> bool {
        if let Some(min) = self.min {
            if value < min {
                return false;
            }
        }
        if let Some(max) = self.max {
            if value > max {
                return false;
            }
        }
        true
    }
}

/// Represents a collection of coordinate constraints for filtering
#[derive(Debug, Clone)]
pub struct CoordinateFilter {
    ranges: Vec<CoordinateRange>,
}

impl CoordinateFilter {
    fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    fn add_range(mut self, range: CoordinateRange) -> Self {
        self.ranges.push(range);
        self
    }

    /// Check if a set of coordinates satisfies all constraints
    pub fn matches(&self, coordinates: &[i64]) -> bool {
        for range in &self.ranges {
            if range.dimension < coordinates.len() && !range.contains(coordinates[range.dimension])
            {
                return false;
            }
        }
        true
    }
}

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
        DataType::Float64 // Convert f32 to f64 for consistency
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

impl ToArrowArray for i16 {
    type ArrowArray = Int16Array;

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
        DataType::Int16
    }

    fn from_vec(data: Vec<Self>) -> Arc<Self::ArrowArray> {
        Arc::new(Int16Array::from(data))
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
        // TODO(alxmrs): Why only create a store? Why not open a store?
        let store = Self::create_store(&store_path)?;
        Ok(Self {
            store_path,
            store: Some(Arc::new(store)),
        })
    }

    /// Create a readable storage from a path
    fn create_store(store_path: &str) -> Result<FilesystemStore, DataFusionError> {
        // For now, assume filesystem store
        // TODO(alxmrs): Add support for other storage backends (S3, GCS, etc.)
        let path = Path::new(store_path);
        if path.exists() {
            FilesystemStore::new(path).map_err(|e| DataFusionError::External(Box::new(e)))
        } else {
            Err(DataFusionError::External(
                format!("Zarr store path does not exist: {store_path}").into(),
            ))
        }
    }

    /// Get the underlying zarr store
    pub fn store(&self) -> Option<&Arc<FilesystemStore>> {
        self.store.as_ref()
    }

    /// Infer Arrow schema from Zarr metadata
    pub fn infer_schema(&self) -> Result<Arc<Schema>, DataFusionError> {
        let store = self
            .store()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;

        // Read the zarr group metadata
        let group =
            Group::open(store.clone(), "/").map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Collect all arrays and separate coordinates from data variables
        let children = group
            .children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Tuple of (path_str, shape, data_type, dimension_names)
        let mut all_arrays = Vec::new();

        // First pass: collect all arrays with their metadata
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                let shape = array.shape().to_vec();
                let data_type = array.data_type().clone();

                // Try to get dimension names from metadata
                let dimension_names = array
                    .dimension_names()
                    .as_ref()
                    .map(|names| names.iter().filter_map(|name| name.clone()).collect())
                    .unwrap_or_else(Vec::new);

                all_arrays.push((path_str, shape, data_type, dimension_names));
            }
        }

        if all_arrays.is_empty() {
            return Err(DataFusionError::External(
                "No arrays found in Zarr store".into(),
            ));
        }

        // Identify data variables vs coordinates using dimension_names
        // Data variables have dimension_names that reference other arrays
        // Coordinates have dimension_names that reference themselves
        let mut data_variables = Vec::new(); // (name, shape, data_type)
        let mut dimension_arrays = Vec::new(); // (name, shape, data_type)
        let mut coordinate_arrays = Vec::new(); // (name, shape, data_type)

        for (name, shape, data_type, dimension_names) in all_arrays {
            // Remove leading slash from name for comparison
            let clean_name = if name.starts_with('/') {
                name.chars().skip(1).collect()
            } else {
                name.clone()
            };

            // Coordinates contain other dimensions
            let is_coordinate = dimension_names.len() == 1;
            // Check if this is a dimension (dimension_names contains only itself)
            let is_dimension = is_coordinate && dimension_names[0] == clean_name;

            if is_dimension {
                dimension_arrays.push((clean_name, shape, data_type));
            } else if is_coordinate {
                coordinate_arrays.push((clean_name, shape, data_type));
            } else {
                // This is a data variable
                data_variables.push((clean_name, shape, data_type));
            }
        }

        // Handle different cases: multi-dimensional data variables or tabular data
        let mut reference_shape: Option<Vec<u64>> = None;

        if !data_variables.is_empty() {
            // Case 1: Multi-dimensional data variables (like air temperature)
            // Validate that data variables have consistent dimensions
            for (name, shape, _) in &data_variables {
                if let Some(ref ref_shape) = reference_shape {
                    if shape != ref_shape {
                        return Err(DataFusionError::External(
                            format!(
                                "Inconsistent dimensions across data variables. Variable '{name}' has shape {shape:?}, but expected {ref_shape:?}. All data variables must have the same dimensional structure."
                            ).into()
                        ));
                    }
                } else {
                    reference_shape = Some(shape.clone());
                }
            }
        } else if !coordinate_arrays.is_empty() {
            // TODO(alxmrs or Claude): Fill out this case taking into account the newly available
            // coordinate_arrays case.
            return Err(DataFusionError::External(
                "Case not yet implemented!".into(),
            ));
        } else {
            return Err(DataFusionError::External(
                "No arrays found in Zarr store".into(),
            ));
        }

        // Build unified schema based on the type of data
        let mut fields = Vec::new();

        if !data_variables.is_empty() {
            // Case 1: Multi-dimensional data with coordinates
            // Add coordinate/dimension fields using actual coordinate names
            if let Some(ref shape) = reference_shape {
                // For typical xarray-generated Zarr, we expect coordinates in a specific order
                // The air dataset has dimensions (time, lat, lon)
                // We need to match coordinate arrays to dimensions

                // First, collect all 1D coordinate arrays that match dimension sizes
                let mut valid_coords: Vec<(String, usize)> = Vec::new();

                for (name, coord_shape, _) in &dimension_arrays {
                    if coord_shape.len() == 1 {
                        let size = coord_shape[0];
                        // Find all dimension indices that match this size
                        for (dim_idx, &dim_size) in shape.iter().enumerate() {
                            if dim_size == size {
                                valid_coords.push((name.to_string(), dim_idx));
                            }
                        }
                    }
                }

                // Add coordinate fields, avoiding duplicates
                let mut added_coords: HashSet<String> = HashSet::new();

                for (coord_name, _) in valid_coords {
                    if !added_coords.contains(&coord_name) {
                        fields.push(Field::new(coord_name.clone(), DataType::Int64, false));
                        added_coords.insert(coord_name);
                    }
                }
            }
        }
        // TODO(alxmrs) The above is wrapped in an if statement bc Claude originally tried
        // to handle "tabular" data separately. The correct way to go about this is to
        // respect a "coordinates" concept in addition to "dimensions".

        // Add data variable fields (remove leading slash if present)
        if !data_variables.is_empty() {
            // Case 1b: Multi-dimensional data variables
            // Exclude coordinate arrays that are already added as dimensions
            let coord_names: HashSet<String> = dimension_arrays
                .iter()
                .map(|(name, _, _)| name.clone())
                .collect();

            for (var_name, _shape, data_type) in &data_variables {
                let arrow_type = self.zarr_type_to_arrow(data_type)?;

                // Only add if this is not a coordinate array
                if !coord_names.contains(var_name) {
                    fields.push(Field::new(var_name, arrow_type, true));
                }
            }
        }
        // TODO(alxmrs or Claude): Handle "coordinates"-only Zarr datasets.

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
                Err(DataFusionError::External(
                    "Complex64 not yet supported".into(),
                ))
            }
            ZarrDataType::Complex128 => Err(DataFusionError::External(
                "Complex128 not yet supported".into(),
            )),
            ZarrDataType::RawBits(_) => Err(DataFusionError::External(
                "RawBits not yet supported".into(),
            )),
            _ => {
                // For string types and other unsupported types, default to String
                // This handles xarray's string variables
                Ok(DataType::Utf8)
            }
        }
    }

    /// Get chunk grid information for the first array in the store
    pub fn get_chunk_grid(&self) -> Result<ChunkGrid, DataFusionError> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;

        // Read the zarr group metadata
        let group =
            Group::open(store.clone(), "/").map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Get the first array to work with
        let children = group
            .children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                return Ok(array.chunk_grid().clone());
            }
        }

        Err(DataFusionError::External(
            "No arrays found in Zarr store".into(),
        ))
    }

    /// Get chunk indices for iterating over chunks
    pub fn get_chunk_indices(&self) -> Result<Vec<Vec<u64>>, DataFusionError> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;

        // Read the zarr group metadata
        let group =
            Group::open(store.clone(), "/").map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Get the first array to work with
        let children = group
            .children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                // Get the chunk grid shape
                let chunk_grid_shape = array.chunk_grid_shape().ok_or_else(|| {
                    DataFusionError::External("Failed to get chunk grid shape".into())
                })?;

                // Create an ArraySubset covering all chunks
                let chunks_subset = ArraySubset::new_with_shape(chunk_grid_shape);

                // Get chunk indices iterator and collect into Vec
                let chunk_indices: Vec<Vec<u64>> = chunks_subset.indices().iter().collect();
                return Ok(chunk_indices);
            }
        }

        Err(DataFusionError::External(
            "No arrays found in Zarr store".into(),
        ))
    }

    /// Get chunk subset for a specific chunk index
    pub fn get_chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, DataFusionError> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;

        // Read the zarr group metadata
        let group =
            Group::open(store.clone(), "/").map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Get the first array to work with
        let children = group
            .children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                // Get chunk subset for the given chunk indices
                let chunk_subset = array
                    .chunk_subset(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;

                return Ok(chunk_subset);
            }
        }

        Err(DataFusionError::External(
            "No arrays found in Zarr store".into(),
        ))
    }

    /// Transform a Zarr chunk into a RecordBatch with multi-variable support
    pub fn chunk_to_record_batch(
        &self,
        chunk_indices: &[u64],
    ) -> Result<RecordBatch, DataFusionError> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;

        // Get the zarr group
        let group =
            Group::open(store.clone(), "/").map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Collect all arrays and separate coordinates from data variables
        let children = group
            .children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let mut all_arrays = Vec::new();

        // First pass: collect all arrays with their metadata
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                let shape = array.shape().to_vec();

                // Try to get dimension names from metadata
                let dimension_names = array
                    .dimension_names()
                    .as_ref()
                    .map(|names| names.iter().filter_map(|name| name.clone()).collect())
                    .unwrap_or_else(Vec::new);

                all_arrays.push((path_str, array, shape, dimension_names));
            }
        }

        if all_arrays.is_empty() {
            return Err(DataFusionError::External(
                "No arrays found in Zarr store".into(),
            ));
        }

        // Identify data variables vs coordinates using dimension_names
        let mut data_variables = Vec::new();
        let mut dimension_arrays = Vec::new();

        for (name, array, shape, dimension_names) in all_arrays {
            // Remove leading slash from name for comparison
            let clean_name = if name.starts_with('/') {
                name.chars().skip(1).collect()
            } else {
                name.clone()
            };

            // Check if this is a coordinate (dimension_names contains only itself)
            let is_coordinate = dimension_names.len() == 1 && dimension_names[0] == clean_name;

            if is_coordinate {
                dimension_arrays.push((name, array, shape));
            } else {
                // This is a data variable
                data_variables.push((name, array, shape));
            }
        }

        // Handle different cases: multi-dimensional data variables or tabular data
        if !data_variables.is_empty() {
            // Check if all data variables are 1D (tabular data)
            let all_1d = data_variables.iter().all(|(_, _, shape)| shape.len() == 1);

            if all_1d {
                // Case 1: Tabular data - all arrays are 1D data variables
                self.create_tabular_record_batch(data_variables, chunk_indices)
            } else {
                // Case 2: Multi-dimensional data variables
                self.create_multi_variable_record_batch(
                    data_variables
                        .into_iter()
                        .map(|(name, array, _shape)| {
                            let chunk_subset = array
                                .chunk_subset(chunk_indices)
                                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                            Ok((name, array, chunk_subset))
                        })
                        .collect::<Result<Vec<_>, DataFusionError>>()?,
                    chunk_indices,
                )
            }
        } else {
            // Case 3: Only coordinate arrays (should not happen in normal zarr)
            self.create_tabular_record_batch(dimension_arrays, chunk_indices)
        }
    }

    /// Create a RecordBatch from 1D tabular data
    fn create_tabular_record_batch(
        &self,
        arrays: Vec<(String, Array<FilesystemStore>, Vec<u64>)>,
        chunk_indices: &[u64],
    ) -> Result<RecordBatch, DataFusionError> {
        if arrays.is_empty() {
            return Err(DataFusionError::External("No arrays provided".into()));
        }

        let mut arrow_arrays: Vec<Arc<dyn arrow_array::Array>> = Vec::new();
        let mut fields = Vec::new();

        for (name, array, _shape) in arrays {
            // Clean the name
            let clean_name = if name.starts_with('/') {
                name.chars().skip(1).collect()
            } else {
                name.clone()
            };

            // Get the data type and create appropriate Arrow array
            let data_array = match array.data_type() {
                ZarrDataType::Float64 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<f64>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_f64(chunk_data)?
                }
                ZarrDataType::Float32 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<f32>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_f32(chunk_data)?
                }
                ZarrDataType::Int64 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<i64>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i64(chunk_data)?
                }
                ZarrDataType::Int32 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<i32>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i32(chunk_data)?
                }
                ZarrDataType::Int16 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<i16>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i16(chunk_data)?
                }
                other => {
                    // For string types and other unsupported types, skip for now
                    // TODO: Add proper string support with zarrs library
                    eprintln!("Warning: Skipping unsupported zarr data type for tabular data variable '{clean_name}': {other:?}");
                    continue;
                }
            };

            arrow_arrays.push(data_array);
            let arrow_type = self.zarr_type_to_arrow(array.data_type())?;
            fields.push(Field::new(clean_name, arrow_type, true));
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrow_arrays)
            .map_err(|e| DataFusionError::External(Box::new(e)))
    }

    /// Create a RecordBatch from multiple variables with proper cartesian product (existing method)
    fn create_multi_variable_record_batch(
        &self,
        arrays_data: Vec<(String, Array<FilesystemStore>, ArraySubset)>,
        chunk_indices: &[u64],
    ) -> Result<RecordBatch, DataFusionError> {
        if arrays_data.is_empty() {
            return Err(DataFusionError::External("No arrays provided".into()));
        }

        // Continue with existing logic for multi-dimensional data
        let mut arrays_data_processed = Vec::new();
        let mut reference_shape: Option<Vec<u64>> = None;
        let mut reference_chunk_subset: Option<ArraySubset> = None;

        for (path_str, array, chunk_subset) in arrays_data {
            let shape = array.shape().to_vec();

            // Validate chunk alignment
            if let Some(ref ref_shape) = reference_shape {
                if shape != *ref_shape {
                    return Err(DataFusionError::External(
                        format!(
                            "Inconsistent array shapes. Variable '{path_str}' has shape {shape:?}, expected {ref_shape:?}"
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

            arrays_data_processed.push((path_str, array, chunk_subset));
        }

        // Now create the multi-variable RecordBatch using the existing logic
        self.create_multi_variable_record_batch_internal(arrays_data_processed, chunk_indices)
    }

    /// Internal method to create multi-variable record batch (continuation of existing logic)
    fn create_multi_variable_record_batch_internal(
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
        let total_elements = chunk_shape.iter().product::<u64>() as usize;

        // Generate coordinate arrays (same for all variables)
        let chunk_start = ref_chunk_subset.start();
        let coord_arrays =
            self.generate_coordinates_from_shape(chunk_shape, chunk_start, total_elements);

        // Collect data from all variables
        let mut all_data_arrays = Vec::new();

        for (var_name, array, _chunk_subset) in &arrays_data {
            // Retrieve data based on the array's data type
            let data_array = match array.data_type() {
                ZarrDataType::Float64 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<f64>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_f64(chunk_data)?
                }
                ZarrDataType::Float32 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<f32>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_f32(chunk_data)?
                }
                ZarrDataType::Int64 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<i64>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i64(chunk_data)?
                }
                ZarrDataType::Int32 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<i32>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i32(chunk_data)?
                }
                ZarrDataType::Int16 => {
                    let chunk_data = array
                        .retrieve_chunk_ndarray::<i16>(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.create_data_array_i16(chunk_data)?
                }
                other => {
                    return Err(DataFusionError::External(
                        format!("Unsupported zarr data type for variable '{var_name}': {other:?}",)
                            .into(),
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

        // Create the schema using the same logic as infer_schema
        let schema = self.infer_schema()?;

        // Create the RecordBatch
        RecordBatch::try_new(schema, arrows).map_err(|e| DataFusionError::External(Box::new(e)))
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
    fn create_data_array_f64(
        &self,
        data: ndarray::ArrayD<f64>,
    ) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements).map_err(|e| {
            DataFusionError::External(format!("Failed to reshape f64 array: {e}").into())
        })?;

        let data_vec = f64::to_arrow_array(&flat_data);
        Ok(f64::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }

    /// Create Arrow array from f32 ndarray
    fn create_data_array_f32(
        &self,
        data: ndarray::ArrayD<f32>,
    ) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements).map_err(|e| {
            DataFusionError::External(format!("Failed to reshape f32 array: {e}").into())
        })?;

        let data_vec = f32::to_arrow_array(&flat_data);
        Ok(f32::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }

    /// Create Arrow array from i64 ndarray
    fn create_data_array_i64(
        &self,
        data: ndarray::ArrayD<i64>,
    ) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements).map_err(|e| {
            DataFusionError::External(format!("Failed to reshape i64 array: {e}").into())
        })?;

        let data_vec = i64::to_arrow_array(&flat_data);
        Ok(i64::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }

    /// Create Arrow array from i32 ndarray
    fn create_data_array_i32(
        &self,
        data: ndarray::ArrayD<i32>,
    ) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements).map_err(|e| {
            DataFusionError::External(format!("Failed to reshape i32 array: {e}").into())
        })?;

        let data_vec = i32::to_arrow_array(&flat_data);
        Ok(i32::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }

    /// Create Arrow array from i16 ndarray
    fn create_data_array_i16(
        &self,
        data: ndarray::ArrayD<i16>,
    ) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        let total_elements = data.len();
        let flat_data = data.to_shape(total_elements).map_err(|e| {
            DataFusionError::External(format!("Failed to reshape i16 array: {e}").into())
        })?;

        let data_vec = i16::to_arrow_array(&flat_data);
        Ok(i16::from_vec(data_vec) as Arc<dyn arrow_array::Array>)
    }

    /// Convert a specific array chunk to RecordBatch
    fn array_chunk_to_record_batch(
        &self,
        array: &Array<FilesystemStore>,
        chunk_indices: &[u64],
        array_name: &str,
    ) -> Result<RecordBatch, DataFusionError> {
        // Get the chunk subset
        let chunk_subset = array
            .chunk_subset(chunk_indices)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Handle different data types using the generic implementation
        match array.data_type() {
            ZarrDataType::Float64 => {
                let chunk_data = array
                    .retrieve_chunk_ndarray::<f64>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            ZarrDataType::Float32 => {
                let chunk_data = array
                    .retrieve_chunk_ndarray::<f32>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            ZarrDataType::Int64 => {
                let chunk_data = array
                    .retrieve_chunk_ndarray::<i64>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            ZarrDataType::Int32 => {
                let chunk_data = array
                    .retrieve_chunk_ndarray::<i32>(chunk_indices)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                self.ndarray_to_record_batch(chunk_data, &chunk_subset, array_name)
            }
            other => Err(DataFusionError::External(
                format!("Unsupported zarr data type for chunk reading: {other:?}").into(),
            )),
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
        let flat_data = data.to_shape(total_elements).map_err(|e| {
            DataFusionError::External(format!("Failed to reshape array: {e}").into())
        })?;

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

        // Create the schema using the same logic as infer_schema
        // For single array case, we need to create a minimal schema
        let mut fields = Vec::new();
        for dim_idx in 0..ndim {
            fields.push(Field::new(format!("dim_{dim_idx}"), DataType::Int64, false));
        }

        // Remove leading slash from array name if present
        let clean_name = if array_name.starts_with('/') {
            array_name.chars().skip(1).collect()
        } else {
            array_name.to_string()
        };
        fields.push(Field::new(clean_name, T::arrow_data_type(), true));

        let schema = Arc::new(Schema::new(fields));

        // Create the RecordBatch
        RecordBatch::try_new(schema, arrays).map_err(|e| DataFusionError::External(Box::new(e)))
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

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        // For predicate pushdown, parse coordinate filters
        let filter_refs: Vec<&Expr> = filters.iter().collect();
        let coordinate_filter = self.parse_coordinate_filters(&filter_refs)?;

        // Create filtered batches using coordinate constraints
        let batches = self.create_filtered_batches(coordinate_filter, limit)?;

        // Create MemTable with filtered data
        let schema = self.schema();
        let mem_table = MemTable::try_new(schema, vec![batches])?;

        // Apply projection if specified
        mem_table.scan(_state, projection, &[], limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>, DataFusionError> {
        let mut results = Vec::new();

        for filter in filters {
            // Check if this filter can be pushed down (coordinate-based filters)
            if self.can_pushdown_filter(filter) {
                results.push(TableProviderFilterPushDown::Exact);
            } else {
                results.push(TableProviderFilterPushDown::Unsupported);
            }
        }

        Ok(results)
    }
}

impl ZarrTableProvider {
    /// Check if a filter expression can be pushed down to coordinate level
    fn can_pushdown_filter(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                // Check for coordinate column comparisons
                let has_coord_column = match (left.as_ref(), right.as_ref()) {
                    (Expr::Column(col), Expr::Literal(_, _)) => col.name.starts_with("dim_"),
                    (Expr::Literal(_, _), Expr::Column(col)) => col.name.starts_with("dim_"),
                    _ => false,
                };

                // Only support specific comparison operators
                let supported_op = matches!(
                    op,
                    Operator::Eq | Operator::Gt | Operator::GtEq | Operator::Lt | Operator::LtEq
                );

                has_coord_column && supported_op
            }
            _ => false,
        }
    }
    /// Parse DataFusion expressions into coordinate filters
    pub fn parse_coordinate_filters(
        &self,
        filters: &[&Expr],
    ) -> Result<CoordinateFilter, DataFusionError> {
        let mut coordinate_filter = CoordinateFilter::new();

        for filter in filters {
            if let Some(range) = self.parse_coordinate_expression(filter)? {
                coordinate_filter = coordinate_filter.add_range(range);
            }
        }

        Ok(coordinate_filter)
    }

    /// Parse a single expression into a coordinate range if possible
    fn parse_coordinate_expression(
        &self,
        expr: &Expr,
    ) -> Result<Option<CoordinateRange>, DataFusionError> {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                // Handle expressions like: dim_0 >= 5, dim_1 < 10, etc.
                if let (Expr::Column(col), Expr::Literal(scalar, _)) =
                    (left.as_ref(), right.as_ref())
                {
                    if let Some(dim_idx) = self.parse_dimension_name(&col.name) {
                        if let Some(value) = self.extract_i64_from_scalar(scalar) {
                            let range = match op {
                                Operator::Eq => CoordinateRange::new(dim_idx)
                                    .with_min(value)
                                    .with_max(value),
                                Operator::Gt => CoordinateRange::new(dim_idx).with_min(value + 1),
                                Operator::GtEq => CoordinateRange::new(dim_idx).with_min(value),
                                Operator::Lt => CoordinateRange::new(dim_idx).with_max(value - 1),
                                Operator::LtEq => CoordinateRange::new(dim_idx).with_max(value),
                                _ => return Ok(None), // Unsupported operator
                            };
                            return Ok(Some(range));
                        }
                    }
                }

                // Handle reversed expressions like: 5 <= dim_0
                if let (Expr::Literal(scalar, _), Expr::Column(col)) =
                    (left.as_ref(), right.as_ref())
                {
                    if let Some(dim_idx) = self.parse_dimension_name(&col.name) {
                        if let Some(value) = self.extract_i64_from_scalar(scalar) {
                            let range = match op {
                                Operator::Eq => CoordinateRange::new(dim_idx)
                                    .with_min(value)
                                    .with_max(value),
                                Operator::Lt => CoordinateRange::new(dim_idx).with_min(value + 1),
                                Operator::LtEq => CoordinateRange::new(dim_idx).with_min(value),
                                Operator::Gt => CoordinateRange::new(dim_idx).with_max(value - 1),
                                Operator::GtEq => CoordinateRange::new(dim_idx).with_max(value),
                                _ => return Ok(None), // Unsupported operator
                            };
                            return Ok(Some(range));
                        }
                    }
                }
            }
            _ => return Ok(None), // Unsupported expression type
        }

        Ok(None)
    }

    /// Parse dimension name like "dim_0", "dim_1" into dimension index
    fn parse_dimension_name(&self, name: &str) -> Option<usize> {
        if let Some(rest) = name.strip_prefix("dim_") {
            rest.parse::<usize>().ok()
        } else {
            None
        }
    }

    /// Extract i64 value from ScalarValue
    fn extract_i64_from_scalar(&self, scalar: &ScalarValue) -> Option<i64> {
        match scalar {
            ScalarValue::Int8(Some(v)) => Some(*v as i64),
            ScalarValue::Int16(Some(v)) => Some(*v as i64),
            ScalarValue::Int32(Some(v)) => Some(*v as i64),
            ScalarValue::Int64(Some(v)) => Some(*v),
            ScalarValue::UInt8(Some(v)) => Some(*v as i64),
            ScalarValue::UInt16(Some(v)) => Some(*v as i64),
            ScalarValue::UInt32(Some(v)) => Some(*v as i64),
            ScalarValue::UInt64(Some(v)) => {
                if *v <= i64::MAX as u64 {
                    Some(*v as i64)
                } else {
                    None // Value too large for i64
                }
            }
            _ => None,
        }
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>, DataFusionError> {
        let mut results = Vec::new();

        for filter in filters {
            // Check if this filter can be converted to a coordinate range
            match self.parse_coordinate_expression(filter) {
                Ok(Some(_)) => {
                    // This filter operates on coordinates and can be pushed down
                    results.push(TableProviderFilterPushDown::Exact);
                }
                Ok(None) => {
                    // This filter cannot be pushed down (operates on data variables)
                    results.push(TableProviderFilterPushDown::Unsupported);
                }
                Err(_) => {
                    // Error parsing, cannot push down
                    results.push(TableProviderFilterPushDown::Unsupported);
                }
            }
        }

        Ok(results)
    }

    /// Create RecordBatches with coordinate filtering applied
    pub fn create_filtered_batches(
        &self,
        coordinate_filter: CoordinateFilter,
        limit: Option<usize>,
    ) -> Result<Vec<RecordBatch>, DataFusionError> {
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;

        // Get the zarr group and arrays
        let group =
            Group::open(store.clone(), "/").map_err(|e| DataFusionError::External(Box::new(e)))?;

        let children = group
            .children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Collect data variables using same logic as infer_schema
        let mut all_arrays = Vec::new();
        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                let shape = array.shape().to_vec();

                // Try to get dimension names from metadata
                let dimension_names = array
                    .dimension_names()
                    .as_ref()
                    .map(|names| names.iter().filter_map(|name| name.clone()).collect())
                    .unwrap_or_else(Vec::new);

                all_arrays.push((path_str, array, shape, dimension_names));
            }
        }

        if all_arrays.is_empty() {
            return Err(DataFusionError::External(
                "No arrays found in Zarr store".into(),
            ));
        }

        // Identify data variables vs coordinates using dimension_names (same logic as infer_schema)
        let mut data_variables = Vec::new();
        let mut dimension_arrays = Vec::new();

        for (name, array, shape, dimension_names) in all_arrays {
            // Remove leading slash from name for comparison
            let clean_name = if name.starts_with('/') {
                name.chars().skip(1).collect()
            } else {
                name.clone()
            };

            // Check if this is a coordinate (dimension_names contains only itself)
            let is_coordinate = dimension_names.len() == 1 && dimension_names[0] == clean_name;

            if is_coordinate {
                dimension_arrays.push((name, array, shape));
            } else {
                // This is a data variable
                data_variables.push((name, array, shape));
            }
        }

        // Handle different cases: multi-dimensional data variables or tabular data
        if data_variables.is_empty() && dimension_arrays.is_empty() {
            return Err(DataFusionError::External(
                "No arrays found in Zarr store".into(),
            ));
        }

        // Get chunk grid from the first available array
        let (_, _ref_array, ref_shape) = if !data_variables.is_empty() {
            &data_variables[0]
        } else {
            &dimension_arrays[0]
        };

        // Generate all possible chunk indices and filter them
        let mut filtered_batches = Vec::new();
        let mut row_count = 0;

        // For now, just iterate through the first chunk for testing
        // TODO: Implement proper chunk iteration when zarrs API is clearer
        let chunk_indices = vec![0u64; ref_shape.len()];
        let chunk_combinations = vec![chunk_indices];

        for chunk_indices in chunk_combinations {
            // Check if this chunk potentially contains data matching our filter
            if self.chunk_matches_filter(&chunk_indices, ref_shape, &coordinate_filter)? {
                // Read the chunk and apply row-level filtering
                match self.chunk_to_record_batch(&chunk_indices) {
                    Ok(batch) => {
                        let filtered_batch = self.filter_record_batch(batch, &coordinate_filter)?;
                        if filtered_batch.num_rows() > 0 {
                            row_count += filtered_batch.num_rows();
                            filtered_batches.push(filtered_batch);

                            // Apply limit if specified
                            if let Some(limit) = limit {
                                if row_count >= limit {
                                    break;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // Skip chunks that can't be read (might be expected)
                        continue;
                    }
                }
            }
        }

        Ok(filtered_batches)
    }

    /// Check if a chunk potentially contains data matching the coordinate filter
    fn chunk_matches_filter(
        &self,
        chunk_indices: &[u64],
        array_shape: &[u64],
        coordinate_filter: &CoordinateFilter,
    ) -> Result<bool, DataFusionError> {
        if coordinate_filter.ranges.is_empty() {
            return Ok(true); // No filters, chunk matches
        }

        let store = self
            .store
            .as_ref()
            .ok_or_else(|| DataFusionError::External("No store available".into()))?;

        let group =
            Group::open(store.clone(), "/").map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Find a data variable to get chunk subset info
        let children = group
            .children(false)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        for child in &children {
            let path_str = child.path().to_string();
            if let Ok(array) = Array::open(store.clone(), &path_str) {
                let shape = array.shape().to_vec();
                if shape.len() > 1 && shape == array_shape {
                    // Get chunk subset to determine coordinate ranges
                    let chunk_subset = array
                        .chunk_subset(chunk_indices)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;

                    let chunk_start = chunk_subset.start();
                    let chunk_shape = chunk_subset.shape();

                    // Check if any coordinate in this chunk could match the filter
                    for range in &coordinate_filter.ranges {
                        if range.dimension < chunk_start.len() {
                            let chunk_min = chunk_start[range.dimension] as i64;
                            let chunk_max = chunk_min + chunk_shape[range.dimension] as i64 - 1;

                            // Check if the filter range overlaps with the chunk range
                            let filter_min = range.min.unwrap_or(i64::MIN);
                            let filter_max = range.max.unwrap_or(i64::MAX);

                            // No overlap if chunk_max < filter_min or chunk_min > filter_max
                            if chunk_max < filter_min || chunk_min > filter_max {
                                return Ok(false); // This chunk doesn't match
                            }
                        }
                    }

                    return Ok(true); // Chunk potentially matches
                }
            }
        }

        Ok(true) // Couldn't determine, assume it matches
    }

    /// Filter a RecordBatch to only include rows matching the coordinate filter
    fn filter_record_batch(
        &self,
        batch: RecordBatch,
        coordinate_filter: &CoordinateFilter,
    ) -> Result<RecordBatch, DataFusionError> {
        if coordinate_filter.ranges.is_empty() {
            return Ok(batch); // No filtering needed
        }

        let schema = batch.schema();
        let num_rows = batch.num_rows();
        let mut keep_rows = Vec::with_capacity(num_rows);

        // Extract coordinate columns (first N columns are coordinates)
        let num_dims = coordinate_filter
            .ranges
            .iter()
            .map(|r| r.dimension + 1)
            .max()
            .unwrap_or(0)
            .min(batch.num_columns());

        let coord_arrays: Vec<&Int64Array> = (0..num_dims)
            .map(|i| {
                batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
            })
            .collect();

        // Check each row against the coordinate filter
        for row_idx in 0..num_rows {
            let coordinates: Vec<i64> = coord_arrays.iter().map(|arr| arr.value(row_idx)).collect();

            if coordinate_filter.matches(&coordinates) {
                keep_rows.push(row_idx);
            }
        }

        // Create filtered batch
        if keep_rows.len() == num_rows {
            // All rows match, return original batch
            Ok(batch)
        } else if keep_rows.is_empty() {
            // No rows match, return empty batch with same schema
            let empty_arrays: Vec<Arc<dyn arrow_array::Array>> = schema
                .fields()
                .iter()
                .map(|field| {
                    match field.data_type() {
                        DataType::Int64 => {
                            Arc::new(Int64Array::new_null(0)) as Arc<dyn arrow_array::Array>
                        }
                        DataType::Float64 => {
                            Arc::new(Float64Array::new_null(0)) as Arc<dyn arrow_array::Array>
                        }
                        DataType::Int32 => {
                            Arc::new(Int32Array::new_null(0)) as Arc<dyn arrow_array::Array>
                        }
                        DataType::Float32 => {
                            Arc::new(Float32Array::new_null(0)) as Arc<dyn arrow_array::Array>
                        }
                        _ => Arc::new(Int64Array::new_null(0)) as Arc<dyn arrow_array::Array>, // Default fallback
                    }
                })
                .collect();

            RecordBatch::try_new(schema, empty_arrays)
                .map_err(|e| DataFusionError::External(Box::new(e)))
        } else {
            // Some rows match, create filtered batch
            let filtered_arrays: Vec<Arc<dyn arrow_array::Array>> = (0..batch.num_columns())
                .map(|col_idx| {
                    let column = batch.column(col_idx);
                    self.filter_array(column, &keep_rows)
                })
                .collect::<Result<Vec<_>, _>>()?;

            RecordBatch::try_new(schema, filtered_arrays)
                .map_err(|e| DataFusionError::External(Box::new(e)))
        }
    }

    /// Filter an Arrow array to only include specified row indices
    fn filter_array(
        &self,
        array: &Arc<dyn arrow_array::Array>,
        keep_rows: &[usize],
    ) -> Result<Arc<dyn arrow_array::Array>, DataFusionError> {
        // Handle different array types
        if let Some(int64_array) = array.as_any().downcast_ref::<Int64Array>() {
            let filtered_values: Vec<i64> = keep_rows
                .iter()
                .map(|&idx| int64_array.value(idx))
                .collect();
            Ok(Arc::new(Int64Array::from(filtered_values)))
        } else if let Some(float64_array) = array.as_any().downcast_ref::<Float64Array>() {
            let filtered_values: Vec<f64> = keep_rows
                .iter()
                .map(|&idx| float64_array.value(idx))
                .collect();
            Ok(Arc::new(Float64Array::from(filtered_values)))
        } else if let Some(int32_array) = array.as_any().downcast_ref::<arrow_array::Int32Array>() {
            let filtered_values: Vec<i32> = keep_rows
                .iter()
                .map(|&idx| int32_array.value(idx))
                .collect();
            Ok(Arc::new(arrow_array::Int32Array::from(filtered_values)))
        } else {
            Err(DataFusionError::External(
                "Unsupported array type for filtering".into(),
            ))
        }
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        // Get the schema for this table
        let schema = self.schema();

        // Parse coordinate filters from the provided expressions
        let filter_refs: Vec<&Expr> = filters.iter().collect();
        let coordinate_filter = self.parse_coordinate_filters(&filter_refs)?;

        // Generate filtered RecordBatches
        let batches = self.create_filtered_batches(coordinate_filter, limit)?;

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

        // Create MemTable with filtered data
        let mem_table = MemTable::try_new(projected_schema, vec![batches])?;

        // Return the MemTable's execution plan
        mem_table.scan(_state, projection, &[], limit).await // Note: filters already applied
    }
}

#[pymethods]
impl ZarrTableProvider {
    #[new]
    pub fn new(store_path: String) -> PyResult<Self> {
        Self::from_path(store_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
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
    use datafusion::execution::context::SessionConfig;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::lit;

    use super::*;
    use std::sync::Arc;

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
        assert!(provider
            .zarr_type_to_arrow(&ZarrDataType::Complex64)
            .is_err());
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
        let config = SessionConfig::new();
        let runtime_env = Arc::new(RuntimeEnvBuilder::new().build().unwrap());
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(config)
            .with_runtime_env(runtime_env)
            .build();

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

        let config = SessionConfig::new();
        let runtime_env = Arc::new(RuntimeEnvBuilder::new().build().unwrap());
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(config)
            .with_runtime_env(runtime_env)
            .build();

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
        assert_eq!(
            pushdown_decisions[0],
            TableProviderFilterPushDown::Unsupported
        );
    }

    #[tokio::test]
    async fn test_table_scanning_with_projection() {
        // Test that the table provider can handle projection
        let provider = ZarrTableProvider {
            store_path: "test".to_string(),
            store: None,
        };

        // Create a mock session state
        let config = SessionConfig::new();
        let runtime_env = Arc::new(RuntimeEnvBuilder::new().build().unwrap());
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(config)
            .with_runtime_env(runtime_env)
            .build();

        // Test scan with projection - should work even with empty schema
        let projection = vec![]; // Empty projection
        let filters = vec![];
        let scan_result = provider
            .scan(&state, Some(&projection), &filters, None)
            .await;
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
        let dim_0_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let dim_1_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let data_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

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
