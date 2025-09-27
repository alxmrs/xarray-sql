use arrow::array::RecordBatch;
use arrow::pyarrow::PyArrowType;
use arrow_schema::Schema;
use arrow_zarr::table::ZarrTable;
use arrow_zarr::zarr_store_opener::ZarrRecordBatchStream;
use datafusion::datasource::TableProvider;
use datafusion::error::DataFusionError;
use futures::stream::TryStreamExt;
use object_store::local::LocalFileSystem;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;
use std::path::PathBuf;
use std::sync::Arc;
use zarrs_object_store::AsyncObjectStore;

/// A simple reader that converts Zarr data to Arrow RecordBatches for Python bindings
#[pyclass(name = "ZarrTableProvider")]
#[derive(Clone, Debug)]
pub struct ZarrTableProvider {
    store_path: String,
}

impl ZarrTableProvider {
    fn from_path(store_path: String) -> Result<Self, DataFusionError> {
        Ok(Self { store_path })
    }

    /// Read all data from the Zarr store and return as PyArrow RecordBatches
    async fn read_to_arrow_async(&self) -> Result<Vec<RecordBatch>, DataFusionError> {
        // Use ZarrTable to get the schema
        let zarr_table = ZarrTable::from_path(self.store_path.clone()).await;
        let schema = zarr_table.schema();

        println!("Inferred schema: {:?}", schema);

        // Create the object store from the path
        let path = PathBuf::from(&self.store_path);
        let store = Arc::new(AsyncObjectStore::new(
            LocalFileSystem::new_with_prefix(path)
                .map_err(|e| DataFusionError::External(Box::new(e)))?,
        ));

        // Create the ZarrRecordBatchStream
        let stream = ZarrRecordBatchStream::try_new(
            store,
            schema,
            None,    // group
            None,    // projection
            1,       // n_partitions
            0,       // partition
        )
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Collect all RecordBatches
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(batches)
    }
}

#[pymethods]
impl ZarrTableProvider {
    #[new]
    pub fn new(store_path: String) -> PyResult<Self> {
        Self::from_path(store_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Read all data and return as a list of PyArrow RecordBatches
    pub fn read_to_arrow<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let store_path = self.store_path.clone();
        future_into_py(py, async move {
            let provider = ZarrTableProvider { store_path };
            let batches = provider
                .read_to_arrow_async()
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Convert to PyArrow objects
            let py_batches: Result<Vec<PyArrowType<RecordBatch>>, PyErr> = batches
                .into_iter()
                .map(|batch| Ok(PyArrowType(batch)))
                .collect();

            py_batches
        })
    }

    /// Get the schema as a PyArrow Schema
    pub fn get_schema<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let store_path = self.store_path.clone();
        future_into_py(py, async move {
            // Use ZarrTable to get the schema
            let zarr_table = ZarrTable::from_path(store_path).await;
            let schema = zarr_table.schema();

            Ok(PyArrowType(Schema::from(schema.as_ref().clone())))
        })
    }
}

#[pymodule]
fn zarrquet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZarrTableProvider>()?;
    Ok(())
}