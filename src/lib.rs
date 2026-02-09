// Licensed under the Apache License, Version 2.0
//!
//! Lazy Arrow stream table provider for xarray-sql.
//!
//! This module provides `LazyArrowStreamTable`, which wraps a Python object
//! implementing `__arrow_c_stream__` and exposes it as a DataFusion TableProvider
//! via the `__datafusion_table_provider__` protocol.
//!
//! ## Key Features
//!
//! - **Lazy evaluation**: Data is not read from the Python stream until query
//!   execution time (during `collect()`), not at registration time.
//!
//! - **Error propagation**: Errors during batch reading are properly propagated
//!   to DataFusion and surfaced to the user.
//!
//! - **Reusable tables**: The factory pattern allows the same table to be queried
//!   multiple times, with fresh data streams created for each query.
//!
//! ## Streaming Behavior
//!
//! Batches are read lazily one at a time during query execution. The GIL is acquired
//! for each batch read, allowing DataFusion to process and potentially filter batches
//! incrementally. This enables processing of larger-than-memory datasets when combined
//! with DataFusion's streaming execution.
//!
//! ## Parallel Execution
//!
//! Each xarray chunk becomes a separate partition, enabling parallel execution across
//! multiple cores. Due to a bug in DataFusion v51.0.0's `collect()` method, aggregation
//! queries should use `to_arrow_table()` instead to ensure complete results.
//! TODO(#107): Upgrading to the latest datafusion-python (52+) should fix this.
//!
//! ## Filter Pushdown (Partition Pruning)
//!
//! When partition metadata is provided, SQL filters on dimension columns (time, lat, lon)
//! automatically prune partitions that can't contain matching rows. For example:
//!
//! ```sql
//! SELECT * FROM air WHERE time > '2020-02-01'
//! ```
//!
//! Will skip loading partitions whose time ranges are entirely before 2020-02-01.
//! Supported operators: `=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `IN`, `AND`, `OR`.

use std::any::Any;
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::pyarrow::FromPyArrow;
use async_stream::try_stream;
use async_trait::async_trait;
use datafusion::catalog::streaming::StreamingTable;
use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result as DFResult, ScalarValue};
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{
    BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::physical_plan::{ExecutionPlan, SendableRecordBatchStream};
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use tokio::runtime::Handle;

// ============================================================================
// Partition Metadata Types for Filter Pushdown
// ============================================================================

// TODO(alxmrs, Claude): Support every valid xarray coordinate type.
/// Scalar value for dimension bounds, supporting common xarray coordinate types.
#[derive(Clone, Debug)]
pub enum ScalarBound {
    /// 64-bit integer (for integer coordinates)
    Int64(i64),
    /// 64-bit float (for lat/lon coordinates)
    Float64(f64),
    /// Nanoseconds since Unix epoch (for datetime64[ns] coordinates)
    TimestampNanos(i64),
}

impl ScalarBound {
    /// Compare this bound with a DataFusion ScalarValue.
    /// Returns None if types are incompatible.
    fn compare_to_scalar(&self, scalar: &ScalarValue) -> Option<std::cmp::Ordering> {
        match (self, scalar) {
            // Integer comparisons
            (ScalarBound::Int64(a), ScalarValue::Int64(Some(b))) => Some(a.cmp(b)),
            (ScalarBound::Int64(a), ScalarValue::Int32(Some(b))) => Some(a.cmp(&(*b as i64))),

            // Float comparisons
            (ScalarBound::Float64(a), ScalarValue::Float64(Some(b))) => a.partial_cmp(b),
            (ScalarBound::Float64(a), ScalarValue::Float32(Some(b))) => a.partial_cmp(&(*b as f64)),

            // Timestamp comparisons - convert to nanoseconds
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampNanosecond(Some(b), _)) => {
                Some(a.cmp(b))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampMicrosecond(Some(b), _)) => {
                Some(a.cmp(&(b * 1_000)))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampMillisecond(Some(b), _)) => {
                Some(a.cmp(&(b * 1_000_000)))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampSecond(Some(b), _)) => {
                Some(a.cmp(&(b * 1_000_000_000)))
            }

            // Incompatible types
            _ => None,
        }
    }
}

/// Range bounds for one dimension in a partition.
#[derive(Clone, Debug)]
pub struct DimensionRange {
    /// The column name (dimension name from xarray)
    pub column_name: String,
    /// Minimum value (inclusive) - first coordinate value in this partition
    pub min: ScalarBound,
    /// Maximum value (inclusive) - last coordinate value in this partition
    pub max: ScalarBound,
}

/// Metadata for a single partition, used for filter-based pruning.
#[derive(Clone, Debug, Default)]
pub struct PartitionMetadata {
    /// Dimension ranges for this partition, keyed by column name
    pub ranges: HashMap<String, DimensionRange>,
}

impl PartitionMetadata {
    /// Get the range for a specific dimension column.
    pub fn get_range(&self, column: &str) -> Option<&DimensionRange> {
        self.ranges.get(column)
    }
}

// ============================================================================
// Custom TableProvider with Filter Pushdown
// ============================================================================

/// A table provider that supports partition pruning via filter pushdown.
///
/// This wraps partition streams with their metadata and implements
/// `TableProvider::supports_filters_pushdown` and partition pruning in `scan()`.
struct PrunableStreamingTable {
    schema: SchemaRef,
    /// Partition streams paired with their coordinate range metadata
    partitions: Vec<(Arc<dyn PartitionStream>, PartitionMetadata)>,
    /// Set of column names that are dimension columns (eligible for pruning)
    dimension_columns: std::collections::HashSet<String>,
}

impl PrunableStreamingTable {
    fn new(
        schema: SchemaRef,
        partitions: Vec<(Arc<dyn PartitionStream>, PartitionMetadata)>,
    ) -> Self {
        // Collect dimension column names from partition metadata
        let dimension_columns: std::collections::HashSet<String> = partitions
            .first()
            .map(|(_, meta)| meta.ranges.keys().cloned().collect())
            .unwrap_or_default();

        Self {
            schema,
            partitions,
            dimension_columns,
        }
    }

    /// Determine which partitions should be included based on filters.
    /// Returns indices of partitions that may contain matching rows.
    fn prune_partitions(&self, filters: &[Expr]) -> Vec<usize> {
        self.partitions
            .iter()
            .enumerate()
            .filter(|(_, (_, meta))| {
                // Include partition unless a filter definitely excludes it
                !filters
                    .iter()
                    .any(|f| self.filter_excludes_partition(f, meta))
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Returns true if this filter definitely excludes the partition.
    /// Conservative: returns false (include) if uncertain.
    fn filter_excludes_partition(&self, expr: &Expr, meta: &PartitionMetadata) -> bool {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                // Handle AND/OR logic
                match op {
                    Operator::And => {
                        // For AND, exclude if either side excludes
                        self.filter_excludes_partition(left, meta)
                            || self.filter_excludes_partition(right, meta)
                    }
                    Operator::Or => {
                        // For OR, exclude only if both sides exclude
                        self.filter_excludes_partition(left, meta)
                            && self.filter_excludes_partition(right, meta)
                    }
                    // Handle comparison operators
                    _ => self.comparison_excludes(left, op, right, meta),
                }
            }
            Expr::Not(inner) => {
                // NOT inverts the logic, but we can't easily invert exclusion
                // Be conservative and don't exclude
                !self.filter_excludes_partition(inner, meta)
            }
            Expr::Between(between) => self.between_excludes(between, meta),
            Expr::InList(in_list) => self.in_list_excludes(in_list, meta),
            // Unknown expression type - be conservative
            _ => false,
        }
    }

    /// Check if a comparison expression excludes this partition.
    fn comparison_excludes(
        &self,
        left: &Expr,
        op: &Operator,
        right: &Expr,
        meta: &PartitionMetadata,
    ) -> bool {
        // Try to extract column and literal from either side
        let (col_name, scalar, flipped) = match (left, right) {
            (Expr::Column(c), Expr::Literal(s, _)) => (c.name.clone(), s, false),
            (Expr::Literal(s, _), Expr::Column(c)) => (c.name.clone(), s, true),
            _ => return false, // Not a simple column-literal comparison
        };

        // Get the dimension range for this column
        let range = match meta.get_range(&col_name) {
            Some(r) => r,
            None => return false, // Not a dimension column, can't prune
        };

        // Flip operator if literal was on left side
        let effective_op = if flipped { flip_operator(op) } else { *op };

        // Determine if partition can be excluded based on operator
        match effective_op {
            // col > literal: exclude if max <= literal
            Operator::Gt => matches!(
                range.max.compare_to_scalar(scalar),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ),
            // col >= literal: exclude if max < literal
            Operator::GtEq => {
                matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Less)
                )
            }
            // col < literal: exclude if min >= literal
            Operator::Lt => matches!(
                range.min.compare_to_scalar(scalar),
                Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
            ),
            // col <= literal: exclude if min > literal
            Operator::LtEq => {
                matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Greater)
                )
            }
            // col = literal: exclude if literal outside [min, max]
            Operator::Eq => {
                let below_min = matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Greater)
                );
                let above_max = matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Less)
                );
                below_min || above_max
            }
            // col != literal: can't exclude (partition may have other values)
            Operator::NotEq => false,
            // Other operators: be conservative
            _ => false,
        }
    }

    /// Check if a BETWEEN expression excludes this partition.
    fn between_excludes(
        &self,
        between: &datafusion::logical_expr::Between,
        meta: &PartitionMetadata,
    ) -> bool {
        if between.negated {
            // NOT BETWEEN is complex, be conservative
            return false;
        }

        // Extract column name
        let col_name = match between.expr.as_ref() {
            Expr::Column(c) => c.name.clone(),
            _ => return false,
        };

        // Get dimension range
        let range = match meta.get_range(&col_name) {
            Some(r) => r,
            None => return false,
        };

        // Extract low and high bounds
        let (low, high) = match (between.low.as_ref(), between.high.as_ref()) {
            (Expr::Literal(l, _), Expr::Literal(h, _)) => (l, h),
            _ => return false,
        };

        // Exclude if partition range doesn't overlap with [low, high]
        // No overlap if: partition.max < low OR partition.min > high
        let max_below_low = matches!(
            range.max.compare_to_scalar(low),
            Some(std::cmp::Ordering::Less)
        );
        let min_above_high = matches!(
            range.min.compare_to_scalar(high),
            Some(std::cmp::Ordering::Greater)
        );

        max_below_low || min_above_high
    }

    /// Check if an IN list expression excludes this partition.
    fn in_list_excludes(&self, in_list: &InList, meta: &PartitionMetadata) -> bool {
        if in_list.negated {
            // NOT IN is complex, be conservative
            return false;
        }

        // Extract column name
        let col_name = match in_list.expr.as_ref() {
            Expr::Column(c) => c.name.clone(),
            _ => return false,
        };

        // Get dimension range
        let range = match meta.get_range(&col_name) {
            Some(r) => r,
            None => return false,
        };

        // Check if any value in the list could be in this partition's range
        let any_in_range = in_list.list.iter().any(|expr| {
            if let Expr::Literal(scalar, _) = expr {
                // Value is in range if: min <= value <= max
                let above_min = matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                );
                let below_max = matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                );
                above_min && below_max
            } else {
                // Non-literal in list, be conservative
                true
            }
        });

        // Exclude only if NO values could be in range
        !any_in_range
    }

    /// Check if an expression is a filter on a dimension column.
    fn is_dimension_filter(&self, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
                Operator::And | Operator::Or => {
                    self.is_dimension_filter(left) || self.is_dimension_filter(right)
                }
                _ => self.expr_references_dimension(left) || self.expr_references_dimension(right),
            },
            Expr::Between(b) => self.expr_references_dimension(&b.expr),
            Expr::InList(i) => self.expr_references_dimension(&i.expr),
            Expr::Not(inner) => self.is_dimension_filter(inner),
            _ => false,
        }
    }

    /// Check if an expression references a dimension column.
    fn expr_references_dimension(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Column(c) => self.dimension_columns.contains(&c.name),
            _ => false,
        }
    }
}

/// Flip a comparison operator (for when literal is on left side).
fn flip_operator(op: &Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => *other,
    }
}

/// Convert a Python object to a ScalarBound.
fn python_to_scalar_bound(obj: &Bound<'_, PyAny>) -> PyResult<ScalarBound> {
    // Try integer first (includes timestamps as nanoseconds)
    if let Ok(val) = obj.extract::<i64>() {
        // Check if this might be a timestamp (very large number)
        // Python passes datetime64[ns] as nanoseconds since epoch
        if val.abs() > 1_000_000_000_000_000 {
            // Likely a nanosecond timestamp
            return Ok(ScalarBound::TimestampNanos(val));
        }
        return Ok(ScalarBound::Int64(val));
    }

    // Try float
    if let Ok(val) = obj.extract::<f64>() {
        return Ok(ScalarBound::Float64(val));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Unsupported type for partition bound: {:?}",
        obj.get_type().name()
    )))
}

/// Convert Python partition metadata dict to Rust PartitionMetadata.
fn convert_python_metadata(
    meta_dict: HashMap<String, (Py<PyAny>, Py<PyAny>)>,
) -> PyResult<PartitionMetadata> {
    Python::attach(|py| {
        let mut ranges = HashMap::new();
        for (dim_name, (min_obj, max_obj)) in meta_dict {
            let min_bound = python_to_scalar_bound(min_obj.bind(py))?;
            let max_bound = python_to_scalar_bound(max_obj.bind(py))?;
            ranges.insert(
                dim_name.clone(),
                DimensionRange {
                    column_name: dim_name,
                    min: min_bound,
                    max: max_bound,
                },
            );
        }
        Ok(PartitionMetadata { ranges })
    })
}

impl Debug for PrunableStreamingTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrunableStreamingTable")
            .field("schema", &self.schema)
            .field("num_partitions", &self.partitions.len())
            .field("dimension_columns", &self.dimension_columns)
            .finish()
    }
}

#[async_trait]
impl TableProvider for PrunableStreamingTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DFResult<Vec<TableProviderFilterPushDown>> {
        // For dimension filters we can do exact pruning at partition level
        // Return Inexact so DataFusion still applies row-level filtering
        Ok(filters
            .iter()
            .map(|expr| {
                if self.is_dimension_filter(expr) {
                    // We can prune partitions but not individual rows within
                    TableProviderFilterPushDown::Inexact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        // Prune partitions based on filters
        let included_indices = self.prune_partitions(filters);

        // Collect only the included partition streams
        let included_partitions: Vec<Arc<dyn PartitionStream>> = included_indices
            .iter()
            .map(|&idx| Arc::clone(&self.partitions[idx].0))
            .collect();

        // Handle empty case - create an empty streaming table
        if included_partitions.is_empty() {
            // Create a streaming table with no partitions
            // DataFusion will return empty result
            let empty_table = StreamingTable::try_new(Arc::clone(&self.schema), vec![])?;
            return empty_table.scan(state, projection, &[], limit).await;
        }

        // Create StreamingTable with the pruned partitions
        let streaming = StreamingTable::try_new(Arc::clone(&self.schema), included_partitions)?;

        // Delegate to StreamingTable for actual execution
        // Pass empty filters since we've already done partition pruning
        // DataFusion will still apply row-level filtering
        streaming.scan(state, projection, &[], limit).await
    }
}

/// A partition stream that wraps a Python factory function that creates streams.
///
/// The factory is called lazily on each `execute()` invocation, allowing
/// the same table to be queried multiple times.
struct PyArrowStreamPartition {
    schema: SchemaRef,
    /// A Python callable (factory) that returns a fresh stream implementing `__arrow_c_stream__`.
    /// Called on each execute() to create a new stream.
    stream_factory: Py<PyAny>,
}

impl PyArrowStreamPartition {
    fn new(stream_factory: Py<PyAny>, schema: SchemaRef) -> Self {
        Self {
            schema,
            stream_factory,
        }
    }
}

impl Debug for PyArrowStreamPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyArrowStreamPartition")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PartitionStream for PyArrowStreamPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let schema = Arc::clone(&self.schema);

        // Clone the factory with the GIL held
        let factory = Python::attach(|py| self.stream_factory.clone_ref(py));

        // Create a lazy stream using try_stream! macro.
        // This is cleaner than manual state management with unfold.
        // Each iteration acquires the GIL and reads one batch.
        let batch_stream = try_stream! {
            // Call factory to get the PyArrow RecordBatchReader
            let reader: Py<PyAny> = Python::attach(|py| {
                factory.call0(py).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to call stream factory: {e}"))
                })
            })?;

            // Read batches until StopIteration
            loop {
                let batch_result = Python::attach(|py| {
                    let bound_reader = reader.bind(py);
                    match bound_reader.call_method0("read_next_batch") {
                        Ok(batch_obj) => {
                            match RecordBatch::from_pyarrow_bound(&batch_obj) {
                                Ok(batch) => Ok(Some(batch)),
                                Err(e) => Err(DataFusionError::Execution(format!(
                                    "Failed to convert batch from PyArrow: {e}"
                                ))),
                            }
                        }
                        Err(e) => {
                            if e.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                                Ok(None) // Stream exhausted normally
                            } else {
                                Err(DataFusionError::Execution(format!(
                                    "Error reading batch from stream: {e}"
                                )))
                            }
                        }
                    }
                });

                match batch_result {
                    Ok(Some(batch)) => yield batch,
                    Ok(None) => break,
                    Err(e) => Err(e)?,
                }
            }
        };

        Box::pin(RecordBatchStreamAdapter::new(schema, batch_stream))
    }
}

/// A lazy table provider that wraps Python stream factory functions.
///
/// This class implements the `__datafusion_table_provider__` protocol, allowing
/// it to be registered with DataFusion's `SessionContext.register_table()`.
///
/// Data is NOT read until query execution time - this enables true lazy evaluation.
/// Each partition has its own factory function that is called on query execution
/// to create a fresh stream, enabling true parallelism in DataFusion.
///
/// ## Filter Pushdown
///
/// When partition metadata is provided, SQL filters on dimension columns (time, lat, lon, etc.)
/// will automatically prune partitions that can't contain matching rows. This dramatically
/// improves query performance for range queries on large datasets.
///
/// # Note
///
/// Due to a bug in DataFusion v51.0.0's `collect()` method, use `to_arrow_table()`
/// instead for aggregation queries to ensure complete results.
///
/// # Example
///
/// ```python
/// from datafusion import SessionContext
/// from xarray_sql import LazyArrowStreamTable
/// import pyarrow as pa
///
/// # Create factories for each partition (chunk)
/// factories = [
///     lambda: pa.RecordBatchReader.from_batches(schema, batches_chunk_0),
///     lambda: pa.RecordBatchReader.from_batches(schema, batches_chunk_1),
/// ]
///
/// # Partition metadata for filter pushdown (optional)
/// # Each dict maps dimension name to (min, max) coordinate values
/// metadata = [
///     {'time': (0, 1000000000), 'lat': (-90.0, 0.0)},  # partition 0
///     {'time': (1000000001, 2000000000), 'lat': (0.0, 90.0)},  # partition 1
/// ]
///
/// # Wrap factories in lazy table with metadata
/// table = LazyArrowStreamTable(factories, schema, metadata)
///
/// # Register with DataFusion
/// ctx = SessionContext()
/// ctx.register_table("air", table)
///
/// # Queries with filters on dimension columns will prune partitions!
/// # This query might only read partition 1:
/// result = ctx.sql("SELECT AVG(air) FROM air WHERE lat > 0").to_arrow_table()
/// ```
#[pyclass(name = "LazyArrowStreamTable")]
struct LazyArrowStreamTable {
    /// The underlying table provider with pruning support
    table: Arc<PrunableStreamingTable>,
}

#[pymethods]
impl LazyArrowStreamTable {
    /// Create a new LazyArrowStreamTable from stream factory functions.
    ///
    /// Args:
    ///     stream_factories: A list of callables, each returning a Python object
    ///             implementing the Arrow PyCapsule interface (`__arrow_c_stream__`).
    ///             Each factory represents one partition, enabling parallel execution.
    ///             Called on each query execution to create fresh streams.
    ///     schema: A PyArrow Schema for the table. Required since the factories
    ///             haven't been called yet.
    ///     partition_metadata: Optional list of dicts mapping dimension names to
    ///             (min, max) tuples. When provided, enables filter pushdown to
    ///             prune partitions based on SQL WHERE clauses.
    ///
    /// Raises:
    ///     TypeError: If the schema is not a valid PyArrow Schema.
    ///     ValueError: If stream_factories is empty or metadata length doesn't match.
    #[new]
    #[pyo3(signature = (stream_factories, schema, partition_metadata=None))]
    fn new(
        stream_factories: &Bound<'_, PyAny>,
        schema: &Bound<'_, PyAny>,
        partition_metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Convert the PyArrow schema to Arrow schema
        use arrow::datatypes::Schema;
        use arrow::pyarrow::FromPyArrow;

        let arrow_schema = Schema::from_pyarrow_bound(schema).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
        })?;
        let schema_ref = Arc::new(arrow_schema);

        // Extract factories from the Python list
        let factories: Vec<Py<PyAny>> = stream_factories.extract().map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "stream_factories must be a list of callables: {e}"
            ))
        })?;

        if factories.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "stream_factories must not be empty",
            ));
        }

        // Extract and convert partition metadata if provided
        let metadata_list: Vec<PartitionMetadata> = if let Some(meta_py) = partition_metadata {
            type MetaDict = HashMap<String, (Py<PyAny>, Py<PyAny>)>;
            let meta_dicts: Vec<MetaDict> = meta_py.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "partition_metadata must be a list of dicts: {e}"
                ))
            })?;

            if meta_dicts.len() != factories.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "partition_metadata length ({}) must match stream_factories length ({})",
                    meta_dicts.len(),
                    factories.len()
                )));
            }

            meta_dicts
                .into_iter()
                .map(convert_python_metadata)
                .collect::<PyResult<Vec<_>>>()?
        } else {
            // No metadata provided - create empty metadata for each partition
            vec![PartitionMetadata::default(); factories.len()]
        };

        // Create partitions with their metadata
        let partitions: Vec<(Arc<dyn PartitionStream>, PartitionMetadata)> = factories
            .into_iter()
            .zip(metadata_list)
            .map(|(factory, meta)| {
                let partition = Arc::new(PyArrowStreamPartition::new(factory, schema_ref.clone()))
                    as Arc<dyn PartitionStream>;
                (partition, meta)
            })
            .collect();

        // Create the PrunableStreamingTable
        let table = PrunableStreamingTable::new(schema_ref, partitions);

        Ok(Self {
            table: Arc::new(table),
        })
    }

    /// Returns a PyCapsule implementing the DataFusion TableProvider FFI.
    ///
    /// This method is called by DataFusion's `register_table()` to get a
    /// foreign table provider that can be used in queries.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        // Create the FFI table provider
        let provider: Arc<dyn TableProvider + Send> = self.table.clone();

        // Try to get the current tokio runtime handle (available when called from DataFusion context)
        let runtime = Handle::try_current().ok();

        // Create FFI wrapper with filter pushdown ENABLED
        let ffi_provider = FFI_TableProvider::new(
            provider, true, // can_support_pushdown_filters = ENABLED
            runtime,
        );

        // Create the capsule name
        let name = CString::new("datafusion_table_provider").unwrap();

        // Create the PyCapsule without a destructor closure
        // The PyCapsule takes ownership of the FFI_TableProvider
        PyCapsule::new(py, ffi_provider, Some(name))
    }

    /// Get the schema of the table as a PyArrow Schema.
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use arrow::pyarrow::ToPyArrow;
        self.table
            .schema()
            .to_pyarrow(py)
            .map(|bound| bound.unbind())
    }

    fn __repr__(&self) -> String {
        format!("LazyArrowStreamTable(schema={:?})", self.table.schema())
    }
}

/// Python module initialization
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LazyArrowStreamTable>()?;
    Ok(())
}
