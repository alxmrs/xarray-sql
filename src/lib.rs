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
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
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
use pyo3::types::{PyCapsule, PyList};
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

            // Timestamp comparisons - convert to nanoseconds.
            // Use checked_mul to avoid silent overflow in release builds;
            // on overflow return None (conservative: include the partition).
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampNanosecond(Some(b), _)) => {
                Some(a.cmp(b))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampMicrosecond(Some(b), _)) => {
                b.checked_mul(1_000).map(|b_ns| a.cmp(&b_ns))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampMillisecond(Some(b), _)) => {
                b.checked_mul(1_000_000).map(|b_ns| a.cmp(&b_ns))
            }
            (ScalarBound::TimestampNanos(a), ScalarValue::TimestampSecond(Some(b), _)) => {
                b.checked_mul(1_000_000_000).map(|b_ns| a.cmp(&b_ns))
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
    /// Partition streams paired with their coordinate range metadata.
    /// Stored as the concrete type so scan() can clone them with a projection.
    partitions: Vec<(Arc<PyArrowStreamPartition>, PartitionMetadata)>,
    /// Set of column names that are dimension columns (eligible for pruning)
    dimension_columns: HashSet<String>,
}

impl PrunableStreamingTable {
    fn new(
        schema: SchemaRef,
        partitions: Vec<(Arc<PyArrowStreamPartition>, PartitionMetadata)>,
    ) -> Self {
        // Collect dimension column names from the first partition that has
        // non-empty metadata. All partitions share the same dimension names,
        // so we only need one representative. Using find_map keeps this O(D)
        // rather than O(N × D) — important when N is in the hundreds of
        // thousands (e.g. hourly chunks of a decades-long climate dataset).
        let dimension_columns: HashSet<String> = partitions
            .iter()
            .find_map(|(_, meta)| {
                if meta.ranges.is_empty() {
                    None
                } else {
                    Some(meta.ranges.keys().cloned().collect())
                }
            })
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
            Expr::Not(_) => {
                // NOT inverts the predicate. We cannot safely derive exclusion
                // from the inner result: if inner returns false (uncertain),
                // !false = true would incorrectly exclude the partition.
                // Example: partition [1,10], NOT (col > 5) ≡ col <= 5 —
                // inner returns false (max=10 > 5), but the partition contains
                // values 1–5 which satisfy col <= 5 and must be included.
                // Be conservative: never exclude on NOT.
                false
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
            // col != literal: can exclude only if range is a single point equal
            // to the literal — every row has that value, so none satisfy !=.
            Operator::NotEq => {
                let min_eq = matches!(
                    range.min.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Equal)
                );
                let max_eq = matches!(
                    range.max.compare_to_scalar(scalar),
                    Some(std::cmp::Ordering::Equal)
                );
                min_eq && max_eq
            }
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

/// Convert a Python object to a ScalarBound using an explicit dtype tag.
fn python_to_scalar_bound(obj: &Bound<'_, PyAny>, dtype_tag: &str) -> PyResult<ScalarBound> {
    match dtype_tag {
        "timestamp_ns" => {
            let val = obj.extract::<i64>()?;
            Ok(ScalarBound::TimestampNanos(val))
        }
        "float64" => {
            let val = obj.extract::<f64>()?;
            Ok(ScalarBound::Float64(val))
        }
        "int64" => {
            let val = obj.extract::<i64>()?;
            Ok(ScalarBound::Int64(val))
        }
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported dtype tag for partition bound: {dtype_tag}"
        ))),
    }
}

/// Convert a bound Python metadata dict to Rust PartitionMetadata.
///
/// Operates on an already-bound reference so no additional GIL acquisition
/// is needed — this is called from within a `#[pymethods]` context where
/// the GIL is already held.
fn convert_python_metadata_from_bound(meta_obj: &Bound<'_, PyAny>) -> PyResult<PartitionMetadata> {
    type MetaDict = HashMap<String, (Py<PyAny>, Py<PyAny>, String)>;
    let meta_dict: MetaDict = meta_obj.extract()?;
    let py = meta_obj.py();
    let mut ranges = HashMap::new();
    for (dim_name, (min_obj, max_obj, dtype_tag)) in meta_dict {
        let min_bound = python_to_scalar_bound(min_obj.bind(py), &dtype_tag)?;
        let max_bound = python_to_scalar_bound(max_obj.bind(py), &dtype_tag)?;
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

        // Handle empty case — all partitions pruned, return empty plan
        if included_indices.is_empty() {
            let empty_table = StreamingTable::try_new(Arc::clone(&self.schema), vec![])?;
            return empty_table.scan(state, projection, filters, limit).await;
        }

        // Determine whether to push projection down to the Python factory.
        //
        // We push when the projection includes at least one data variable
        // (non-dimension column), because xarray can selectively load only
        // the requested data arrays while dimension coordinates come for free
        // via to_dataframe()'s MultiIndex.
        //
        // We do NOT push when:
        //   - projection is None (load everything — factory receives None)
        //   - projection is Some([]) (COUNT(*) — let StreamingTable handle)
        //   - projection contains only dimension columns (ds.to_dataframe()
        //     needs at least one data variable; dimensions are always loaded)
        let push_projection = match projection {
            Some(indices) if !indices.is_empty() => indices
                .iter()
                .any(|&i| !self.dimension_columns.contains(self.schema.field(i).name())),
            _ => false,
        };

        if push_projection {
            let indices = projection.unwrap();

            // Build the projected schema (only the requested fields)
            let proj_fields: Vec<_> = indices
                .iter()
                .map(|&i| self.schema.field(i).clone())
                .collect();
            let projected_schema = Arc::new(Schema::new(proj_fields));

            // Collect the requested column names to send to the factory
            let proj_col_names: Vec<String> = indices
                .iter()
                .map(|&i| self.schema.field(i).name().to_string())
                .collect();

            // Clone each pruned partition with the projection baked in.
            // The factory will receive proj_col_names and load only those vars.
            let projected_partitions: Vec<Arc<dyn PartitionStream>> = included_indices
                .iter()
                .map(|&idx| {
                    Arc::new(self.partitions[idx].0.clone_with_projection(
                        proj_col_names.clone(),
                        Arc::clone(&projected_schema),
                    )) as Arc<dyn PartitionStream>
                })
                .collect();

            // StreamingTable already has the projected schema — pass None for
            // projection so it doesn't wrap the stream in a redundant ProjectionExec.
            let streaming = StreamingTable::try_new(projected_schema, projected_partitions)?;
            streaming.scan(state, None, &[], limit).await
        } else {
            // No projection pushdown — factory is called with None (loads all
            // columns). StreamingTable applies projection via ProjectionExec.
            let included_partitions: Vec<Arc<dyn PartitionStream>> = included_indices
                .iter()
                .map(|&idx| Arc::clone(&self.partitions[idx].0) as Arc<dyn PartitionStream>)
                .collect();
            let streaming = StreamingTable::try_new(Arc::clone(&self.schema), included_partitions)?;
            streaming.scan(state, projection, &[], limit).await
        }
    }
}

/// A partition stream that wraps a Python factory function that creates streams.
///
/// The factory is called lazily on each `execute()` invocation, allowing
/// the same table to be queried multiple times.
///
/// When `projection` is set, the factory is called with that list of column
/// names so that xarray only loads the requested data variables rather than
/// materializing every variable in the dataset.
struct PyArrowStreamPartition {
    schema: SchemaRef,
    /// A Python callable (factory) that returns a fresh stream.
    /// Signature: `make_stream(projection_names: Optional[List[str]]) -> RecordBatchReader`
    stream_factory: Py<PyAny>,
    /// Column names to pass to the factory. `None` means load all columns.
    projection: Option<Vec<String>>,
}

impl PyArrowStreamPartition {
    fn new(stream_factory: Py<PyAny>, schema: SchemaRef) -> Self {
        Self {
            schema,
            stream_factory,
            projection: None,
        }
    }

    /// Create a new partition with a baked-in column projection.
    ///
    /// The factory reference is cloned (requires the GIL) and the new
    /// partition uses `projected_schema` so the stream it produces has only
    /// the requested columns.
    fn clone_with_projection(&self, projection: Vec<String>, projected_schema: SchemaRef) -> Self {
        let factory = Python::attach(|py| self.stream_factory.clone_ref(py));
        Self {
            schema: projected_schema,
            stream_factory: factory,
            projection: Some(projection),
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

        // Clone the factory and projection with the GIL held
        let factory = Python::attach(|py| self.stream_factory.clone_ref(py));
        let projection = self.projection.clone();

        // Create a lazy stream using try_stream! macro.
        // The GIL is acquired only for the duration of each Python call
        // and released between batches.  After each yielded batch we call
        // yield_now() to explicitly suspend this task, giving the Tokio
        // executor a chance to poll other partition streams (which can then
        // acquire the GIL and make progress in parallel).
        let batch_stream = try_stream! {
            // Call factory with the projection argument.
            // `projection` is either a Python list of column names or None
            // (load all columns).  The factory always receives exactly one arg
            // so it can distinguish "no projection" from "empty projection".
            let reader: Py<PyAny> = Python::attach(|py| {
                let proj_arg = match &projection {
                    Some(cols) => PyList::new(py, cols.iter().map(|s| s.as_str()))
                        .map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to build projection list: {e}"
                            ))
                        })?
                        .into_any(),
                    None => py.None().into_bound(py).into_any(),
                };
                factory.call1(py, (proj_arg,)).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to call stream factory: {e}"))
                })
            })?;

            // Read batches until StopIteration.
            // The GIL is released between iterations; yield_now() ensures
            // other async tasks (i.e., other partitions) are scheduled
            // before this stream is polled again.
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
                    Ok(Some(batch)) => {
                        yield batch;
                        // Yield to the executor so that other partition
                        // streams can acquire the GIL and make progress.
                        tokio::task::yield_now().await;
                    }
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
/// SQL filters on dimension columns (time, lat, lon, etc.) automatically prune
/// partitions that can't contain matching rows when metadata is supplied.
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
/// import pyarrow as pa
///
/// schema = pa.schema([("time", pa.int64()), ("air", pa.float32())])
///
/// # Each element is a (factory_callable, metadata_dict) pair.
/// # metadata_dict maps dim name -> (min, max, dtype_str); use {} for no pruning.
/// def make_partitions():
///     yield (lambda: pa.RecordBatchReader.from_batches(schema, batches_0),
///            {"time": (0, 1_000_000_000, "int64")})
///     yield (lambda: pa.RecordBatchReader.from_batches(schema, batches_1),
///            {"time": (1_000_000_001, 2_000_000_000, "int64")})
///
/// table = LazyArrowStreamTable(make_partitions(), schema)
///
/// ctx = SessionContext()
/// ctx.register_table("air", table)
/// result = ctx.sql("SELECT AVG(air) FROM air WHERE time > 500000000").to_arrow_table()
/// ```
#[pyclass(name = "LazyArrowStreamTable")]
struct LazyArrowStreamTable {
    /// The underlying table provider with pruning support
    table: Arc<PrunableStreamingTable>,
}

#[pymethods]
impl LazyArrowStreamTable {
    /// Create a new LazyArrowStreamTable from an iterable of partition pairs.
    ///
    /// Args:
    ///     partitions: Any Python iterable yielding ``(factory, metadata_dict)``
    ///             pairs, where:
    ///             - ``factory`` is a zero-argument callable returning a
    ///               ``pa.RecordBatchReader`` (called lazily at query time).
    ///             - ``metadata_dict`` is a ``dict[str, tuple[Any, Any, str]]``
    ///               mapping dimension name to ``(min, max, dtype_str)``; pass
    ///               ``{}`` to skip pruning for a partition.
    ///             Generators are accepted, so partition state can be produced
    ///             one item at a time and released after Rust stores it.
    ///     schema: A PyArrow Schema for the table.
    ///
    /// Raises:
    ///     TypeError: If the schema is not a valid PyArrow Schema.
    ///     ValueError: If the partitions iterable is empty.
    #[new]
    #[pyo3(signature = (partitions, schema))]
    fn new(partitions: &Bound<'_, PyAny>, schema: &Bound<'_, PyAny>) -> PyResult<Self> {
        use arrow::datatypes::Schema;
        use arrow::pyarrow::FromPyArrow;

        let arrow_schema = Schema::from_pyarrow_bound(schema).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Failed to convert schema: {e}"))
        })?;
        let schema_ref = Arc::new(arrow_schema);

        // Consume the Python iterable one item at a time.
        // All GIL-bound work happens here, in a single GIL-held context,
        // eliminating the per-partition Python::attach() calls of the old
        // three-list approach.  Python can release each block dict, factory
        // closure, and metadata dict as soon as Rust has ingested them.
        // Stored as Arc<PyArrowStreamPartition> (not erased to dyn PartitionStream)
        // so that scan() can clone them with a projection at query time.
        let mut partition_list: Vec<(Arc<PyArrowStreamPartition>, PartitionMetadata)> = Vec::new();
        for item_result in partitions.try_iter()? {
            let item = item_result?;
            let (factory_obj, meta_obj): (Py<PyAny>, Py<PyAny>) = item.extract().map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "each partition must be a (factory, metadata_dict) tuple: {e}"
                ))
            })?;
            let meta = convert_python_metadata_from_bound(meta_obj.bind(partitions.py()))?;
            let partition = Arc::new(PyArrowStreamPartition::new(factory_obj, schema_ref.clone()));
            partition_list.push((partition, meta));
        }

        if partition_list.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "partitions iterable must not be empty",
            ));
        }

        let table = PrunableStreamingTable::new(schema_ref, partition_list);
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
