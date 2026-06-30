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
//! multiple cores.
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

use arrow::array::{make_array, ArrayData, ArrayRef, BooleanArray, RecordBatch};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow::pyarrow::FromPyArrow;
use arrow::pyarrow::ToPyArrow;
use async_stream::try_stream;
use async_trait::async_trait;
use datafusion::catalog::memory::MemorySchemaProvider;
use datafusion::catalog::streaming::StreamingTable;
use datafusion::catalog::Session;
use datafusion::common::stats::Precision;
use datafusion::common::{Column, ColumnStatistics, Statistics};
use datafusion::common::{DataFusionError, Result as DFResult, ScalarValue, TableReference};
use datafusion::config::ConfigOptions;
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{
    BinaryExpr, ColumnarValue, Expr, Operator, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature, TableProviderFilterPushDown, TableType, Volatility,
};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::pruning::{PruningPredicate, PruningStatistics};
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::physical_plan::{
    displayable, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::prelude::{col, lit, DataFrame, SessionContext};
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use datafusion_physical_expr_common::physical_expr::snapshot_physical_expr;
use futures::StreamExt;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyTuple};
use tokio::runtime::Runtime;

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
    /// Exact number of rows in this partition (product of the chunk's
    /// per-dimension sizes). `None` when the producer did not supply it.
    /// Used to report exact `Statistics::num_rows` to the optimizer so
    /// cost-based rules (join build-side selection, broadcast vs. shuffle)
    /// have real cardinalities instead of guesses. xarray knows this
    /// exactly — it is the product of the partition's dimension lengths —
    /// so unlike most table providers these statistics are not estimates.
    pub num_rows: Option<usize>,
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
    /// Stored behind the `ProjectableStream` trait so `PrunableStreamingTable`
    /// is not coupled to `PyArrowStreamPartition`.
    partitions: Vec<(Arc<dyn ProjectableStream>, PartitionMetadata)>,
    /// Set of column names that are dimension columns (eligible for pruning)
    dimension_columns: HashSet<String>,
}

impl PrunableStreamingTable {
    fn new(
        schema: SchemaRef,
        partitions: Vec<(Arc<dyn ProjectableStream>, PartitionMetadata)>,
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

/// Extension trait for partition streams that support column projection.
///
/// Implemented by `PyArrowStreamPartition` so that `PrunableStreamingTable`
/// can push projections to Python factories without coupling to the concrete type.
/// Any new stream implementation (e.g. for non-Python backends) can implement this
/// trait and be used with `PrunableStreamingTable` directly.
trait ProjectableStream: PartitionStream + Debug {
    /// Return a new stream that emits only the specified columns.
    fn clone_with_projection(
        &self,
        projection: Arc<[String]>,
        projected_schema: SchemaRef,
    ) -> Arc<dyn PartitionStream>;

    /// Clone this stream as a generic `PartitionStream` Arc.
    fn clone_as_stream(&self) -> Arc<dyn PartitionStream>;
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
    Ok(PartitionMetadata {
        ranges,
        num_rows: None,
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

        // Exact per-partition row counts for the partitions that survive
        // pruning, in scan order. These feed `XarrayScanExec`'s statistics so
        // the optimizer sees real cardinalities.
        let included_metas: Vec<&PartitionMetadata> = included_indices
            .iter()
            .map(|&idx| &self.partitions[idx].1)
            .collect();
        let partition_rows: Vec<Precision<usize>> = included_metas
            .iter()
            .map(|meta| match meta.num_rows {
                Some(n) => Precision::Exact(n),
                None => Precision::Absent,
            })
            .collect();
        // Owned copies of the surviving partitions' coordinate ranges, so the
        // scan node can re-prune at execution time against a dynamic filter.
        let owned_metas: Vec<PartitionMetadata> =
            included_metas.iter().map(|&m| m.clone()).collect();

        // Handle empty case — all partitions pruned, return empty plan
        if included_indices.is_empty() {
            let empty_table = StreamingTable::try_new(Arc::clone(&self.schema), vec![])?;
            let inner = empty_table.scan(state, projection, filters, limit).await?;
            let stats = build_scan_statistics(inner.schema().as_ref(), &included_metas);
            return Ok(Arc::new(XarrayScanExec::new(
                inner,
                stats,
                partition_rows,
                owned_metas,
            )));
        }

        // Determine whether to push projection down to the Python factory.
        //
        // We push when the projection includes at least one data variable
        // (non-dimension column), because xarray can selectively load only
        // the requested data arrays while dimension coordinates are always
        // available via xarray's coordinate system.
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

            // Collect the requested column names to send to the factory.
            // Stored in an Arc so each clone_with_projection call shares the
            // same allocation via an atomic refcount increment (no N Vec copies).
            let proj_col_names: Arc<[String]> = indices
                .iter()
                .map(|&i| self.schema.field(i).name().to_string())
                .collect::<Vec<_>>()
                .into();

            // Clone each pruned partition with the projection baked in.
            // The factory will receive proj_col_names and load only those vars.
            let projected_partitions: Vec<Arc<dyn PartitionStream>> = included_indices
                .iter()
                .map(|&idx| {
                    self.partitions[idx].0.clone_with_projection(
                        Arc::clone(&proj_col_names),
                        Arc::clone(&projected_schema),
                    )
                })
                .collect();

            // StreamingTable already has the projected schema — pass None for
            // projection so it doesn't wrap the stream in a redundant ProjectionExec.
            let streaming = StreamingTable::try_new(projected_schema, projected_partitions)?;
            let inner = streaming.scan(state, None, filters, limit).await?;
            let stats = build_scan_statistics(inner.schema().as_ref(), &included_metas);
            Ok(Arc::new(XarrayScanExec::new(
                inner,
                stats,
                partition_rows,
                owned_metas,
            )))
        } else {
            // No projection pushdown — factory is called with None (loads all
            // columns). StreamingTable applies projection via ProjectionExec.
            let included_partitions: Vec<Arc<dyn PartitionStream>> = included_indices
                .iter()
                .map(|&idx| self.partitions[idx].0.clone_as_stream())
                .collect();
            let streaming = StreamingTable::try_new(Arc::clone(&self.schema), included_partitions)?;
            let inner = streaming.scan(state, projection, filters, limit).await?;
            let stats = build_scan_statistics(inner.schema().as_ref(), &included_metas);
            Ok(Arc::new(XarrayScanExec::new(
                inner,
                stats,
                partition_rows,
                owned_metas,
            )))
        }
    }
}

// ============================================================================
// Exact Statistics + Scan Wrapper
// ============================================================================

/// Sum a set of optional per-partition row counts into a `Precision<usize>`.
///
/// Exact only when *every* partition reports a count; if any is missing we
/// return `Absent` rather than an under-count, so the optimizer never sees a
/// cardinality smaller than reality.
fn sum_row_counts<'a>(metas: impl Iterator<Item = &'a PartitionMetadata>) -> Precision<usize> {
    let mut total: usize = 0;
    for meta in metas {
        match meta.num_rows {
            Some(n) => total += n,
            None => return Precision::Absent,
        }
    }
    Precision::Exact(total)
}

/// Fold two same-variant `ScalarBound`s, keeping the smaller (`keep_min`) or
/// larger one. Returns `None` if the variants differ (never expected within a
/// single dimension) so the caller can fall back to unknown.
fn fold_bound(a: &ScalarBound, b: &ScalarBound, keep_min: bool) -> Option<ScalarBound> {
    let ord = match (a, b) {
        (ScalarBound::Int64(x), ScalarBound::Int64(y)) => x.partial_cmp(y),
        (ScalarBound::Float64(x), ScalarBound::Float64(y)) => x.partial_cmp(y),
        (ScalarBound::TimestampNanos(x), ScalarBound::TimestampNanos(y)) => x.partial_cmp(y),
        _ => return None,
    }?;
    let take_a = if keep_min {
        ord != std::cmp::Ordering::Greater
    } else {
        ord != std::cmp::Ordering::Less
    };
    Some(if take_a { a.clone() } else { b.clone() })
}

/// Convert a coordinate bound into a `ScalarValue` matching a column's Arrow
/// type, so column statistics line up with the schema. Returns `None` for
/// type combinations we don't convert exactly (e.g. timestamp unit scaling),
/// in which case the column is left without min/max rather than risk a wrong
/// value.
fn bound_to_scalar(bound: &ScalarBound, dtype: &DataType) -> Option<ScalarValue> {
    match (bound, dtype) {
        (ScalarBound::Int64(v), DataType::Int64) => Some(ScalarValue::Int64(Some(*v))),
        (ScalarBound::Int64(v), DataType::Int32) => {
            i32::try_from(*v).ok().map(|x| ScalarValue::Int32(Some(x)))
        }
        (ScalarBound::Float64(v), DataType::Float64) => Some(ScalarValue::Float64(Some(*v))),
        (ScalarBound::Float64(v), DataType::Float32) => Some(ScalarValue::Float32(Some(*v as f32))),
        // Timestamp columns are intentionally left without min/max for now:
        // bounds are stored in nanoseconds but the column may be a different
        // unit, and an unscaled value would be wrong. num_rows already covers
        // the cost model; exact timestamp bounds can come with the dynamic
        // filter work that actually consumes them.
        _ => None,
    }
}

/// Build table-level `Statistics` for a scan over the given partitions.
///
/// `num_rows` is exact (the summed product of chunk dimension sizes). For
/// numeric dimension columns we also surface exact min/max, folded across the
/// included partitions — these are the join/filter key columns, and unlike
/// most providers the bounds are exact coordinate values, not estimates.
fn build_scan_statistics(output_schema: &Schema, metas: &[&PartitionMetadata]) -> Statistics {
    let mut stats = Statistics::new_unknown(output_schema);
    stats.num_rows = sum_row_counts(metas.iter().copied());

    for (col_idx, field) in output_schema.fields().iter().enumerate() {
        // Fold this column's min/max across every partition that has a range
        // for it. All partitions share the same bound variant per dimension.
        let mut folded: Option<(ScalarBound, ScalarBound)> = None;
        for meta in metas {
            if let Some(range) = meta.ranges.get(field.name()) {
                folded = Some(match folded {
                    None => (range.min.clone(), range.max.clone()),
                    Some((lo, hi)) => (
                        fold_bound(&lo, &range.min, true).unwrap_or(lo),
                        fold_bound(&hi, &range.max, false).unwrap_or(hi),
                    ),
                });
            }
        }

        if let Some((lo, hi)) = folded {
            let dtype = field.data_type();
            if let (Some(min), Some(max)) =
                (bound_to_scalar(&lo, dtype), bound_to_scalar(&hi, dtype))
            {
                stats.column_statistics[col_idx] = ColumnStatistics {
                    null_count: Precision::Exact(0),
                    max_value: Precision::Exact(max),
                    min_value: Precision::Exact(min),
                    sum_value: Precision::Absent,
                    distinct_count: Precision::Absent,
                    byte_size: Precision::Absent,
                };
            }
        }
    }

    stats
}

/// A thin scan operator that wraps an inner `StreamingTableExec` and reports
/// exact `Statistics` to the query optimizer.
///
/// Execution, schema, ordering, and partitioning are delegated verbatim to the
/// inner plan (so projection mechanics are reused unchanged); the only thing
/// this node adds is real cardinality. When consumed natively (not across the
/// FFI boundary, which drops statistics entirely), this is what lets
/// DataFusion's cost-based `JoinSelection` rule pick a sensible build side and
/// broadcast-vs-shuffle strategy.
struct XarrayScanExec {
    inner: Arc<dyn ExecutionPlan>,
    /// Output schema (== `inner.schema()`), cached for pruning predicates.
    schema: SchemaRef,
    statistics: Statistics,
    /// Exact row count per output partition (parallel to `inner` partitions),
    /// so `partition_statistics(Some(i))` is exact too.
    partition_rows: Vec<Precision<usize>>,
    /// Coordinate-range metadata per output partition (parallel to `inner`
    /// partitions), used to skip partitions a dynamic filter can't match.
    metas: Vec<PartitionMetadata>,
    /// Dynamic filters accepted from joins/TopK during the post-optimization
    /// filter-pushdown phase. Empty until a parent pushes one in.
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl XarrayScanExec {
    fn new(
        inner: Arc<dyn ExecutionPlan>,
        statistics: Statistics,
        partition_rows: Vec<Precision<usize>>,
        metas: Vec<PartitionMetadata>,
    ) -> Self {
        let schema = inner.schema();
        Self {
            inner,
            schema,
            statistics,
            partition_rows,
            metas,
            dynamic_filters: Vec::new(),
        }
    }

    /// Clone this node carrying an additional set of dynamic filters.
    fn with_dynamic_filters(&self, filters: Vec<Arc<dyn PhysicalExpr>>) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            schema: Arc::clone(&self.schema),
            statistics: self.statistics.clone(),
            partition_rows: self.partition_rows.clone(),
            metas: self.metas.clone(),
            dynamic_filters: filters,
        }
    }
}

impl Debug for XarrayScanExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("XarrayScanExec")
            .field("num_partitions", &self.metas.len())
            .field("num_dynamic_filters", &self.dynamic_filters.len())
            .finish()
    }
}

impl DisplayAs for XarrayScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let n = self.dynamic_filters.len();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "XarrayScanExec: rows={:?}", self.statistics.num_rows)?;
                if n > 0 {
                    write!(f, ", dynamic_filters={n}")?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                write!(f, "rows={:?}", self.statistics.num_rows)
            }
        }
    }
}

/// `PruningStatistics` view over a single partition's coordinate bounds, so a
/// dynamic filter snapshot can be evaluated against it to decide skipping.
struct SinglePartitionStats<'a> {
    meta: &'a PartitionMetadata,
    schema: &'a SchemaRef,
}

impl SinglePartitionStats<'_> {
    /// One-element array of this partition's min (or max) for `column`, typed to
    /// the column, or `None` if the column has no usable numeric bound.
    fn bound_array(&self, column: &Column, want_min: bool) -> Option<ArrayRef> {
        let field = self.schema.field_with_name(&column.name).ok()?;
        let dtype = field.data_type();
        let range = self.meta.ranges.get(&column.name)?;
        let bound = if want_min { &range.min } else { &range.max };
        let scalar = bound_to_scalar(bound, dtype)?;
        ScalarValue::iter_to_array(std::iter::once(scalar)).ok()
    }
}

impl PruningStatistics for SinglePartitionStats<'_> {
    fn min_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound_array(column, true)
    }
    fn max_values(&self, column: &Column) -> Option<ArrayRef> {
        self.bound_array(column, false)
    }
    fn num_containers(&self) -> usize {
        1
    }
    fn null_counts(&self, _column: &Column) -> Option<ArrayRef> {
        None
    }
    fn row_counts(&self, _column: &Column) -> Option<ArrayRef> {
        None
    }
    fn contained(
        &self,
        _column: &Column,
        _values: &std::collections::HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}

/// Returns true if every row of `meta`'s partition is provably excluded by at
/// least one of `filters` (evaluated at their current/snapshot value).
///
/// Conservative: any uncertainty (a predicate `PruningPredicate` can't model, a
/// column without bounds, an error) keeps the partition. Correctness does not
/// depend on this — it only skips work a join/TopK would discard anyway — so a
/// missed prune costs time, never results.
fn partition_pruned(
    filters: &[Arc<dyn PhysicalExpr>],
    meta: &PartitionMetadata,
    schema: &SchemaRef,
) -> bool {
    let stats = SinglePartitionStats { meta, schema };
    for filter in filters {
        // Snapshot resolves any DynamicFilterPhysicalExpr to its current bounds.
        let Ok(snapshot) = snapshot_physical_expr(Arc::clone(filter)) else {
            continue;
        };
        let Ok(predicate) = PruningPredicate::try_new(snapshot, Arc::clone(schema)) else {
            continue;
        };
        if let Ok(keep) = predicate.prune(&stats) {
            // One container in, one bool out: false means "cannot match".
            if keep.first() == Some(&false) {
                return true;
            }
        }
    }
    false
}

#[async_trait]
impl ExecutionPlan for XarrayScanExec {
    fn name(&self) -> &str {
        "XarrayScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        // Delegate partitioning + output ordering + boundedness to the inner
        // StreamingTableExec. This is also how declared coordinate ordering
        // (when present) reaches the optimizer.
        self.inner.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // A scan is a leaf; the inner plan is an execution detail, not a child
        // the optimizer should rewrite.
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        ctx: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Fast path: no dynamic filters, just stream the inner partition.
        if self.dynamic_filters.is_empty() {
            return self.inner.execute(partition, ctx);
        }

        // Otherwise defer the prune decision to first poll. By the time the
        // hash join probes (and this stream is polled) the build side has run
        // and updated the dynamic filter's bounds, so the snapshot is final.
        // Skipping here means the partition's Python factory is never called —
        // no remote read for a partition that cannot match.
        let inner = Arc::clone(&self.inner);
        let schema = Arc::clone(&self.schema);
        let filters = self.dynamic_filters.clone();
        let meta = self.metas[partition].clone();

        let stream = try_stream! {
            if partition_pruned(&filters, &meta, &schema) {
                return;
            }
            let mut s = inner.execute(partition, ctx)?;
            while let Some(batch) = s.next().await {
                yield batch?;
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }

    fn statistics(&self) -> DFResult<Statistics> {
        Ok(self.statistics.clone())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Statistics> {
        match partition {
            None => Ok(self.statistics.clone()),
            Some(i) => {
                let mut s = self.statistics.clone();
                s.num_rows = self
                    .partition_rows
                    .get(i)
                    .cloned()
                    .unwrap_or(Precision::Absent);
                Ok(s)
            }
        }
    }

    /// Accept dynamic filters (join/TopK) pushed in during the post phase.
    ///
    /// We mark them `Yes` so the producing join activates its bounds
    /// accumulator and keeps updating the filter at runtime. This is safe even
    /// though we only prune at partition granularity: a hash join still matches
    /// every surviving row, so the filter is a pure optimization here. Static
    /// (`Pre`-phase) filters are left to the existing logical pushdown.
    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        if phase != FilterPushdownPhase::Post || child_pushdown_result.parent_filters.is_empty() {
            return Ok(FilterPushdownPropagation::if_all(child_pushdown_result));
        }

        // Re-wrap each filter through `with_new_children`. For a
        // `DynamicFilterPhysicalExpr` this clones the shared `inner` Arc,
        // registering this scan as a *consumer* of the filter — which is what
        // makes the producing hash join treat it as "used" (`is_used()`) and
        // therefore actually compute and publish the build-side bounds at
        // runtime. Without this the captured filter stays at its initial
        // `true` value and never prunes.
        let parent_filters: Vec<Arc<dyn PhysicalExpr>> = child_pushdown_result
            .parent_filters
            .iter()
            .map(|f| {
                let original = Arc::clone(&f.filter);
                let children: Vec<Arc<dyn PhysicalExpr>> =
                    original.children().into_iter().cloned().collect();
                Arc::clone(&original)
                    .with_new_children(children)
                    .unwrap_or(original)
            })
            .collect();

        let supports = vec![PushedDown::Yes; parent_filters.len()];
        let updated = self.with_dynamic_filters(parent_filters);
        Ok(FilterPushdownPropagation {
            filters: supports,
            updated_node: Some(Arc::new(updated)),
        })
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
    ///
    /// Wrapped in `Arc` so `ProjectableStream::clone_with_projection` can share
    /// the same Python object across projected partitions without acquiring the
    /// GIL — only an atomic reference-count increment is needed.
    stream_factory: Arc<Py<PyAny>>,
    /// Column names to pass to the factory. `None` means load all columns.
    /// Stored as `Arc<[String]>` so multiple projected clones share one allocation.
    projection: Option<Arc<[String]>>,
}

impl PyArrowStreamPartition {
    fn new(stream_factory: Py<PyAny>, schema: SchemaRef) -> Self {
        Self {
            schema,
            stream_factory: Arc::new(stream_factory),
            projection: None,
        }
    }
}

impl ProjectableStream for PyArrowStreamPartition {
    /// Return a new partition that emits only the given columns.
    ///
    /// Clones the factory `Arc` (atomic refcount increment, no GIL) so the
    /// same Python callable is shared across all projected partitions.
    fn clone_with_projection(
        &self,
        projection: Arc<[String]>,
        projected_schema: SchemaRef,
    ) -> Arc<dyn PartitionStream> {
        Arc::new(Self {
            schema: projected_schema,
            stream_factory: Arc::clone(&self.stream_factory),
            projection: Some(projection),
        })
    }

    fn clone_as_stream(&self) -> Arc<dyn PartitionStream> {
        Arc::new(Self {
            schema: Arc::clone(&self.schema),
            stream_factory: Arc::clone(&self.stream_factory),
            projection: self.projection.clone(),
        })
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

        // Clone the factory Arc (no GIL needed) and the projection list.
        let factory = Arc::clone(&self.stream_factory);
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

// ============================================================================
// FFI Helpers
// ============================================================================

/// Extract an `FFI_LogicalExtensionCodec` from a Python session object.
///
/// DataFusion 52 passes the `SessionContext` to `__datafusion_table_provider__`
/// so that the provider can obtain the codec needed for physical-plan
/// serialisation across the FFI boundary.  The session exposes this via
/// `__datafusion_logical_extension_codec__()`, which returns a PyCapsule
/// named `"datafusion_logical_extension_codec"`.
///
/// Mirrors the helper in the official datafusion-python FFI example
/// (`examples/datafusion-ffi-example/src/utils.rs`).
fn ffi_logical_codec_from_pycapsule(
    session: Bound<'_, PyAny>,
) -> PyResult<FFI_LogicalExtensionCodec> {
    let attr = "__datafusion_logical_extension_codec__";
    let capsule = if session.hasattr(attr)? {
        session.getattr(attr)?.call0()?
    } else {
        session
    };

    let capsule = capsule.downcast::<PyCapsule>().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "session did not produce a PyCapsule for the logical extension codec: {e}"
        ))
    })?;

    let capsule_name = capsule.name()?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "datafusion_logical_extension_codec PyCapsule has no name set",
        )
    })?;
    let capsule_name = capsule_name.to_str()?;
    if capsule_name != "datafusion_logical_extension_codec" {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "expected capsule name 'datafusion_logical_extension_codec', got '{capsule_name}'"
        )));
    }

    // SAFETY: The capsule was produced by datafusion-python and contains a
    // valid FFI_LogicalExtensionCodec (#[repr(C)] StableAbi struct).
    let codec = unsafe { capsule.reference::<FFI_LogicalExtensionCodec>() };
    Ok(codec.clone())
}

// ============================================================================
// Python-visible Table Class
// ============================================================================

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
        // Stored as Arc<dyn ProjectableStream> so PrunableStreamingTable
        // is decoupled from PyArrowStreamPartition.
        let mut partition_list: Vec<(Arc<dyn ProjectableStream>, PartitionMetadata)> = Vec::new();
        for item_result in partitions.try_iter()? {
            let item = item_result?;
            // Accept either ``(factory, metadata)`` (legacy) or
            // ``(factory, metadata, num_rows)`` (preferred — carries the exact
            // partition row count for statistics). Try the 3-tuple first.
            let (factory_obj, meta_obj, num_rows): (Py<PyAny>, Py<PyAny>, Option<usize>) =
                match item.extract::<(Py<PyAny>, Py<PyAny>, usize)>() {
                    Ok((f, m, n)) => (f, m, Some(n)),
                    Err(_) => {
                        let (f, m): (Py<PyAny>, Py<PyAny>) = item.extract().map_err(|e| {
                            pyo3::exceptions::PyTypeError::new_err(format!(
                                "each partition must be a (factory, metadata_dict) or \
                                 (factory, metadata_dict, num_rows) tuple: {e}"
                            ))
                        })?;
                        (f, m, None)
                    }
                };
            let mut meta = convert_python_metadata_from_bound(meta_obj.bind(partitions.py()))?;
            meta.num_rows = num_rows;
            let partition: Arc<dyn ProjectableStream> =
                Arc::new(PyArrowStreamPartition::new(factory_obj, schema_ref.clone()));
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
    ///
    /// In DataFusion 52+, the caller passes `session` (a `SessionContext`)
    /// so that the provider can access task-context and codec information
    /// needed for physical plan serialisation across the FFI boundary.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = ffi_logical_codec_from_pycapsule(session)?;

        let provider: Arc<dyn TableProvider + Send> = self.table.clone();

        let ffi_provider = FFI_TableProvider::new_with_ffi_codec(provider, true, None, codec);

        let name = CString::new("datafusion_table_provider").unwrap();
        PyCapsule::new(py, ffi_provider, Some(name))
    }

    /// Get the schema of the table as a PyArrow Schema.
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.table
            .schema()
            .to_pyarrow(py)
            .map(|bound| bound.unbind())
    }

    fn __repr__(&self) -> String {
        format!("LazyArrowStreamTable(schema={:?})", self.table.schema())
    }
}

// ============================================================================
// Native Execution Context
// ============================================================================

/// Convert a DataFusion error into a Python exception.
fn df_err_to_py(e: DataFusionError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("DataFusion error: {e}"))
}

/// A DataFusion `SessionContext` that owns its tables *natively* (in-process,
/// same compiled DataFusion), rather than across the FFI boundary.
///
/// This matters because `datafusion-ffi` does not forward `Statistics` or
/// dynamic-filter pushdown across the boundary: a `ForeignExecutionPlan`
/// reports unknown statistics regardless of what the provider knows. Consuming
/// `PrunableStreamingTable` here, as a native `Arc<dyn TableProvider>`, lets the
/// optimizer see the exact cardinalities `XarrayScanExec` reports, accept
/// join-driven dynamic filters, and (later) custom physical rules.
///
/// Queries are returned as a lazy [`NativeDataFrame`]: nothing executes until
/// the result is streamed, and it is streamed in batches rather than
/// materialised, so a reduction over a petabyte-scale store never holds the
/// whole input (or output) in memory at once.
#[pyclass(name = "NativeContext")]
struct NativeContext {
    ctx: SessionContext,
    rt: Arc<Runtime>,
}

#[pymethods]
impl NativeContext {
    #[new]
    fn new() -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "failed to create tokio runtime: {e}"
            ))
        })?;
        Ok(Self {
            ctx: SessionContext::new(),
            rt: Arc::new(rt),
        })
    }

    /// Register a `LazyArrowStreamTable` natively under `name`.
    ///
    /// Unlike the FFI path (`register_table` with the capsule), this hands the
    /// provider directly to the in-process `SessionContext`, so its statistics
    /// and physical-plan capabilities are visible to the optimizer.
    ///
    /// `name` may be schema-qualified (e.g. `"era5.surface"`), so a dataset
    /// whose variables span several dimension groups registers as a SQL
    /// namespace just like the FFI path. The schema is created on demand.
    fn register_table(&self, name: &str, table: &LazyArrowStreamTable) -> PyResult<()> {
        let provider: Arc<dyn TableProvider> = table.table.clone();
        let reference = TableReference::from(name);

        // For a qualified name, make sure the target schema exists first —
        // DataFusion won't auto-create it.
        if let Some(schema_name) = reference.schema() {
            if let Some(catalog) = self.ctx.catalog("datafusion") {
                if catalog.schema(schema_name).is_none() {
                    catalog
                        .register_schema(schema_name, Arc::new(MemorySchemaProvider::new()))
                        .map_err(df_err_to_py)?;
                }
            }
        }

        self.ctx
            .register_table(reference, provider)
            .map_err(df_err_to_py)?;
        Ok(())
    }

    /// Register a Python callable as a native scalar UDF.
    ///
    /// The callable receives one PyArrow `Array` per argument and returns a
    /// PyArrow `Array`. This is how the `cftime()` filter helper — a Python
    /// function backed by the `cftime` library — becomes available inside the
    /// native engine, which cannot consume a `datafusion-python` UDF across the
    /// FFI boundary. `input_types` and `return_type` are PyArrow `DataType`s.
    fn register_scalar_udf(
        &self,
        name: &str,
        func: Py<PyAny>,
        input_types: Vec<Bound<'_, PyAny>>,
        return_type: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let in_types = input_types
            .iter()
            .map(DataType::from_pyarrow_bound)
            .collect::<PyResult<Vec<_>>>()?;
        let ret = DataType::from_pyarrow_bound(&return_type)?;
        let udf = PyScalarUdf::new(name.to_string(), in_types, ret, func);
        self.ctx.register_udf(ScalarUDF::new_from_impl(udf));
        Ok(())
    }

    /// Plan a SQL query and return a *lazy* `NativeDataFrame`.
    ///
    /// Planning runs now (so errors surface immediately) but no data is read
    /// until the frame is streamed.
    fn sql(&self, py: Python<'_>, query: &str) -> PyResult<NativeDataFrame> {
        let df = py
            .detach(|| self.rt.block_on(async { self.ctx.sql(query).await }))
            .map_err(df_err_to_py)?;
        Ok(NativeDataFrame {
            df,
            rt: Arc::clone(&self.rt),
        })
    }

    /// Return the physical plan for `query` as a string, with statistics shown.
    ///
    /// Internal: used by tests to confirm that exact cardinalities reach the
    /// optimizer (the statistics line is absent on the FFI path).
    fn explain(&self, py: Python<'_>, query: &str) -> PyResult<String> {
        py.detach(|| {
            self.rt.block_on(async {
                let df = self.ctx.sql(query).await?;
                let plan = df.create_physical_plan().await?;
                let rendered = displayable(plan.as_ref())
                    .set_show_statistics(true)
                    .indent(true)
                    .to_string();
                Ok::<_, DataFusionError>(rendered)
            })
        })
        .map_err(df_err_to_py)
    }
}

/// A lazy handle to a planned query on a [`NativeContext`].
///
/// Mirrors the slice of the `datafusion-python` `DataFrame` API that the xarray
/// round-trip needs — `schema()`, `execute_stream()`, plus column projection
/// and coordinate filtering for the chunked (lazy) reconstruction path — but
/// every consumer is streaming, so it scales to stores larger than memory.
#[pyclass(name = "NativeDataFrame")]
struct NativeDataFrame {
    df: DataFrame,
    rt: Arc<Runtime>,
}

#[pymethods]
impl NativeDataFrame {
    /// The PyArrow schema of the result (no execution).
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.df
            .schema()
            .inner()
            .as_ref()
            .to_pyarrow(py)
            .map(|b| b.unbind())
    }

    /// Begin streaming the result. Returns an iterator of PyArrow RecordBatches.
    ///
    /// The GIL is released while each batch is produced, so DataFusion's worker
    /// threads can re-acquire it to pull from the Python-backed partition
    /// streams. Batches are yielded one at a time — the full result is never
    /// collected.
    fn execute_stream(&self, py: Python<'_>) -> PyResult<NativeRecordBatchStream> {
        let df = self.df.clone();
        let stream = py
            .detach(|| self.rt.block_on(async { df.execute_stream().await }))
            .map_err(df_err_to_py)?;
        Ok(NativeRecordBatchStream {
            stream: Some(stream),
            rt: Arc::clone(&self.rt),
        })
    }

    /// Project to a subset of columns by name (lazy).
    fn select_columns(&self, columns: Vec<String>) -> PyResult<NativeDataFrame> {
        let exprs: Vec<Expr> = columns.iter().map(|c| col(c.as_str())).collect();
        let df = self.df.clone().select(exprs).map_err(df_err_to_py)?;
        Ok(NativeDataFrame {
            df,
            rt: Arc::clone(&self.rt),
        })
    }

    /// The distinct values of `column`, ascending (lazy).
    ///
    /// Used to discover a dimension's coordinate values for the chunked
    /// round-trip; the scan projects to this single column and skips data
    /// variables, so discovery reads coordinates only.
    fn distinct_sorted(&self, column: String) -> PyResult<NativeDataFrame> {
        let c = col(column.as_str());
        let df = self
            .df
            .clone()
            .select(vec![c.clone()])
            .map_err(df_err_to_py)?
            .distinct()
            .map_err(df_err_to_py)?
            .sort(vec![c.sort(true, false)])
            .map_err(df_err_to_py)?;
        Ok(NativeDataFrame {
            df,
            rt: Arc::clone(&self.rt),
        })
    }

    /// Keep only rows whose `column` is one of `values` (lazy).
    ///
    /// Used by the chunked round-trip to read a single output chunk: the
    /// coordinate predicate pushes into the scan and prunes partitions, so each
    /// chunk reads only the partitions it overlaps. `dtype_tag` matches the
    /// tags used for partition bounds (`int64`, `float64`, `timestamp_ns`).
    fn filter_in(
        &self,
        column: String,
        values: &Bound<'_, PyAny>,
        dtype_tag: &str,
    ) -> PyResult<NativeDataFrame> {
        let scalars = python_values_to_scalars(values, dtype_tag)?;
        let list: Vec<Expr> = scalars.into_iter().map(lit).collect();
        if list.is_empty() {
            return Ok(NativeDataFrame {
                df: self.df.clone(),
                rt: Arc::clone(&self.rt),
            });
        }
        let predicate = col(column.as_str()).in_list(list, false);
        let df = self.df.clone().filter(predicate).map_err(df_err_to_py)?;
        Ok(NativeDataFrame {
            df,
            rt: Arc::clone(&self.rt),
        })
    }
}

/// Convert a Python sequence of coordinate values into typed `ScalarValue`s.
fn python_values_to_scalars(
    values: &Bound<'_, PyAny>,
    dtype_tag: &str,
) -> PyResult<Vec<ScalarValue>> {
    let mut out = Vec::new();
    for item in values.try_iter()? {
        let item = item?;
        let scalar = match dtype_tag {
            "timestamp_ns" => ScalarValue::TimestampNanosecond(Some(item.extract::<i64>()?), None),
            "float64" => ScalarValue::Float64(Some(item.extract::<f64>()?)),
            "int64" => ScalarValue::Int64(Some(item.extract::<i64>()?)),
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported dtype tag for filter value: {dtype_tag}"
                )))
            }
        };
        out.push(scalar);
    }
    Ok(out)
}

/// A synchronous Python iterator over a DataFusion record-batch stream.
///
/// `unsendable` because a `SendableRecordBatchStream` is `Send` but not `Sync`
/// and is only ever advanced from the single Python thread that owns it.
#[pyclass(name = "NativeRecordBatchStream", unsendable)]
struct NativeRecordBatchStream {
    stream: Option<SendableRecordBatchStream>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl NativeRecordBatchStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Pull the next batch (or signal end of iteration).
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(stream) = self.stream.as_mut() else {
            return Ok(None);
        };
        let rt = Arc::clone(&self.rt);
        let next = py.detach(|| rt.block_on(async { stream.next().await }));
        match next {
            Some(Ok(batch)) => Ok(Some(batch.to_pyarrow(py)?.unbind())),
            Some(Err(e)) => {
                self.stream = None;
                Err(df_err_to_py(e))
            }
            None => {
                self.stream = None;
                Ok(None)
            }
        }
    }
}

/// A native DataFusion scalar UDF backed by a Python callable.
///
/// Each invocation converts the argument arrays to PyArrow, calls the Python
/// function (acquiring the GIL), and converts the PyArrow result back to Arrow.
/// Used to expose the `cftime()` filter helper to the native engine.
#[derive(Debug)]
struct PyScalarUdf {
    name: String,
    signature: Signature,
    return_type: DataType,
    /// `Arc` so the closure moved into `Python::attach` shares the callable
    /// without acquiring the GIL just to clone it.
    func: Arc<Py<PyAny>>,
}

impl PyScalarUdf {
    fn new(
        name: String,
        input_types: Vec<DataType>,
        return_type: DataType,
        func: Py<PyAny>,
    ) -> Self {
        Self {
            name,
            signature: Signature::exact(input_types, Volatility::Immutable),
            return_type,
            func: Arc::new(func),
        }
    }
}

// `ScalarUDFImpl` requires `Eq`/`Hash` (via `DynEq`/`DynHash`), which we can't
// derive through the `Py` callable. Identity is its signature — name, argument
// types, and return type — which is all the optimizer needs to compare UDFs.
impl PartialEq for PyScalarUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.signature == other.signature
            && self.return_type == other.return_type
    }
}

impl Eq for PyScalarUdf {}

impl std::hash::Hash for PyScalarUdf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.return_type.hash(state);
    }
}

impl ScalarUDFImpl for PyScalarUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DFResult<DataType> {
        Ok(self.return_type.clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DFResult<ColumnarValue> {
        let num_rows = args.number_rows;
        // Materialise every argument (scalars broadcast to `num_rows`) so the
        // Python function always sees full arrays.
        let arrays: Vec<ArrayRef> = args
            .args
            .into_iter()
            .map(|cv| cv.into_array(num_rows))
            .collect::<DFResult<_>>()?;

        let func = Arc::clone(&self.func);
        let result = Python::attach(|py| -> DFResult<ArrayRef> {
            let py_args = arrays
                .iter()
                .map(|a| a.to_data().to_pyarrow(py))
                .collect::<PyResult<Vec<_>>>()
                .and_then(|args| PyTuple::new(py, args))
                .map_err(|e| DataFusionError::Execution(format!("UDF arg conversion: {e}")))?;
            let out = func.call1(py, py_args).map_err(|e| {
                DataFusionError::Execution(format!("UDF '{}' failed: {e}", self.name))
            })?;
            let data = ArrayData::from_pyarrow_bound(out.bind(py))
                .map_err(|e| DataFusionError::Execution(format!("UDF result conversion: {e}")))?;
            Ok(make_array(data))
        })?;

        Ok(ColumnarValue::Array(result))
    }
}

/// Python module initialization
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LazyArrowStreamTable>()?;
    m.add_class::<NativeContext>()?;
    m.add_class::<NativeDataFrame>()?;
    m.add_class::<NativeRecordBatchStream>()?;
    Ok(())
}
