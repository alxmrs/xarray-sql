"""Lazy Arrow stream reader for xarray Datasets with projection pushdown.

This module provides XarrayRecordBatchReader, which implements the Arrow
PyCapsule Interface (__arrow_c_stream__) to enable zero-copy, lazy streaming
of xarray data to DataFusion and other Arrow consumers.

Key features:
- Lazy evaluation: data is only read when consumed
- Projection pushdown: only requested columns are processed
- Memory efficient: processes one chunk at a time
"""

from __future__ import annotations

import typing as t

import pyarrow as pa
import xarray as xr

from .df import Block, Chunks, block_slices, pivot, _parse_schema

if t.TYPE_CHECKING:
  from ._native import LazyArrowStreamTable


class XarrayRecordBatchReader:
  """A lazy Arrow stream reader for xarray Datasets with projection support.

  Implements the Arrow PyCapsule Interface (__arrow_c_stream__) to enable
  zero-copy, lazy streaming of xarray data to DataFusion and other Arrow
  consumers.

  The key property is that xarray blocks are only converted to Arrow
  RecordBatches when the consumer calls get_next (e.g., during DataFusion's
  collect()), NOT when the reader is created or registered.

  Supports projection pushdown: when a projection is specified, only those
  columns are read from xarray, reducing memory usage and processing time.

  Attributes:
      schema: The Arrow schema for the stream.

  Example:
      >>> import xarray as xr
      >>> from xarray_sql import XarrayRecordBatchReader
      >>> ds = xr.tutorial.open_dataset('air_temperature')
      >>> reader = XarrayRecordBatchReader(ds, chunks={'time': 240})
      >>> # At this point, NO data has been read from xarray
      >>> # Data is only read when consumed:
      >>> import pyarrow as pa
      >>> pa_reader = pa.RecordBatchReader.from_stream(reader)
      >>> for batch in pa_reader:
      ...     print(batch.num_rows)  # Data read here
  """

  def __init__(
      self,
      ds: xr.Dataset,
      chunks: Chunks = None,
      *,
      data_vars: t.Optional[t.Sequence[str]] = None,
      projection: t.Optional[t.Sequence[int]] = None,
      _iteration_callback: t.Optional[t.Callable[[Block], None]] = None,
      _columns_callback: t.Optional[t.Callable[[t.List[str]], None]] = None,
  ):
    """Initialize the lazy reader.

    Args:
        ds: An xarray Dataset. All data_vars must share the same dimensions.
        chunks: Xarray-like chunks specification. If not provided, uses
            the Dataset's existing chunks.
        data_vars: Optional list of data variable names to include. If provided,
            only these variables will be read. This is for explicit filtering
            at registration time. For query-time filtering, use projection.
        projection: Optional list of column indices to include. This is used
            for query-time projection pushdown from DataFusion. Column indices
            refer to the full schema (dims + data_vars in order).
        _iteration_callback: Internal callback for testing. Called with
            each block dict just before it's converted to Arrow.
        _columns_callback: Internal callback for testing projection pushdown.
            Called with the list of column names that will be processed.
    """
    # Store original dataset for schema reference
    self._original_ds = ds

    # Apply explicit data_vars filtering first (registration-time)
    if data_vars is not None:
      missing = set(data_vars) - set(ds.data_vars)
      if missing:
        raise ValueError(
            f"Requested data_vars not found in Dataset: {missing}. "
            f"Available: {list(ds.data_vars)}"
        )
      ds = ds[list(data_vars)]

    # Parse the full schema before any projection
    full_schema = _parse_schema(ds)
    full_column_names = [field.name for field in full_schema]

    # Apply projection (query-time column filtering)
    # Handle empty projection [] specially - this is used for COUNT(*) queries
    # where DataFusion doesn't need any columns, just the row count.
    self._empty_projection = projection is not None and len(projection) == 0
    if self._empty_projection:
      # Empty projection: empty schema, but we still need to produce row counts
      self._schema = pa.schema([])
      self._projected_column_names = []
    elif projection is not None:
      # Get projected column names
      projected_names = [full_column_names[i] for i in projection]

      # Separate dims and data_vars
      dim_names = list(ds.sizes.keys())
      projected_dims = [n for n in projected_names if n in dim_names]
      projected_data_vars = [n for n in projected_names if n in ds.data_vars]

      # Filter dataset to only include projected data_vars
      if projected_data_vars:
        ds = ds[projected_data_vars]
      else:
        # Edge case: only dims selected, keep one data_var to have valid dataset
        # This shouldn't normally happen, but handle it gracefully
        ds = ds[[list(ds.data_vars)[0]]]

      # Build projected schema in the correct order
      projected_fields = [full_schema.field(i) for i in projection]
      self._schema = pa.schema(projected_fields)
      self._projected_column_names = projected_names
    else:
      self._schema = full_schema
      self._projected_column_names = full_column_names

    self._ds = ds
    self._chunks = chunks
    self._iteration_callback = _iteration_callback
    self._columns_callback = _columns_callback
    self._consumed = False
    self._projection = projection

    # Validate dimensions
    if len(ds.data_vars) > 0:
      fst = next(iter(ds.data_vars.values())).dims
      if not all(da.dims == fst for da in ds.data_vars.values()):
        raise ValueError(
            "All dimensions must be equal. Please filter data_vars in the Dataset."
        )

  @property
  def schema(self) -> pa.Schema:
    """The Arrow schema for this stream."""
    return self._schema

  def _generate_batches(self) -> t.Iterator[pa.RecordBatch]:
    """Generate RecordBatches lazily from xarray blocks.

    This generator is only consumed when the Arrow stream's get_next
    is called, ensuring true lazy evaluation.
    """
    # Notify callback of columns being processed
    if self._columns_callback is not None:
      self._columns_callback(self._projected_column_names)

    for block in block_slices(self._ds, self._chunks):
      # Call the iteration callback if provided (for testing)
      if self._iteration_callback is not None:
        self._iteration_callback(block)

      # Convert this block to a RecordBatch
      block_ds = self._ds.isel(block)
      df = pivot(block_ds)
      num_rows = len(df)

      # Handle empty projection (COUNT(*) queries)
      if self._empty_projection:
        # Create a 0-column batch with the correct row count
        # PyArrow supports this via pa.RecordBatch.from_pydict with empty dict
        # but we need to set num_rows explicitly
        empty_batch = pa.RecordBatch.from_pydict({}, schema=self._schema)
        # Unfortunately from_pydict creates 0-row batch. We need to create
        # a batch with num_rows rows. Use a workaround with struct array.
        if num_rows > 0:
          # Create a struct array with num_rows elements, then create batch
          struct_arr = pa.StructArray.from_arrays([], names=[], mask=pa.array([False] * num_rows))
          yield pa.RecordBatch.from_struct_array(struct_arr)
        else:
          yield empty_batch
        continue

      # If we have a projection, reorder/filter columns to match projected schema
      if self._projection is not None:
        # Select only the columns in the projected schema, in correct order
        projected_col_names = [field.name for field in self._schema]
        # Filter to columns that exist in df
        available_cols = [c for c in projected_col_names if c in df.columns]
        df = df[available_cols]

      yield pa.RecordBatch.from_pandas(df, schema=self._schema)

  def __arrow_c_stream__(
      self, requested_schema: t.Optional[object] = None
  ) -> object:
    """Export as Arrow C Stream via PyCapsule.

    This method is called by Arrow consumers (like DataFusion) to get
    a C-level stream interface. The actual data iteration only begins
    when the consumer calls get_next on the stream.

    Args:
        requested_schema: Optional schema for type casting. Currently
            passed through to PyArrow's implementation.

    Returns:
        PyCapsule containing ArrowArrayStream pointer with name
        "arrow_array_stream".

    Raises:
        RuntimeError: If the stream has already been consumed.
    """
    if self._consumed:
      raise RuntimeError(
          "Stream already consumed. XarrayRecordBatchReader can only "
          "be iterated once. Create a new reader for additional iterations."
      )
    self._consumed = True

    # Create a PyArrow RecordBatchReader from our generator
    reader = pa.RecordBatchReader.from_batches(
        self._schema, self._generate_batches()
    )

    # Delegate to PyArrow's __arrow_c_stream__ implementation
    return reader.__arrow_c_stream__(requested_schema)

  def __arrow_c_schema__(
      self, requested_schema: t.Optional[object] = None
  ) -> object:
    """Export the schema as Arrow C Schema via PyCapsule.

    This allows consumers to inspect the schema without consuming the stream.

    Args:
        requested_schema: Optional schema for negotiation (unused).

    Returns:
        PyCapsule containing ArrowSchema pointer.
    """
    return self._schema.__arrow_c_schema__()


def read_xarray_lazy(
    ds: xr.Dataset,
    chunks: Chunks = None,
    *,
    data_vars: t.Optional[t.Sequence[str]] = None,
) -> XarrayRecordBatchReader:
  """Create a lazy Arrow stream reader from an xarray Dataset.

  This is the recommended way to register xarray data with DataFusion
  for lazy evaluation. Data is only read when queries are executed
  (e.g., during collect()), not during registration.

  Args:
      ds: An xarray Dataset. All data_vars must share the same dimensions.
      chunks: Xarray-like chunks specification. If not provided, uses
          the Dataset's existing chunks.
      data_vars: Optional list of data variable names to include. If provided,
          only these variables will be read, enabling early column filtering
          for more efficient queries. If None, all data_vars are included.

  Returns:
      An XarrayRecordBatchReader that implements __arrow_c_stream__.

  Example:
      >>> from datafusion import SessionContext
      >>> import xarray as xr
      >>> from xarray_sql import read_xarray_lazy, LazyArrowStreamTable
      >>>
      >>> ds = xr.tutorial.open_dataset('air_temperature')
      >>> reader = read_xarray_lazy(ds, chunks={'time': 240})
      >>> table = LazyArrowStreamTable(reader)
      >>>
      >>> ctx = SessionContext()
      >>> ctx.register_table_provider('air', table)
      >>>
      >>> # Data is only read here, during collect()
      >>> result = ctx.sql('SELECT AVG(air) FROM air').collect()
  """
  return XarrayRecordBatchReader(ds, chunks, data_vars=data_vars)


def read_xarray_table(
    ds: xr.Dataset,
    chunks: Chunks = None,
    *,
    data_vars: t.Optional[t.Sequence[str]] = None,
    _iteration_callback: t.Optional[t.Callable[[Block], None]] = None,
    _columns_callback: t.Optional[t.Callable[[t.List[str]], None]] = None,
) -> "LazyArrowStreamTable":
  """Create a lazy DataFusion table from an xarray Dataset.

  This is the simplest way to register xarray data with DataFusion.
  Data is only read when queries are executed (during collect()),
  not during registration. The table can be queried multiple times.

  **Projection Pushdown**: When a query only needs certain columns
  (e.g., `SELECT temperature FROM weather`), only those columns are
  read from xarray. This happens automatically - no user configuration
  needed.

  Args:
      ds: An xarray Dataset. All data_vars must share the same dimensions.
      chunks: Xarray-like chunks specification. If not provided, uses
          the Dataset's existing chunks.
      data_vars: Optional list of data variable names to include. This is
          for explicit registration-time filtering when you KNOW certain
          columns will never be needed. For automatic query-time filtering,
          just register with all columns and let projection pushdown work.
      _iteration_callback: Internal callback for testing. Called with
          each block dict just before it's converted to Arrow.
      _columns_callback: Internal callback for testing projection pushdown.
          Called with the list of column names that will be processed.

  Returns:
      A LazyArrowStreamTable ready for registration with DataFusion.

  Example:
      >>> from datafusion import SessionContext
      >>> import xarray as xr
      >>> from xarray_sql import read_xarray_table
      >>>
      >>> ds = xr.tutorial.open_dataset('air_temperature')
      >>> # Register with all columns
      >>> table = read_xarray_table(ds, chunks={'time': 240})
      >>>
      >>> ctx = SessionContext()
      >>> ctx.register_table_provider('air', table)
      >>>
      >>> # Only 'air' column is read from xarray (projection pushdown)
      >>> result = ctx.sql('SELECT AVG(air) FROM air').collect()
      >>>
      >>> # Different query, different columns read
      >>> result2 = ctx.sql('SELECT time, lat FROM air LIMIT 10').collect()
  """
  from ._native import LazyArrowStreamTable

  # Apply explicit data_vars filtering if specified (registration-time)
  if data_vars is not None:
    missing = set(data_vars) - set(ds.data_vars)
    if missing:
      raise ValueError(
          f"Requested data_vars not found in Dataset: {missing}. "
          f"Available: {list(ds.data_vars)}"
      )
    ds = ds[list(data_vars)]

  # Get schema from (possibly filtered) dataset
  schema = _parse_schema(ds)

  # Create a factory function that accepts projection (query-time filtering)
  # This factory is called by the Rust code at query execution time
  def make_stream(projection: t.Optional[t.List[int]] = None) -> XarrayRecordBatchReader:
    """Create a reader with optional projection for query-time column filtering.

    Args:
        projection: List of column indices to include, or None for all columns.
            This is passed by DataFusion at query execution time.

    Returns:
        An XarrayRecordBatchReader configured for the requested projection.
    """
    return XarrayRecordBatchReader(
        ds,
        chunks,
        projection=projection,
        _iteration_callback=_iteration_callback,
        _columns_callback=_columns_callback,
    )

  return LazyArrowStreamTable(make_stream, schema)
