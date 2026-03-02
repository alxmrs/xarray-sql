"""Lazy Arrow stream reader for xarray Datasets.

This module provides XarrayRecordBatchReader, which implements the Arrow
PyCapsule Interface (__arrow_c_stream__) to enable zero-copy, lazy streaming
of xarray data to DataFusion and other Arrow consumers.

The implementation delegates to PyArrow's RecordBatchReader for the
actual stream implementation, wrapping xarray block iteration in a generator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pyarrow as pa
import xarray as xr

from .df import (
    Block,
    Chunks,
    _parse_schema,
    block_slices,
    dataset_to_record_batch,
    partition_metadata,
)

if TYPE_CHECKING:
  from ._native import LazyArrowStreamTable


class XarrayRecordBatchReader:
  """A lazy Arrow stream reader for xarray Datasets.

  Implements the Arrow PyCapsule Interface (__arrow_c_stream__) to enable
  zero-copy, lazy streaming of xarray data to DataFusion and other Arrow
  consumers.

  The key property is that xarray blocks are only converted to Arrow
  RecordBatches when the consumer calls get_next (e.g., during DataFusion's
  collect()), NOT when the reader is created or registered.

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
      _iteration_callback: Callable[[Block], None] | None = None,
  ):
    """Initialize the lazy reader.

    Args:
        ds: An xarray Dataset. All data_vars must share the same dimensions.
        chunks: Xarray-like chunks specification. If not provided, uses
            the Dataset's existing chunks.
        _iteration_callback: Internal callback for testing. Called with
            each block dict just before it's converted to Arrow. This
            allows tests to track when iteration actually occurs.
    """
    self._ds = ds
    self._chunks = chunks
    self._schema = _parse_schema(ds)
    self._iteration_callback = _iteration_callback
    self._consumed = False

    # Validate dimensions
    fst = next(iter(ds.values())).dims
    if not all(da.dims == fst for da in ds.values()):
      raise ValueError(
          "All dimensions must be equal. Please filter data_vars in the Dataset."
      )

  @property
  def schema(self) -> pa.Schema:
    """The Arrow schema for this stream."""
    return self._schema

  def _generate_batches(self) -> Iterator[pa.RecordBatch]:
    """Generate RecordBatches lazily from xarray blocks.

    This generator is only consumed when the Arrow stream's get_next
    is called, ensuring true lazy evaluation.
    """
    for block in block_slices(self._ds, self._chunks):
      # Call the iteration callback if provided (for testing)
      if self._iteration_callback is not None:
        self._iteration_callback(block)

      # Convert this block to a RecordBatch directly from numpy arrays,
      # bypassing the pandas round-trip for lower peak memory usage.
      yield dataset_to_record_batch(self._ds.isel(block), self._schema)

  def __arrow_c_stream__(
      self, requested_schema: object | None = None
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
    # The generator is NOT consumed here - only when get_next is called
    reader = pa.RecordBatchReader.from_batches(
        self._schema, self._generate_batches()
    )

    # Delegate to PyArrow's __arrow_c_stream__ implementation
    return reader.__arrow_c_stream__(requested_schema)

  def __arrow_c_schema__(
      self, requested_schema: object | None = None
  ) -> object:
    """Export the schema as Arrow C Schema via PyCapsule.

    This allows consumers to inspect the schema without consuming the stream.

    Args:
        requested_schema: Optional schema for negotiation (unused).

    Returns:
        PyCapsule containing ArrowSchema pointer.
    """
    return self._schema.__arrow_c_schema__()


def read_xarray(ds: xr.Dataset, chunks: Chunks = None) -> pa.RecordBatchReader:
  """Pivots an Xarray Dataset into a PyArrow Table, partitioned by chunks.

  Args:
    ds: An Xarray Dataset. All `data_vars` must share the same dimensions.
    chunks: Xarray-like chunks. If not provided, will default to the Dataset's
     chunks. The product of the chunk sizes becomes the standard length of each
     dataframe partition.

  Returns:
    A PyArrow RecordBatchReader, which is a table representation of the input
    Dataset.
  """
  reader = XarrayRecordBatchReader(ds, chunks=chunks)
  return pa.RecordBatchReader.from_stream(reader)


def read_xarray_table(
    ds: xr.Dataset,
    chunks: Chunks = None,
    *,
    _iteration_callback: Callable[[Block], None] | None = None,
) -> "LazyArrowStreamTable":
  """Create a lazy DataFusion table from an xarray Dataset.

  This is the simplest way to register xarray data with DataFusion.
  Data is only read when queries are executed, not during registration.
  The table can be queried multiple times.

  Each chunk becomes a separate partition, enabling DataFusion's parallel
  execution across multiple cores.

  Filter Pushdown:
      SQL queries with WHERE clauses on dimension columns (time, lat, lon, etc.)
      automatically prune partitions that can't contain matching rows. For example:

          # This query will skip loading partitions with time < '2020-02-01'
          result = ctx.sql('SELECT * FROM air WHERE time > \"2020-02-01\"').to_arrow_table()

      Supported operators: =, <, >, <=, >=, BETWEEN, IN, AND, OR.

  Note:
      Due to a bug in DataFusion v51.0.0's collect() method, use
      `to_arrow_table()` instead of `collect()` for aggregation queries
      to ensure complete results. This should be fixed in datafusion-python 52+.

  Args:
      ds: An xarray Dataset. All data_vars must share the same dimensions.
      chunks: Xarray-like chunks specification. If not provided, uses
          the Dataset's existing chunks.
      _iteration_callback: Internal callback for testing. Called with
          each block dict just before it's converted to Arrow.

  Returns:
      A LazyArrowStreamTable ready for registration with DataFusion.

  Example:
      >>> from datafusion import SessionContext
      >>> import xarray as xr
      >>> from xarray_sql import read_xarray_table
      >>>
      >>> ds = xr.tutorial.open_dataset('air_temperature')
      >>> table = read_xarray_table(ds, chunks={'time': 240})
      >>>
      >>> ctx = SessionContext()
      >>> ctx.register_table('air', table)
      >>>
      >>> # Data is only read here, during query execution
      >>> # Filters on 'time' will prune partitions automatically!
      >>> result = ctx.sql('SELECT AVG(air) FROM air').to_arrow_table()
  """
  from ._native import LazyArrowStreamTable

  # Get schema from dataset without creating a stream
  schema = _parse_schema(ds)

  # Compute block slices - need list for metadata computation
  blocks = list(block_slices(ds, chunks))

  # Compute partition metadata for filter pushdown
  # This extracts min/max coordinate values per partition
  metadata = partition_metadata(ds, blocks)

  # Create a factory function for each block (partition)
  # Each factory produces a RecordBatchReader for its specific chunk
  def make_partition_factory(
      block: Block,
  ) -> Callable[[], pa.RecordBatchReader]:
    """Create a factory function for a specific block/chunk."""

    def make_stream() -> pa.RecordBatchReader:
      # Call the iteration callback if provided (for testing)
      if _iteration_callback is not None:
        _iteration_callback(block)

      # Convert this block to Arrow directly from numpy arrays,
      # bypassing the pandas round-trip for lower peak memory usage.
      batch = dataset_to_record_batch(ds.isel(block), schema)
      return pa.RecordBatchReader.from_batches(schema, [batch])

    return make_stream

  # Create one factory per block
  factories = [make_partition_factory(block) for block in blocks]

  # Pass factories, schema, and metadata to Rust
  return LazyArrowStreamTable(factories, schema, metadata)
