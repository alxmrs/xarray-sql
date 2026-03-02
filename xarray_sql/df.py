import itertools
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

Block = dict[Hashable, slice]
Chunks = dict[str, int] | None


# Borrowed from Xarray
def _get_chunk_slicer(
    dim: Hashable, chunk_index: Mapping, chunk_bounds: Mapping
):
  if dim in chunk_index:
    which_chunk = chunk_index[dim]
    return slice(
        chunk_bounds[dim][which_chunk], chunk_bounds[dim][which_chunk + 1]
    )
  return slice(None)


# Adapted from Xarray `map_blocks` implementation.
def block_slices(ds: xr.Dataset, chunks: Chunks = None) -> Iterator[Block]:
  """Compute block slices for a chunked Dataset."""
  if chunks is not None:
    for_chunking = ds.copy(data=None, deep=False).chunk(chunks)
    chunks = for_chunking.chunks
    del for_chunking
  else:
    chunks = ds.chunks

  assert chunks, "Dataset `ds` must be chunked or `chunks` must be provided."

  # chunks is Dict[str, Tuple[int, ...]] from xarray
  chunk_bounds = {
      dim: np.cumsum((0,) + tuple(c))  # type: ignore[arg-type]
      for dim, c in chunks.items()
  }
  ichunk = {dim: range(len(tuple(c))) for dim, c in chunks.items()}  # type: ignore[arg-type]
  ick, icv = zip(*ichunk.items())  # Makes same order of keys and val.
  chunk_idxs = (dict(zip(ick, i)) for i in itertools.product(*icv))
  blocks = (
      {
          dim: _get_chunk_slicer(dim, chunk_index, chunk_bounds)
          for dim in ds.dims
      }
      for chunk_index in chunk_idxs
  )
  yield from blocks


def explode(ds: xr.Dataset, chunks: Chunks = None) -> Iterator[xr.Dataset]:
  """Explodes a dataset into its chunks."""
  yield from (ds.isel(b) for b in block_slices(ds, chunks=chunks))


def _block_len(block: Block) -> int:
  return int(np.prod([v.stop - v.start for v in block.values()]))


def from_map_batched(
    func: Callable[..., pd.DataFrame],
    *iterables,
    args: tuple | None = None,
    schema: pa.Schema = None,
    **kwargs,
) -> pa.RecordBatchReader:
  """Create a PyArrow RecordBatchReader by mapping a function over iterables.

  This is equivalent to dask's from_map but returns a PyArrow
  RecordBatchReader that can be used with DataFusion. It iterates over
  RecordBatches which are created via the `func` one-at-a-time.

  Args:
    func: Function to apply to each element of the iterables. Currently, the function
      must return a Pandas DataFrame.
    *iterables: Iterable objects to map the function over.
    schema: Optional schema needed for the RecordBatchReader.
    args: Additional positional arguments to pass to func.
    **kwargs: Additional keyword arguments to pass to func.

  Returns:
    A PyArrow RecordBatchReader containing the stream of RecordBatches.
  """
  if args is None:
    args = ()

  def map_batches():
    for items in zip(*iterables):
      df = func(*items, *args, **kwargs)
      yield pa.RecordBatch.from_pandas(df, schema=schema)

  return pa.RecordBatchReader.from_batches(schema, map_batches())


def from_map(
    func: Callable, *iterables, args: tuple | None = None, **kwargs
) -> pa.Table:
  """Create a PyArrow Table by mapping a function over iterables.

  This is equivalent to dask's from_map but returns a PyArrow Table
  that can be used with DataFusion instead of a Dask DataFrame.

  Args:
    func: Function to apply to each element of the iterables.
    *iterables: Iterable objects to map the function over.
    args: Additional positional arguments to pass to func.
    **kwargs: Additional keyword arguments to pass to func.

  Returns:
    A PyArrow Table containing the concatenated results.
  """
  if args is None:
    args = ()

  # Apply the function to each combination of iterable elements
  results = []
  for items in zip(*iterables) if len(iterables) > 1 else iterables[0]:
    if isinstance(items, tuple):
      result = func(*items, *args, **kwargs)
    else:
      result = func(items, *args, **kwargs)

    # Convert result to PyArrow Table
    if isinstance(result, pd.DataFrame):
      pa_table = pa.Table.from_pandas(result)
    elif isinstance(result, pa.Table):
      pa_table = result
    else:
      # Try to convert to pandas first, then to PyArrow
      try:
        df = pd.DataFrame(result)
        pa_table = pa.Table.from_pandas(df)
      except Exception as e:
        raise ValueError(
            f"Cannot convert function result to PyArrow Table: {e}"
        )

    results.append(pa_table)

  # Concatenate all results
  if not results:
    raise ValueError("No results to concatenate")

  return pa.concat_tables(results)


def pivot(ds: xr.Dataset) -> pd.DataFrame:
  """Converts an xarray Dataset to a pandas DataFrame."""
  return ds.to_dataframe().reset_index()  # type: ignore[no-any-return]


def dataset_to_record_batch(
    ds: xr.Dataset, schema: pa.Schema
) -> pa.RecordBatch:
  """Convert an xarray Dataset partition to an Arrow RecordBatch.

  Builds the RecordBatch directly from numpy arrays, bypassing the pandas
  round-trip (to_dataframe → reset_index → from_pandas) used by pivot().
  For large partitions this reduces peak memory from ~5× to ~2× the
  partition size.

  Dimension coordinates are broadcast to the full partition shape and
  ravelled. np.broadcast_to() is zero-copy; the ravel() forces one copy
  per coordinate (unavoidable, since broadcast arrays are non-contiguous).
  Data variable arrays are ravelled in-place — a zero-copy view when the
  underlying array is already C-contiguous (the common case for numpy-backed
  xarray datasets).

  Args:
      ds: A partition-sized xarray Dataset (already sliced via isel).
      schema: The Arrow schema for the output, as produced by _parse_schema.
          Column order in the output matches schema field order.

  Returns:
      A RecordBatch with one column per dimension coordinate and data
      variable, in schema order.
  """
  # Use the data variable's dimension order as canonical so coordinate
  # broadcasts and data variable ravels use the same layout. All data
  # variables are validated to share the same dims tuple.
  if ds.data_vars:
    first_var = next(iter(ds.data_vars.values()))
    dim_names = list(first_var.dims)
    shape = first_var.shape
  else:
    dim_names = list(ds.sizes.keys())
    shape = tuple(ds.sizes[d] for d in dim_names)

  arrays = []
  for field in schema:
    name = field.name
    if name in ds.coords and name in ds.dims:
      # Broadcast 1-D coordinate to the full N-D partition shape, then ravel.
      axis = dim_names.index(name)
      coord = ds.coords[name].values
      reshape = [1] * len(shape)
      reshape[axis] = coord.shape[0]
      arr = np.broadcast_to(coord.reshape(reshape), shape).ravel()
      arrays.append(pa.array(arr, type=field.type))
    else:
      # Data variable: ravel to 1-D (zero-copy for C-contiguous arrays).
      arrays.append(pa.array(ds[name].values.ravel(), type=field.type))

  return pa.RecordBatch.from_arrays(arrays, schema=schema)


#: Default number of rows per emitted Arrow RecordBatch.
#: 64 K rows balances DataFusion pipeline depth against per-batch overhead.
DEFAULT_BATCH_SIZE: int = 65_536


def iter_record_batches(
    ds: xr.Dataset,
    schema: pa.Schema,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
  """Yield RecordBatches of at most *batch_size* rows from a partition Dataset.

  Unlike :func:`dataset_to_record_batch`, which materialises the entire
  partition as one batch, this generator emits smaller batches so that
  DataFusion can begin filtering and aggregating before the full partition
  is loaded.  Peak memory per batch is O(batch_size) for coordinate columns
  and O(partition_size) for data-variable columns (which must be loaded in
  full from storage).

  Coordinate values are computed per batch via strided index arithmetic —
  no broadcast array spanning the whole partition is ever allocated.  Data
  variable flat arrays are loaded once (triggering any remote I/O) and then
  sliced as zero-copy views for each batch.

  Args:
      ds: A partition-sized xarray Dataset (already sliced via isel).
      schema: The Arrow schema for the output, as produced by _parse_schema.
      batch_size: Maximum number of rows per yielded RecordBatch.

  Yields:
      RecordBatches in schema column order, covering all rows of the
      partition exactly once.
  """
  if ds.data_vars:
    first_var = next(iter(ds.data_vars.values()))
    dim_names = list(first_var.dims)
    shape = first_var.shape
  else:
    dim_names = list(ds.sizes.keys())
    shape = tuple(ds.sizes[d] for d in dim_names)

  total_rows = int(np.prod(shape))

  # Preload small 1-D coordinate arrays (negligible memory).
  coord_values = {name: ds.coords[name].values for name in dim_names}

  # C-order stride for each dimension: stride[k] = prod(shape[k+1:]).
  # Flat row index i → coordinate index for dim k: (i // stride[k]) % shape[k].
  strides = [int(np.prod(shape[k + 1 :])) for k in range(len(shape))]

  # Load data-variable arrays fully (triggers Dask/Zarr compute once).
  # ravel() is a zero-copy view for C-contiguous arrays.
  data_arrays = {}
  for field in schema:
    if field.name not in ds.dims:
      data_arrays[field.name] = ds[field.name].values.ravel()

  for row_start in range(0, total_rows, batch_size):
    row_end = min(row_start + batch_size, total_rows)
    row_idx = np.arange(row_start, row_end)

    arrays = []
    for field in schema:
      name = field.name
      if name in ds.coords and name in ds.dims:
        k = dim_names.index(name)
        coord_idx = (row_idx // strides[k]) % shape[k]
        arrays.append(pa.array(coord_values[name][coord_idx], type=field.type))
      else:
        arrays.append(
            pa.array(data_arrays[name][row_start:row_end], type=field.type)
        )

    yield pa.RecordBatch.from_arrays(arrays, schema=schema)


def _parse_schema(ds) -> pa.Schema:
  """Extracts a `pa.Schema` from the Dataset, treating dims and data_vars as columns."""
  columns = []

  for coord_name, coord_var in ds.coords.items():
    # Only include dimension coordinates
    if coord_name in ds.dims:
      pa_type = pa.from_numpy_dtype(coord_var.dtype)
      columns.append(pa.field(coord_name, pa_type))

  for var_name, var in ds.data_vars.items():
    pa_type = pa.from_numpy_dtype(var.dtype)
    columns.append(pa.field(var_name, pa_type))

  return pa.schema(columns)


# Type alias for partition metadata: maps dimension name to (min, max, dtype_str) values
PartitionBounds = dict[str, tuple[Any, Any, str]]


def _block_metadata(coord_arrays: dict, block: Block) -> PartitionBounds:
  """Compute min/max coordinate values for a single partition block.

  Args:
      coord_arrays: Pre-materialised coordinate arrays keyed by dimension name
          string.  Hoist this outside any loop to avoid repeated remote I/O
          for Zarr-backed datasets.
      block: A single block slice dict from block_slices().

  Returns:
      Dict mapping dimension name to (min_value, max_value, dtype_str).
      Dimensions with an empty slice are omitted; the Rust pruning logic
      treats missing dimensions conservatively (never prunes on them).
  """
  ranges: PartitionBounds = {}
  for dim, slc in block.items():
    coord_values = coord_arrays[str(dim)][slc]
    if len(coord_values) > 0:
      first, last = coord_values[0], coord_values[-1]
      if first <= last:
        min_val, max_val = first, last
      else:
        min_val, max_val = last, first

      if isinstance(min_val, (np.datetime64, pd.Timestamp)):
        min_val = int(pd.Timestamp(min_val).value)
        max_val = int(pd.Timestamp(max_val).value)
        ranges[str(dim)] = (min_val, max_val, "timestamp_ns")
      elif hasattr(min_val, "item"):
        min_val = min_val.item()
        max_val = max_val.item()
        dtype = "float64" if isinstance(min_val, float) else "int64"
        ranges[str(dim)] = (min_val, max_val, dtype)
      else:
        dtype = "float64" if isinstance(min_val, float) else "int64"
        ranges[str(dim)] = (min_val, max_val, dtype)
  return ranges


def partition_metadata(
    ds: xr.Dataset, blocks: list[Block]
) -> list[PartitionBounds]:
  """Compute min/max coordinate values for each partition.

  This metadata enables filter pushdown: SQL queries with WHERE clauses
  on dimension columns can prune partitions that can't contain matching rows.

  Args:
      ds: The xarray Dataset containing coordinate values.
      blocks: List of block slices from block_slices().

  Returns:
      List of dicts mapping dimension name to (min_value, max_value, dtype_str)
      tuples.
      - For datetime64, values are nanoseconds since Unix epoch (int64),
        dtype_str is "timestamp_ns"
      - For numeric types, values are Python int or float, dtype_str is
        "int64" or "float64"

  Note:
      If a partition has an empty slice for a dimension, that dimension is
      omitted from the partition's metadata. The Rust pruning logic treats
      missing dimensions conservatively (never prunes on them).
  """
  # Hoist coordinate array reads outside the partition loop.
  # ds.coords[dim].values materializes the full array on every call; doing it
  # N_partitions × N_dims times is wasteful and, for remote Zarr-backed datasets
  # (e.g. ARCO-ERA5 on GCS), may trigger repeated network I/O.
  coord_arrays = {str(dim): ds.coords[dim].values for dim in ds.dims}
  return [_block_metadata(coord_arrays, block) for block in blocks]
