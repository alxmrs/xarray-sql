import itertools
import typing as t
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from datafusion.context import ArrowStreamExportable

Block = t.Dict[t.Hashable, slice]
Chunks = t.Optional[t.Dict[str, int]]


# Borrowed from Xarray
def _get_chunk_slicer(
    dim: t.Hashable, chunk_index: t.Mapping, chunk_bounds: t.Mapping
):
  if dim in chunk_index:
    which_chunk = chunk_index[dim]
    return slice(
        chunk_bounds[dim][which_chunk], chunk_bounds[dim][which_chunk + 1]
    )
  return slice(None)


# Adapted from Xarray `map_blocks` implementation.
def block_slices(ds: xr.Dataset, chunks: Chunks = None) -> t.Iterator[Block]:
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


def explode(ds: xr.Dataset, chunks: Chunks = None) -> t.Iterator[xr.Dataset]:
  """Explodes a dataset into its chunks."""
  yield from (ds.isel(b) for b in block_slices(ds, chunks=chunks))


def _block_len(block: Block) -> int:
  return int(np.prod([v.stop - v.start for v in block.values()]))


def from_map_batched(
    func: t.Callable[..., pd.DataFrame],
    *iterables,
    args: t.Optional[t.Tuple] = None,
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
    func: t.Callable, *iterables, args: t.Optional[t.Tuple] = None, **kwargs
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
PartitionBounds = t.Dict[str, t.Tuple[t.Any, t.Any, str]]


def partition_metadata(
    ds: xr.Dataset, blocks: t.List[Block]
) -> t.List[PartitionBounds]:
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

  metadata = []
  for block in blocks:
    ranges: PartitionBounds = {}
    for dim, slc in block.items():
      coord_values = coord_arrays[str(dim)][slc]
      if len(coord_values) > 0:
        # Use endpoints for the common monotonic case (O(1)).
        # xarray/CF-convention dimension coordinates are almost always
        # monotonic; even for descending axes (e.g. latitude 90→-90)
        # first/last gives the correct bounds after the min/max swap below.
        first, last = coord_values[0], coord_values[-1]
        if first <= last:
          min_val, max_val = first, last
        else:
          min_val, max_val = last, first

        # Convert numpy scalar types to Python native types
        # This is required for PyO3 FFI conversion
        if isinstance(min_val, (np.datetime64, pd.Timestamp)):
          # Convert datetime to nanoseconds since epoch
          min_val = int(pd.Timestamp(min_val).value)
          max_val = int(pd.Timestamp(max_val).value)
          ranges[str(dim)] = (min_val, max_val, "timestamp_ns")
        elif hasattr(min_val, "item"):
          # numpy scalar -> Python native
          min_val = min_val.item()
          max_val = max_val.item()
          dtype = "float64" if isinstance(min_val, float) else "int64"
          ranges[str(dim)] = (min_val, max_val, dtype)
        else:
          dtype = "float64" if isinstance(min_val, float) else "int64"
          ranges[str(dim)] = (min_val, max_val, dtype)
    metadata.append(ranges)
  return metadata
