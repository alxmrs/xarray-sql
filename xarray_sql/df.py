import itertools
import typing as t

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

Block = t.Dict[str, slice]
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

  chunk_bounds = {dim: np.cumsum((0,) + c) for dim, c in chunks.items()}
  ichunk = {dim: range(len(c)) for dim, c in chunks.items()}
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
  return np.prod([v.stop - v.start for v in block.values()])


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


def read_xarray(ds: xr.Dataset, chunks: Chunks = None) -> pa.Table:
  """Pivots an Xarray Dataset into a PyArrow Table, partitioned by chunks.

  Args:
    ds: An Xarray Dataset. All `data_vars` mush share the same dimensions.
    chunks: Xarray-like chunks. If not provided, will default to the Dataset's
     chunks. The product of the chunk sizes becomes the standard length of each
     dataframe partition.

  Returns:
    A PyArrow Table, which is a table representation of the input Dataset.
  """
  fst = next(iter(ds.values())).dims
  assert all(
      da.dims == fst for da in ds.values()
  ), "All dimensions must be equal. Please filter data_vars in the Dataset."

  blocks = list(block_slices(ds, chunks))

  def pivot(b: Block) -> pd.DataFrame:
    return ds.isel(b).to_dataframe().reset_index()

  return from_map(pivot, blocks)
