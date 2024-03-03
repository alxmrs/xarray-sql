import itertools
import typing as t

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.dataframe.io import from_map

from . import core

Block = t.Dict[str, slice]
Chunks = t.Dict[str, int]

# Turn on Dask-Expr
dask.config.set({'dataframe.query-planning-warning': False})
dask.config.set({"dataframe.query-planning": True})
pd.options.mode.copy_on_write = True

# Borrowed from Xarray
def _get_chunk_slicer(dim: t.Hashable, chunk_index: t.Mapping,
                      chunk_bounds: t.Mapping):
  if dim in chunk_index:
    which_chunk = chunk_index[dim]
    return slice(chunk_bounds[dim][which_chunk],
                 chunk_bounds[dim][which_chunk + 1])
  return slice(None)


# Adapted from Xarray `map_blocks` implementation.
def block_slices(
    ds: xr.Dataset,
    chunks: t.Optional[Chunks] = None
) -> t.Iterator[Block]:
  """Compute block slices for a chunked Dataset."""
  if chunks is not None:
    for_chunking = ds.copy(data=None, deep=False).chunk(chunks)
    chunks = for_chunking.chunks
    del for_chunking
  else:
    chunks = ds.chunks

  assert chunks, 'Dataset `ds` must be chunked or `chunks` must be provided.'

  chunk_bounds = {
    dim: np.cumsum((0,) + c) for dim, c in chunks.items()
  }
  ichunk = {dim: range(len(c)) for dim, c in chunks.items()}
  ick, icv = zip(*ichunk.items())  # Makes same order of keys and val.
  chunk_idxs = (
    dict(zip(ick, i)) for i in itertools.product(*icv)
  )
  blocks = (
    {
      dim: _get_chunk_slicer(dim, chunk_index, chunk_bounds)
      for dim in ds.dims
    }
    for chunk_index in chunk_idxs
  )
  yield from blocks


def explode(
    ds: xr.Dataset,
    chunks: t.Optional[Chunks] = None
) -> t.Iterator[xr.Dataset]:
  """Explodes a dataset into its chunks."""
  yield from (ds.isel(b) for b in block_slices(ds, chunks=chunks))


def to_pd(ds: xr.Dataset, bounded=True) -> pd.DataFrame:
  columns = core.get_columns(ds)
  if bounded:
    df = pd.DataFrame(core.unravel(ds), columns=columns)
    for c in columns:
      df[c] = df[c].astype(ds[c].dtype)
    return df
  else:
    data = core.unbounded_unravel(ds)
    return pd.DataFrame.from_records(data)


def _block_len(block: Block) -> int:
  return np.prod([v.stop - v.start for v in block.values()])


def to_dd(ds: xr.Dataset, chunks: t.Optional[Chunks] = None) -> dd.DataFrame:
  """Unravel a Dataset into a Dataframe, partitioned by chunks.

  Args:
    ds: An Xarray Dataset. All `data_vars` mush share the same dimensions.
    chunks: Xarray-like chunks. If not provided, will default to the Dataset's
     chunks. The product of the chunk sizes becomes the standard length of each
     dataframe partition.

  Returns:
    A Dask Dataframe, which is a table representation of the input Dataset.
  """
  blocks = list(block_slices(ds, chunks))

  block_lengths = [_block_len(b) for b in blocks]
  divisions = tuple(np.cumsum([0] + block_lengths))  # 0 ==> start partition.

  def f(b: Block) -> pd.DataFrame:
    return to_pd(ds.isel(b), bounded=False)

  # Token is needed to prevent Dask from spending too many cycles calculating
  # it's own token from the constituent parts.
  token = (
    f'xarray-Dataset-' 
    f'{"_".join(list(ds.dims.keys()))}'
    f'__'
    f'{"_".join(list(ds.data_vars.keys()))}'
  )

  columns = core.get_columns(ds)

  # TODO(#18): Is it possible to pass the length (known now) here?
  meta = {
    c: ds[c].dtype for c in columns
  }

  return from_map(
    f,
    blocks,
    meta=meta,
    divisions=divisions,
    token=token,
  )
