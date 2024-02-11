import itertools
import typing as t

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from dask.dataframe.io import from_map

from . import core as qr


# Borrowed from Xarray
def _get_chunk_slicer(dim: t.Hashable, chunk_index: t.Mapping,
                      chunk_bounds: t.Mapping):
  if dim in chunk_index:
    which_chunk = chunk_index[dim]
    return slice(chunk_bounds[dim][which_chunk],
                 chunk_bounds[dim][which_chunk + 1])
  return slice(None)


# Adapted from Xarray `map_blocks` implementation.
def explode(ds: xr.Dataset, chunks=None) -> t.Iterator[xr.Dataset]:
  """Explodes a dataset into its chunks."""
  if chunks is not None:
    ds.chunk(chunks)
  assert ds.chunks, 'Dataset must be chunked'

  chunk_bounds = {
    dim: np.cumsum((0,) + chunks_v) for dim, chunks_v in ds.chunks.items()
  }
  ichunk = {dim: range(len(chunks_v)) for dim, chunks_v in ds.chunks.items()}
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

  yield from (ds.isel(b) for b in blocks)


# TODO(alxmrs): Does this need to be ichunked?
def to_pd(ds: xr.Dataset) -> pd.DataFrame:
  columns = list(ds.dims.keys()) + list(ds.data_vars.keys())
  return pd.DataFrame(qr.unravel(ds), columns=columns)


def to_dd(ds: xr.Dataset) -> dd.DataFrame:
  dss = explode(ds)

  # TODO(alxmrs): Add partition info -- https://docs.dask.org/en/latest/dataframe-design.html#partitions
  return from_map(
    to_pd,
    dss,
  )
