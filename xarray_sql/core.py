import itertools
import typing as t

import numpy as np
import xarray as xr

Row = list[t.Any]
Block = dict[t.Hashable, slice]
Chunks = t.Optional[t.Dict[t.Hashable, int]]

# deprecated
def get_columns(ds: xr.Dataset) -> list[t.Hashable]:
  return list(ds.sizes.keys()) + list(ds.data_vars.keys())


# Deprecated
def unravel(ds: xr.Dataset) -> t.Iterator[Row]:
  dim_keys, dim_vals = zip(*ds.sizes.items())

  for idx in itertools.product(*(range(d) for d in dim_vals)):
    coord_idx = dict(zip(dim_keys, idx))
    data = ds.isel(coord_idx)
    coord_data = [ds.coords[v][coord_idx[v]] for v in dim_keys]
    row = [v.values for v in coord_data + list(data.data_vars.values())]
    yield row


# Deprecated
def unbounded_unravel(ds: xr.Dataset) -> np.ndarray:
  """Unravel with unbounded memory (as a NumPy Array)."""
  dim_keys, dim_vals = zip(*ds.sizes.items())
  columns = get_columns(ds)

  N = np.prod([d for d in dim_vals])

  out = np.recarray((N,), dtype=[(c, ds[c].dtype) for c in columns])

  for name, da in ds.items():
    out[name] = da.values.ravel()

  prod_vals = (ds.coords[k].values for k in dim_keys)
  coords = np.array(np.meshgrid(*prod_vals), dtype=int).T.reshape(
      -1, len(dim_keys)
  )

  for i, d in enumerate(dim_keys):
    out[d] = coords[:, i]

  return out


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
