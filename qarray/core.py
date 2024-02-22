import functools
import itertools
import operator
import typing as t

import numpy as np
import xarray as xr

Row = t.List[t.Any]


def get_columns(ds: xr.Dataset) -> t.List[str]:
  return list(ds.dims.keys()) + list(ds.data_vars.keys())


def unravel(ds: xr.Dataset) -> t.Iterator[Row]:
  dim_keys, dim_vals = zip(*ds.dims.items())

  for idx in itertools.product(*(range(d) for d in dim_vals)):
    coord_idx = dict(zip(dim_keys, idx))
    data = ds.isel(coord_idx)
    coord_data = [ds.coords[v][coord_idx[v]] for v in dim_keys]
    row = [v.values for v in coord_data + list(data.data_vars.values())]
    yield row


def unbounded_unravel(ds: xr.Dataset) -> np.ndarray:
  """Unravel with unbounded memory (as a NumPy Array)."""
  dim_keys, dim_vals = zip(*ds.dims.items())
  columns = get_columns(ds)

  N = np.prod([d for d in dim_vals])

  out = np.recarray((N,), dtype=[(c, ds[c].dtype) for c in columns])

  for name, da in ds.items():
    out[name] = da.values.ravel()

  prod_vals = (ds.coords[k].values for k in dim_keys)
  coords = (
    np.array(np.meshgrid(*prod_vals), dtype=int).T
    .reshape(-1, len(dim_keys))
  )

  for i, d in enumerate(dim_keys):
    out[d] = coords[:, i]

  return out


def _index_to_position(index: int, dimensions: t.List[int]) -> t.List[int]:
  """Converts a table index into a position within an nd-array."""
  # Authored by ChatGPT from the following prompt:
  #   """
  #   Lets say I have an integer “index”. It represents the index of an array
  #   where the last index value is the product of the dimensions of the array.
  #   Each index value corresponds to a cell in the array. The array can be D
  #   dimensions, and usually D is 3 or 4. In Python, how do I convert the
  #   integer “index” into the position of the cell in the array (i.e. a list
  #   of integers of length D where each value represents the position along
  #   the dimension)?
  #   """
  position = []
  for dim in reversed(dimensions):
    position.insert(0, index % dim)
    index //= dim
  return position


def _unbox(array):
  """When a numpy array is a scalar, it extracts the value."""
  try:
    return array.item()
  except ValueError:
    return array