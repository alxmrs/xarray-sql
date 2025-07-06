import itertools
import typing as t

import numpy as np
import xarray as xr

Row = t.List[t.Any]


# deprecated
def get_columns(ds: xr.Dataset) -> t.List[str]:
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
