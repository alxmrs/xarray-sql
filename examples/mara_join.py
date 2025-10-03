#!/usr/bin/env python3
# /// script
# dependencies = [
#   "arraylake",
#   "icechunk",
#   "xarray",
#   "xarray-sql",
#   "pandas",
# ]
# ///

"""Demo of a (spatial) join using Xarray-SQL and DataFusion for MARA."""
import pandas as pd
import xarray as xr
import xarray_sql as xql
from arraylake import Client

# Login and get access to EarthMover's Temporal ERA5 dataset.
client = Client()
client.login()
repo = client.get_repo("earthmover-public/era5-surface-aws")
ds = xr.open_zarr(repo.readonly_session("main").store, group="temporal", chunks=None, zarr_format=3, consolidated=False)

# Create a DataFusion context and register the datasets as tables.
ctx = xql.XarrayContext()
ctx.from_dataset('era5', ds.isel(time=8736*2), chunks=dict(latitude=12, longitude=12))
ctx.from_pandas(
  pd.read_feather(
    'https://github.com/wildlife-dynamics/ecoscope/raw/refs/heads/master/'
    'tests/sample_data/vector/movebank_data.feather'
  ),
  'movebank',
)

# TODO(alxmrs): When I'm not on cellular internet, explore the datasets and write the
#  join in SQL.
