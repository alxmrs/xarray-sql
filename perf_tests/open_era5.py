#!/usr/bin/env python3

import xarray as xr
import xarray_sql as qr

# Requires authenticating with GCP
era5_ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
    chunks={'time': 240, 'level': 1},
)
era5_wind_df = qr.read_xarray(
    era5_ds[['u_component_of_wind', 'v_component_of_wind']]
)

print(era5_wind_df.columns)
