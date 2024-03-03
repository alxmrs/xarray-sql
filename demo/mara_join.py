#!/usr/bin/env python3
"""Demo of a spatial join using Qarray and dask_geopandas for MARA."""
import dask.dataframe as dd
import dask_geopandas as gdd
import xarray as xr
import qarray as qr


mv_df = dd.read_csv(
  'https://raw.githubusercontent.com/wildlife-dynamics/ecoscope/master/tests/'
  'sample_data/vector/movbank_data.csv',
)

mv_df['timestamp'] = dd.to_datetime(mv_df['timestamp'])
mv_df.set_index('timestamp', drop=False, sort=True)
mv_df['geometry'] = gdd.points_from_xy(
  mv_df, 'location-long', 'location-lat', crs=4326
)
timerange = slice(
  mv_df.timestamp.min().compute(),
  mv_df.timestamp.max().compute(),
)
mv_gdf = gdd.from_dask_dataframe(mv_df, 'geometry')

# For MARA, we'd replace this with an Xee call.
era5_ds = xr.open_zarr(
  'gs://gcp-public-data-arco-era5/ar/'
  '1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
  chunks={'time': 240, 'level': 1}
)
print('era5 dataset opened.')
era5_wind_ds = era5_ds[['u_component_of_wind', 'v_component_of_wind']].sel(
  time=timerange,
  level=1000,  # surface level only.
)
era5_wind_df = qr.read_xarray(era5_wind_ds)
# What is the CRS?
era5_wind_df['geometry'] = gdd.points_from_xy(
  era5_wind_df, 'longitude', 'latitude',
)
era5_wind_gdf = gdd.from_dask_dataframe(era5_wind_df, 'geometry')

print('beginning spatial join')
# Only an inner spatial join is supported right now (in dask_geopandas).
intersection = era5_wind_gdf.sjoin(mv_gdf).compute()
print(intersection)

