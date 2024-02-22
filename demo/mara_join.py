"""Demo of a spatial join using Qarray and dask_geopandas for MARA."""
import dask.dataframe as dd
import dask_geopandas as gdd
import xarray as xr
import qarray as qr


mv_df = dd.read_csv(
  'https://raw.githubusercontent.com/wildlife-dynamics/ecoscope/master/tests/'
  'sample_data/vector/movbank_data.csv'
)

mv_df['timestamp'] = dd.to_datetime(mv_df['timestamp'], utc=True)
mv_df['geometry'] = gdd.points_from_xy(
  mv_df['location-long'], mv_df['location-lat']
)
# What is the CRS?
mv_gdf = gdd.from_dask_dataframe(mv_df, 'geometry')

era5_ds = xr.open_zarr(
  'gs://gcp-public-data-arco-era5/ar/'
  '1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
  chunks={'time': 240, 'level': 1}
)
era5_wind_ds = era5_ds[['u_component_of_wind', 'v_component_of_wind']].sel(
  time=slice(mv_gdf['timestamp'].min(), mv_gdf['timestamp'].max())
)
era5_wind_df = qr.to_dd(era5_wind_ds)
# What is the CRS?
era5_wind_df['geometry'] = gdd.points_from_xy(
  era5_wind_df['longitude'], era5_wind_df['latitude'], era5_wind_df['level']
)
era5_wind_gdf = gdd.from_dask_dataframe(era5_wind_df, 'geometry')


# Only an inner spatial join is supported right now (in dask_geopandas).
intersection = mv_gdf.sjoin(era5_wind_gdf)
print(intersection.head())

