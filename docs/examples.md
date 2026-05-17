# Examples

```python
import xarray as xr
import xarray_sql as xql

ds = xr.tutorial.open_dataset('air_temperature')

ctx = xql.XarrayContext()
ctx.from_dataset('air', ds, chunks=dict(time=24))

result = ctx.sql('''
  SELECT
    "lat", "lon", AVG("air") as air_avg
  FROM
    "air"
  GROUP BY
   "lat", "lon"
''')

df = result.to_pandas()
df.head()
```

## Mixed-dimension datasets: ARCO-ERA5

When a Dataset has variables with differing dimensions (e.g. surface fields on
`(time, latitude, longitude)` and atmospheric fields on
`(time, level, latitude, longitude)`), `from_dataset` splits them into one
table per dimension group, registered together under a SQL schema named after
the first argument. [ARCO-ERA5][arco-era5] is a good example: 262 of its
variables are surface fields and 11 are atmospheric.

Use the `table_names` kwarg to give each dimension group a friendly name:

```python
import xarray as xr
import xarray_sql as xql

# Open ARCO-ERA5 directly from GCS (anonymous read).
url = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full = xr.open_zarr(url, chunks=None, storage_options={'token': 'anon'})

# Small slice for the example: a few hours over the NYC area, two pressure levels,
# one variable from each dimension group.
ds = full[['2m_temperature', 'temperature']].sel(
    time=slice('2020-01-01', '2020-01-01T05'),
    latitude=slice(40, 39),
    longitude=slice(286, 287),  # ERA5 uses 0–360° longitudes
    level=[500, 850],
).chunk({'time': 3})

ctx = xql.XarrayContext()
ctx.from_dataset('era5', ds, table_names={
    ('time', 'latitude', 'longitude'): 'surface',
    ('time', 'level', 'latitude', 'longitude'): 'atmosphere',
})
# Registers two tables under a SQL schema named 'era5': 'surface' and 'atmosphere'.

# Query the surface table.
ctx.sql('''
  SELECT AVG("2m_temperature") - 273.15 AS avg_c
  FROM era5.surface
''').to_pandas()

# Query the atmospheric table.
ctx.sql('''
  SELECT level, AVG(temperature) - 273.15 AS avg_c
  FROM era5.atmosphere
  GROUP BY level
  ORDER BY level
''').to_pandas()
```

If you omit `table_names`, each table is named by joining its dimension names
with underscores, e.g. `era5.time_latitude_longitude` and
`era5.time_level_latitude_longitude`.

A runnable version of this example lives at
[`perf_tests/era5_nyc_avg_temp.py`](../perf_tests/era5_nyc_avg_temp.py).

[arco-era5]: https://github.com/google-research/arco-era5
