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

Open a year of ARCO-ERA5 and let SQL `WHERE` clauses do the filtering — the
library prunes time partitions and pushes dimension-column filters down. Use
the `table_names` kwarg to give each dimension group a friendly name:

```python
import xarray as xr
import xarray_sql as xql

# Open ARCO-ERA5 directly from GCS (anonymous read).
url = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full = xr.open_zarr(url, chunks=None, storage_options={'token': 'anon'})

# A full year of hourly ERA5 — all 273 variables. No spatial slicing on the
# xarray side; SQL WHERE clauses below express the filters. `chunks={'time': 1}`
# aligns Dask chunks to native Zarr chunks of shape (1, 37, 721, 1440) so
# chunk reads from GCS happen concurrently.
#
# Heads up: 262 of those variables are surface and 11 are atmospheric. The
# library pushes column projection down, so SELECT only fetches what you ask
# for — but `SELECT * FROM era5.surface` would try to pull every variable
# across the year (terabytes from GCS). Always SELECT specific columns.
ds = full.sel(time='2020').chunk({'time': 1})

ctx = xql.XarrayContext()
ctx.from_dataset('era5', ds, table_names={
    ('time', 'latitude', 'longitude'): 'surface',
    ('time', 'level', 'latitude', 'longitude'): 'atmosphere',
})
# Registers two tables under a SQL schema named 'era5': 'surface' and 'atmosphere'.

# Average 2m-temperature over the NYC area on the morning of 2020-01-01.
ctx.sql('''
  SELECT AVG("2m_temperature") - 273.15 AS avg_c
  FROM era5.surface
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
    AND latitude  BETWEEN 39 AND 40
    AND longitude BETWEEN 286 AND 287
''').to_pandas()

# Average temperature per pressure level, globally — the standard
# atmospheric temperature profile. Scans ~230M rows.
ctx.sql('''
  SELECT level, AVG(temperature) - 273.15 AS avg_c
  FROM era5.atmosphere
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
  GROUP BY level
  ORDER BY level DESC  -- surface (1000 hPa) first
''').to_pandas()
```

If you omit `table_names`, each table is named by joining its dimension names
with underscores, e.g. `era5.time_latitude_longitude` and
`era5.time_level_latitude_longitude`.

## GOES satellite imagery (scalar variables)

Real-world stores often mix gridded data with scalar (0-dimensional) metadata.
GOES satellite imagery, for example, pairs `(y, x)` image bands with dozens of
scalar variables such as `goes_imager_projection`. `from_dataset` groups all the
scalars into a single one-row table named `scalar`:

```python
import xarray as xr
from xarray_sql import XarrayContext

# A GOES-16 ABI cloud-and-moisture file: (y, x) bands + scalar metadata.
ds = xr.open_dataset('OR_ABI-L2-MCMIPM1-M6_G16_....nc').chunk({'y': 250, 'x': 250})

ctx = XarrayContext()
ctx.from_dataset('goes', ds)

ctx.sql('SELECT COUNT(*) FROM goes.y_x')   # the gridded bands
ctx.sql('SELECT * FROM goes.scalar')       # one row of metadata
```

Override the default name like any other group with `table_names={(): 'metadata'}`.

A runnable version of the ERA5 example lives at
[`perf_tests/era5_temp_profile.py`](../perf_tests/era5_temp_profile.py).

[arco-era5]: https://github.com/google-research/arco-era5
