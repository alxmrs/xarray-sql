# qarray

_Query Xarray via SQL_

## What is this?

This is an experiment that provides an SQL interface for raster data.

```python
import xarray as xr
import qarray as qr

ds = xr.tutorial.open_dataset('air_temperature')

# The same as a dask-sql Context; i.e. an Apache DataFusion Context.
c = qr.Context(ds)

df = c.sql('''
  SELECT
    "lat", "lon", AVG("air") as air_total
  FROM 
    "air" 
  GROUP BY
   "lat", "lon"
''')

# A table of the average temperature for each location across time.
df.compute()

# Alternatively, you can just create the DataFrame from the Dataset:
df = qr.read_xarray(ds)
df.head()
```

To put it succinctly, we "pivot" Xarray Datasets so we can treat them like
tables so we can run SQL queries against them.

## Why build this?

A few reasons:

* Even though SQL is the lingua franca of data, scientific datasets are often
  inaccessible to non-scientists.
* Joining tabular data with raster data is both common and difficult when it
  could be easy.
* There are many Xarray-openeable datasets that would be nice to query via SQL,
  e.g. from [Google Earth Engine](https://github.com/google/Xee)
  or [Pangeo Forge](https://pangeo-forge.org/).

This is a light-weight way to prove the value of the interface.

## How does it work?

All chunks in an Xarray Dataset are transformed into a Dask DataFrame via
`from_map()` and `to_dataframe()`. For SQL support, we just use `dask-sql`.
That's it!

## What are the current limitations?

Dask doesn't support
`MultiIndex`s ([dask/dask#1493](https://github.com/dask/dask/issues/1493)). If
it did, I suspect performance for many types of queries would greatly improve.

Further, while this does play well with `dask-geopandas` (for geospatial query
support), certain types of operations don't quite match standard geopandas.
Spatial joins come to mind as a killer feature, but only inner joins are
supported ([geopandas/dask-geopandas#72](https://github.com/geopandas/dask-geopandas/issues/72))
.

## What would a deeper integration look like?

I have a few ideas so far. One approach involves parsing SQL to apply operations
directly on the Xarray Dataset. This approach is being
pursued [here](https://github.com/google/weather-tools/tree/main/xql), as `xql`.

Deeper still: I was thinking we could make like a virtual filesystem for parquet
that would internally map data to ([virtual](https://fsspec.github.io/kerchunk/)
?) Zarr. Raster-backed virtual parquet would open up integrations to numerous
tools, including dask, pyarrow, duckdb, and BigQuery. More thoughts on this
in [#4](https://github.com/alxmrs/qarray/issues/4).
