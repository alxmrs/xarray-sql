# qarray

_Query Xarray via SQL_

## What is this?

This is an experiment to provide a SQL interface for raster data.

```python
import xarray as xr
import qarray as qr

ds = xr.tutorial.open_dataset('air_temperature')

# The same as a dask-sql Context; i.e. an Apache DataFusion Context.
c = qr.Context(ds)
c.create_table('air', ds)

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

Succinctly, we "pivot" Xarray Datasets to treat them like
tables so we can run SQL queries against them.

## Why build this?

A few reasons:

* Even though SQL is the lingua franca of data, scientific datasets are often
  inaccessible to non-scientists.
* Joining tabular data with raster data is common yet difficult. It
  could be easy.
* There are many cloud-native, Xarray-openable datasets, 
  from [Google Earth Engine](https://github.com/google/Xee)
  to [Pangeo Forge](https://pangeo-forge.org/). Wouldn’t it be great if these
  were also SQL-accessible? How can the bridge be built with minimal effort? 

This is a light-weight way to prove the value of the interface.

## How does it work?

All chunks in an Xarray Dataset are transformed into a Dask DataFrame via
`from_map()` and `to_dataframe()`. For SQL support, we just use `dask-sql`.
That's it!

## Why does this work?

Underneath Xarray, Dask, and Pandas, there are NumPy arrays. These are
paged in chucks and represented contiguously in memory. It is only a 
matter of metadata that breaks them up into ndarrays. `to_dataframe()`
just changes this metadata (via a `ravel()`/`reshape()`), back into a
column amenable to a DataFrame. 

There is added overhead from duplicating dimensions as columns, which
we see as worth the convenience of DataFrames. 

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
