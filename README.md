# qarray

_Query Xarray with SQL._

## What is this?

This is an experiment to try to provide an SQL interface for raster data. If all
goes well, the user experience will look like the following:

```python
import xarray as xr
import qarray as qr
import pandas as pd

ds = xr.open_dataset(...)

# A SQLAlchemy wrapper for an open Xarray Dataset. 
con = qr.Connection(ds, **qr_kwargs)

df = pd.read_sql(
    "select lat, lon, time, level, dtm, t2m, cape from <conuri> limit 1000",
    con=con
)
```

## Why build this?

A few quick reasons:

* SQL is the lingua franca of data. The only thing more democratizing is
  spreadsheets.
* Scientific datasets often have analytics use-cases that are ignored by modern 
  tooling.
* There are a lot of Xarray-openeable datasets that I would have like to make
  available to my users via SQL, i.e. from [GEE](https://github.com/google/xee)
  or [Pangeo](https://pangeo-forge.org/).
  
This seems like a light-weight way to prove the value of the interface before 
investing in a deeper integration.

## What would a deeper integration look like?

I have a few ideas so far. I was thinking if we could make like a virtual 
filesystem for parquet that would internally map data to ([virtual](https://fsspec.github.io/kerchunk/)?) 
Zarr. Raster-backed virtual parquet would open up integrations to numerous 
tools, including pandas, duckdb, and BigQuery.