# xarray-sql

_Query Xarray with SQL_

[![ci](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml)
[![lint](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml)

```shell
pip install xarray-sql
```

## What is this?

This is an experiment to provide a SQL interface for raster data.

```python
import xarray as xr
import xarray_sql as qr

ds = xr.tutorial.open_dataset('air_temperature')

# The same as a dask-sql Context; i.e. an Apache DataFusion Context.
c = qr.XarrayContext()
c.from_dataset('air', ds, chunks=dict(time=24))

df = c.sql('''
  SELECT
    "lat", "lon", AVG("air") as air_total
  FROM 
    "air" 
  GROUP BY
   "lat", "lon"
''')

# A table of the average temperature for each location across time.
df.to_pandas()

# Alternatively, you can just create the DataFrame from the Dataset:
df = qr.read_xarray(ds).to_pandas()
df.head()
```

Succinctly, we "pivot" Xarray Datasets to treat them like tables so we can run
SQL queries against them.

## Why build this?

A few reasons:

* Even though SQL is the lingua franca of data, scientific datasets are often
  inaccessible to non-scientists (SQL users).
* Joining tabular data with raster data is common yet difficult. It could be
  easy.
* There are many cloud-native, Xarray-openable datasets,
  from [Google Earth Engine](https://github.com/google/Xee)
  to [Pangeo Forge](https://pangeo-forge.org/). Wouldn’t it be great if these
  were also SQL-accessible? How can the bridge be built with minimal effort?

This is a light-weight way to prove the value of the interface.

The larger goal is to explore the hypothesis that [Pangeo](https://pangeo.io/)
is a scientific database. Here, xarray-sql can be thought of as a missing DB 
front end.

## How does it work?

All chunks in an Xarray Dataset are transformed into a Dask DataFrame via
`from_map()` and `to_dataframe()`. For SQL support, we just use `dask-sql`.
That's it!

_2025 update_: This library now implements a dask-like `from_map` interface in 
pure `datafusion` and `pyarrow`, but works with the same principle!

## Why does this work?

Underneath Xarray, Dask, and Pandas, there are NumPy arrays. These are paged in
chunks and represented contiguously in memory. It is only a matter of metadata
that breaks them up into ndarrays. `to_dataframe()`
just changes this metadata (via a `ravel()`/`reshape()`), back into a column
amenable to a DataFrame.

There is added overhead from duplicating dimensions as columns, which we see as
worth the convenience of DataFrames.

## What are the current limitations?

_2025 update_: TBD, `datafusion` provides a whole new world!

## What would a deeper integration look like?

I have a few ideas so far. One approach involves applying operations directly on
Xarray Datasets. This approach is being pursued
[here](https://github.com/google/weather-tools/tree/main/xql), as `xql`.

Deeper still: I was thinking we could make
a [virtual](https://fsspec.github.io/kerchunk/)
filesystem for parquet that would internally map to Zarr. Raster-backed virtual
parquet would open up integrations to numerous tools like dask, pyarrow, duckdb,
and BigQuery. More thoughts on this
in [#4](https://github.com/alxmrs/xarray-sql/issues/4).

## Sponsors & Contributors

I want to give a special thanks to the following folks and institutions:

- Pramod Gupta and the Anthromet Team at Google Research for the problem
  formation and design inspiration.
- Jake Wall and AI2/Ecoscope for compute resources and key use cases.
- Charles Stern, Stephan Hoyer, Alexander Kmoch, Wei Ji, and Qiusheng Wu
  for the early review and discussion of this project.

## License

```
Copyright 2024 Alexander Merose

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Some sources are re-distributed from Google LLC
via https://github.com/google/Xee (also Apache-2.0 License) with and without
modification (specifically, Github Actions workflows). These files are subject
to the original copyright; they include the original license header comment as
well as a note to indicate modifications (when appropriate).
