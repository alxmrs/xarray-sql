# xarray-sql

_Query [Xarray](https://xarray.dev/) with SQL_

[![ci](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml)
[![lint](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml)
[![ci-build](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-build.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-build.yml)
[![rust-lint](https://github.com/alxmrs/xarray-sql/actions/workflows/rust-lint.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/rust-lint.yml)

```shell
pip install xarray-sql
```

## What is this?

This is an experiment to provide a SQL interface for array datasets.

```python
import xarray as xr
import xarray_sql as xql

ds = xr.tutorial.open_dataset('air_temperature')

# The same as a dask-sql Context; i.e. an Apache DataFusion Context.
ctx = xql.XarrayContext()
ctx.from_dataset('air', ds, chunks=dict(time=24))  # the dataset needs to be chunked!
# data is only materialized when we make a query.

result = ctx.sql('''
  SELECT
    "lat", "lon", AVG("air") as air_avg
  FROM
    "air"
  GROUP BY
   "lat", "lon"
''')
# DataFrame()
# +------+-------+--------------------+
# | lat  | lon   | air_avg            |
# +------+-------+--------------------+
# | 75.0 | 205.0 | 259.88662671232834 |
# | 75.0 | 207.5 | 259.48268150684896 |
# | 75.0 | 230.0 | 258.9192123287667  |
# | 75.0 | 275.0 | 257.07574315068456 |
# | 75.0 | 322.5 | 250.11792123287654 |
# | 75.0 | 325.0 | 250.81590068493134 |
# | 72.5 | 205.0 | 262.74933904109537 |
# | 72.5 | 207.5 | 262.5384315068488  |
# | 72.5 | 230.0 | 260.82879452054743 |
# | 72.5 | 275.0 | 257.3063321917804  |
# +------+-------+--------------------+
# Data truncated.

# The full query is only made when we call `collect()`, or, in this case,
# `to_pandas()`.
df = result.to_pandas()
df.head()
#     lat    lon     air_avg
# 0  75.0  232.5  258.836188
# 1  75.0  247.5  257.716171
# 2  75.0  262.5  257.347959
# 3  75.0  277.5  257.671308
# 4  72.5  232.5  260.654401
```

Succinctly, we "pivot" Xarray Datasets (with consistent dimensions) to treat them like tables so we can run
SQL queries against them.

## Why build this?

A few reasons:

* Even though SQL is the lingua franca of data, scientific datasets are often
  inaccessible to non-scientists (SQL users).
* Joining tabular data with raster data is common yet difficult. It could be
  easy.
* There are many cloud-native, Xarray-openable datasets,
  from [Google Earth Engine](https://github.com/google/Xee)
  to the [Source Cooperative](https://source.coop/products?tags=zarr). Wouldn’t it be great if these
  were also SQL-accessible? How can the bridge be built with minimal effort?

This is a light-weight way to prove the value of the interface.

The larger goal is to explore the hypothesis that the [Pangeo
ecosystem is a scientific database](https://www.hytradboi.com/2025/c18b8cdc-fd17-4099-9c03-eb107217f627-pangeo-is-a-database). Here, xarray-sql can be thought of as a missing
DB front end.

## How does it work?

All chunks in a Xarray Dataset are transformed into a Dask DataFrame via
`from_map()` and `to_dataframe()`. For SQL support, we just use `dask-sql`.
That's it!

_2025 update_: This library now implements a Dask-like `from_map` interface in
pure DataFusion and PyArrow, but works with the same principle!

_2026 update_: Instead of `from_map()`, we make factory functions from blocks of
Xarray datasets that return RecordBatchReaders. These feed into a Rust-based
DataFusion `TableProvider`. Every chunk is uses the Arrow in memory format to
translate between Python and Rust. Even still, the core of what makes this idea
work is the core `pivot()` operation from where this project began!

## Why does this work?

Underneath Xarray, Dask, and Pandas, there are NumPy arrays. These are paged in
chunks and represented contiguously in memory. It is only a matter of metadata
that breaks them up into ndarrays. `pivot()`, which uses `to_dataframe()`,
just changes this metadata (via a `ravel()`/`reshape()`), back into a column
amenable to a DataFrame. We take advantage of this light weight metadata change to
make chunked information scannable by a DB engine (DataFusion).

## What are the current limitations?

TBD, DataFusion provides a whole new world! Currently, we're looking for
early users – "tire kickers", if you will. We'd love your input to shape the direction of this
project! Please, give this a try and [file issues](https://github.com/alxmrs/xarray-sql/issues) as
you see fit. Check out our [contributing guide](CONTRIBUTING.md), too 😉.

I can say that for now, the library is oriented towards making whole scans of
Xarray Datasets. Common filter optimizations (even basic ones like an `.sel()` on
core dimensions, let alone predicate push downs) are not fully implemented yet.
However, these operations and more are on our roadmap.

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

_2025 update_: Something like this is being built across a few projects! The ones I know about are:
- [CartoDB's Raquet](https://github.com/CartoDB/raquet)
- The DataFusion community's [arrow-zarr](https://github.com/datafusion-contrib/arrow-zarr)

As of writing, this project is [amid integrating](https://github.com/alxmrs/xarray-sql/pull/69) a
rust-based DataFusion backend provided by arrow-zarr.

## Roadmap

- [x] ~Lazy evaluation via the pyarrow Dataset interface [#93](https://github.com/alxmrs/xarray-sql/issues/93).~ _Implemented in [#100](https://github.com/alxmrs/xarray-sql/pull/100)_
- [ ] Support proper parallelism via proper partition handling on the rust/datafusion side. [#106](https://github.com/alxmrs/xarray-sql/issues/106)
- [ ] Support core datafusion optimizations to scan less data, like [104](https://github.com/alxmrs/xarray-sql/issues/104), ...
- [ ] Translate a single Zarr to a collection of tables via DataFusion's catalog interface [#85](https://github.com/alxmrs/xarray-sql/issues/85).
- [ ] Distributed beyond a single node through the DataFusion integration with Ray Datasets [#68](https://github.com/alxmrs/xarray-sql/issues/68) or Apache Ballista [#98](https://github.com/alxmrs/xarray-sql/issues/98).
- [ ] Demo: calculate Sea Surface Temperature from 1940 - Present in SQL [#36](https://github.com/alxmrs/xarray-sql/issues/36).
- [ ] Provide an option to integrate DataFusion directly to Zarr via Rust [#4](https://github.com/alxmrs/xarray-sql/issues/4).
- [ ] (To be formally announced eventually): The 100 Trillion Row Challenge [#34](https://github.com/alxmrs/xarray-sql/issues/34).

## Sponsors & Contributors

I want to give a special thanks to the following folks and institutions:

- Pramod Gupta and the Anthromet Team at Google Research for the problem
  formation and design inspiration.
- Jake Wall and AI2/Ecoscope for compute resources and key use cases.
- Charles Stern, Stephan Hoyer, Alexander Kmoch, Wei Ji, and Qiusheng Wu
  for the early review and discussion of this project.
- Tom Nichols, Kyle Barron, Tom White, and Maxime Dion for the [Array Working
  Group](https://discourse.pangeo.io/t/new-working-group-for-distributed-array-computing/2734)
  and DataFusion-specific collaboration.
- The gracious volunteer data science students at [UCSD's DS3](https://www.ds3atucsd.com/) org,
  who are working to make this library better.


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

All vendored code has proper license attribution.
