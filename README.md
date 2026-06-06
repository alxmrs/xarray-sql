# xarray-sql

_Query [Xarray](https://xarray.dev/) with SQL_

[![ci](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci.yml)
[![lint](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/lint.yml)
[![ci-build](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-build.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-build.yml)
[![ci-rust](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-rust.yml/badge.svg)](https://github.com/alxmrs/xarray-sql/actions/workflows/ci-rust.yml)

```shell
pip install xarray-sql
```

## What is this?

This is an experiment to provide a SQL interface for array datasets.

```python
import xarray as xr
import xarray_sql as xql


# Open a year of ARCO-ERA5 — all 273 variables. Selecting a year up front
# keeps Dask's partition setup cheap before any chunks are read from GCS.
ds = (
  xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
               chunks=dict(time=1),
               storage_options={'token': 'anon'})  # Anonymous read from the public GCS bucket — no auth required.
  .sel(time='2020')
)

ctx = xql.XarrayContext()
ctx.from_dataset('era5', ds, table_names={
    ('time', 'latitude', 'longitude'): 'surface',
    ('time', 'level', 'latitude', 'longitude'): 'atmosphere',
})
# Registration: ~0.5s for a full year of hourly ERA5, all variables.


# Heads up: ARCO-ERA5 has 262 surface + 11 atmospheric variables. The library
# pushes column projection down to Zarr, so SELECT only fetches what you ask
# for — but `SELECT * FROM era5.surface` would try to pull every variable
# across the year (terabytes from GCS). 
#  ---> Always SELECT specific columns. <---

# Average 2m-temperature over NYC on the morning of 2020-01-01. The library
# pushes WHERE clauses on dimension columns down to partition pruning.
ctx.sql('''
  SELECT AVG("2m_temperature") - 273.15 AS avg_c
  FROM era5.surface
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
    AND latitude  BETWEEN 39 AND 40
    AND longitude BETWEEN 286 AND 287  -- ERA5 uses 0-360 longitudes
''').to_pandas()
#       avg_c
# 0  8.640069

# Average temperature per pressure level, globally. 
ctx.sql('''
  SELECT level, AVG(temperature) - 273.15 AS avg_c
  FROM era5.atmosphere
  WHERE time BETWEEN TIMESTAMP '2020-01-01'
                 AND TIMESTAMP '2020-01-01 05:00:00'
  GROUP BY level
  ORDER BY level DESC
''').to_pandas()
#     level      avg_c
# 0    1000   6.621012   ← surface
# 1     975   5.185638
# 2     950   4.028429
# 3     925   3.082812
# 4     900   2.210917
# 5     875   1.395018
# 6     850   0.634267
# 7     825  -0.210372
# 8     800  -1.181075
# 9     775  -2.306465
# 10    750  -3.535534
# 11    700  -6.241685
# 12    650  -9.236364
# 13    600 -12.580938
# 14    550 -16.335386
# 15    500 -20.643604
# 16    450 -25.573401
# 17    400 -31.156920
# 18    350 -37.400552
# 19    300 -43.852607
# 20    250 -49.322132
# 21    225 -51.569113
# 22    200 -53.693248
# 23    175 -55.890484
# 24    150 -58.382290
# 25    125 -61.091916
# 26    100 -63.624885   ← tropopause
# 27     70 -63.182300
# 28     50 -60.124845
# 29     30 -55.986327
# 30     20 -52.433089
# 31     10 -44.140750
# 32      7 -38.707350
# 33      5 -32.621999
# 34      3 -21.509175
# 35      2 -13.355764
# 36      1  -9.020513   ← top of atmosphere
```

_(A runnable version of this example lives at
[`perf_tests/era5_temp_profile.py`](perf_tests/era5_temp_profile.py).)_

Succinctly, we "pivot" Xarray Datasets to treat them like tables so we can run
SQL queries against them. 

## Round-tripping back to Xarray

`ctx.sql(...)` returns an `XarrayDataFrame` that exposes `.to_pandas()`
(unchanged) and a new `.to_dataset()` for converting the result back into
an `xr.Dataset`. The reverse path is **lazy by default**: the returned
Dataset is backed by an `xarray.backends.BackendArray` that translates
xarray indexers into DataFusion `filter` expressions and consumes the
filtered DataFrame via `execute_stream`. Arrow `RecordBatch` es scatter
directly into a preallocated numpy buffer with no pandas hop, so only
the slab actually accessed is materialized.

```python
out = ctx.sql('SELECT * FROM "air"').to_dataset()
# <xarray.Dataset>
# Dimensions:  (time: 2920, lat: 25, lon: 53)
# Coordinates:
#   * time     (time) datetime64[ns] ...
#   * lat      (lat) float32 ...
#   * lon      (lon) float32 ...
# Data variables:
#     air      (time, lat, lon) float32 ...

# Slicing pushes down into DataFusion; only the requested slab is
# materialized.
slab = out["air"].isel(time=0).values

# For full eager materialization, call .compute().
eager = out.compute()
```

`dimension_columns` defaults to the dims of the single registered Dataset
on the context (or the one named via `template_table=` when several are
registered). Variable attrs, dataset attrs, non-dimension coordinates,
and dim-coordinate dtype are recovered from the registered Dataset
automatically.

For filtered queries that return only part of the original extent, pass
`sparsity="template"` to reindex back to the full grid with NaN
fills:

```python
out = ctx.sql(
    'SELECT * FROM "air" WHERE lat > 50'
).to_dataset(sparsity="template")
# Full lat range restored; cells with lat <= 50 are NaN.
```

Aggregation queries (e.g. `AVG(air) AS air_avg ... GROUP BY lat, lon`)
materialize once because their output does not align with the source dim
structure; the aggregation path is also Arrow-native (no pandas
intermediates). Pass `dimension_columns=[...]` explicitly when an
aggregation drops a dim.

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

_2026 update_: Instead of `from_map()`, we create a way to translate Xarray chunks
into Arrow RecordBatches. We pass a Python callback into a DataFusion `TableProvider`
that lets the DB engine translate the underlying Dataset arrays into DataFusion partitions.
Ultimately, the initial insight of the `pivot()` function -- that any ndarray can be
translated into a 2D table -- underlies this performant query mechanism. 

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

_2026 update_: A colleague and I are experimenting with native Zarr RDBMS engines. Check out:

- [Zarr-Datafusion](https://lib.rs/crates/zarr-datafusion)
- [DuckDB-Zarr](https://github.com/alxmrs/duckdb-zarr)

## Roadmap

- [x] ~Lazy evaluation via the pyarrow Dataset interface [#93](https://github.com/alxmrs/xarray-sql/issues/93).~ _Implemented in [#100](https://github.com/alxmrs/xarray-sql/pull/100)_
- [x] Support proper parallelism via proper partition handling on the rust/datafusion side. [#106](https://github.com/alxmrs/xarray-sql/issues/106)
- [x] Support core datafusion optimizations to scan less data, like [104](https://github.com/alxmrs/xarray-sql/issues/104), ...
- [x] Translate a single Zarr to a collection of tables [#85](https://github.com/alxmrs/xarray-sql/issues/85).
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
- Andrew Huang for the sense of taste he brings to the project and consummate code
  changes.


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
