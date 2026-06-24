# Geospatial operations are relational operations

A working hypothesis, and a slightly radical one: **the core operations of
geospatial and climate analysis — the ones we reach for an array library to
perform — are, underneath, relational operations.** Climatologies, anomalies,
zonal means, spectral indices, forecast skill, even regridding: each maps onto
ordinary SQL — `GROUP BY`, `JOIN`, window functions, `CASE`, and the occasional
scalar UDF.

The array paradigm (NumPy, Xarray, Dask) is a wonderful *interface* for these
operations. But it is not the only one, and for a large and growing audience —
the people fluent in SQL rather than in `apply_ufunc` and rechunking — it is not
the most accessible one. [`xarray-sql`](../README.md) lets you pose these
questions in SQL and answers them with a real query engine (DataFusion),
complete with partition pruning and projection pushdown. The datasets are
registered *lazily* — nothing is read or column-selected up front; each query
pulls only the variable and the partitions it needs, and value filters are
passed as bound **query parameters** rather than formatted into the SQL string.
And because a gridded result is still gridded data, every query here round-trips
its answer straight back to an `xarray.Dataset` (via `to_dataset`) — SQL in, an
array out, ready to plot or save.

This page makes the argument case by case. Every claim below is backed by a
runnable script in [`benchmarks/geospatial/`](../benchmarks/geospatial/) that
poses the operation in SQL and **asserts the answer matches an xarray/array
reference** to floating-point tolerance. The point is not that "SQL is faster";
the point is that the SQL reads like the *definition* of the operation and
computes the same numbers — at ERA5's real 0.25° global resolution.

## The mapping

| Operation | The "array" framing | The relational reality | Script |
|-----------|---------------------|------------------------|--------|
| Spectral index (NDVI) | `apply_ufunc` over a raster | column arithmetic | [`01_ndvi.py`](../benchmarks/geospatial/01_ndvi.py) |
| Climatology | rechunk → grouped reduction | `GROUP BY lat, lon, hour-of-day` | [`02_climatology.py`](../benchmarks/geospatial/02_climatology.py) |
| Zonal mean | reduce over lon/time axes | `GROUP BY lat` | [`03_zonal_mean.py`](../benchmarks/geospatial/03_zonal_mean.py) |
| Anomaly | grouped broadcast-subtract | climatology CTE self-`JOIN` | [`04_anomaly.py`](../benchmarks/geospatial/04_anomaly.py) |
| Forecast skill (RMSE) | align valid/init/lead, reduce | forecast↔truth `JOIN` on `valid_time` | [`05_forecast_skill.py`](../benchmarks/geospatial/05_forecast_skill.py) |
| Zonal stats over regions | rasterize polygons + mask | raster × vector range `JOIN` | [`06_zonal_vector.py`](../benchmarks/geospatial/06_zonal_vector.py) |
| Reprojection | per-pixel CRS transform | scalar **UDF** (`ST_Transform`-style) | [`07_reproject_udf.py`](../benchmarks/geospatial/07_reproject_udf.py) |
| Regridding | interpolation to a new grid | sparse-weight table `JOIN` | [`08_regrid_weights.py`](../benchmarks/geospatial/08_regrid_weights.py) |

## 1. A pixel-wise formula is a column expression

NDVI is `(NIR − Red) / (NIR + Red)`, per pixel. The array idiom broadcasts a
ufunc over the raster. But "one output per pixel, computed from that pixel's
bands" is the definition of a SQL projection:

```sql
SELECT x, y, (nir - red) / (nir + red) AS ndvi
FROM scene
ORDER BY y, x
```

Invalid pixels need no special handling: xarray decodes the band's `_FillValue`
to `NaN` on open, and `NaN` propagates through the arithmetic on both sides, so
the masking is free.

[`01_ndvi.py`](../benchmarks/geospatial/01_ndvi.py) runs this against a **real
Sentinel-2 L2A scene in Zarr** — discovered with `pystac-client` and opened the
canonical way with `xr.open_datatree` (ESA's EOPF sample service) — and matches
xarray's `apply_ufunc`-style result over a million pixels.

## 2. A climatology is a `GROUP BY` over the cycle

A climatology is the average value for each time-of-cycle at each location. In
the array world this is the canonical painful workload — load native chunks,
*rechunk* so all of time lands in one chunk, reduce, rechunk back. The
rechunking serves the array layout, not the question. The question is:

```sql
SELECT latitude, longitude, date_part('hour', time) AS hour,
       AVG("2m_temperature")
FROM era5 GROUP BY latitude, longitude, date_part('hour', time)
```

The grouping keys are the dimensions you keep; everything else is reduced. No
layout to reason about. [`02_climatology.py`](../benchmarks/geospatial/02_climatology.py)
computes the **diurnal cycle** of ERA5 2m-temperature over a region — averaging
each cell by hour of day — and matches `da.groupby("time.hour").mean()` across
~500k cells.

A **zonal mean** ([`03_zonal_mean.py`](../benchmarks/geospatial/03_zonal_mean.py))
is the same idea with fewer keys: the axes you "reduce over" are simply the
columns you don't `GROUP BY`.

## 3. Broadcasting a normal back onto observations is a `JOIN`

An anomaly subtracts each cell's climatological normal from every matching
observation. Xarray expresses the realignment with grouped broadcasting
(`ds.groupby("time.hour") - climatology`). That realignment — *attach each
cell's normal to every timestep that shares its key* — is a JOIN on the
grouping key:

```sql
WITH clim AS (
  SELECT latitude, longitude, date_part('hour', time) AS hour,
         AVG("2m_temperature") AS clim_t
  FROM era5 GROUP BY latitude, longitude, date_part('hour', time)
)
SELECT a.time, a.latitude, a.longitude,
       a."2m_temperature" - c.clim_t AS anomaly
FROM era5 a JOIN clim c
  ON a.latitude = c.latitude AND a.longitude = c.longitude
 AND date_part('hour', a.time) = c.hour
```

[`04_anomaly.py`](../benchmarks/geospatial/04_anomaly.py) computes the
climatology once (the CTE) and joins it back to every observation.

## 4. Forecast evaluation is a `JOIN` on valid time + aggregate

This is the real workload of [WeatherBench 2](https://weatherbench2.readthedocs.io/):
scoring machine-learning weather models — **Pangu-Weather** and **GraphCast** —
against ERA5 ground truth. A forecast is indexed by *initialization time* and
*lead time* (`prediction_timedelta`); the truth is indexed by *valid time*.
Evaluation aligns them by `valid_time = init + lead` and reduces the error to
RMSE as a function of lead.

That alignment is a relational JOIN, and `valid_time = init + lead` is just
timestamp + duration arithmetic the engine does natively:

```sql
SELECT f.model, f.prediction_timedelta AS lead,
       SQRT(AVG(POWER(f."2m_temperature" - e."2m_temperature", 2))) AS rmse
FROM forecasts f
JOIN era5 e
  ON  e.time = f.time + f.prediction_timedelta   -- valid_time = init + lead
  AND e.latitude  = f.latitude
  AND e.longitude = f.longitude
GROUP BY f.model, f.prediction_timedelta
```

Both models are stacked along a `model` dimension into one forecast table, so
the query scores them together by grouping on a `model` *column* — no table name
formatted into the SQL. The entire evaluation — temporal alignment across three
time axes, spatial matching, and the score — is one JOIN and one aggregate.
[`05_forecast_skill.py`](../benchmarks/geospatial/05_forecast_skill.py) runs it
for both models, matches an xarray reference, and reproduces the published result
that GraphCast edges out Pangu at every lead — the classic "error grows with
horizon" curve (≈0.3 K at 6 h rising to ≈2.5 K at 9 days):

```
  lead (days)     Pangu   GraphCast
         0.25     0.336       0.296
         5.25     1.469       1.228
         9.25     2.814       2.380
```

## 5. Raster × vector zonal statistics is a range `JOIN`

"Average the raster inside each region" is the canonical raster-meets-vector
task. The array idiom rasterizes each polygon to a mask and reduces under it. But
a region is a row in a table of bounds, and "pixel inside region" is a range
predicate — so zonal statistics is a JOIN:

```sql
SELECT r.region, AVG(a."2m_temperature") - 273.15 AS avg_c
FROM era5.surface a JOIN regions r
  ON  a.latitude  BETWEEN r.lat_min AND r.lat_max
  AND a.longitude BETWEEN r.lon_min AND r.lon_max
WHERE a.time BETWEEN TIMESTAMP '2020-06-01' AND TIMESTAMP '2020-06-01 23:00:00'
GROUP BY r.region
```

This is the README's promise — *joining tabular data with raster data* — made
literal: the raster is the full ERA5 archive (the `WHERE` prunes it to a day),
the regions are a second SQL table, and the spatial relationship is an ordinary
`BETWEEN`. See [`06_zonal_vector.py`](../benchmarks/geospatial/06_zonal_vector.py)
— it reports e.g. Sahara 33 °C vs Greenland −8 °C for a June day. (Rectangular
regions keep the demo dependency-free; arbitrary polygons are the natural next
step, via a point-in-polygon UDF — see below.)

## 6. The hard cases: where a UDF fits, and where it doesn't

Reprojection and regridding are the operations most wedded to the array
paradigm. They split cleanly along one line: **is the operation row-independent?**

**Reprojection is.** Moving a coordinate from one CRS to another depends only on
that coordinate, so it is a *scalar function* — exactly what PostGIS and
DuckDB-spatial already ship as `ST_Transform`. We register a PROJ-backed scalar
UDF (mirroring the `cftime()` UDF already in `xarray_sql/cftime.py`) and
reproject in SQL:

```sql
SELECT x, y, reproject(x, y)['lon'] AS lon, reproject(x, y)['lat'] AS lat
FROM grid
```

[`07_reproject_udf.py`](../benchmarks/geospatial/07_reproject_udf.py) matches
pyproj to 1e-9. Two honest caveats, both documented in the script: PROJ's
context is not thread-safe (so the UDF returns both coordinates from *one* call
and runs on a single partition), and reprojection moves coordinates without
resampling onto a grid — which is the next operation.

**Regridding is not** row-independent: each output cell is a weighted blend of
several input cells. That is a *many-to-many* relationship — and a many-to-many
weighted blend is a sparse matrix–vector product, which is a `JOIN` against a
weight table plus a weighted `GROUP BY`:

```sql
SELECT w.dst_id, SUM(s.value * w.weight) AS regridded
FROM weights w JOIN src s ON s.cell_id = w.src_id
GROUP BY w.dst_id
```

[`08_regrid_weights.py`](../benchmarks/geospatial/08_regrid_weights.py) matches
xarray's bilinear `.interp()` exactly. So regridding does not weaken the thesis —
it is the most relational operation of all.

## Where the array paradigm still earns its keep

The honest boundary is **weight generation**. Applying a regridding is a join;
*computing* the weights — cell overlaps for conservative remapping, stencils and
spherical geometry for bilinear, the whole machinery of xESMF/ESMF — is genuinely
geometric work that arrays (and specialized libraries) do well. The relational
view does not replace that; it consumes its output. The division of labor is
clean and, we think, the right one:

> **Arrays compute the geometry (the weights). SQL applies it (the join).**

Likewise, the array libraries remain the right tool for building the inputs in
the first place — opening Zarr, decoding CF metadata, the numerics of generating
a weight matrix. `xarray-sql` sits downstream of all that as a query front-end:
once the data is openable as an `xarray.Dataset`, these everyday operations are
expressible — and accessible — as SQL.

## Running the suite

```shell
python benchmarks/geospatial/02_climatology.py   # inside the repo
uv run benchmarks/geospatial/02_climatology.py   # standalone (PEP 723 deps)
```

Each script prints its SQL, runs the array reference, and asserts the two agree.
See [`benchmarks/geospatial/README.md`](../benchmarks/geospatial/README.md) for
the full list and dataset notes.
