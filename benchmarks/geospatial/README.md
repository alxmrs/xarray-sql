# Geospatial SQL benchmarks

**Thesis:** the core geospatial operations we assume require an *array* paradigm
are, underneath, **relational** operations — `GROUP BY`, `JOIN`, window
functions, and `CASE`. Each script here takes one such operation, expresses it
in SQL against [`xarray-sql`](../../README.md), and **proves the SQL answer
matches an xarray/array reference implementation** (`numpy.assert_allclose`).
Wall-clock and peak memory are reported too, but the headline is correctness +
clarity of the SQL.

This suite is *expressibility-first*: the point is that the SQL reads like the
plain-English definition of the operation, and computes the same numbers.

## The cases

| # | Case | Array mental model | Relational reality |
|---|------|--------------------|--------------------|
| 01 | `01_ndvi.py` | `apply_ufunc` over a raster | column arithmetic |
| 02 | `02_climatology.py` | rechunk → grouped reduction | `GROUP BY lat, lon, hour-of-day` |
| 03 | `03_zonal_mean.py` | reduce over lon/time axes | `GROUP BY latitude` |
| 04 | `04_anomaly.py` | climatology broadcast-subtract | climatology CTE self-`JOIN` |
| 05 | `05_forecast_skill.py` | align valid/init/lead, reduce | forecast↔truth `JOIN` on `valid_time` + aggregate |
| 06 | `06_zonal_vector.py` | rasterize + mask per region | range `JOIN` raster↔regions |
| 07 | `07_reproject_udf.py` | per-pixel CRS transform | scalar **UDF** (`reproject()`), à la PostGIS `ST_Transform` |
| 08 | `08_regrid_weights.py` | interpolation to a new grid | sparse-weight table `JOIN` + weighted `GROUP BY` |

Cases 01–06 show operations that are *natively* relational. Cases 07–08 are the
"hardest" array operations — reprojection and regridding — and show where a UDF
fits (a per-row coordinate transform) versus where the operation is really a
sparse matrix multiply expressed as a `JOIN`. See
[`docs/geospatial.md`](../../docs/geospatial.md) for the full narrative,
including *where the array paradigm still earns its keep* (generating the
interpolation weights — the geometry — which SQL applies but does not compute).

## Datasets

- **01 NDVI** — a real Sentinel-2 L2A scene in **Zarr** from the ESA EOPF sample
  service, discovered with `pystac-client` and opened with `xr.open_datatree`
  (bands B04/B08). Requires network; skips cleanly if offline.
- **02–06** — the full **[ARCO-ERA5](https://github.com/google-research/arco-era5)**
  archive (0.25° global, ~1.3M hourly timesteps, 273 variables) read anonymously
  from a public GCS bucket. Cases 03 and 06 register the *whole* archive and let
  SQL `WHERE` prune it to a one-day window (the partition-pruning demo); cases
  02/04 read a bounded region/time window into memory once and compare
  SQL against the same array. All require network (`gcsfs`); skip cleanly
  offline. Each ERA5 case takes roughly a minute, dominated by the GCS read.
- **05 forecast skill** — the **[WeatherBench 2](https://weatherbench2.readthedocs.io/)**
  Pangu-Weather, GraphCast, and ERA5 datasets at a coarse 64×32 grid, scoring
  both ML models against ERA5 ground truth. Network-backed; runs in seconds
  because the grid is small.
- **07–08** — small/synthetic grids plus precomputed regrid weights, so they run
  without heavy geospatial dependencies (ESMF/ESMPy).

## Running

Inside the repo, use the project environment (so `import xarray_sql` resolves to
the locally built native extension):

```shell
python benchmarks/geospatial/03_zonal_mean.py
```

Each script also carries [PEP 723 / `uv` inline metadata](https://docs.astral.sh/uv/guides/scripts/),
so it can be run standalone against the published `xarray-sql` wheel:

```shell
uv run benchmarks/geospatial/03_zonal_mean.py
```

A passing case prints a `✅ … SQL matches array reference` line; a mismatch
raises `AssertionError` and exits non-zero. Cases that need an unavailable
dataset/dependency print `⏭ SKIPPED` and exit 0.

Shared helpers (timing, peak memory, the `assert_allclose` wrapper, SQL echo)
live in [`_harness.py`](_harness.py).
