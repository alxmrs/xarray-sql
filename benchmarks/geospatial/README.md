# Geospatial SQL benchmarks

**Thesis:** the core geospatial operations we assume require an *array* paradigm
are, underneath, **relational** operations — `GROUP BY`, `JOIN`, window
functions, and `CASE`. Each script here takes one such operation, expresses it
in SQL against [`xarray-sql`](../../README.md), and **proves the SQL answer
matches a plain-xarray reference** to floating-point tolerance. Wall-clock and
peak memory are reported too, but the headline is correctness + clarity of the
SQL.

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
  from a public GCS bucket. Each case opens the *whole* archive lazily, so a query
  reads only the variable and the window it asks for — never the other 272
  variables or the rest of the timesteps. All require network (`gcsfs`) and skip
  cleanly offline; each takes roughly one to a few minutes, dominated by the read.
- **05 forecast skill** — the **[WeatherBench 2](https://weatherbench2.readthedocs.io/)**
  Pangu-Weather, GraphCast, and ERA5 datasets at a coarse 64×32 grid, scoring
  both ML models against ERA5 ground truth. Network-backed; runs in seconds
  because the grid is small.
- **07–08** — the **Earth Engine** catalog via [Xee](https://github.com/google/Xee).
  07 reprojects a UTM grid and validates the SQL transform against Earth Engine's
  *own* per-pixel lon/lat (`ee.Image.pixelLonLat()`) — an independent reprojection
  reference, not PROJ-vs-PROJ. 08 regrids real **SRTM elevation** (Sierra Nevada)
  and validates against xarray's bilinear `.interp()`. Both run against Earth
  Engine using your existing `gcloud` login, and skip cleanly without it.

## Running

Run a single case, or the whole suite, from any directory:

```shell
uv run benchmarks/geospatial/03_zonal_mean.py   # one case
benchmarks/geospatial/run_all.sh                # all of them
```

Each script carries [PEP 723 / `uv` inline metadata](https://docs.astral.sh/uv/guides/scripts/)
and runs against the `xarray-sql` in this checkout.

A passing case prints a `✅ … SQL matches xarray reference` line and the result
as an xarray repr; a mismatch raises `AssertionError` and exits non-zero. Cases
that need data or credentials you don't have print `⏭ SKIPPED` and exit 0.

Shared helpers — timing, peak memory, the result check and its printout, SQL
echo — live in [`_harness.py`](_harness.py).
