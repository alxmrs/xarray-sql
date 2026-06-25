#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "pyproj",
#   "pyarrow",
#   "xee",
#   "earthengine-api",
#   "shapely",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Reprojection — a per-pixel CRS transform is a scalar UDF (à la ST_Transform).

Reprojection moves coordinates from one CRS to another (here UTM zone 10N,
EPSG:32610, → lon/lat, EPSG:4326). Crucially it is **row-independent**: each
pixel's new coordinate depends only on its own old coordinate. That is exactly
the shape of a SQL *scalar UDF*, and it is precisely how the geospatial SQL
world already does it — PostGIS ``ST_Transform`` and DuckDB-spatial
``ST_Transform`` are scalar PROJ wrappers.

So we register a PROJ-backed scalar UDF and reproject in SQL::

    SELECT x, y, reproject(x, y)['lon'] AS lon, reproject(x, y)['lat'] AS lat
    FROM grid

**The reference is Earth Engine itself.** There is *one* dataset: a single UTM
grid opened through [Xee](https://github.com/google/Xee) carrying
``ee.Image.pixelLonLat()``. Each pixel arrives with two things — its UTM ``x``/
``y`` (the grid coordinates, our SQL input) and Earth Engine's *own* per-pixel
``longitude``/``latitude`` (data variables, the reference). So we are not
opening the same image twice in two CRS; we feed the UTM coordinates to the PROJ
UDF and check the lon/lat it returns against EE's independently-computed lon/lat
for the *same* pixels. The reference is a different geodesy engine, not PROJ
again, and they agree to sub-metre precision.

PROJ's context is not thread-safe and DataFusion evaluates projection
expressions concurrently, so we return *both* coordinates from one
struct-returning UDF and keep the source in a single chunk (one serial UDF).

Requires Earth Engine access: ``earthengine authenticate`` once, then an
initialized project (set ``EARTHENGINE_PROJECT``). Skips cleanly otherwise.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyproj
import xarray as xr
from datafusion import udf

import xarray_sql as xql

from _harness import (
    CaseSkipped,
    assert_grid_close,
    initialize_earth_engine,
    measured,
    run_case,
    show_result,
    show_sql,
)

_SRC_CRS, _DST_CRS = "EPSG:32610", "EPSG:4326"  # UTM zone 10N → lon/lat
# A 1° box over the San Francisco Bay area, well inside UTM zone 10N.
_AOI = (-122.6, 37.4, -121.6, 38.4)
_SCALE_M = 2_000  # 2 km pixels → a ~50×60 grid


def register_reproject_udf(
    ctx, src_crs: str, dst_crs: str, name: str = "reproject"
) -> None:
    """Register a ``reproject(x, y) -> {lon, lat}`` PROJ scalar UDF.

    Mirrors ``xarray_sql.cftime.make_cftime_udf``: a vectorized scalar UDF over
    Arrow arrays. ``always_xy=True`` keeps argument order (easting, northing) →
    (lon, lat) regardless of CRS axis conventions. Like PostGIS/DuckDB
    ``ST_Transform``, it returns *both* output coordinates from one call — here
    as an Arrow struct, so callers write ``reproject(x, y)['lon']``.

    Returning a struct (rather than two separate UDFs) is deliberate: PROJ's
    context is not thread-safe, and DataFusion evaluates independent projection
    expressions concurrently — two PROJ UDFs in one SELECT race and crash. One
    struct-returning UDF does the transform exactly once per row, on one thread.
    """
    ret = pa.struct([("lon", pa.float64()), ("lat", pa.float64())])

    def _fn(x: pa.Array, y: pa.Array) -> pa.Array:
        # Build the Transformer inside the call so it lives on the worker
        # thread that uses it (PROJ contexts are thread-bound).
        transformer = pyproj.Transformer.from_crs(
            src_crs, dst_crs, always_xy=True
        )
        xs = np.asarray(x.to_numpy(zero_copy_only=False), dtype="float64")
        ys = np.asarray(y.to_numpy(zero_copy_only=False), dtype="float64")
        lon, lat = transformer.transform(xs, ys)
        return pa.StructArray.from_arrays(
            [
                pa.array(np.asarray(lon, "float64")),
                pa.array(np.asarray(lat, "float64")),
            ],
            names=["lon", "lat"],
        )

    ctx.register_udf(
        udf(_fn, [pa.float64(), pa.float64()], ret, "immutable", name)
    )


def _open_ee_lonlat_grid() -> xr.Dataset:
    """Open ``ee.Image.pixelLonLat()`` on a UTM grid via Xee.

    Earth Engine evaluates ``pixelLonLat`` on the requested UTM grid, so each
    pixel carries its UTM ``x``/``y`` (coordinates) and EE's own ``longitude`` /
    ``latitude`` (data variables) — the independent reprojection reference.
    """
    try:
        import shapely.geometry as sgeom
        from xee import helpers
    except ImportError as exc:  # pragma: no cover
        raise CaseSkipped(
            "Earth Engine support needs `pip install earthengine-api xee`"
        ) from exc

    ee = initialize_earth_engine()

    # fit_geometry builds the pixel grid (crs, crs_transform, shape_2d) Xee's
    # backend expects — here a UTM grid at _SCALE_M metres covering the AOI.
    grid = helpers.fit_geometry(
        sgeom.box(*_AOI),
        geometry_crs="EPSG:4326",
        grid_crs=_SRC_CRS,
        grid_scale=(float(_SCALE_M), float(_SCALE_M)),
    )
    ic = ee.ImageCollection([ee.Image.pixelLonLat()])
    ds = xr.open_dataset(ic, engine="ee", **grid)
    # One image → a length-1 time axis; drop it. Xee gives x/y coordinates (UTM
    # metres) and longitude/latitude data variables (EE's per-pixel geodesy).
    return ds.isel(time=0).load()


def main() -> None:
    ds = _open_ee_lonlat_grid()
    n = ds.sizes["y"] * ds.sizes["x"]
    print(
        f"  EE pixelLonLat on UTM grid {dict(ds.sizes)}  ({n:,} pixels)  "
        f"{_SRC_CRS} → {_DST_CRS}"
    )

    ctx = xql.XarrayContext()
    # Single chunk → single partition → serial UDF (PROJ is not thread-safe).
    ctx.from_dataset(
        "grid", ds, chunks={"y": ds.sizes["y"], "x": ds.sizes["x"]}
    )
    register_reproject_udf(ctx, _SRC_CRS, _DST_CRS)

    sql = """
        SELECT x, y,
               reproject(x, y)['lon'] AS lon,
               reproject(x, y)['lat'] AS lat
        FROM grid
        ORDER BY y, x
    """
    show_sql(sql)

    for _ in measured("SQL reprojection (PROJ scalar UDF)"):
        got = ctx.sql(sql).to_dataset(dims=["y", "x"])

    # Reference: Earth Engine's own per-pixel lon/lat (independent of PROJ).
    # EE and PROJ are separate implementations, so compare at ~1e-5° (~1 m).
    assert_grid_close(
        "reprojected longitude", got.lon, ds.longitude, rtol=0, atol=1e-5
    )
    assert_grid_close(
        "reprojected latitude", got.lat, ds.latitude, rtol=0, atol=1e-5
    )

    show_result(got)

    corner = got.isel(x=0, y=0)
    print(
        f"\n  Corner check: UTM ({float(corner.x):.0f}, {float(corner.y):.0f}) → "
        f"lon {float(corner.lon):.4f}, lat {float(corner.lat):.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Reprojection: PROJ scalar UDF vs Earth Engine")
    )
