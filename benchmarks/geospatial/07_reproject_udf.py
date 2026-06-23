# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "pyproj",
#   "pyarrow",
# ]
# ///
"""Reprojection — a per-pixel CRS transform is a scalar UDF (à la ST_Transform).

Reprojection moves coordinates from one CRS to another (here UTM zone 32N,
EPSG:32632, → lon/lat, EPSG:4326). Crucially it is **row-independent**: each
pixel's new coordinate depends only on its own old coordinate. That is exactly
the shape of a SQL *scalar UDF*, and it is precisely how the geospatial SQL
world already does it — PostGIS ``ST_Transform`` and DuckDB-spatial
``ST_Transform`` are scalar PROJ wrappers.

So we register a PROJ-backed scalar UDF and reproject in SQL::

    SELECT x, y, utm_to_lon(x, y) AS lon, utm_to_lat(x, y) AS lat
    FROM grid

The UDF (built on ``pyproj``, vectorized over each Arrow batch) mirrors the
``cftime()`` UDF already shipped in xarray-sql (see ``xarray_sql/cftime.py``);
it could graduate into the package as ``xql.register_reproject_udfs``.

**What this does *not* do:** it moves the coordinates, it does not resample onto
a regular target grid. Producing a gridded product in the new CRS still needs
interpolation — which is case 08, where regridding turns out to be a JOIN
against a weight table rather than a scalar UDF.

**An honest caveat on threading:** PROJ's transformation context is not
thread-safe, and DataFusion evaluates independent projection expressions on
concurrent worker threads. Two separate PROJ UDFs in one ``SELECT`` (one for
lon, one for lat) race and segfault. The fix is to return *both* coordinates
from a **single** struct-returning UDF (so PROJ is touched once per row), and
to keep the source in one chunk so that single UDF runs serially. A
production-grade ``ST_Transform`` would additionally give each worker thread its
own PROJ context; the point here is the *shape* of the operation — a scalar
UDF — not the parallel execution.

No network: a synthetic UTM grid over the extent of a Sentinel-2 tile.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyproj
import xarray as xr
from datafusion import udf

import xarray_sql as xql

from _harness import check_close, run_case, show_sql, timed


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


def _utm_grid() -> xr.Dataset:
    """A synthetic field on a UTM grid matching a Sentinel-2 T32T tile extent."""
    # EPSG:32632 extent of tile T32TLQ (from the STAC proj:bbox), coarsened.
    x = np.linspace(300_000.0, 409_800.0, 60, dtype="float64")
    y = np.linspace(5_000_040.0, 4_890_240.0, 60, dtype="float64")
    elevation = np.zeros(
        (y.size, x.size), dtype="float64"
    )  # value is irrelevant
    # Single chunk → single partition → serial UDF (PROJ is not thread-safe).
    return xr.Dataset(
        {"elevation": (["y", "x"], elevation)},
        coords={"y": y, "x": x},
    ).chunk({"y": 60, "x": 60})


def main() -> None:
    src_crs, dst_crs = "EPSG:32632", "EPSG:4326"
    ds = _utm_grid()
    print(
        f"  UTM grid {dict(ds.sizes)}  ({ds.sizes['y'] * ds.sizes['x']:,} points)  "
        f"{src_crs} → {dst_crs}"
    )

    ctx = xql.XarrayContext()
    ctx.from_dataset("grid", ds, chunks={"y": 60, "x": 60})
    register_reproject_udf(ctx, src_crs, dst_crs)

    sql = """
        SELECT x, y,
               reproject(x, y)['lon'] AS lon,
               reproject(x, y)['lat'] AS lat
        FROM grid
        ORDER BY y, x
    """
    show_sql(sql)

    with timed("SQL reprojection (PROJ scalar UDF)"):
        got = ctx.sql(sql).to_pandas()

    # Array reference: the same PROJ transform applied to the full grid.
    with timed("pyproj reference"):
        transformer = pyproj.Transformer.from_crs(
            src_crs, dst_crs, always_xy=True
        )
        xx, yy = np.meshgrid(ds.x.values, ds.y.values)
        ref_lon, ref_lat = transformer.transform(xx.ravel(), yy.ravel())

    got = got.sort_values(["y", "x"]).reset_index(drop=True)
    # The reference is built in (y, x) row-major order; align by sorting both.
    order = np.lexsort((xx.ravel(), yy.ravel()))
    check_close(
        "reprojected longitude", got["lon"], ref_lon[order], rtol=0, atol=1e-9
    )
    check_close(
        "reprojected latitude", got["lat"], ref_lat[order], rtol=0, atol=1e-9
    )

    print(
        f"\n  Corner check: UTM ({got.x.iloc[0]:.0f}, {got.y.iloc[0]:.0f}) → "
        f"lon {got.lon.iloc[0]:.4f}, lat {got.lat.iloc[0]:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Reprojection: PROJ scalar UDF (ST_Transform)")
    )
