#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "zarr>=3",
#   "numpy",
# ]
# ///
"""NDVI + cloud/nodata masking — "apply_ufunc over a raster" is column algebra.

The Normalized Difference Vegetation Index is the workhorse of optical remote
sensing: ``NDVI = (NIR - Red) / (NIR + Red)``, computed per pixel, with invalid
pixels (nodata / clouds) masked out. The array paradigm reaches for
``xarray.apply_ufunc`` (the coiled/benchmarks #1545 "vectorized operations"
case) to broadcast this over a whole scene.

But a per-pixel formula over two bands is just *column arithmetic over two
columns*, and "mask the invalid pixels" is just ``CASE WHEN``::

    SELECT x, y,
           CASE WHEN red = 0 OR nir = 0 THEN NULL
                ELSE (nir_refl - red_refl) / (nir_refl + red_refl)
           END AS ndvi
    FROM scene

Each pixel is one row; the ufunc is the SELECT expression. No broadcasting
rules, no apply_ufunc signature — the SQL *is* the index definition.

Dataset: a real Sentinel-2 L2A scene in **Zarr** from the ESA EOPF sample
service (bands B04=red, B08=NIR at 10 m, stored as uint16 with
``reflectance = DN * 0.0001 - 0.1``). We read one cloud-light window so the
case stays bounded. Requires network; skips cleanly if the store is offline.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import xarray_sql as xql

from _harness import CaseSkipped, check_close, run_case, show_sql, timed

# A specific Sentinel-2 L2A product from the EOPF sample service (STAC
# collection ``sentinel-2-l2a``, tile T32TLQ over the Italian Alps). Hard-coded
# so the case needs no STAC client; if the sample service rotates this product
# out, the open below fails and the case skips.
_BASE = (
    "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:"
    "202505-s02msil2a/03/products/cpm_v256/"
    "S2B_MSIL2A_20250503T102559_N0511_R108_T32TLQ_20250503T132157.zarr"
)
_R10M = _BASE + "/measurements/reflectance/r10m"

# proj:transform of the 10 m grid (EPSG:32632), from the STAC asset metadata.
_X0, _Y0, _RES = 300_000.0, 5_000_040.0, 10.0

# Reflectance encoding (STAC raster:scale / raster:offset), DN -> reflectance.
_SCALE, _OFFSET = 0.0001, -0.1

# A 1024×1024 window (~105 km²) offset into the tile to land on vegetated
# terrain (median NDVI ≈ 0.33 in early May) rather than the nodata border.
_ROW0, _COL0, _N = 2_500, 6_500, 1024


def _load_scene() -> xr.Dataset:
    """Read the B04/B08 window from the EOPF Zarr store into an xr.Dataset."""
    try:
        import zarr

        b04 = zarr.open_array(_R10M + "/b04", mode="r")
        b08 = zarr.open_array(_R10M + "/b08", mode="r")
        rows = slice(_ROW0, _ROW0 + _N)
        cols = slice(_COL0, _COL0 + _N)
        red = np.asarray(b04[rows, cols])
        nir = np.asarray(b08[rows, cols])
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"EOPF Sentinel-2 Zarr unavailable ({exc})") from exc

    # Pixel-center coordinates from the affine transform.
    x = _X0 + (np.arange(_COL0, _COL0 + _N) + 0.5) * _RES
    y = _Y0 - (np.arange(_ROW0, _ROW0 + _N) + 0.5) * _RES
    return xr.Dataset(
        {"red": (["y", "x"], red), "nir": (["y", "x"], nir)},
        coords={"y": y.astype("float64"), "x": x.astype("float64")},
    ).chunk({"y": 256, "x": 256})


def main() -> None:
    ds = _load_scene()
    n = ds.sizes["y"] * ds.sizes["x"]
    print(
        f"  Sentinel-2 L2A scene window: {dict(ds.sizes)}  "
        f"({n:,} pixels, B04=red/B08=NIR, uint16 DN)"
    )

    ctx = xql.XarrayContext()
    ctx.from_dataset("scene", ds, chunks={"y": 256, "x": 256})

    # DN -> reflectance happens inline; CASE masks nodata (DN == 0).
    sql = """
        SELECT x, y,
               CASE
                 WHEN red = 0 OR nir = 0 THEN NULL
                 ELSE ( (CAST(nir AS DOUBLE) * 0.0001 - 0.1)
                      - (CAST(red AS DOUBLE) * 0.0001 - 0.1) )
                    / ( (CAST(nir AS DOUBLE) * 0.0001 - 0.1)
                      + (CAST(red AS DOUBLE) * 0.0001 - 0.1) )
               END AS ndvi
        FROM scene
    """
    show_sql(sql)

    with timed("SQL NDVI"):
        got = ctx.sql(sql).to_dataset(dims=["y", "x"]).ndvi

    # Array reference: the same formula via xarray broadcasting + .where mask.
    with timed("xarray reference"):
        red = ds.red.astype("float64") * _SCALE + _OFFSET
        nir = ds.nir.astype("float64") * _SCALE + _OFFSET
        ref = ((nir - red) / (nir + red)).where((ds.red != 0) & (ds.nir != 0))

    # Align both to the same (y, x) grid before comparing.
    got = got.sortby(["y", "x"])
    ref = ref.sortby(["y", "x"])
    check_close("NDVI (per-pixel)", got, ref, rtol=1e-6, atol=1e-6)

    valid = np.isfinite(got.values)
    print(
        f"\n  NDVI over {valid.sum():,} valid pixels: "
        f"min {np.nanmin(got.values):.3f}, "
        f"mean {np.nanmean(got.values):.3f}, "
        f"max {np.nanmax(got.values):.3f}"
    )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "NDVI: per-pixel column algebra + CASE mask")
    )
