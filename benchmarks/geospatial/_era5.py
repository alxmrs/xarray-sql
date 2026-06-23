"""Shared ARCO-ERA5 access for the geospatial benchmarks.

[ARCO-ERA5](https://github.com/google-research/arco-era5) is the ECMWF ERA5
reanalysis re-published as analysis-ready, cloud-optimized Zarr on a public GCS
bucket: 273 variables, hourly, 0.25° global (721×1440), since 1940 — about 1.3
million timesteps. We open the *whole* archive (no time/space slicing on the
xarray side) and let SQL ``WHERE`` clauses prune it. That is the demo: the
table is the full reanalysis; the query reads only the window it asks for.

These cases require anonymous GCS access (``gcsfs``); they skip cleanly when
the network or bucket is unavailable.
"""

from __future__ import annotations

import xarray as xr

import xarray_sql as xql

from _harness import CaseSkipped

URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

#: Friendly names for ERA5's two dimension groups (surface vs. atmosphere).
TABLE_NAMES = {
    ("time", "latitude", "longitude"): "surface",
    ("time", "level", "latitude", "longitude"): "atmosphere",
}


def open_era5() -> xr.Dataset:
    """Open the full ARCO-ERA5 archive (lazy, dask off), or skip if unreachable."""
    try:
        import gcsfs  # noqa: F401 — required by the gs:// protocol

        return xr.open_zarr(URL, chunks=None, storage_options={"token": "anon"})
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"ARCO-ERA5 unavailable ({exc})") from exc


def register_era5(ctx: xql.XarrayContext, ds: xr.Dataset, *, chunks=None):
    """Register ERA5 as ``era5.surface`` / ``era5.atmosphere`` tables."""
    ctx.from_dataset(
        "era5", ds, chunks=chunks or {"time": 6}, table_names=TABLE_NAMES
    )
    return ctx


def load_window(
    ds: xr.Dataset,
    var: str,
    *,
    time,
    latitude=None,
    longitude=None,
) -> xr.Dataset:
    """Read a bounded ERA5 window for ``var`` into memory as a 1-variable Dataset.

    Multi-timestep cases (climatology, anomaly, forecast skill) read their window
    *once* into memory here, then run both the SQL and its array reference
    against the same in-memory data — keeping I/O bounded and the comparison
    apples-to-apples. ERA5 latitude is descending, so pass
    ``latitude=slice(north, south)``; longitude is 0–360°E ascending.
    """
    da = ds[var].sel(time=time)
    if latitude is not None:
        da = da.sel(latitude=latitude)
    if longitude is not None:
        da = da.sel(longitude=longitude)
    return da.to_dataset().load()
