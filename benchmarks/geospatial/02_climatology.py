# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "gcsfs",
#   "zarr>=3",
# ]
# ///
"""Diurnal climatology — the "rechunk + grouped reduction" that is a GROUP BY.

A *climatology* is the average value for each time-of-cycle, computed
independently at every location: "what is the typical temperature here at
06:00?" In the array paradigm (and in the coiled/benchmarks #1545 write-up)
this is the canonical painful workload — load native Zarr chunks, *rechunk* to
put all of time in one chunk ("pencils"), run a grouped reduction over the
calendar, then rechunk back to "pancakes" for output.

The rechunking exists only to serve the array layout. The *operation* is::

    SELECT latitude, longitude, hour_of_day, AVG("2m_temperature")
    GROUP BY latitude, longitude, hour_of_day

Group by location and time-of-cycle, average the rest — the same answer as
``da.groupby("time.hour").mean()``. ERA5 is hourly, so grouping by hour of day
gives a clean 24-bin **diurnal cycle**, one sample per day in the window.

Dataset: **ARCO-ERA5** 2m-temperature over a North-American box for a few days,
opened and sliced with a single fluent xarray chain.
"""

from __future__ import annotations

import xarray as xr

import xarray_sql as xql

from _harness import CaseSkipped, assert_grid_close, run_case, show_sql, timed

_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# A few days over a CONUS-ish box (ERA5 latitude descends; lon is 0–360°E).
_TIME = slice("2020-06-01", "2020-06-03T23")
_LAT = slice(50.0, 25.0)
_LON = slice(235.0, 290.0)


def main() -> None:
    # Open the full ARCO-ERA5 archive and slice to the window in one chain:
    # pick the variable, select the box and days, and read it into memory.
    try:
        import gcsfs  # noqa: F401 — required by the gs:// protocol

        with timed("open + slice ERA5 window"):
            ds = (
                xr.open_zarr(
                    _URL, chunks=None, storage_options={"token": "anon"}
                )[["2m_temperature"]]
                .sel(time=_TIME, latitude=_LAT, longitude=_LON)
                .load()
            )
    except Exception as exc:  # noqa: BLE001 — any failure → skip, not crash
        raise CaseSkipped(f"ARCO-ERA5 unavailable ({exc})") from exc

    print(
        f"  ERA5 2m_temperature window: {dict(ds.sizes)}  "
        f"(diurnal climatology over {ds.sizes['latitude']}×{ds.sizes['longitude']} cells)"
    )

    ctx = xql.XarrayContext()
    ctx.from_dataset("era5", ds, chunks={"time": 24})

    sql = """
        SELECT latitude,
               longitude,
               date_part('hour', time) AS hour,
               AVG("2m_temperature") - 273.15 AS clim_c
        FROM era5
        GROUP BY latitude, longitude, date_part('hour', time)
        ORDER BY latitude DESC, longitude, hour
    """
    show_sql(sql)

    # A climatology is a gridded product, so round-trip the result back to an
    # xarray Dataset keyed by (latitude, longitude, hour) — how it is used.
    with timed("SQL diurnal climatology"):
        got = ctx.sql(sql).to_dataset(dims=["latitude", "longitude", "hour"])

    # Array reference: the textbook groupby-over-the-cycle reduction, in °C.
    with timed("xarray reference"):
        ref = ds["2m_temperature"].groupby("time.hour").mean("time") - 273.15

    assert_grid_close(
        "diurnal climatology (°C)", got.clim_c, ref, rtol=1e-4, atol=1e-2
    )

    print(f"\n  climatology Dataset: {dict(got.sizes)}")


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Climatology: GROUP BY lat, lon, hour (ARCO-ERA5)")
    )
