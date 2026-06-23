# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "pandas",
#   "gcsfs",
#   "zarr>=3",
# ]
# ///
"""Diurnal climatology — the "rechunk + grouped reduction" that is a GROUP BY.

A *climatology* is the average value for each time-of-cycle, computed
independently at every location: "what is the typical temperature here at 06:00
local cycle?" In the array paradigm (and in the coiled/benchmarks #1545
write-up) this is the canonical painful workload — load native Zarr chunks,
*rechunk* to put all of time in one chunk ("pencils"), run a grouped reduction
over the calendar, then rechunk back to "pancakes" for output.

The rechunking exists only to serve the array layout. The *operation* is::

    SELECT latitude, longitude, hour_of_day, AVG("2m_temperature")
    GROUP BY latitude, longitude, hour_of_day

Group by location and time-of-cycle, average the rest. Same answer as
``da.groupby("time.hour").mean()`` — here the **diurnal cycle** (mean
temperature by hour of day) over a region, at ERA5's 0.25° resolution.

Dataset: **ARCO-ERA5** 2m-temperature over a North-American box for a few days,
read once into memory (see ``_era5.load_window``).
"""

from __future__ import annotations

import xarray_sql as xql

from _era5 import load_window, open_era5
from _harness import check_close, run_case, show_sql, timed

# A few days over a CONUS-ish box (ERA5 latitude descends; lon is 0–360°E).
_TIME = slice("2020-06-01", "2020-06-03T23")
_LAT = slice(50.0, 25.0)
_LON = slice(235.0, 290.0)


def main() -> None:
    full = open_era5()
    with timed("read ERA5 window into memory"):
        ds = load_window(
            full, "2m_temperature", time=_TIME, latitude=_LAT, longitude=_LON
        )
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
    """
    show_sql(sql)

    with timed("SQL diurnal climatology"):
        got = ctx.sql(sql).to_pandas()

    # Array reference: the textbook groupby-over-the-cycle reduction.
    with timed("xarray reference"):
        clim = ds["2m_temperature"].groupby("time.hour").mean("time") - 273.15
        ref = clim.to_dataframe(name="clim_c").reset_index()

    merged = got.merge(
        ref, on=["latitude", "longitude", "hour"], validate="one_to_one"
    )
    assert len(merged) == len(got) == len(ref), (
        f"row-count mismatch: sql={len(got)} ref={len(ref)} merged={len(merged)}"
    )
    check_close(
        "diurnal climatology (°C)",
        merged["clim_c_x"],
        merged["clim_c_y"],
        rtol=1e-4,
        atol=1e-3,
    )

    print(
        f"\n  {len(got):,} climatology cells (latitude × longitude × hour-of-day)."
    )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Climatology: GROUP BY lat, lon, hour (ARCO-ERA5)")
    )
