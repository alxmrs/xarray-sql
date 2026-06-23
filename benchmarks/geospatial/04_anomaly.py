# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "gcsfs",
#   "zarr>=3",
# ]
# ///
"""Temperature anomaly — "broadcast-subtract the climatology" is a self-JOIN.

An *anomaly* is the departure of each observation from its climatological
normal: ``anomaly(t) = T(t) − climatology(hour-of-day(t))`` at each cell. The
array paradigm computes the climatology, then leans on xarray's grouped
broadcasting to line it back up with every timestep:
``ds.groupby("time.hour") - climatology``.

That broadcast — "attach each cell's normal back onto every matching timestep" —
is exactly a relational **JOIN** on the grouping key. So the anomaly is a
climatology CTE joined back to the raw observations::

    WITH clim AS (SELECT latitude, longitude, hour, AVG(T) ... GROUP BY ...)
    SELECT a.T - c.clim_t AS anomaly
    FROM era5 a JOIN clim c
      ON (a.latitude, a.longitude, hour(a.time)) = (c.latitude, c.longitude, c.hour)

Dataset: **ARCO-ERA5** 2m-temperature over a North-American box for a few days,
opened and sliced with a single fluent xarray chain (anomalies vs the diurnal
cycle of that window).
"""

from __future__ import annotations

import xarray as xr

import xarray_sql as xql

from _harness import CaseSkipped, assert_grid_close, run_case, show_sql, timed

_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_TIME = slice("2020-06-01", "2020-06-03T23")
_LAT = slice(50.0, 25.0)
_LON = slice(235.0, 290.0)


def main() -> None:
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
        f"  ERA5 2m_temperature window: {dict(ds.sizes)}  (anomaly vs diurnal normals)"
    )

    ctx = xql.XarrayContext()
    ctx.from_dataset("era5", ds, chunks={"time": 24})

    sql = """
        WITH clim AS (
            SELECT latitude, longitude,
                   date_part('hour', time) AS hour,
                   AVG("2m_temperature") AS clim_t
            FROM era5
            GROUP BY latitude, longitude, date_part('hour', time)
        )
        SELECT a.time, a.latitude, a.longitude,
               a."2m_temperature" - c.clim_t AS anomaly
        FROM era5 a
        JOIN clim c
          ON a.latitude = c.latitude
         AND a.longitude = c.longitude
         AND date_part('hour', a.time) = c.hour
        ORDER BY a.time, a.latitude DESC, a.longitude
    """
    show_sql(sql)

    # The anomaly is a gridded field; round-trip it to (time, lat, lon).
    with timed("SQL anomaly (climatology CTE self-join)"):
        got = ctx.sql(sql).to_dataset(dims=["time", "latitude", "longitude"])

    # Array reference: grouped broadcast-subtract, in pure xarray.
    with timed("xarray reference"):
        grouped = ds["2m_temperature"].groupby("time.hour")
        ref = grouped - grouped.mean("time")

    assert_grid_close(
        "anomaly (T − diurnal climatology)",
        got.anomaly,
        ref,
        rtol=1e-3,
        atol=1e-2,
    )

    print(f"\n  anomaly Dataset: {dict(got.sizes)}")


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Anomaly: climatology CTE self-JOIN (ARCO-ERA5)")
    )
