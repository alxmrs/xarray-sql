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
read once into memory (anomalies vs the diurnal cycle of that window).
"""

from __future__ import annotations

import xarray_sql as xql

from _era5 import load_window, open_era5
from _harness import check_close, run_case, show_sql, timed

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
        f"(anomaly vs diurnal normals)"
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
    """
    show_sql(sql)

    with timed("SQL anomaly (climatology CTE self-join)"):
        got = ctx.sql(sql).to_pandas()

    # Array reference: grouped broadcast-subtract.
    with timed("xarray reference"):
        clim = ds["2m_temperature"].groupby("time.hour").mean("time")
        anom = ds["2m_temperature"].groupby("time.hour") - clim
        ref = anom.to_dataframe(name="anomaly").reset_index()

    merged = got.merge(
        ref,
        on=["time", "latitude", "longitude"],
        suffixes=("_sql", "_ref"),
        validate="one_to_one",
    )
    assert len(merged) == len(got) == len(ref), (
        f"row-count mismatch: sql={len(got)} ref={len(ref)} merged={len(merged)}"
    )
    check_close(
        "anomaly (T − diurnal climatology)",
        merged["anomaly_sql"],
        merged["anomaly_ref"],
        rtol=1e-4,
        atol=1e-3,
    )

    print(f"\n  {len(got):,} hourly anomalies over the window.")


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Anomaly: climatology CTE self-JOIN (ARCO-ERA5)")
    )
