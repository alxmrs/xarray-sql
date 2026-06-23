# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "gcsfs",
#   "zarr>=3",
# ]
# ///
"""Zonal mean — the array reduction that is secretly a GROUP BY.

A *zonal mean* averages a field around each circle of latitude (over all
longitudes, and here over a day of hours too), collapsing a 3-D field to a 1-D
profile of value-vs-latitude — the classic pole-to-pole temperature curve. In
the array paradigm this is ``da.mean(dim=["longitude", "time"])``, a reduction
over two axes.

Relationally it is nothing more than::

    SELECT latitude, AVG("2m_temperature") GROUP BY latitude

The "axes" we reduce over are just the columns we *don't* group by. Same answer,
and the SQL reads like the plain-English definition of a zonal mean.

Dataset: the full **ARCO-ERA5** archive (0.25° global, 1.3M hourly timesteps).
The table is the whole reanalysis; ``WHERE time …`` prunes it to one day, and
the GROUP BY produces a 721-point global temperature profile.
"""

from __future__ import annotations


import xarray_sql as xql

from _era5 import open_era5, register_era5
from _harness import check_close, run_case, show_sql, timed

# One day of hourly data, global; the WHERE below prunes ERA5 to this window.
_DAY = "2020-06-01"


def main() -> None:
    ds = open_era5()
    print(
        f"  ARCO-ERA5: {ds.sizes['time']:,} hourly timesteps, "
        f"{ds.sizes['latitude']}×{ds.sizes['longitude']} grid, "
        f"{len(ds.data_vars)} variables (no pre-slicing)"
    )

    ctx = xql.XarrayContext()
    with timed("register full ERA5"):
        register_era5(ctx, ds)

    sql = f"""
        SELECT latitude,
               AVG("2m_temperature") - 273.15 AS air_mean_c
        FROM era5.surface
        WHERE time BETWEEN TIMESTAMP '{_DAY}'
                       AND TIMESTAMP '{_DAY} 23:00:00'
        GROUP BY latitude
        ORDER BY latitude DESC
    """
    show_sql(sql)

    with timed("SQL zonal mean (WHERE-pruned to one day)"):
        got = ctx.sql(sql).to_pandas()

    # Array reference: reduce the same day over the two un-grouped axes.
    with timed("xarray reference"):
        window = ds["2m_temperature"].sel(time=_DAY)
        ref = (window.mean(dim=["longitude", "time"]) - 273.15).sortby(
            "latitude", ascending=False
        )

    check_close(
        "zonal mean (2m_temp vs latitude, °C)",
        got["air_mean_c"],
        ref,
        rtol=1e-4,
        atol=1e-3,
    )

    print("\n  Global temperature profile (every 10th parallel):")
    print(got.iloc[::72].to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Zonal mean: GROUP BY latitude (ARCO-ERA5)")
    )
