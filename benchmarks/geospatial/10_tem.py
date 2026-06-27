# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "gcsfs",
#   "zarr>=3",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Transformed Eulerian Mean: zonal means and eddy fluxes are a GROUP BY.

The Transformed Eulerian Mean (TEM) is a standard atmospheric-circulation
diagnostic: average the flow around each latitude circle, then measure how the
departures from that average (the eddies) carry momentum and heat. In the array
paradigm it is ``ds.mean("longitude")`` plus a few ``(x - x_bar)`` products
averaged again over longitude.

Every piece of that is relational. A zonal mean is ``GROUP BY latitude``
collapsing longitude. An eddy flux such as the momentum flux
``u'v' = mean((u - u_bar)(v - v_bar))`` is, by the covariance identity, just
``AVG(u*v) - AVG(u)*AVG(v)``: one grouped pass, no self-join. So the whole
diagnostic is::

    SELECT time, level, latitude,
           AVG(u) AS u_bar, ...,
           AVG(u*v) - AVG(u)*AVG(v) AS upvp,   -- eddy momentum flux u'v'
           AVG(v*t) - AVG(v)*AVG(t) AS vptp,   -- eddy heat flux v't'
           AVG(u*w) - AVG(u)*AVG(w) AS upwp    -- vertical momentum flux u'w'
    FROM era5 GROUP BY time, level, latitude

This is the diagnostic dcherian raised in the Large Scale Geospatial Benchmarks
discussion (coiled/benchmarks #1545); the SQL reads like its textbook definition.

Dataset: the full ARCO-ERA5 archive (0.25 degree, 37 pressure levels), opened
lazily, so the query reads only u, v, T, w on the requested levels and timestep.
Validated against the same diagnostic computed in pure xarray.
"""

from __future__ import annotations

import datetime

import xarray as xr

import xarray_sql as xql

from _harness import (
    CaseSkipped,
    assert_grid_close,
    measured,
    run_case,
    show_result,
    show_sql,
    timed,
)

_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
_T = datetime.datetime(2020, 6, 1, 12)
# Three representative pressure levels (hPa): upper jet, mid, lower troposphere.
_LEVELS = (250, 500, 850)
_VARS = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "vertical_velocity",
]
# Timestep and levels are bound as query parameters, not formatted into the SQL.
_PARAMS = {"t": _T, "l1": _LEVELS[0], "l2": _LEVELS[1], "l3": _LEVELS[2]}


def main() -> None:
    try:
        import gcsfs  # noqa: F401  (required by the gs:// protocol)

        ds = xr.open_zarr(_URL, chunks=None, storage_options={"token": "anon"})
    except Exception as exc:  # noqa: BLE001  (any failure skips, not crash)
        raise CaseSkipped(f"ARCO-ERA5 unavailable ({exc})") from exc

    print(
        f"  ARCO-ERA5: {ds.sizes['time']:,} timesteps x {ds.sizes['level']} levels "
        f"x {ds.sizes['latitude']}x{ds.sizes['longitude']} (no pre-slicing)"
    )

    ctx = xql.XarrayContext()
    with timed("register full ERA5 (lazy)"):
        ctx.from_dataset(
            "era5",
            ds,
            chunks={"time": 1},
            table_names={
                ("time", "latitude", "longitude"): "surface",
                ("time", "level", "latitude", "longitude"): "atmosphere",
            },
        )

    sql = """
        WITH f AS (
            SELECT time, level, latitude,
                   "u_component_of_wind" AS u,
                   "v_component_of_wind" AS v,
                   "temperature"         AS t,
                   "vertical_velocity"   AS w
            FROM era5.atmosphere
            WHERE time = $t
              AND level IN ($l1, $l2, $l3)
        )
        SELECT time, level, latitude,
               AVG(u) AS u_bar,
               AVG(v) AS v_bar,
               AVG(t) AS t_bar,
               AVG(w) AS w_bar,
               AVG(u * v) - AVG(u) * AVG(v) AS upvp,
               AVG(v * t) - AVG(v) * AVG(t) AS vptp,
               AVG(u * w) - AVG(u) * AVG(w) AS upwp
        FROM f
        GROUP BY time, level, latitude
        ORDER BY time, level, latitude
    """
    show_sql(sql)

    # Round-trip the diagnostic to a (time, level, latitude) Dataset.
    for _ in measured("SQL TEM (zonal means + eddy covariances, lazy read)"):
        got = ctx.sql(sql, param_values=_PARAMS).to_dataset(
            dims=["time", "level", "latitude"]
        )

    # Array reference: the same TEM diagnostic in pure xarray.
    for _ in measured("xarray reference"):
        sub = ds[_VARS].sel(time=[_T], level=list(_LEVELS))
        u, v, t, w = (sub[n] for n in _VARS)

        def zm(x: xr.DataArray) -> xr.DataArray:
            return x.mean("longitude")

        u_bar, v_bar, t_bar, w_bar = zm(u), zm(v), zm(t), zm(w)
        ref_upvp = zm((u - u_bar) * (v - v_bar))
        ref_vptp = zm((v - v_bar) * (t - t_bar))
        ref_upwp = zm((u - u_bar) * (w - w_bar))

    # Tolerance covers the SQL one-pass covariance (AVG(u*v) - AVG(u)*AVG(v))
    # against the two-pass xarray reference on float32 ERA5 fields.
    assert_grid_close(
        "zonal-mean u (u_bar)", got.u_bar, u_bar, rtol=1e-3, atol=1e-2
    )
    assert_grid_close(
        "eddy momentum flux u'v'", got.upvp, ref_upvp, rtol=1e-3, atol=1e-2
    )
    assert_grid_close(
        "eddy heat flux v't'", got.vptp, ref_vptp, rtol=1e-3, atol=1e-2
    )
    assert_grid_close(
        "vertical flux u'w'", got.upwp, ref_upwp, rtol=1e-3, atol=1e-2
    )

    show_result(got)


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "TEM: zonal means + eddy fluxes as GROUP BY (ARCO-ERA5)")
    )
