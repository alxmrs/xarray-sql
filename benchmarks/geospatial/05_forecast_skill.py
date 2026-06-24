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
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Forecast skill — scoring ML weather models against ERA5 is a JOIN + aggregate.

Scoring the **Pangu-Weather** and **GraphCast** machine-learning forecast models
against ERA5 ground truth is the headline workload of
[WeatherBench 2](https://weatherbench2.readthedocs.io/). A forecast is indexed by
*initialization time* and *lead time* (``prediction_timedelta``); the truth is
indexed by *valid time*. Evaluation aligns them by ``valid_time = init + lead``
and reduces the error to RMSE as a function of lead — the classic "error grows
with forecast horizon" curve.

That alignment is a relational **JOIN**, and ``valid_time = init + lead`` is just
timestamp + duration arithmetic the engine does natively::

    SELECT f.model, f.prediction_timedelta AS lead,
           SQRT(AVG(POWER(f.t - e.t, 2))) AS rmse
    FROM forecasts f
    JOIN era5 e
      ON  e.time = f.time + f.prediction_timedelta   -- valid_time = init + lead
      AND e.latitude  = f.latitude
      AND e.longitude = f.longitude
    GROUP BY f.model, f.prediction_timedelta

We stack the two models along a ``model`` dimension into a single forecast
table, so the query groups by a ``model`` *column* — no table name is formatted
into the SQL. Nothing is loaded up front either: the forecasts and ERA5 are
registered lazily, and the JOIN reads only what it needs at query time.

Datasets: WeatherBench 2 **Pangu**, **GraphCast**, and **ERA5** at a coarse
64×32 grid (so the demo is small and fast), read from the public ``gs://
weatherbench2`` bucket. Requires network; skips cleanly offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import xarray_sql as xql

from _harness import CaseSkipped, assert_grid_close, run_case, show_sql, timed

_SO = {"token": "anon"}
_GRID = "64x32_equiangular_conservative"
_ERA5 = f"gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-{_GRID}.zarr"
_PANGU = f"gs://weatherbench2/datasets/pangu/2018-2022_0012_{_GRID}.zarr"
_GRAPHCAST = (
    "gs://weatherbench2/datasets/graphcast/2020/"
    f"date_range_2019-11-16_2021-02-01_12_hours-{_GRID}.zarr"
)
_VAR = "2m_temperature"
_INIT = slice("2020-01-01", "2020-01-10")  # 20 init times (12-hourly)


def _open(url: str) -> xr.Dataset:
    try:
        import gcsfs  # noqa: F401

        # decode_timedelta=True: forecasts store prediction_timedelta as a
        # real duration (and it silences xarray's decode-timedelta warning).
        return xr.open_zarr(
            url, chunks=None, storage_options=_SO, decode_timedelta=True
        )
    except Exception as exc:  # noqa: BLE001
        raise CaseSkipped(f"WeatherBench2 unavailable ({exc})") from exc


def _reference_rmse(forecasts: xr.Dataset, truth: xr.Dataset) -> xr.DataArray:
    """xarray reference: per (model, lead), align truth at valid_time, take RMSE.

    The 64×32 windows are tiny, so the reference loads them and reduces in
    memory; the SQL side above stays lazy.
    """
    f = forecasts[_VAR].load()
    e = truth[_VAR].load()
    leads = f.prediction_timedelta.values
    per_lead = []
    for lead in leads:
        e_at_valid = e.sel(time=f.time.values + lead)  # (init, lat, lon)
        diff = f.sel(prediction_timedelta=lead) - e_at_valid.values
        per_lead.append(
            np.sqrt((diff**2).mean(["time", "latitude", "longitude"]))
        )
    return (
        xr.concat(per_lead, dim="lead")
        .assign_coords(lead=leads)
        .transpose("model", "lead")
    )


def main() -> None:
    # Open everything lazily — no .load() here.
    era5 = _open(_ERA5)

    # The two models store different pressure-level sets (Pangu 13, GraphCast
    # 37), so we keep the common surface field 2m_temperature and stack the
    # models along a `model` dimension into one forecast table. Snap the grid
    # onto ERA5's exact coordinates (same 64×32 grid) so the equality JOIN is
    # bit-safe across the Zarr stores.
    pangu = _open(_PANGU)[[_VAR]].sel(time=_INIT)
    graphcast = _open(_GRAPHCAST)[[_VAR]].sel(time=_INIT)
    forecasts = xr.concat([pangu, graphcast], dim="model").assign_coords(
        model=["pangu", "graphcast"],
        latitude=era5.latitude.values,
        longitude=era5.longitude.values,
    )

    # ERA5 truth must span every valid time (last init + longest lead); bound it
    # lazily so the JOIN does not scan the whole 1959–2023 record.
    valid_max = (
        pangu.time.values.max() + pangu.prediction_timedelta.values.max()
    )
    truth = era5[[_VAR]].sel(time=slice(_INIT.start, pd.Timestamp(valid_max)))

    print(
        f"  64×32 2m_temperature | init {_INIT.start}…{_INIT.stop} "
        f"({pangu.sizes['time']} inits × {pangu.sizes['prediction_timedelta']} "
        f"leads × 2 models)"
    )

    ctx = xql.XarrayContext()
    ctx.from_dataset("forecasts", forecasts, chunks={"time": 6})
    ctx.from_dataset("era5", truth, chunks={"time": 100})

    sql = """
        SELECT f.model,
               f.prediction_timedelta AS lead,
               SQRT(AVG(POWER(
                   CAST(f."2m_temperature" AS DOUBLE) - e."2m_temperature", 2
               ))) AS rmse
        FROM forecasts f
        JOIN era5 e
          ON  e.time = f.time + f.prediction_timedelta  -- valid = init + lead
          AND e.latitude  = f.latitude
          AND e.longitude = f.longitude
        GROUP BY f.model, f.prediction_timedelta
        ORDER BY f.model, lead
    """
    show_sql(sql)

    with timed("SQL RMSE by (model, lead) — lazy JOIN"):
        got = ctx.sql(sql).to_dataset(dims=["model", "lead"]).rmse

    with timed("xarray reference"):
        ref = _reference_rmse(forecasts, truth)

    assert_grid_close("RMSE(model, lead)", got, ref, rtol=1e-4, atol=1e-3)

    # Headline table: error growth with forecast horizon, both models.
    lead_days = got["lead"].values / np.timedelta64(1, "D")
    pangu_rmse = got.sel(model="pangu")
    graphcast_rmse = got.sel(model="graphcast")
    print("\n  2m-temperature RMSE (K) vs lead time — lower is better:")
    print(f"  {'lead (days)':>12} {'Pangu':>9} {'GraphCast':>11}")
    for i in range(0, len(lead_days), 4):
        print(
            f"  {lead_days[i]:>12.2f} "
            f"{float(pangu_rmse.isel(lead=i)):>9.3f} "
            f"{float(graphcast_rmse.isel(lead=i)):>11.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Forecast skill: Pangu vs GraphCast vs ERA5 (WB2)")
    )
