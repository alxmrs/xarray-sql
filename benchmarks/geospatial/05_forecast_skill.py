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
"""Forecast skill — scoring ML weather models against ERA5 is a JOIN + aggregate.

This is the real thing: scoring the **Pangu-Weather** and **GraphCast** machine-
learning forecast models against ERA5 ground truth, the headline workload of
[WeatherBench 2](https://weatherbench2.readthedocs.io/). A forecast is indexed by
*initialization time* and *lead time* (``prediction_timedelta``); the truth is
indexed by *valid time*. Evaluation aligns them by ``valid_time = init + lead``
and reduces the error to RMSE as a function of lead — the classic "error grows
with forecast horizon" curve.

That alignment is a relational **JOIN**, and ``valid_time = init + lead`` is just
timestamp + duration arithmetic the engine does natively::

    SELECT f.prediction_timedelta AS lead,
           SQRT(AVG(POWER(f.t - e.t, 2))) AS rmse
    FROM forecast f
    JOIN era5 e
      ON  e.time = f.time + f.prediction_timedelta   -- valid_time = init + lead
      AND e.latitude  = f.latitude
      AND e.longitude = f.longitude
    GROUP BY f.prediction_timedelta

The whole evaluation — temporal alignment, spatial matching, the score — is one
JOIN and one aggregate. We run it for both models and check each against a NumPy
reference.

Datasets: WeatherBench 2 **Pangu**, **GraphCast**, and **ERA5** at a coarse
64×32 grid (so the demo is small and fast), read from the public ``gs://
weatherbench2`` bucket. Requires network; skips cleanly offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import xarray_sql as xql

from _harness import CaseSkipped, check_close, run_case, show_sql, timed

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


def _load_forecast(url: str, grid: xr.Dataset) -> xr.Dataset:
    """Load a model's 2m-temperature over the init window, all lead times.

    Snaps lat/lon onto ERA5's exact coordinate arrays (identical 64×32 grid) so
    the equality JOIN on coordinates is bit-safe across the two Zarr stores.
    """
    da = _open(url)[_VAR].sel(time=_INIT)
    da = da.assign_coords(
        latitude=grid.latitude.values, longitude=grid.longitude.values
    )
    return da.to_dataset().load()


def _rmse_sql(table: str) -> str:
    """The forecast↔truth JOIN + RMSE-per-lead query for one model ``table``."""
    return f"""
        SELECT f.prediction_timedelta AS lead,
               SQRT(AVG(POWER(
                   CAST(f."{_VAR}" AS DOUBLE) - e."{_VAR}", 2))) AS rmse
        FROM {table} f
        JOIN era5 e
          ON  e.time = f.time + f.prediction_timedelta  -- valid = init + lead
          AND e.latitude  = f.latitude
          AND e.longitude = f.longitude
        GROUP BY f.prediction_timedelta
        ORDER BY lead
    """


def _rmse_by_lead(ctx: xql.XarrayContext, table: str) -> pd.DataFrame:
    return ctx.sql(_rmse_sql(table)).to_pandas()


def _reference_rmse(fc: xr.Dataset, truth: xr.Dataset) -> np.ndarray:
    """NumPy reference: for each lead, align truth at valid_time and take RMSE."""
    f = fc[_VAR]
    e = truth[_VAR]
    out = []
    for lead in f.prediction_timedelta.values:
        valid = f.time.values + lead
        e_at_valid = e.sel(time=valid).values  # (init, lat, lon)
        diff = f.sel(prediction_timedelta=lead).values - e_at_valid
        out.append(float(np.sqrt(np.mean(diff**2))))
    return np.array(out)


def main() -> None:
    era5_full = _open(_ERA5)

    # Load the forecasts first; snap their grid to ERA5's exact coordinates.
    with timed("read Pangu + GraphCast forecasts"):
        pangu = _load_forecast(_PANGU, era5_full)
        graphcast = _load_forecast(_GRAPHCAST, era5_full)

    # ERA5 truth must span every valid time: window start through the last
    # init plus the longest lead, so the JOIN keeps every forecast pair.
    valid_max = (
        pangu.time.values.max() + pangu.prediction_timedelta.values.max()
    )
    with timed("read ERA5 truth window"):
        truth = (
            era5_full[[_VAR]]
            .sel(time=slice(_INIT.start, pd.Timestamp(valid_max)))
            .load()
        )

    print(
        f"  64×32 2m_temperature | init {_INIT.start}…{_INIT.stop} "
        f"({pangu.sizes['time']} inits × {pangu.sizes['prediction_timedelta']} leads)"
    )

    ctx = xql.XarrayContext()
    ctx.from_dataset("era5", truth, chunks={"time": 100})
    ctx.from_dataset("pangu", pangu, chunks={"time": 20})
    ctx.from_dataset("graphcast", graphcast, chunks={"time": 20})

    print("\n  Forecast↔ERA5 JOIN on valid_time = init + lead, RMSE per lead:")
    show_sql(_rmse_sql("pangu"))

    results = {}
    for name, fc in [("pangu", pangu), ("graphcast", graphcast)]:
        with timed(f"SQL RMSE-by-lead: {name}"):
            got = _rmse_by_lead(ctx, name)
        with timed(f"NumPy reference: {name}"):
            ref = _reference_rmse(fc, truth)
        check_close(
            f"{name} RMSE(lead)", got["rmse"], ref, rtol=1e-4, atol=1e-3
        )
        results[name] = got

    # Headline table: error growth with forecast horizon, both models.
    leads = results["pangu"]["lead"].dt.total_seconds() / 86400.0
    print("\n  2m-temperature RMSE (K) vs lead time — lower is better:")
    print(f"  {'lead (days)':>12} {'Pangu':>9} {'GraphCast':>11}")
    for i in range(0, len(leads), 4):
        print(
            f"  {leads.iloc[i]:>12.2f} "
            f"{results['pangu']['rmse'].iloc[i]:>9.3f} "
            f"{results['graphcast']['rmse'].iloc[i]:>11.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(
        run_case(main, "Forecast skill: Pangu vs GraphCast vs ERA5 (WB2)")
    )
