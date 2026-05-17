#!/usr/bin/env python3
"""Surface and global-atmospheric temperatures on 2020-01-01, in SQL.

Two queries against ARCO-ERA5 on the morning of January 1, 2020:

  * **Surface (local).** Average 2m-temperature over a small grid covering the
    New York City area for the first six hours.
  * **Atmosphere (global).** Average temperature per pressure level, computed
    over the entire planet for the same six hours — a classic atmospheric
    temperature profile (surface around 1000 hPa is warmest, tropopause near
    100 hPa is coldest).

Both queries express their filters entirely in SQL: ``xr.open_zarr`` is given
a single calendar year and no spatial slicing. The library's table provider
prunes time partitions for ``WHERE time …`` filters, and pushes ``WHERE
latitude/longitude …`` down to dimension columns.

ARCO-ERA5's atmospheric variables are stored in native Zarr chunks of shape
``(1, 37, 721, 1440)`` — about 150 MB per hour. We align Dask chunks to that
shape with ``chunks={'time': 1}`` so chunks fetch from GCS concurrently. The
global atmospheric query scans ~230M rows after pruning.

The Zarr is read anonymously from the public GCS bucket — no auth required.
"""

import time

import xarray as xr

import xarray_sql as xql


URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def main() -> None:
    full = xr.open_zarr(URL, chunks=None, storage_options={"token": "anon"})

    # Open a full calendar year — all 273 variables. No spatial slicing on
    # the xarray side; SQL WHERE clauses below express the filters.
    #
    # Heads up: the library pushes column projection down to Zarr, so SELECT
    # only fetches what you ask for — but `SELECT * FROM era5.surface` would
    # try to read every variable across the year (terabytes from GCS).
    # Always SELECT specific columns.
    ds = full.sel(time="2020").chunk({"time": 1})
    print(
        "ARCO-ERA5 opened: year 2020, "
        f"{ds.sizes['time']:,} hourly time steps, "
        f"{len(ds.data_vars)} variables (no spatial pre-slicing)."
    )

    ctx = xql.XarrayContext()
    t0 = time.perf_counter()
    ctx.from_dataset(
        "era5",
        ds,
        table_names={
            ("time", "latitude", "longitude"): "surface",
            ("time", "level", "latitude", "longitude"): "atmosphere",
        },
    )
    print(f"Registration: {time.perf_counter() - t0:.2f}s")
    ctx.sql("SELECT 1").to_pandas()  # warm the planner

    print("\nAverage 2m-temperature over NYC, 2020-01-01 00:00-05:00 UTC (°C):")
    t0 = time.perf_counter()
    surface = ctx.sql(
        """
        SELECT AVG("2m_temperature") - 273.15 AS avg_c
        FROM era5.surface
        WHERE time BETWEEN TIMESTAMP '2020-01-01'
                       AND TIMESTAMP '2020-01-01 05:00:00'
          AND latitude  BETWEEN 39 AND 40
          AND longitude BETWEEN 286 AND 287  -- ERA5 uses 0-360 longitudes
        """
    ).to_pandas()
    print(surface)
    print(f"  ({time.perf_counter() - t0:.2f}s)")

    print(
        "\nAverage temperature per pressure level, globally, "
        "2020-01-01 00:00-05:00 UTC (°C):"
    )
    t0 = time.perf_counter()
    profile = ctx.sql(
        """
        SELECT level, AVG(temperature) - 273.15 AS avg_c
        FROM era5.atmosphere
        WHERE time BETWEEN TIMESTAMP '2020-01-01'
                       AND TIMESTAMP '2020-01-01 05:00:00'
        GROUP BY level
        ORDER BY level DESC  -- surface (1000 hPa) first, top of atmosphere last
        """
    ).to_pandas()
    print(profile.to_string(index=False))
    print(f"  ({time.perf_counter() - t0:.2f}s, ~230M rows scanned)")


if __name__ == "__main__":
    main()
