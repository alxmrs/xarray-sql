#!/usr/bin/env python3
"""Surface and atmospheric temperatures on 2020-01-01, in SQL.

Three queries against the full ARCO-ERA5 archive, all filtered to the morning
of January 1, 2020 entirely in SQL:

  * **Surface point (local).** Average 2m-temperature over a small grid box
    covering the New York City area for the first six hours — a scalar.
  * **Atmospheric profile (global).** Average temperature per pressure level
    over the whole planet — a classic profile (surface ~1000 hPa warmest,
    tropopause coldest), returned as a DataFrame.
  * **Surface map (global).** Average 2m-temperature per grid point,
    reconstructed back into an ``xr.Dataset`` (a 721x1440 raster) via
    ``.to_dataset()``.

Nothing is sliced on the xarray side: ``xr.open_zarr`` opens the entire archive
with Dask turned off (``chunks=None``), and ``from_dataset`` partitions it with
``chunks={'time': 1}``. Every filter is expressed in SQL — the table provider
prunes time partitions for ``WHERE time …`` and pushes ``WHERE
latitude/longitude …`` down to the dimension columns — so each query reads only
the partitions it needs.

ARCO-ERA5's atmospheric variables are stored in native Zarr chunks of shape
``(1, 37, 721, 1440)`` (~150 MB/hour); ``chunks={'time': 1}`` aligns partitions
to that shape so they fetch from GCS concurrently. The global atmospheric query
scans ~230M rows after pruning.

The Zarr is read anonymously from the public GCS bucket — no auth required.
"""

import time

import xarray as xr

import xarray_sql as xql


URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def main() -> None:
    # Open the whole archive with Dask turned off. SQL WHERE clauses below
    # express every filter; nothing is pre-sliced on the xarray side.
    #
    # Heads up: the library pushes column projection down to Zarr, so SELECT
    # only fetches what you ask for — but `SELECT * FROM era5.surface` would
    # try to read every variable across the archive (terabytes from GCS).
    # Always SELECT specific columns.
    ds = xr.open_zarr(URL, chunks=None, storage_options={"token": "anon"})
    print(
        f"ARCO-ERA5 opened: {ds.sizes['time']:,} hourly time steps, "
        f"{len(ds.data_vars)} variables (no Dask, no pre-slicing)."
    )

    ctx = xql.XarrayContext()
    t0 = time.perf_counter()
    # `chunks` is required here: `ds` is not Dask-backed, so the partition grid
    # is given explicitly (one partition per hourly time step).
    ctx.from_dataset(
        "era5",
        ds,
        chunks={"time": 1},
        table_names={
            ("time", "latitude", "longitude"): "surface",
            ("time", "level", "latitude", "longitude"): "atmosphere",
        },
    )
    print(f"Registration: {time.perf_counter() - t0:.2f}s")
    ctx.sql("SELECT 1").to_pandas()  # warm the planner

    # 1. Surface point: average 2m-temperature over NYC. WHERE clauses on
    #    dimension columns prune partitions and push down to Zarr.
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

    # 2. Atmospheric profile: average temperature per pressure level, globally.
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

    # 3. Surface map: average 2m-temperature per grid point, reconstructed back
    #    into an xr.Dataset (a 721x1440 global raster). `template=ds` recovers
    #    coordinate metadata; the aggregation result is materialized eagerly.
    print(
        "\nAverage 2m-temperature per grid point, globally, "
        "2020-01-01 00:00-05:00 UTC -> xr.Dataset:"
    )
    t0 = time.perf_counter()
    temp_map = ctx.sql(
        """
        SELECT latitude, longitude, AVG("2m_temperature") - 273.15 AS avg_c
        FROM era5.surface
        WHERE time BETWEEN TIMESTAMP '2020-01-01'
                       AND TIMESTAMP '2020-01-01 05:00:00'
        GROUP BY latitude, longitude
        ORDER BY latitude DESC, longitude
        """
    ).to_dataset(dims=["latitude", "longitude"], template=ds)
    print(temp_map)
    print(f"  ({time.perf_counter() - t0:.2f}s, ~6M rows scanned)")


if __name__ == "__main__":
    main()
