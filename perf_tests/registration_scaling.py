#!/usr/bin/env python3
"""Registration time vs. native chunk count (issue #174).

`read_xarray_table` used to create one scan partition per native xarray chunk,
making registration O(num_chunks): ~25 us/partition, so a finely chunked store
(e.g. a GOES-16 variable with ~59M native chunks) took tens of minutes just to
register, if it finished at all.

Native chunks are now coalesced into at most `target_partitions` scan
partitions, so registration cost is bounded regardless of how finely the store
is chunked. This script registers a synthetic dataset at a range of native
chunk counts and prints registration time with coalescing on (bounded) vs. off
(`target_partitions=None`, the historical O(num_chunks) behavior).

No network or large memory needed: the dataset is tiny per-chunk; only the
*number* of chunks grows, which is exactly what drives registration cost.

Run: python perf_tests/registration_scaling.py
"""

import time

import numpy as np
import pandas as pd
import xarray as xr

import xarray_sql as xql
from xarray_sql.df import DEFAULT_TARGET_PARTITIONS


def make_dataset(n_time_chunks: int) -> xr.Dataset:
    """A (time, lat, lon) dataset with `n_time_chunks` native time chunks."""
    time_coord = pd.date_range("2000-01-01", periods=n_time_chunks, freq="h")
    lat = np.linspace(-90, 90, 4)
    lon = np.linspace(-180, 180, 4)
    data = np.zeros((n_time_chunks, 4, 4), dtype="float32")
    ds = xr.Dataset(
        {"v": (["time", "lat", "lon"], data)},
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )
    # One native chunk per time step -> n_time_chunks native partitions.
    return ds.chunk({"time": 1, "lat": 4, "lon": 4})


def time_registration(ds: xr.Dataset, target_partitions) -> float:
    t0 = time.perf_counter()
    xql.read_xarray_table(ds, target_partitions=target_partitions)
    return time.perf_counter() - t0


def main() -> None:
    print(
        f"{'native chunks':>14} {'coalesced (s)':>14} {'uncoalesced (s)':>16}"
    )
    print("-" * 48)
    for n in (1_000, 10_000, 100_000, 1_000_000):
        ds = make_dataset(n)
        coalesced = time_registration(ds, DEFAULT_TARGET_PARTITIONS)
        uncoalesced = time_registration(ds, None)
        print(f"{n:>14,} {coalesced:>14.3f} {uncoalesced:>16.3f}")

    print(
        "\nCoalesced registration stays flat (bounded by target_partitions); "
        "uncoalesced grows linearly with the native chunk count."
    )


if __name__ == "__main__":
    main()
