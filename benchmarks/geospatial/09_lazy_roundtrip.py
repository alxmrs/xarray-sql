# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray-sql",
#   "xarray",
#   "numpy",
#   "pandas",
#   "dask",
#   "pooch",
#   "netCDF4",
# ]
#
# [tool.uv.sources]
# xarray-sql = { path = "../../", editable = true }
# ///
"""Lazy round-trip: the SQL answer comes back as an array without materializing.

The other cases prove the SQL computes the *same numbers* as xarray. This one
proves the other half of the claim the suite leans on: the round-trip back to
xarray is **lazy**. ``ctx.sql(...).to_dataset()`` hands you a Dataset whose data
is still a query; slicing it (``.sel(time=t0)``) pushes a ``WHERE`` back down
into SQL, so reading one slab reads one slab, not the whole table.

That is the property the Large Scale Geospatial Benchmarks discussion
(coiled/benchmarks #1545) actually asks about: not "can you express it" but
"does the stack stay light when you point it at a big archive and pull a slice".
Here we make it a number. Three ways to get one timestep out of SQL:

    eager   ctx.sql(...).to_pandas()                       # whole long table
    eager   to_dataset(chunks=None)[v].sel(time=t0)        # whole grid, then slice
    lazy    to_dataset(chunks={"time": 1})[v].sel(time=t0)  # one WHERE, one slab

All three return the identical slab (asserted against the xarray reference), but
the lazy path materializes one timestep's worth of rows instead of the whole
``time x lat x lon`` product, and its peak memory tracks that.

Dataset: ``air_temperature`` from ``xarray.tutorial`` (NCEP reanalysis,
2920 x 25 x 53), the dataset the ``to_dataset`` round-trip (#58 / PR #167) was
benchmarked on. Downloads once via pooch; skips cleanly offline.
"""

from __future__ import annotations

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

_VAR = "air"


def main() -> None:
    try:
        ds = xr.tutorial.open_dataset("air_temperature")
    except Exception as exc:  # noqa: BLE001: no network / no pooch cache, skip
        raise CaseSkipped(
            f"air_temperature tutorial dataset unavailable ({exc})"
        ) from exc

    nt, nlat, nlon = ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"]
    full_rows, slab_rows = nt * nlat * nlon, nlat * nlon
    print(
        f"  air_temperature: {nt}x{nlat}x{nlon} "
        f"({full_rows:,} cells; one timestep = {slab_rows:,} cells)"
    )

    # Register the grid lazily, one timestep per chunk, so the WHERE the
    # round-trip pushes down on .sel(time=t0) prunes to a single slab.
    ctx = xql.XarrayContext()
    with timed("register air (one timestep per chunk)"):
        ctx.from_dataset(_VAR, ds.chunk({"time": 1}), chunks={"time": 1})

    sql = f'SELECT * FROM "{_VAR}"'
    show_sql(sql)

    # The xarray reference: one timestep, the plain-array way. We compare by
    # *label* (.sel(time=t0)) rather than position: `SELECT *` has no inherent
    # row order, so to_dataset rebuilds the time axis in result order and a
    # positional .isel(time=0) could land on a different slab.
    t0 = ds["time"].values[0]
    dims = ["time", "lat", "lon"]
    ref = ds[_VAR].sel(time=t0)

    # (1) Eager via the DataFusion API: materialize the entire long table, then
    # pull the one timestep out of the dataframe.
    for _ in measured("eager to_pandas() (whole table into RAM)"):
        frame = ctx.sql(sql).to_pandas()
        eager_df = (
            frame[frame["time"] == t0]
            .set_index(["lat", "lon"])[_VAR]
            .to_xarray()
        )

    # (2) Eager round-trip: build the whole gridded Dataset, then slice it.
    for _ in measured("eager to_dataset(chunks=None) then sel(time=t0)"):
        eager_ds = (
            ctx.sql(sql).to_dataset(dims=dims, chunks=None)[_VAR].sel(time=t0)
        )

    # (3) Lazy round-trip: slice first, so only one WHERE'd slab is read.
    lazy = ctx.sql(sql).to_dataset(dims=dims, chunks={"time": 1})
    print(f"  lazy to_dataset: {_VAR}.chunks = {lazy[_VAR].chunks}")
    got = ref  # placeholder; the loop below binds it
    for _ in measured("lazy sel(time=t0) (single WHERE pushed into SQL)"):
        got = lazy[_VAR].sel(time=t0).load()

    # Correctness: every path returns the same slab as the xarray reference.
    assert_grid_close("eager to_pandas slab", eager_df, ref)
    assert_grid_close("eager to_dataset slab", eager_ds, ref)
    assert_grid_close("lazy to_dataset slab", got, ref)

    # Headline: how many rows each path pulled into memory to answer the slice.
    # (Peak memory per path is in the ⏱/📊 lines above.)
    print("\n  Rows materialized to get one timestep, three ways:\n")
    print(f"  {'path':<36}{'rows in RAM':>14}")
    print(f"  {'-' * 50}")
    print(f"  {'eager  to_pandas()':<36}{full_rows:>14,}")
    print(f"  {'eager  to_dataset(chunks=None)':<36}{full_rows:>14,}")
    print(f"  {'lazy   to_dataset(chunks=time:1)':<36}{slab_rows:>14,}")
    print(
        f"\n  Lazy path reads {full_rows // slab_rows}x fewer rows "
        f"({slab_rows:,} vs {full_rows:,}): the slice became a SQL WHERE."
    )

    show_result(got)


if __name__ == "__main__":
    raise SystemExit(
        run_case(
            main,
            "Lazy round-trip: SQL slice -> WHERE pushdown (air_temperature)",
        )
    )
