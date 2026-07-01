"""Tests for the native (non-FFI) execution engine.

The native engine registers tables into an in-process DataFusion
``SessionContext`` compiled into the extension, bypassing the FFI boundary
that drops table statistics. These tests check (a) result parity with the
default FFI engine and (b) that exact cardinalities now reach the optimizer,
which is the whole point of the native path.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_sql import XarrayContext


@pytest.fixture
def grid_ds():
    """A small, fully in-memory gridded dataset (no network)."""
    nt, nlat, nlon = 8, 6, 5
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"air": (("time", "lat", "lon"), rng.standard_normal((nt, nlat, nlon)))},
        coords={
            "time": np.arange(nt),
            "lat": np.linspace(10, 50, nlat),
            "lon": np.linspace(0, 30, nlon),
        },
    )


def test_engine_validation():
    with pytest.raises(ValueError, match="engine must be"):
        XarrayContext(engine="bogus")


def test_native_select_parity(grid_ds):
    ffi = XarrayContext()
    ffi.from_dataset("air", grid_ds, chunks={"time": 3})
    nat = XarrayContext(engine="native")
    nat.from_dataset("air", grid_ds, chunks={"time": 3})

    q = 'SELECT lat, lon, "air" FROM air'
    a = ffi.sql(q).to_pandas().sort_values(["lat", "lon", "air"]).reset_index(drop=True)
    b = nat.sql(q).to_pandas().sort_values(["lat", "lon", "air"]).reset_index(drop=True)
    assert a.shape == b.shape
    np.testing.assert_allclose(a["air"].to_numpy(), b["air"].to_numpy())


def test_native_groupby_parity(grid_ds):
    nat = XarrayContext(engine="native")
    nat.from_dataset("air", grid_ds, chunks={"time": 3})

    got = (
        nat.sql("SELECT lat, lon, AVG(air) AS a FROM air GROUP BY lat, lon")
        .to_pandas()
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )
    ref = (
        grid_ds["air"]
        .mean("time")
        .to_dataframe()
        .reset_index()
        .rename(columns={"air": "a"})
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )
    assert len(got) == grid_ds.sizes["lat"] * grid_ds.sizes["lon"]
    np.testing.assert_allclose(got["a"].to_numpy(), ref["a"].to_numpy())


def test_native_to_dataset_roundtrip(grid_ds):
    nat = XarrayContext(engine="native")
    nat.from_dataset("air", grid_ds, chunks={"time": 3})

    ds = nat.sql(
        "SELECT lat, lon, AVG(air) AS air FROM air GROUP BY lat, lon"
    ).to_dataset(dims=["lat", "lon"])
    assert isinstance(ds, xr.Dataset)
    assert set(ds.dims) == {"lat", "lon"}
    ref = grid_ds["air"].mean("time")
    # GROUP BY returns rows in hash order, so sort both by coordinate label
    # before comparing raw values.
    got = ds["air"].sortby(["lat", "lon"]).transpose("lat", "lon")
    ref = ref.sortby(["lat", "lon"]).transpose("lat", "lon")
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy())


def test_native_lazy_chunked_roundtrip(grid_ds):
    """to_dataset(chunks=...) is lazy: dask-backed arrays, correct on compute."""
    dask = pytest.importorskip("dask")  # noqa: F841
    nat = XarrayContext(engine="native")
    nat.from_dataset("air", grid_ds, chunks={"time": 3})

    out = nat.sql("SELECT time, lat, lon, air FROM air").to_dataset(
        dims=["time", "lat", "lon"], chunks={"time": 3}
    )
    # The result variable is lazy (a chunked dask array), not a dense ndarray.
    assert out["air"].chunks is not None
    assert type(out["air"].data).__module__.startswith("dask")

    # Computing a slice reads lazily and matches the reference.
    got = out["air"].sel(time=slice(3, 5)).compute().transpose("time", "lat", "lon")
    ref = grid_ds["air"].sel(time=slice(3, 5)).transpose("time", "lat", "lon")
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy())

    # Full materialisation also matches.
    full = out["air"].compute().transpose("time", "lat", "lon")
    np.testing.assert_allclose(
        full.to_numpy(), grid_ds["air"].transpose("time", "lat", "lon").to_numpy()
    )


def test_native_sql_returns_lazy_frame():
    """NativeContext.sql plans lazily and streams — it does not collect."""
    from xarray_sql._native import NativeContext
    from xarray_sql.reader import read_xarray_table

    ds = xr.Dataset(
        {"v": (("x",), np.arange(10, dtype="float64"))},
        coords={"x": np.arange(10)},
    )
    nc = NativeContext()
    nc.register_table("t", read_xarray_table(ds, chunks={"x": 5}))
    frame = nc.sql("SELECT x, v FROM t")
    # A lazy frame exposes schema without executing, and streams batches.
    assert [f.name for f in frame.schema()] == ["x", "v"]
    total = sum(b.num_rows for b in frame.execute_stream())
    assert total == 10


def test_native_statistics_in_plan(grid_ds):
    """Exact row counts must appear at the scan and propagate upward."""
    nat = XarrayContext(engine="native")
    nat.from_dataset("air", grid_ds, chunks={"time": 3})
    plan = nat._explain_native("SELECT lat, lon, AVG(air) FROM air GROUP BY lat, lon")
    total = grid_ds.sizes["time"] * grid_ds.sizes["lat"] * grid_ds.sizes["lon"]
    assert f"rows=Exact({total})" in plan
    # The exact count is not dropped to Absent above the scan.
    assert "Rows=Absent" not in plan.splitlines()[-1]


def test_native_column_minmax_in_plan(grid_ds):
    """Numeric dimension columns get exact min/max coordinate bounds."""
    nat = XarrayContext(engine="native")
    nat.from_dataset("air", grid_ds, chunks={"time": 3})
    plan = nat._explain_native("SELECT lat, lon, air FROM air")
    scan = next(l for l in plan.splitlines() if "XarrayScanExec" in l)

    def fmt(v: float) -> str:
        # DataFusion's ScalarValue display drops a trailing ".0".
        return str(int(v)) if v == int(v) else str(v)

    lat_min, lat_max = float(grid_ds.lat.min()), float(grid_ds.lat.max())
    lon_min, lon_max = float(grid_ds.lon.min()), float(grid_ds.lon.max())
    assert f"Min=Exact(Float64({fmt(lat_min)}))" in scan
    assert f"Max=Exact(Float64({fmt(lat_max)}))" in scan
    assert f"Min=Exact(Float64({fmt(lon_min)}))" in scan
    assert f"Max=Exact(Float64({fmt(lon_max)}))" in scan


def test_native_join_picks_small_build_side():
    """With exact statistics the optimizer broadcasts the smaller table.

    A big (time x lat x lon) table joined to a small (lat x lon) weight table
    should plan as a CollectLeft hash join with the small table on the build
    side. Without statistics (the FFI path) the optimizer cannot know which
    side is smaller and falls back to a Partitioned join.
    """
    rng = np.random.default_rng(0)
    big = xr.Dataset(
        {"t": (("time", "lat", "lon"), rng.standard_normal((200, 8, 8)))},
        coords={"time": np.arange(200), "lat": np.arange(8), "lon": np.arange(8)},
    )
    small = xr.Dataset(
        {"w": (("lat", "lon"), rng.standard_normal((8, 8)))},
        coords={"lat": np.arange(8), "lon": np.arange(8)},
    )
    nat = XarrayContext(engine="native")
    nat.from_dataset("big", big, chunks={"time": 50})
    nat.from_dataset("small", small, chunks={"lat": 8})

    plan = nat._explain_native(
        "SELECT b.time, SUM(b.t * s.w) AS x FROM big b "
        "JOIN small s ON b.lat=s.lat AND b.lon=s.lon GROUP BY b.time"
    )
    assert "HashJoinExec: mode=CollectLeft" in plan


def test_native_multigroup_not_supported(weather_dataset):
    """Datasets that split into a namespace are FFI-only for now."""
    # Build a two-group dataset: a 3-D var and a 2-D var.
    ds = weather_dataset
    two_group = ds.assign(
        surface_pressure=(("lat", "lon"), ds["temperature"].isel(time=0, level=0).data)
    )
    nat = XarrayContext(engine="native")
    with pytest.raises(NotImplementedError, match="native engine"):
        nat.from_dataset("wx", two_group, chunks={"time": 3})
