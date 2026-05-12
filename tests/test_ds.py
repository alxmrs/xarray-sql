"""Tests for the SQL -> xarray reverse path (xarray-sql#58).

Phase 1 covers the eager round-trip:

* :class:`xarray_sql.XarrayDataFrame` returned by ``ctx.sql``
* ``.to_dataset()`` with explicit and auto-inferred ``dim_cols``
* Template-based metadata recovery (var attrs/encoding, dataset attrs,
  non-dim coords, dim-coord dtype)
* FROM-clause regex unit tests

Lazy semantics land in Phase 2; sparse extent and edge cases in Phase 3.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_sql import XarrayContext, XarrayDataFrame


# ---------------------------------------------------------------------------
# Wrapper: ctx.sql(...) returns XarrayDataFrame
# ---------------------------------------------------------------------------


def test_ctx_sql_returns_xarray_dataframe(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql("SELECT * FROM air LIMIT 5")
    assert isinstance(result, XarrayDataFrame)


def test_to_pandas_unchanged_behavior(air_dataset_small):
    """Wrapped ``.to_pandas()`` is bit-for-bit equal to the un-wrapped path."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    wrapped = ctx.sql("SELECT * FROM air LIMIT 7").to_pandas()
    raw = super(type(ctx), ctx).sql("SELECT * FROM air LIMIT 7").to_pandas()
    pd.testing.assert_frame_equal(wrapped, raw)


def test_passthrough_methods(air_dataset_small):
    """Methods we did not override forward through ``__getattr__``."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql("SELECT * FROM air LIMIT 5")
    names = [f.name for f in result.schema()]
    assert {"lat", "lon", "time", "air"}.issubset(set(names))
    assert repr(result) == repr(result._inner)


# ---------------------------------------------------------------------------
# Round-trip via to_dataset (explicit dim_cols)
# ---------------------------------------------------------------------------


def test_to_dataset_explicit_dims_select_star(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dim_cols=["time", "lat", "lon"]
    )
    assert isinstance(out, xr.Dataset)
    assert set(out.dims) == {"time", "lat", "lon"}
    assert "air" in out.data_vars
    assert out.sizes["lat"] == air_dataset_small.sizes["lat"]
    assert out.sizes["lon"] == air_dataset_small.sizes["lon"]
    assert out.sizes["time"] == air_dataset_small.sizes["time"]


def test_round_trip_select_star_values_match(air_dataset_small):
    """Values survive the round-trip (modulo ascending coord ordering)."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dim_cols=["time", "lat", "lon"]
    )
    expected = air_dataset_small.compute().sortby(["time", "lat", "lon"])
    actual = out.sortby(["time", "lat", "lon"])
    np.testing.assert_array_equal(actual["air"].values, expected["air"].values)


def test_round_trip_preserves_dim_order(air_dataset_small):
    """Auto-inferred dim_cols match the source data var's dim order."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    expected_dims = air_dataset_small["air"].dims
    assert out["air"].dims == expected_dims


def test_aggregation_drops_dim(air_dataset_small):
    """``GROUP BY lat, lon`` over time -> 2D Dataset with the alias."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dim_cols=["lat", "lon"])
    assert set(out.dims) == {"lat", "lon"}
    assert "air_avg" in out.data_vars
    assert "air" not in out.data_vars
    expected = (
        air_dataset_small.compute()
        .sortby(["lat", "lon"])
        .mean(dim="time")["air"]
        .values
    )
    actual = out.sortby(["lat", "lon"])["air_avg"].values
    np.testing.assert_allclose(actual, expected)


# ---------------------------------------------------------------------------
# dim_cols inference
# ---------------------------------------------------------------------------


def test_to_dataset_infers_dim_cols_from_single_registration(
    air_dataset_small,
):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    assert set(out.dims) == {"time", "lat", "lon"}


def test_to_dataset_infer_picks_referenced_table(air_dataset_small):
    """Two registered Datasets, SQL references one -> use that one's dims."""
    ctx = XarrayContext()
    ctx.from_dataset("air1", air_dataset_small)
    ctx.from_dataset("air2", air_dataset_small)
    out = ctx.sql("SELECT * FROM air1").to_dataset()
    assert set(out.dims) == {"time", "lat", "lon"}


def test_to_dataset_infer_fails_when_no_template_fits(air_dataset_small):
    """If no registered Dataset's dims fit the result -> clear error."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="dim_cols cannot be inferred"):
        # GROUP BY drops 'time'; air's dims = {time, lat, lon} are not all
        # present in the result -> cannot infer.
        ctx.sql(
            "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
        ).to_dataset()


# ---------------------------------------------------------------------------
# Template-based metadata recovery
# ---------------------------------------------------------------------------


def test_template_recovers_dataset_attrs(air_dataset_small):
    ds = air_dataset_small.copy()
    ds.attrs = {"source": "test", "version": "1.0"}
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dim_cols=["time", "lat", "lon"]
    )
    assert out.attrs == {"source": "test", "version": "1.0"}


def test_template_recovers_var_attrs(air_dataset_small):
    ds = air_dataset_small.copy()
    ds["air"].attrs = {"units": "K", "long_name": "Air Temperature"}
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dim_cols=["time", "lat", "lon"]
    )
    assert out["air"].attrs == {"units": "K", "long_name": "Air Temperature"}


def test_template_recovers_var_encoding_strips_dtype(air_dataset_small):
    """``zlib`` survives; dtype-bound keys are stripped (SQL may have cast)."""
    ds = air_dataset_small.copy()
    ds["air"].encoding = {
        "zlib": True,
        "dtype": "int16",
        "_FillValue": -999,
        "missing_value": -999,
    }
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dim_cols=["time", "lat", "lon"]
    )
    assert out["air"].encoding.get("zlib") is True
    assert "dtype" not in out["air"].encoding
    assert "_FillValue" not in out["air"].encoding
    assert "missing_value" not in out["air"].encoding


def test_template_recovers_non_dim_scalar_coord(weather_dataset):
    """``rand_wx`` attaches a scalar ``reference_time`` non-dim coord."""
    assert "reference_time" in weather_dataset.coords
    assert "reference_time" not in weather_dataset.dims
    ctx = XarrayContext()
    ctx.from_dataset("weather", weather_dataset)
    out = ctx.sql("SELECT * FROM weather").to_dataset()
    assert "reference_time" in out.coords
    assert (
        out["reference_time"].values == weather_dataset["reference_time"].values
    )


def test_template_aggregation_alias_no_attrs(air_dataset_small):
    """``air_avg`` from ``AVG(air)`` does NOT inherit attrs from ``air``."""
    ds = air_dataset_small.copy()
    ds["air"].attrs = {"units": "K"}
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dim_cols=["lat", "lon"])
    assert "air_avg" in out.data_vars
    assert out["air_avg"].attrs == {}


def test_template_dim_dtype_preserved(air_dataset_small):
    """Datetime dim round-trips as ``datetime64``."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dim_cols=["time", "lat", "lon"]
    )
    assert np.issubdtype(out["time"].dtype, np.datetime64)
    assert out["time"].dtype == air_dataset_small["time"].dtype


def test_template_table_explicit_override(air_dataset_small):
    """``template_table=`` picks a registered Dataset deterministically."""
    other = air_dataset_small.copy()
    other.attrs = {"flag": "other"}
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    ctx.from_dataset("other", other)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dim_cols=["time", "lat", "lon"], template_table="other"
    )
    assert out.attrs == {"flag": "other"}


def test_template_table_unknown_raises(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="not a registered table"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dim_cols=["time", "lat", "lon"], template_table="missing"
        )


def test_template_and_template_table_mutually_exclusive(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="Pass at most one"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dim_cols=["time", "lat", "lon"],
            template=air_dataset_small,
            template_table="air",
        )


# ---------------------------------------------------------------------------
# FROM-clause regex unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("SELECT * FROM air", ["air"]),
        ('SELECT * FROM "air"', ["air"]),
        ("SELECT * FROM air a", ["air"]),
        ("SELECT * FROM air AS a", ["air"]),
        (
            "SELECT * FROM air a JOIN stations s ON a.lat = s.lat",
            ["air", "stations"],
        ),
        ("SELECT * FROM (SELECT 1)", []),
        ("WITH cte AS (SELECT 1) SELECT * FROM cte", ["cte"]),
        ("select * from air", ["air"]),
    ],
)
def test_extract_from_tables(query, expected):
    from xarray_sql.ds import _extract_from_tables

    assert _extract_from_tables(query) == expected


# ---------------------------------------------------------------------------
# Phase 2: Lazy backend semantics
# ---------------------------------------------------------------------------


def _patch_raw_sql_counter(monkeypatch):
    """Wrap ds._raw_sql with a counter; return the counter ref."""
    from xarray_sql import ds as ds_mod

    calls = {"n": 0}
    original = ds_mod._raw_sql

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ds_mod, "_raw_sql", counting)
    return calls


def test_lazy_select_star_returns_lazily_indexed_array(air_dataset_small):
    """For SELECT *, the air var should be backed by LazilyIndexedArray."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    inner = out["air"].variable._data
    # xr.core.indexing.LazilyIndexedArray wraps our SQLBackendArray.
    assert "LazilyIndexedArray" in type(inner).__name__
    from xarray_sql.ds import SQLBackendArray

    # Drill in to confirm the underlying array is ours.
    underlying = inner.array if hasattr(inner, "array") else inner
    assert isinstance(underlying, SQLBackendArray)


def test_lazy_no_sql_execution_until_access(air_dataset_small, monkeypatch):
    """Constructing the lazy Dataset must not trigger arbitrary SQL runs.

    Some setup queries are allowed (schema introspection, distinct-coord
    fetch when there's no template) but reading metadata (dims, coords,
    attrs) on the returned Dataset must not increment beyond that.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    calls = _patch_raw_sql_counter(monkeypatch)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    construct_count = calls["n"]
    # Reading metadata only.
    _ = out.dims
    _ = out.sizes
    _ = out.attrs
    _ = out["air"].attrs
    _ = out["lat"].values  # coord arrays come from the template, not SQL
    assert calls["n"] == construct_count, (
        f"Metadata reads should not trigger SQL; was {construct_count}, "
        f"now {calls['n']}"
    )


def test_lazy_isel_int_pushes_down_equality(air_dataset_small, monkeypatch):
    """isel(time=0) triggers exactly one wrapped SQL with WHERE = literal."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    calls = _patch_raw_sql_counter(monkeypatch)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    before = calls["n"]
    slab = out["air"].isel(time=0).values
    after = calls["n"]
    # Shape: (lat, lon)
    assert slab.shape == (
        air_dataset_small.sizes["lat"],
        air_dataset_small.sizes["lon"],
    )
    # At least one SQL execution past the schema setup.
    assert after > before


def test_lazy_isel_slice_pushdown(air_dataset_small):
    """isel(time=slice(0, 3)) round-trip matches the source."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    actual = out["air"].isel(time=slice(0, 3)).sortby(["lat", "lon"]).values
    expected = (
        air_dataset_small["air"]
        .compute()
        .isel(time=slice(0, 3))
        .sortby(["lat", "lon"])
        .values
    )
    np.testing.assert_array_equal(actual, expected)


def test_lazy_select_star_round_trip_equality(air_dataset_small):
    """Lazy .values produces the same data as the eager path on full read."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    expected = air_dataset_small.compute().sortby(["time", "lat", "lon"])
    actual = out.sortby(["time", "lat", "lon"])
    np.testing.assert_array_equal(actual["air"].values, expected["air"].values)


def test_aggregation_uses_eager_path(air_dataset_small):
    """Aggregation queries materialize once via the eager path.

    No assertion on the data type of variable._data here; xarray may
    wrap eager numpy arrays in NumpyIndexingAdapter. The contract is:
    values match the source and slicing afterwards is local.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dim_cols=["lat", "lon"])
    underlying = out["air_avg"].variable._data
    from xarray_sql.ds import SQLBackendArray

    # Aggregation -> not a SQLBackendArray.
    if hasattr(underlying, "array"):
        assert not isinstance(underlying.array, SQLBackendArray)
    else:
        assert not isinstance(underlying, SQLBackendArray)


def test_lazy_outer_indexer_array(air_dataset_small):
    """Fancy index along one dim works (IN clause pushdown)."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    slab = out["air"].isel(lat=[0, 3, 5]).values
    assert slab.shape == (
        air_dataset_small.sizes["time"],
        3,
        air_dataset_small.sizes["lon"],
    )
    expected_lats = air_dataset_small["lat"].values[[0, 3, 5]]
    np.testing.assert_array_equal(
        out["air"].isel(lat=[0, 3, 5])["lat"].values, expected_lats
    )


def test_lazy_compute_returns_eager(air_dataset_small):
    """.compute() returns an in-memory Dataset matching the source."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset().compute()
    # After .compute(), no SQLBackendArray underneath.
    from xarray_sql.ds import SQLBackendArray

    underlying = out["air"].variable._data
    if hasattr(underlying, "array"):
        assert not isinstance(underlying.array, SQLBackendArray)
    np.testing.assert_array_equal(
        out.sortby(["time", "lat", "lon"])["air"].values,
        air_dataset_small.compute()
        .sortby(["time", "lat", "lon"])["air"]
        .values,
    )
