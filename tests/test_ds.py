"""Tests for the SQL -> xarray reverse path.

Covers:

* :class:`xarray_sql.XarrayDataFrame` returned by ``ctx.sql`` -- wrapper
  behavior and DataFusion-method passthrough.
* ``.to_dataset()`` round-trips: lazy default, explicit aggregation
  cases, ``dimension_columns`` auto-inference vs explicit.
* Template-based metadata recovery (var attrs / encoding, dataset
  attrs, non-dim coords, dim-coord dtype).
* Sparsity handling and edge cases (null dim rows, fill_value
  dtype behavior, vectorized indexer fallback).
"""

from typing import Any

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
    from datafusion import SessionContext

    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    wrapped = ctx.sql("SELECT * FROM air LIMIT 7").to_pandas()
    raw = SessionContext.sql(ctx, "SELECT * FROM air LIMIT 7").to_pandas()
    pd.testing.assert_frame_equal(wrapped, raw)


def test_passthrough_methods(air_dataset_small):
    """DataFusion methods we did not override forward via ``__getattr__``."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql("SELECT * FROM air LIMIT 5")
    names = [f.name for f in result.schema()]
    assert {"lat", "lon", "time", "air"}.issubset(set(names))


# ---------------------------------------------------------------------------
# Round-trip via to_dataset (explicit dimension_columns)
# ---------------------------------------------------------------------------


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Strip ``encoding`` from a Dataset and all its variables.

    Round-trip identity tests should not be coupled to encoding choices,
    since ``apply_template`` deliberately drops dtype-bound keys.
    """
    ds = ds.copy()
    for v in ds.variables.values():
        v.encoding.clear()
    ds.encoding.clear()
    return ds


@pytest.mark.parametrize(
    "fixture_name",
    ["air_dataset_small", "weather_dataset", "synthetic_dataset"],
)
def test_round_trip_identity(request, fixture_name):
    """``SELECT *`` round-trips to a Dataset that is ``assert_identical``
    to the source: values, dims, coord values, dtypes, non-dim coords,
    and attrs all match (modulo coord ordering, which we normalize on
    both sides). One test covers what was previously eight narrow checks.
    """
    source = request.getfixturevalue(fixture_name).copy()
    source.attrs["round_trip_marker"] = "yes"
    first_var = next(iter(source.data_vars))
    source[first_var].attrs["units"] = "test_units"

    ctx = XarrayContext()
    ctx.from_dataset("t", source)
    out = ctx.sql("SELECT * FROM t").to_dataset().compute()

    sort_keys = list(out.dims)
    actual = _clear_encoding(out.sortby(sort_keys))
    expected = _clear_encoding(source.compute().sortby(sort_keys))
    xr.testing.assert_identical(actual, expected)


def test_aggregation_drops_dim(air_dataset_small):
    """``GROUP BY lat, lon`` over time -> 2D Dataset with the alias."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dimension_columns=["lat", "lon"])
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
# dimension_columns inference
# ---------------------------------------------------------------------------


def test_to_dataset_multi_registered_requires_explicit_template(
    air_dataset_small,
):
    """With more than one registered Dataset, no SQL parsing means the
    caller must disambiguate via ``template_table=``."""
    ctx = XarrayContext()
    ctx.from_dataset("air1", air_dataset_small)
    ctx.from_dataset("air2", air_dataset_small)
    out = ctx.sql("SELECT * FROM air1").to_dataset(template_table="air1")
    assert set(out.dims) == {"time", "lat", "lon"}


def test_to_dataset_infer_fails_when_no_template_fits(air_dataset_small):
    """If no registered Dataset's dims fit the result -> clear error."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(
        ValueError, match="dimension_columns cannot be inferred"
    ):
        # GROUP BY drops 'time'; air's dims = {time, lat, lon} are not all
        # present in the result -> cannot infer.
        ctx.sql(
            "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
        ).to_dataset()


# ---------------------------------------------------------------------------
# Template-based metadata recovery
# ---------------------------------------------------------------------------


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
        dimension_columns=["time", "lat", "lon"]
    )
    assert out["air"].encoding.get("zlib") is True
    assert "dtype" not in out["air"].encoding
    assert "_FillValue" not in out["air"].encoding
    assert "missing_value" not in out["air"].encoding


def test_template_aggregation_alias_no_attrs(air_dataset_small):
    """``air_avg`` from ``AVG(air)`` does NOT inherit attrs from ``air``."""
    ds = air_dataset_small.copy()
    ds["air"].attrs = {"units": "K"}
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dimension_columns=["lat", "lon"])
    assert "air_avg" in out.data_vars
    assert out["air_avg"].attrs == {}


def test_template_table_explicit_override(air_dataset_small):
    """``template_table=`` picks a registered Dataset deterministically."""
    other = air_dataset_small.copy()
    other.attrs = {"flag": "other"}
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    ctx.from_dataset("other", other)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dimension_columns=["time", "lat", "lon"], template_table="other"
    )
    assert out.attrs == {"flag": "other"}


def test_template_table_unknown_raises(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="not a registered table"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dimension_columns=["time", "lat", "lon"], template_table="missing"
        )


def test_template_and_template_table_mutually_exclusive(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="Pass at most one"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dimension_columns=["time", "lat", "lon"],
            template=air_dataset_small,
            template_table="air",
        )


# ---------------------------------------------------------------------------
# Lazy backend semantics
# ---------------------------------------------------------------------------


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


def test_lazy_no_slab_access_until_data_read(air_dataset_small, monkeypatch):
    """Reading metadata (``dims``, ``coords``, ``attrs``) on the lazy
    Dataset must not trigger ``SQLBackendArray._raw_getitem`` (the slab
    materialization entry point)."""
    from xarray_sql import ds as ds_mod

    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    calls = {"n": 0}
    original = ds_mod.SQLBackendArray._raw_getitem

    def counting(self, key):
        calls["n"] += 1
        return original(self, key)

    monkeypatch.setattr(ds_mod.SQLBackendArray, "_raw_getitem", counting)
    _ = out.dims
    _ = out.sizes
    _ = out.attrs
    _ = out["air"].attrs
    _ = out["lat"].values  # coord arrays already in memory from construction
    assert calls["n"] == 0, (
        f"Metadata reads should not trigger slab access; got {calls['n']}"
    )


def test_lazy_isel_int_pushes_down_equality(air_dataset_small, monkeypatch):
    """``isel(time=0)`` triggers the lazy backend (one ``_raw_getitem`` call)."""
    from xarray_sql import ds as ds_mod

    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    calls = {"n": 0}
    original = ds_mod.SQLBackendArray._raw_getitem

    def counting(self, key):
        calls["n"] += 1
        return original(self, key)

    monkeypatch.setattr(ds_mod.SQLBackendArray, "_raw_getitem", counting)
    slab = out["air"].isel(time=0).values
    assert slab.shape == (
        air_dataset_small.sizes["lat"],
        air_dataset_small.sizes["lon"],
    )
    assert calls["n"] >= 1


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


def test_aggregation_uses_lazy_backend(air_dataset_small):
    """Aggregation queries return a lazy Dataset just like SELECT *.

    Pushdown and laziness are orthogonal: an aggregation can't push the
    request indexer into a useful filter, but the result is still
    streamed via execute_stream on first access. The user-visible
    contract is values match the source's ``mean(dim="time")``.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dimension_columns=["lat", "lon"])
    from xarray_sql.ds import SQLBackendArray

    inner = out["air_avg"].variable._data
    underlying = inner.array if hasattr(inner, "array") else inner
    assert isinstance(underlying, SQLBackendArray)
    expected = (
        air_dataset_small.compute()
        .sortby(["lat", "lon"])
        .mean(dim="time")["air"]
        .values
    )
    actual = out.sortby(["lat", "lon"])["air_avg"].values
    np.testing.assert_allclose(actual, expected)


def test_lazy_outer_indexer_array(air_dataset_small):
    """Fancy index along one dim works (IN clause pushdown).

    Compare lazy output against the eager-computed equivalent rather than
    the source Dataset directly, because the lazy path derives coord
    arrays from SELECT DISTINCT (always ascending) regardless of the
    source's coord ordering.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    lazy = ctx.sql("SELECT * FROM air").to_dataset()
    eager = ctx.sql("SELECT * FROM air").to_dataset().compute()
    indices = [0, 3, 5]
    lazy_slab = lazy["air"].isel(lat=indices)
    eager_slab = eager["air"].isel(lat=indices)
    np.testing.assert_array_equal(
        lazy_slab["lat"].values, eager_slab["lat"].values
    )
    np.testing.assert_array_equal(lazy_slab.values, eager_slab.values)


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


# ---------------------------------------------------------------------------
# Sparsity handling and edge cases
# ---------------------------------------------------------------------------


def test_sparsity_result_default_filters_lazy(air_dataset_small):
    """Default sparsity='result' keeps only filtered coords (lazy path)."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    threshold = float(air_dataset_small["lat"].values[5])
    out = ctx.sql(f"SELECT * FROM air WHERE lat > {threshold}").to_dataset()
    assert (out["lat"].values > threshold).all()
    assert out.sizes["lat"] < air_dataset_small.sizes["lat"]


def test_sparsity_template_full_grid(air_dataset_small):
    """sparsity='template' reindexes to the full grid with NaN fills."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    threshold = float(air_dataset_small["lat"].values[5])
    out = ctx.sql(f"SELECT * FROM air WHERE lat > {threshold}").to_dataset(
        sparsity="template"
    )
    assert out.sizes["lat"] == air_dataset_small.sizes["lat"]
    lat_vals = out["lat"].values
    below_mask = lat_vals <= threshold
    above_mask = lat_vals > threshold
    below = out["air"].isel(lat=np.where(below_mask)[0])
    above = out["air"].isel(lat=np.where(above_mask)[0])
    assert np.isnan(below.values).all()
    assert not np.isnan(above.values).any()


def test_sparsity_template_requires_template(air_dataset_small):
    """No resolvable template -> sparsity='template' raises."""
    other = air_dataset_small.copy()
    ctx = XarrayContext()
    # Two registrations so auto-resolve returns None.
    ctx.from_dataset("a", air_dataset_small)
    ctx.from_dataset("b", other)
    with pytest.raises(ValueError, match="requires template= to be supplied"):
        ctx.sql("SELECT * FROM a").to_dataset(
            dimension_columns=["time", "lat", "lon"],
            sparsity="template",
        )


def test_sparsity_invalid_value_raises(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="sparsity must be"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dimension_columns=["time", "lat", "lon"],
            sparsity="bogus",  # type: ignore[arg-type]
        )


def test_sparsity_template_with_aggregation(air_dataset_small):
    """sparsity='template' on an aggregation respects dimension_columns subset."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    threshold = float(air_dataset_small["lat"].values[5])
    out = ctx.sql(
        f"""
        SELECT lat, lon, AVG(air) AS air_avg
        FROM air
        WHERE lat > {threshold}
        GROUP BY lat, lon
        """
    ).to_dataset(dimension_columns=["lat", "lon"], sparsity="template")
    assert out.sizes["lat"] == air_dataset_small.sizes["lat"]
    assert "time" not in out.dims
    below_mask = out["lat"].values <= threshold
    below = out["air_avg"].isel(lat=np.where(below_mask)[0])
    assert np.isnan(below.values).all()


def test_fill_value_int_upcasts_to_float():
    """fill_value=NaN forces float upcast on int columns -- documented."""
    ds = xr.Dataset(
        {"v": (("lat", "lon"), np.arange(6, dtype=np.int64).reshape(3, 2))},
        coords={"lat": [0, 1, 2], "lon": [10, 11]},
    ).chunk({"lat": 3})
    ctx = XarrayContext()
    ctx.from_dataset("t", ds)
    out = ctx.sql("SELECT * FROM t WHERE lat > 0").to_dataset(
        sparsity="template"
    )
    assert np.issubdtype(out["v"].dtype, np.floating)
    assert np.isnan(out["v"].sel(lat=0).values).all()


def test_fill_value_custom_preserves_int(air_dataset_small):
    """Passing a typed sentinel preserves the data var's int dtype."""
    # Build a small int-valued Dataset, register, filter out part of the
    # extent, and reindex back with an int fill_value via sparsity.
    source = xr.Dataset(
        {
            "v": (
                ("lat", "lon"),
                np.arange(6, dtype=np.int64).reshape(3, 2) + 1,
            ),
        },
        coords={"lat": [0, 1, 2], "lon": [10, 11]},
    ).chunk({"lat": 3})
    ctx = XarrayContext()
    ctx.from_dataset("t", source)
    out = ctx.sql("SELECT * FROM t WHERE lat > 0").to_dataset(
        sparsity="template", fill_value=-1
    )
    assert np.issubdtype(out["v"].dtype, np.integer)
    assert (out["v"].sel(lat=0).values == -1).all()
    assert out["v"].sel(lat=2, lon=11).item() == 6


def test_sparsity_template_then_metadata(air_dataset_small):
    """sparsity='template' composes with template metadata recovery."""
    ds = air_dataset_small.copy()
    ds.attrs = {"src": "tmpl"}
    ds["air"].attrs = {"units": "K"}
    ctx = XarrayContext()
    ctx.from_dataset("air", ds)
    threshold = float(ds["lat"].values[5])
    out = ctx.sql(f"SELECT * FROM air WHERE lat > {threshold}").to_dataset(
        sparsity="template"
    )
    assert out.attrs == {"src": "tmpl"}
    assert out["air"].attrs == {"units": "K"}
    assert out.sizes["lat"] == ds.sizes["lat"]


def test_to_dataset_explicit_template_overrides_auto_resolve(
    air_dataset_small,
):
    """Explicit template= wins over the auto-resolved FROM-clause table."""
    other = air_dataset_small.copy()
    other.attrs = {"flag": "explicit"}
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)  # registered, but...
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dimension_columns=["time", "lat", "lon"], template=other
    )
    assert out.attrs == {"flag": "explicit"}


def test_vectorized_indexer_falls_back_via_xarray_adapter(
    air_dataset_small,
):
    """VectorizedIndexer paths through xarray's adapter to outer + gather.

    Our SQLBackendArray declares ``IndexingSupport.OUTER``, so xarray's
    ``explicit_indexing_adapter`` converts vectorized indexers into a
    series of outer reads followed by an in-memory numpy gather. The
    public contract: values match the eager-computed equivalent.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    lazy = ctx.sql("SELECT * FROM air").to_dataset()
    eager = ctx.sql("SELECT * FROM air").to_dataset().compute()

    # Vectorized indexing: pick (time, lat) pairs along a new dim "point".
    points_t = xr.DataArray([0, 3, 1], dims="point")
    points_lat = xr.DataArray([2, 0, 5], dims="point")
    lazy_pts = lazy["air"].isel(time=points_t, lat=points_lat).values
    eager_pts = eager["air"].isel(time=points_t, lat=points_lat).values
    np.testing.assert_array_equal(lazy_pts, eager_pts)
    # Shape: (point=3, lon=53).
    assert lazy_pts.shape == (3, air_dataset_small.sizes["lon"])


def test_full_dim_slice_omits_filter_for_full_dims(
    air_dataset_small, monkeypatch
):
    """``isel(time=0)`` filters on ``time`` only; full-extent dims contribute
    no filter predicate (verified by intercepting ``DataFrame.filter``)."""
    from xarray_sql import ds as ds_mod

    calls: list[Any] = []
    original_filter = ds_mod.SQLBackendArray._raw_getitem

    def trace(self, key):
        # Intercept by patching the inner DataFrame's filter on each call.
        captured_inner = self._inner_df

        def capture_filter(expr):
            calls.append(expr)
            return original_inner_filter(expr)

        original_inner_filter = captured_inner.filter
        captured_inner.filter = capture_filter  # type: ignore[method-assign]
        try:
            return original_filter(self, key)
        finally:
            captured_inner.filter = original_inner_filter  # restore

    monkeypatch.setattr(ds_mod.SQLBackendArray, "_raw_getitem", trace)

    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset()
    calls.clear()
    _ = out["air"].isel(time=0).values
    # Exactly one filter call (one slab access); the predicate's repr
    # should reference "time" but not "lat" or "lon" (they cover the
    # full extent and get omitted).
    assert len(calls) == 1
    predicate_repr = repr(calls[0])
    assert "time" in predicate_repr
    assert "lat" not in predicate_repr
    assert "lon" not in predicate_repr
