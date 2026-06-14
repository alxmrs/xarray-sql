"""Tests for the SQL -> xarray reverse path.

Covers the user-facing contract of ``ctx.sql(...).to_dataset(...)``:

* Wrapper behavior on the object returned by ``ctx.sql`` and DataFusion
  method passthrough.
* Round-trip identity across varied source Datasets (one parametrized
  ``assert_identical`` test, not eight per-aspect checks).
* Aggregation, ``dimension_columns`` inference, and the template /
  ``template_table`` resolution rules with their error paths.
* Sparsity handling and ``fill_value`` dtype behavior.
* The vectorized-indexer fallback through xarray's adapter.

The tests favor checking the user-visible contract (values, dims,
attrs) over the implementation path (call counts, internal class
identity), so the suite stays useful as the lazy backend evolves.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xarray_sql import XarrayContext
from xarray_sql.ds import XarrayDataFrame


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
# Round-trip identity (parametrized over local + tutorial datasets)
# ---------------------------------------------------------------------------


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Strip ``encoding`` from a Dataset and all its variables.

    Round-trip identity tests should not be coupled to encoding choices,
    since template-recovery deliberately drops dtype-bound keys.
    """
    ds = ds.copy()
    for v in ds.variables.values():
        v.encoding.clear()
    ds.encoding.clear()
    return ds


def _load_tutorial(name: str) -> xr.Dataset | None:
    """Return a small xarray tutorial Dataset, or None when unavailable.

    Used to widen round-trip coverage beyond the conftest fixtures without
    requiring network in CI. Pooch caches downloads locally on first run.
    """
    try:
        return xr.tutorial.open_dataset(name)
    except (OSError, ValueError, ImportError):
        return None


@pytest.mark.parametrize(
    "fixture_name",
    ["air_dataset_small", "weather_dataset", "synthetic_dataset", "eraint_uvz"],
)
def test_round_trip_identity(request, fixture_name):
    """``SELECT *`` round-trips to a Dataset that is ``assert_identical``
    to the source: values, dims, coord values, dtypes, non-dim coords,
    and attrs all match (modulo coord ordering, normalized on both
    sides). One test covers what was previously a fan of narrow checks,
    parametrized over local fixtures and one xarray tutorial dataset.
    """
    if fixture_name == "eraint_uvz":
        source = _load_tutorial("eraint_uvz")
        if source is None:
            pytest.skip("eraint_uvz tutorial dataset unavailable")
        source = source.chunk()
    else:
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


def test_barrier_query_scans_source_once(air_dataset_small):
    """A barrier plan (aggregation) executes the source exactly once.

    The lazy scan path re-runs the whole upstream plan for every coordinate
    discovery and every variable access; for an aggregation -- which cannot push
    an indexer filter below the GROUP BY -- that is pure re-computation of an
    expensive scan. ``to_dataset()`` on a barrier plan must instead make a
    single streamed pass over the source, and ``.compute()`` must trigger no
    further reads.
    """
    from xarray_sql.df import block_slices
    from xarray_sql.reader import read_xarray_table

    reads: list = []
    table = read_xarray_table(
        air_dataset_small,
        chunks={"time": 6},
        _iteration_callback=lambda block, proj: reads.append(block),
    )
    n_partitions = len(list(block_slices(air_dataset_small, {"time": 6})))

    ctx = XarrayContext()
    ctx.register_table("air", table)
    ctx._registered_datasets["air"] = air_dataset_small

    out = ctx.sql(
        "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
    ).to_dataset(dimension_columns=["lat", "lon"])
    reads_after_construct = len(reads)
    out.compute()
    reads_after_compute = len(reads)

    # Exactly one pass over the source (each partition read once) ...
    assert reads_after_construct == n_partitions
    # ... and computing the materialized result re-reads nothing.
    assert reads_after_compute == reads_after_construct


# ---------------------------------------------------------------------------
# dimension_columns / template resolution rules
# ---------------------------------------------------------------------------


def test_to_dataset_multi_registered_requires_explicit_template(
    air_dataset_small,
):
    """With more than one registered Dataset, the caller must
    disambiguate via ``template_table=``."""
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
        ctx.sql(
            "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
        ).to_dataset()


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


def test_to_dataset_explicit_template_overrides_auto_resolve(
    air_dataset_small,
):
    """Explicit template= wins over the auto-resolved FROM-clause table."""
    other = air_dataset_small.copy()
    other.attrs = {"flag": "explicit"}
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset(
        dimension_columns=["time", "lat", "lon"], template=other
    )
    assert out.attrs == {"flag": "explicit"}


# ---------------------------------------------------------------------------
# Lazy backend: value-level contract (not call counts)
# ---------------------------------------------------------------------------


def test_lazy_isel_int_round_trip(air_dataset_small):
    """``isel(time=0)`` on the lazy result matches the eager equivalent."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    lazy = ctx.sql("SELECT * FROM air").to_dataset()
    eager = lazy.compute()
    actual = lazy["air"].isel(time=0).sortby(["lat", "lon"]).values
    expected = eager["air"].isel(time=0).sortby(["lat", "lon"]).values
    np.testing.assert_array_equal(actual, expected)


def test_lazy_isel_slice_round_trip(air_dataset_small):
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


def test_lazy_outer_indexer_array(air_dataset_small):
    """Fancy index along one dim works (IN-equivalent pushdown)."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    lazy = ctx.sql("SELECT * FROM air").to_dataset()
    eager = lazy.compute()
    indices = [0, 3, 5]
    np.testing.assert_array_equal(
        lazy["air"].isel(lat=indices).values,
        eager["air"].isel(lat=indices).values,
    )


def test_lazy_compute_returns_eager(air_dataset_small):
    """``.compute()`` returns an in-memory Dataset matching the source."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql("SELECT * FROM air").to_dataset().compute()
    np.testing.assert_array_equal(
        out.sortby(["time", "lat", "lon"])["air"].values,
        air_dataset_small.compute()
        .sortby(["time", "lat", "lon"])["air"]
        .values,
    )


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
    eager = lazy.compute()

    points_t = xr.DataArray([0, 3, 1], dims="point")
    points_lat = xr.DataArray([2, 0, 5], dims="point")
    np.testing.assert_array_equal(
        lazy["air"].isel(time=points_t, lat=points_lat).values,
        eager["air"].isel(time=points_t, lat=points_lat).values,
    )


# ---------------------------------------------------------------------------
# Sparsity handling and fill_value
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
