"""Tests for the SQL -> xarray reverse path.

Covers the user-facing contract of ``ctx.sql(...).to_dataset(...)``:

* Wrapper behavior on the object returned by ``ctx.sql`` (method passthrough,
  ``to_pandas`` equivalence).
* Round-trip identity across the eager and chunked paths (one parametrized
  ``assert_identical`` test).
* Aggregation behavior: dim reduction, single-scan execution, ``ORDER BY``
  direction, and the ``chunks`` argument (eager / inherit / ``"auto"``).
* ``dims`` inference and ``template`` resolution (name or Dataset), with error
  paths and metadata recovery.
* Indexing the chunked backend, sparsity handling, and ``fill_value`` dtype.

The tests favor the user-visible contract (values, dims, attrs) over the
implementation path, so the suite stays useful as the backend evolves.
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


def test_sql_returns_wrapper_that_forwards_methods(air_dataset_small):
    """``ctx.sql`` returns an ``XarrayDataFrame`` that forwards un-overridden
    DataFusion methods (e.g. ``schema()``) via ``__getattr__``."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    result = ctx.sql("SELECT * FROM air LIMIT 5")
    assert isinstance(result, XarrayDataFrame)
    names = [f.name for f in result.schema()]
    assert {"lat", "lon", "time", "air"}.issubset(set(names))


def test_to_pandas_unchanged_behavior(air_dataset_small):
    """Wrapped ``.to_pandas()`` is bit-for-bit equal to the un-wrapped path."""
    from datafusion import SessionContext

    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    wrapped = ctx.sql("SELECT * FROM air LIMIT 7").to_pandas()
    raw = SessionContext.sql(ctx, "SELECT * FROM air LIMIT 7").to_pandas()
    pd.testing.assert_frame_equal(wrapped, raw)


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


@pytest.mark.parametrize(
    "fixture_name",
    # ``air`` exercises the eager path (single-chunk source); ``weather``
    # exercises the chunked path and adds datetime + non-dim coordinates.
    ["air_dataset_small", "weather_dataset"],
)
def test_round_trip_identity(request, fixture_name):
    """``SELECT *`` round-trips to a Dataset that is ``assert_identical``
    to the source: values, dims, coord values, dtypes, non-dim coords,
    and attrs all match (modulo coord ordering, normalized on both
    sides).
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
    ).to_dataset(dims=["lat", "lon"])
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
    ).to_dataset(dims=["lat", "lon"])
    reads_after_construct = len(reads)
    out.compute()
    reads_after_compute = len(reads)

    # Exactly one pass over the source (each partition read once) ...
    assert reads_after_construct == n_partitions
    # ... and computing the materialized result re-reads nothing.
    assert reads_after_compute == reads_after_construct


def test_order_by_direction_sets_dim_order(air_dataset_small):
    """A barrier query's ORDER BY direction carries through to the Dataset
    dimension order, rather than being force-sorted ascending.

    ``ORDER BY lat DESC`` must yield a strictly descending ``lat`` dimension,
    with data still correctly aligned to those (descending) coordinates.
    """
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    out = ctx.sql(
        "SELECT lat, AVG(air) AS air_avg FROM air GROUP BY lat ORDER BY lat DESC"
    ).to_dataset(dims=["lat"])

    lat = out["lat"].values
    assert (np.diff(lat) < 0).all(), f"expected descending lat, got {lat}"

    # Values stay aligned to the descending coordinate (scatter handles order).
    expected = (
        air_dataset_small.compute().mean(dim=["time", "lon"])["air"].sortby(
            "lat", ascending=False
        )
    )
    np.testing.assert_allclose(out["air_avg"].values, expected.values)


def test_chunks_argument_controls_partitioning(synthetic_dataset):
    """``chunks`` controls eager-vs-chunked and inherits the source grid.

    The default ``"inherit"`` reuses the source's genuinely multi-chunk
    dimensions, so the output chunk grid maps onto the source partitions;
    ``chunks=None`` forces an eager, in-memory result. Both reproduce the source.
    """
    import dask.array as da

    ctx = XarrayContext()
    ctx.from_dataset("t", synthetic_dataset)
    var = next(iter(synthetic_dataset.data_vars))

    inherited = ctx.sql("SELECT * FROM t").to_dataset()
    assert isinstance(inherited[var].data, da.Array)
    # Output time chunks align to the source's time partitions.
    assert (
        inherited.chunksizes["time"] == synthetic_dataset.chunksizes["time"]
    )

    eager = ctx.sql("SELECT * FROM t").to_dataset(chunks=None)
    assert not isinstance(eager[var].data, da.Array)

    xr.testing.assert_allclose(
        inherited.compute().sortby(["time", "lat", "lon"]),
        synthetic_dataset.compute().sortby(["time", "lat", "lon"]),
    )


def test_chunks_auto_snaps_to_source_partitions():
    """``chunks="auto"`` coarsens to the byte budget but snaps chunk boundaries
    to whole source partitions (so no chunk splits a source partition)."""
    import dask

    # 12 source partitions of size 2 along time.
    ds = xr.Dataset(
        {"v": (("time", "x"), np.arange(24 * 4, dtype="float64").reshape(24, 4))},
        coords={"time": np.arange(24), "x": np.arange(4)},
    ).chunk({"time": 2})
    ctx = XarrayContext()
    ctx.from_dataset("t", ds)

    # block bytes = 8 * 2(time) * 4(x) = 64; target 192 -> merge 3 partitions.
    with dask.config.set({"array.chunk-size": "192B"}):
        out = ctx.sql("SELECT * FROM t").to_dataset(chunks="auto")

    time_chunks = out.chunksizes["time"]
    assert all(c % 2 == 0 for c in time_chunks)  # aligned to source size 2
    assert time_chunks[0] > 2  # genuinely coarsened
    assert len(time_chunks) < 12  # fewer chunks than source partitions

    xr.testing.assert_allclose(
        out.compute().sortby(["time", "x"]),
        ds.compute().sortby(["time", "x"]),
    )


# ---------------------------------------------------------------------------
# dimension_columns / template resolution rules
# ---------------------------------------------------------------------------


def test_to_dataset_infer_fails_when_no_template_fits(air_dataset_small):
    """If no registered Dataset's dims fit the result -> clear error."""
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(
        ValueError, match="dims cannot be inferred"
    ):
        ctx.sql(
            "SELECT lat, lon, AVG(air) AS air_avg FROM air GROUP BY lat, lon"
        ).to_dataset()


def test_template_accepts_name_or_dataset(air_dataset_small):
    """``template=`` accepts either a registered table name or a Dataset
    object, with equivalent metadata recovery."""
    other = air_dataset_small.copy()
    other.attrs = {"flag": "other"}
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    ctx.from_dataset("other", other)

    by_name = ctx.sql("SELECT * FROM air").to_dataset(
        dims=["time", "lat", "lon"], template="other"
    )
    by_object = ctx.sql("SELECT * FROM air").to_dataset(
        dims=["time", "lat", "lon"], template=other
    )
    assert by_name.attrs == {"flag": "other"}
    assert by_object.attrs == {"flag": "other"}


def test_template_unknown_name_raises(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="not a registered table"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dims=["time", "lat", "lon"], template="missing"
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
        dims=["time", "lat", "lon"]
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
    ).to_dataset(dims=["lat", "lon"])
    assert "air_avg" in out.data_vars
    assert out["air_avg"].attrs == {}


# ---------------------------------------------------------------------------
# Chunked (lazy SQLBackendArray) backend: value-level contract
# ---------------------------------------------------------------------------


def test_chunked_backend_indexing_matches_eager(air_dataset_small):
    """Indexing a chunked result (the lazy ``SQLBackendArray`` path, reached by
    passing ``chunks=``) matches the eager equivalent across every indexer kind.

    ``chunks=`` forces the dask-wrapped backend whose chunks read their
    coordinate range via DataFusion filter pushdown; this exercises the int,
    slice, outer-array, and vectorized indexer translations -- the last via
    xarray's ``IndexingSupport.OUTER`` adapter (outer reads + numpy gather).
    """
    import dask.array as da

    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    chunked = ctx.sql("SELECT * FROM air").to_dataset(chunks={"time": 4})
    assert isinstance(chunked["air"].data, da.Array)  # genuinely lazy/chunked
    # Compare lazy indexing against computing-then-indexing the SAME Dataset, so
    # both sides share one coordinate order (positional indexers stay aligned).
    eager = chunked.compute()

    def s(da_):  # normalize coord order before comparing values
        return da_.sortby(["lat", "lon"]).values

    # int indexer
    np.testing.assert_array_equal(
        s(chunked["air"].isel(time=0)), s(eager["air"].isel(time=0))
    )
    # slice indexer
    np.testing.assert_array_equal(
        s(chunked["air"].isel(time=slice(0, 3))),
        s(eager["air"].isel(time=slice(0, 3))),
    )
    # outer (fancy) array indexer
    np.testing.assert_array_equal(
        chunked["air"].isel(lat=[0, 3, 5]).values,
        eager["air"].isel(lat=[0, 3, 5]).values,
    )
    # vectorized indexer (xarray adapter -> outer + gather)
    pt = xr.DataArray([0, 3, 1], dims="point")
    pl = xr.DataArray([2, 0, 5], dims="point")
    np.testing.assert_array_equal(
        chunked["air"].isel(time=pt, lat=pl).values,
        eager["air"].isel(time=pt, lat=pl).values,
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
            dims=["time", "lat", "lon"],
            sparsity="template",
        )


def test_sparsity_invalid_value_raises(air_dataset_small):
    ctx = XarrayContext()
    ctx.from_dataset("air", air_dataset_small)
    with pytest.raises(ValueError, match="sparsity must be"):
        ctx.sql("SELECT * FROM air").to_dataset(
            dims=["time", "lat", "lon"],
            sparsity="bogus",  # type: ignore[arg-type]
        )


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
