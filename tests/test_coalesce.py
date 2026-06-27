"""Tests for bounded-partition registration (issue #174).

`read_xarray_table` used to create one DataFusion scan partition per native
xarray chunk, making registration O(num_chunks) and intractable on finely
chunked stores (e.g. ~59M partitions for one GOES-16 variable).

These tests pin the fix: native chunks are coalesced into a bounded number of
scan partitions while query results stay identical. They cover the group-size
algorithm, the block coalescing/tiling, and the end-to-end behavior via
`read_xarray_table` and `XarrayContext.from_dataset`.
"""

import math
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datafusion import SessionContext

from xarray_sql import XarrayContext
from xarray_sql.df import (
    DEFAULT_TARGET_PARTITIONS,
    block_slices,
    coalesce_group_sizes,
    coalesced_blocks,
)
from xarray_sql.reader import read_xarray_table


def _resulting_partition_count(chunk_counts, groups):
    """Partitions produced by merging ``groups[d]`` native chunks per dim."""
    product = 1
    for dim, count in chunk_counts.items():
        product *= math.ceil(count / groups[dim])
    return product


def _block_key(block):
    """Order-independent hashable identity for a block slice dict."""
    return tuple(
        sorted((str(dim), slc.start, slc.stop) for dim, slc in block.items())
    )


@pytest.fixture
def finely_chunked():
    """200 native chunks: time chunked to 1 step over a small spatial grid."""
    np.random.seed(0)
    n_time = 200
    time = pd.date_range("2020-01-01", periods=n_time, freq="h")
    lat = np.linspace(-10, 10, 4)
    lon = np.linspace(-10, 10, 4)
    data = np.random.rand(n_time, 4, 4).astype("float32")
    ds = xr.Dataset(
        {"t2m": (["time", "lat", "lon"], data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds.chunk({"time": 1, "lat": 4, "lon": 4})


class TestCoalesceGroupSizes:
    """Unit tests for the per-dimension group-size algorithm."""

    def test_default_constant_is_sane(self):
        assert isinstance(DEFAULT_TARGET_PARTITIONS, int)
        assert DEFAULT_TARGET_PARTITIONS >= 1024

    def test_identity_when_under_target(self):
        counts = {"time": 4, "lat": 2}
        assert coalesce_group_sizes(counts, 16_384) == {"time": 1, "lat": 1}

    def test_none_target_disables_coalescing(self):
        counts = {"time": 100_000, "lat": 50}
        assert coalesce_group_sizes(counts, None) == {"time": 1, "lat": 1}

    def test_empty_counts(self):
        assert coalesce_group_sizes({}, 16_384) == {}

    def test_single_chunk(self):
        assert coalesce_group_sizes({"time": 1}, 10) == {"time": 1}

    def test_bounds_product_for_goes_case(self):
        counts = {"time": 102_988, "lat": 24, "lon": 24}
        groups = coalesce_group_sizes(counts, 16_384)
        assert _resulting_partition_count(counts, groups) <= 16_384

    def test_spatial_dims_not_collapsed_at_generous_target(self):
        # Spatial pruning must survive: lat/lon keep more than one partition.
        counts = {"time": 102_988, "lat": 24, "lon": 24}
        groups = coalesce_group_sizes(counts, 16_384)
        assert math.ceil(counts["lat"] / groups["lat"]) > 1
        assert math.ceil(counts["lon"] / groups["lon"]) > 1

    def test_balanced_reduces_largest_dimension_most(self):
        counts = {"time": 102_988, "lat": 24, "lon": 24}
        groups = coalesce_group_sizes(counts, 1_000)
        assert groups["time"] > groups["lat"]
        assert _resulting_partition_count(counts, groups) <= 1_000

    def test_tight_fit_uses_most_of_the_budget(self):
        # The allocation should hug the target from below, not waste most of
        # the budget (which would coarsen pruning more than necessary).
        counts = {"time": 102_988, "lat": 24, "lon": 24}
        groups = coalesce_group_sizes(counts, 16_384)
        total = _resulting_partition_count(counts, groups)
        assert total <= 16_384
        assert total >= 16_384 // 2

    def test_target_below_dimension_count_terminates(self):
        counts = {dim: 10 for dim in "abcde"}
        groups = coalesce_group_sizes(counts, 3)
        assert _resulting_partition_count(counts, groups) <= 3

    def test_huge_input_completes_quickly(self):
        # An O(num_chunks) implementation would hang/OOM on 10**12 chunks.
        counts = {"a": 10**6, "b": 10**6}
        groups = coalesce_group_sizes(counts, 1_000)
        assert _resulting_partition_count(counts, groups) <= 1_000


class TestCoalescedBlocks:
    """The coalesced block generator must tile the dataset exactly."""

    def test_identity_matches_block_slices_under_target(self, finely_chunked):
        supers = [
            super_block
            for super_block, _subs in coalesced_blocks(
                finely_chunked, None, 10**9
            )
        ]
        assert supers == list(block_slices(finely_chunked))

    def test_subblocks_tile_dataset_exactly(self, finely_chunked):
        native = list(block_slices(finely_chunked))
        collected = []
        for _super_block, subs in coalesced_blocks(finely_chunked, None, 10):
            collected.extend(subs())
        assert sorted(map(_block_key, collected)) == sorted(
            map(_block_key, native)
        )

    def test_partition_count_bounded(self, finely_chunked):
        partitions = list(coalesced_blocks(finely_chunked, None, 10))
        assert len(partitions) <= 10
        assert len(partitions) < len(list(block_slices(finely_chunked)))

    def test_super_block_is_bounding_slice(self, finely_chunked):
        for super_block, subs in coalesced_blocks(finely_chunked, None, 7):
            subs = list(subs())
            for dim, slc in super_block.items():
                assert slc.start == min(s[dim].start for s in subs)
                assert slc.stop == max(s[dim].stop for s in subs)

    def test_scalar_dataset_single_block(self):
        ds = xr.Dataset({"x": 5})
        partitions = list(coalesced_blocks(ds, None, 10))
        assert len(partitions) == 1
        super_block, subs = partitions[0]
        assert super_block == {}
        assert list(subs()) == [{}]


class _Tracker:
    """Records every (coalesced) partition scanned during a query."""

    def __init__(self):
        self.count = 0
        self.blocks = []

    def __call__(self, block, projection=None):
        self.count += 1
        self.blocks.append(block)


def _count_scanned_partitions(ds, sql, **kwargs):
    tracker = _Tracker()
    table = read_xarray_table(ds, _iteration_callback=tracker, **kwargs)
    ctx = SessionContext()
    ctx.register_table("t", table)
    ctx.sql(sql).collect()
    return tracker.count


class TestReadXarrayTableCoalescing:
    """End-to-end behavior through read_xarray_table."""

    def test_default_is_noop_for_small_datasets(self, finely_chunked):
        # 200 native chunks << default target -> one partition per chunk.
        scanned = _count_scanned_partitions(
            finely_chunked, "SELECT COUNT(*) FROM t"
        )
        assert scanned == len(list(block_slices(finely_chunked)))

    def test_target_partitions_bounds_partition_count(self, finely_chunked):
        scanned = _count_scanned_partitions(
            finely_chunked, "SELECT COUNT(*) FROM t", target_partitions=8
        )
        assert scanned <= 8
        assert scanned < len(list(block_slices(finely_chunked)))

    def test_target_none_is_one_partition_per_chunk(self, finely_chunked):
        scanned = _count_scanned_partitions(
            finely_chunked, "SELECT COUNT(*) FROM t", target_partitions=None
        )
        assert scanned == len(list(block_slices(finely_chunked)))

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT COUNT(*) AS n, MIN(t2m) AS mn, MAX(t2m) AS mx FROM t",
            "SELECT COUNT(*) AS n, MIN(t2m) AS mn, MAX(t2m) AS mx "
            "FROM t WHERE time > '2020-01-04'",
            "SELECT COUNT(*) AS n FROM t WHERE lat > 0",
        ],
    )
    def test_results_identical_coalesced_vs_uncoalesced(
        self, finely_chunked, sql
    ):
        def run(target):
            table = read_xarray_table(finely_chunked, target_partitions=target)
            ctx = SessionContext()
            ctx.register_table("t", table)
            return ctx.sql(sql).to_pandas()

        pd.testing.assert_frame_equal(run(8), run(None))


class TestCoalescedMemory:
    """Coalescing must not inflate per-partition memory."""

    def test_single_partition_streams_native_subblocks(self):
        # Build the dataset BEFORE tracemalloc.start() so its source arrays are
        # not counted (mirrors test_read_xarray_table_memory_bounds).
        np.random.seed(1)
        # Spatial chunk large enough that one native chunk dominates the fixed
        # registration overhead (lazy native-module import, coord arrays).
        n_time, n_lat, n_lon = 120, 128, 128
        ds = xr.Dataset(
            {
                "a": (
                    ["time", "lat", "lon"],
                    np.random.rand(n_time, n_lat, n_lon).astype("float32"),
                ),
                "b": (
                    ["time", "lat", "lon"],
                    np.random.rand(n_time, n_lat, n_lon).astype("float32"),
                ),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=n_time, freq="h"),
                "lat": np.linspace(-90, 90, n_lat),
                "lon": np.linspace(-180, 180, n_lon),
            },
        ).chunk({"time": 1, "lat": n_lat, "lon": n_lon})  # 300 native chunks

        full = ds.nbytes
        one_chunk = ds.isel(next(block_slices(ds))).nbytes

        tracemalloc.stop()  # reset any state from a previously-failed test
        tracemalloc.start()
        try:
            # target=1: a single partition spanning all 300 native chunks.
            table = read_xarray_table(ds, target_partitions=1)
            reg_size, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()

            # Registration holds only coord arrays + metadata, not data.
            assert reg_size < one_chunk, (
                f"registration held {reg_size} bytes >= one chunk "
                f"{one_chunk}: data loaded eagerly"
            )

            ctx = SessionContext()
            ctx.register_table("t", table)
            ctx.sql("SELECT COUNT(*) FROM t").collect()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # If the single partition materialised its whole super-block (the entire
        # dataset), peak would approach `full`. Streaming native sub-blocks one
        # at a time keeps it far below that.
        assert peak < full, (
            f"query peak {peak} >= full dataset {full}: the coalesced "
            "partition materialised its whole super-block instead of streaming "
            "native sub-blocks"
        )


class TestFromDatasetCoalescing:
    """XarrayContext.from_dataset must thread target_partitions through."""

    def test_from_dataset_accepts_target_and_preserves_results(
        self, finely_chunked
    ):
        def run(target):
            ctx = XarrayContext()
            ctx.from_dataset("t", finely_chunked, target_partitions=target)
            return ctx.sql(
                "SELECT COUNT(*) AS n, MIN(t2m) AS mn, MAX(t2m) AS mx FROM t"
            ).to_pandas()

        pd.testing.assert_frame_equal(run(8), run(None))


class TestExplicitChunksCapping:
    """An explicit chunks= is still capped at target (safety net)."""

    def _unchunked(self):
        np.random.seed(2)
        n_time = 500
        return xr.Dataset(
            {"v": (["time", "x"], np.random.rand(n_time, 2).astype("float32"))},
            coords={
                "time": pd.date_range("2020-01-01", periods=n_time, freq="h"),
                "x": [0, 1],
            },
        )

    def test_explicit_fine_chunks_are_capped(self):
        ds = self._unchunked()
        # chunks={'time': 1} would make 500 native partitions; target caps it.
        scanned = _count_scanned_partitions(
            ds,
            "SELECT COUNT(*) FROM t",
            chunks={"time": 1, "x": 2},
            target_partitions=8,
        )
        assert scanned <= 8

    def test_explicit_chunks_not_capped_when_target_none(self):
        ds = self._unchunked()
        scanned = _count_scanned_partitions(
            ds,
            "SELECT COUNT(*) FROM t",
            chunks={"time": 1, "x": 2},
            target_partitions=None,
        )
        assert scanned == 500


class TestCftimeCoalescing:
    """Non-Gregorian cftime metadata (cft.partition_bounds) under coalescing.

    A 360_day calendar maps to int64 columns with a ``cftime()`` filter UDF;
    its partition bounds go through a different code path than numeric/datetime
    coords, so coalescing it must still leave query results (and pruning)
    unchanged.
    """

    def _dataset(self):
        time = xr.date_range(
            "2000-01-01",
            periods=360,
            freq="D",
            calendar="360_day",
            use_cftime=True,
        )
        return xr.Dataset(
            {"v": (["time", "x"], np.random.rand(360, 2).astype("float32"))},
            coords={"time": time, "x": [0, 1]},
        ).chunk({"time": 1, "x": 2})  # 360 native chunks

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT COUNT(*) AS n FROM c",
            "SELECT COUNT(*) AS n FROM c WHERE time >= cftime('2000-07-01')",
        ],
    )
    def test_360day_results_identical_coalesced_vs_uncoalesced(self, sql):
        ds = self._dataset()

        def run(target):
            ctx = XarrayContext()
            ctx.from_dataset("c", ds, target_partitions=target)
            return ctx.sql(sql).to_pandas()

        pd.testing.assert_frame_equal(run(4), run(None))
