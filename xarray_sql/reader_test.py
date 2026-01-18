"""Tests for XarrayRecordBatchReader lazy streaming behavior.

These tests verify that XarrayRecordBatchReader provides true lazy evaluation:
- No data iteration during reader creation
- No data iteration during DataFusion table registration (using LazyArrowStreamTable)
- Data iteration ONLY occurs during query execution (collect())

The lazy streaming is achieved via the Rust LazyArrowStreamTable class which
implements the __datafusion_table_provider__ protocol using StreamingTable.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import SessionContext

from ._native import LazyArrowStreamTable
from .reader import XarrayRecordBatchReader, read_xarray_lazy, read_xarray_table


@pytest.fixture
def small_ds():
  """Create a small dataset for testing."""
  np.random.seed(42)
  time = pd.date_range("2020-01-01", periods=100, freq="h")
  lat = np.linspace(-90, 90, 10)
  lon = np.linspace(-180, 180, 10)

  data = np.random.rand(100, 10, 10).astype(np.float32)

  return xr.Dataset(
      {"temperature": (["time", "lat", "lon"], data)},
      coords={"time": time, "lat": lat, "lon": lon},
  )


class IterationTracker:
  """Tracks when iteration occurs for testing lazy evaluation."""

  def __init__(self):
    self.iteration_count = 0
    self.blocks_seen = []

  def __call__(self, block):
    self.iteration_count += 1
    self.blocks_seen.append(block)

  def reset(self):
    self.iteration_count = 0
    self.blocks_seen = []


class TestXarrayRecordBatchReaderCreation:
  """Tests that reader creation does NOT trigger data iteration."""

  def test_reader_creation_does_not_iterate(self, small_ds):
    """Creating a reader should NOT iterate through any data."""
    tracker = IterationTracker()

    reader = XarrayRecordBatchReader(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    assert tracker.iteration_count == 0, (
        f"Expected 0 iterations during reader creation, "
        f"but got {tracker.iteration_count}"
    )

  def test_read_xarray_lazy_does_not_iterate(self, small_ds):
    """read_xarray_lazy() should NOT iterate through any data."""
    # We can't use the callback with the public API, but we can verify
    # basic behavior - no exceptions and no data loaded
    reader = read_xarray_lazy(small_ds, chunks={"time": 25})

    assert reader is not None
    assert reader.schema is not None
    assert not reader._consumed

  def test_schema_access_does_not_iterate(self, small_ds):
    """Accessing the schema should NOT trigger iteration."""
    tracker = IterationTracker()

    reader = XarrayRecordBatchReader(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    # Access schema
    _ = reader.schema
    _ = reader.__arrow_c_schema__()

    assert tracker.iteration_count == 0, (
        f"Expected 0 iterations when accessing schema, "
        f"but got {tracker.iteration_count}"
    )


class TestDataFusionRegistration:
  """Tests that DataFusion table registration does NOT trigger iteration.

  These tests use read_xarray_table with register_table_provider()
  to achieve true lazy evaluation.
  """

  def test_register_table_does_not_iterate(self, small_ds):
    """Registering a LazyArrowStreamTable should NOT iterate data.

    This is the KEY test for lazy evaluation. LazyArrowStreamTable wraps
    a factory and implements __datafusion_table_provider__ with StreamingTable,
    ensuring data is only read during query execution.
    """
    tracker = IterationTracker()

    # Use read_xarray_table which creates a factory-based table
    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    assert tracker.iteration_count == 0, (
        f"LAZY EVALUATION FAILED: Expected 0 iterations during "
        f"register_table_provider(), but got {tracker.iteration_count}."
    )

  def test_lazy_table_creation_does_not_iterate(self, small_ds):
    """Creating a LazyArrowStreamTable should NOT iterate data."""
    tracker = IterationTracker()

    # Use read_xarray_table which creates a factory-based table
    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    assert tracker.iteration_count == 0, (
        f"LAZY EVALUATION FAILED: Expected 0 iterations during "
        f"LazyArrowStreamTable creation, but got {tracker.iteration_count}."
    )

  def test_sql_planning_does_not_iterate(self, small_ds):
    """Creating a SQL query plan should NOT iterate data."""
    tracker = IterationTracker()

    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    # Create a query but don't execute it
    query = ctx.sql("SELECT AVG(temperature) FROM test_table")

    # Just creating the query shouldn't iterate
    assert tracker.iteration_count == 0, (
        f"Expected 0 iterations during SQL planning, "
        f"but got {tracker.iteration_count}. "
        f"DataFusion may be scanning data during query planning."
    )


class TestDataFusionCollect:
  """Tests that data iteration ONLY occurs during collect().

  These tests use read_xarray_table to verify lazy evaluation.
  """

  def test_collect_triggers_iteration(self, small_ds):
    """collect() should trigger data iteration."""
    tracker = IterationTracker()

    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    # Verify no iteration yet (lazy registration)
    iteration_before_collect = tracker.iteration_count
    assert (
        iteration_before_collect == 0
    ), "Should have 0 iterations before collect"

    # Now collect - this SHOULD iterate
    result = ctx.sql("SELECT * FROM test_table LIMIT 10").collect()

    assert tracker.iteration_count > 0, (
        f"Expected iterations during collect(), but got 0. "
        f"Data was never read!"
    )
    assert (
        tracker.iteration_count > iteration_before_collect
    ), f"Expected more iterations after collect()"

  def test_full_query_iterates_all_blocks(self, small_ds):
    """A query that reads all data should iterate all blocks."""
    tracker = IterationTracker()

    chunks = {"time": 25}
    table = read_xarray_table(
        small_ds,
        chunks=chunks,
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    # Run a query that needs to scan all data
    result = ctx.sql("SELECT COUNT(*) FROM test_table").collect()

    # With time=100 and chunks=25, we expect 4 blocks
    expected_blocks = 100 // 25
    assert tracker.iteration_count == expected_blocks, (
        f"Expected {expected_blocks} block iterations, "
        f"but got {tracker.iteration_count}"
    )

  def test_aggregation_query_iterates_correctly(self, small_ds):
    """Aggregation queries should iterate all necessary blocks."""
    tracker = IterationTracker()

    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    # Run aggregation
    result = ctx.sql(
        "SELECT lat, AVG(temperature) as avg_temp "
        "FROM test_table GROUP BY lat"
    ).collect()

    # Should have iterated some blocks
    assert tracker.iteration_count > 0
    assert len(result) > 0


class TestLazyEvaluationEndToEnd:
  """End-to-end tests verifying lazy evaluation through the full pipeline.

  These tests use read_xarray_table to achieve true lazy evaluation.
  """

  def test_lazy_evaluation_sequence(self, small_ds):
    """Verify the exact sequence of lazy evaluation stages.

    This is the comprehensive test that proves true lazy evaluation:
    1. Table creation: 0 iterations
    2. Table registration: 0 iterations
    3. Query planning: 0 iterations
    4. collect(): N iterations (where N = number of blocks)
    """
    tracker = IterationTracker()

    # Stage 1: Table creation (with factory)
    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )
    iterations_after_table = tracker.iteration_count
    assert iterations_after_table == 0, (
        f"Stage 1 FAILED: Table creation triggered "
        f"{iterations_after_table} iterations"
    )

    # Stage 2: Table registration
    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)
    iterations_after_registration = tracker.iteration_count
    assert iterations_after_registration == 0, (
        f"Stage 2 FAILED: Table registration triggered "
        f"{iterations_after_registration} iterations"
    )

    # Stage 3: Query planning
    query = ctx.sql("SELECT * FROM test_table")
    iterations_after_planning = tracker.iteration_count
    assert iterations_after_planning == 0, (
        f"Stage 3 FAILED: Query planning triggered "
        f"{iterations_after_planning} iterations"
    )

    # Stage 4: collect() - NOW iteration should happen
    result = query.collect()
    iterations_after_collect = tracker.iteration_count
    assert (
        iterations_after_collect > 0
    ), f"Stage 4 FAILED: collect() triggered 0 iterations - no data was read!"

    # Verify we got the expected number of blocks (100 time steps / 25 = 4)
    expected_blocks = 4
    assert (
        iterations_after_collect == expected_blocks
    ), f"Expected {expected_blocks} iterations, got {iterations_after_collect}"

  def test_multiple_queries_on_same_table(self, small_ds):
    """Same table can be queried multiple times with fresh iteration each time."""
    tracker = IterationTracker()

    table = read_xarray_table(
        small_ds,
        chunks={"time": 50},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    # First query
    ctx.sql("SELECT COUNT(*) FROM test_table").collect()
    first_query_iterations = tracker.iteration_count
    assert first_query_iterations > 0, "First query should iterate"

    # Second query on same table - should iterate again
    ctx.sql("SELECT AVG(temperature) FROM test_table").collect()
    second_query_iterations = tracker.iteration_count
    assert (
        second_query_iterations > first_query_iterations
    ), "Second query should trigger additional iterations"

  def test_stream_consumed_error(self, small_ds):
    """Once consumed, a single XarrayRecordBatchReader should not be reusable."""
    reader = XarrayRecordBatchReader(small_ds, chunks={"time": 25})

    # Consume the reader by converting to a PyArrow reader and reading
    import pyarrow as pa

    pa_reader = pa.RecordBatchReader.from_stream(reader)
    _ = pa_reader.read_all()

    # Reader is now consumed, calling __arrow_c_stream__ again should fail
    with pytest.raises(RuntimeError, match="already consumed"):
      reader.__arrow_c_stream__()


class TestDataIntegrity:
  """Tests that verify data correctness alongside lazy evaluation.

  These tests use read_xarray_table for lazy streaming.
  """

  def test_query_results_are_correct(self, small_ds):
    """Verify that lazy evaluation produces correct results."""
    table = read_xarray_table(small_ds, chunks={"time": 25})

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    # Get count
    result = ctx.sql("SELECT COUNT(*) as cnt FROM test_table").collect()
    count = result[0].to_pandas()["cnt"].iloc[0]

    # Expected: 100 time steps * 10 lat * 10 lon = 10,000 rows
    expected_count = 100 * 10 * 10
    assert (
        count == expected_count
    ), f"Expected {expected_count} rows, got {count}"

  def test_aggregation_results_are_correct(self, small_ds):
    """Verify aggregation produces correct results."""
    table = read_xarray_table(small_ds, chunks={"time": 25})

    ctx = SessionContext()
    ctx.register_table_provider("test_table", table)

    # Get average temperature
    result = ctx.sql(
        "SELECT AVG(temperature) as avg_temp FROM test_table"
    ).collect()
    avg_temp = result[0].to_pandas()["avg_temp"].iloc[0]

    # With seed 42 and random data in [0, 1), average should be ~0.5
    assert (
        0.4 < avg_temp < 0.6
    ), f"Expected average temperature ~0.5, got {avg_temp}"


class TestPyArrowInterop:
  """Tests for PyArrow interoperability."""

  def test_from_stream_does_not_iterate(self, small_ds):
    """pa.RecordBatchReader.from_stream() should not iterate."""
    tracker = IterationTracker()

    reader = XarrayRecordBatchReader(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    # Create PyArrow reader from our stream
    pa_reader = pa.RecordBatchReader.from_stream(reader)

    assert tracker.iteration_count == 0, (
        f"Expected 0 iterations when creating PyArrow reader, "
        f"but got {tracker.iteration_count}"
    )

  def test_pyarrow_iteration_triggers_callbacks(self, small_ds):
    """Iterating via PyArrow should trigger our callbacks."""
    tracker = IterationTracker()

    reader = XarrayRecordBatchReader(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    pa_reader = pa.RecordBatchReader.from_stream(reader)

    # Now iterate
    for batch in pa_reader:
      pass

    assert (
        tracker.iteration_count == 4
    ), f"Expected 4 iterations, got {tracker.iteration_count}"

  def test_read_all_iterates_all(self, small_ds):
    """read_all() should iterate through all blocks."""
    tracker = IterationTracker()

    reader = XarrayRecordBatchReader(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    pa_reader = pa.RecordBatchReader.from_stream(reader)
    table = pa_reader.read_all()

    assert tracker.iteration_count == 4
    assert len(table) == 100 * 10 * 10
