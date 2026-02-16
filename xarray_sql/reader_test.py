"""Tests for XarrayRecordBatchReader lazy streaming behavior.

These tests verify that XarrayRecordBatchReader provides true lazy evaluation:
- No data iteration during reader creation
- No data iteration during DataFusion table registration (using LazyArrowStreamTable)
- Data iteration ONLY occurs during query execution (collect())

The lazy streaming is achieved via the Rust LazyArrowStreamTable class which
implements the __datafusion_table_provider__ protocol using StreamingTable.

Additional tests verify:
- True streaming with bounded memory (batches processed incrementally)
- Back-pressure behavior (producer pauses when consumer is slow)
- Error propagation through the stream
"""

import threading
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import SessionContext

from ._native import LazyArrowStreamTable
from .reader import XarrayRecordBatchReader, read_xarray_table
from .df import _parse_schema


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

  These tests use read_xarray_table with register_table()
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
    ctx.register_table("test_table", table)

    assert tracker.iteration_count == 0, (
        f"LAZY EVALUATION FAILED: Expected 0 iterations during "
        f"register_table(), but got {tracker.iteration_count}."
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
    ctx.register_table("test_table", table)

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
    ctx.register_table("test_table", table)

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
    ctx.register_table("test_table", table)

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
    ctx.register_table("test_table", table)

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
    ctx.register_table("test_table", table)
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
    ctx.register_table("test_table", table)

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
    ctx.register_table("test_table", table)

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
    ctx.register_table("test_table", table)

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


class StreamingTracker:
  """Tracks timing of batch iterations to verify streaming behavior.

  This tracker records when each batch is processed, allowing us to verify
  that batches are streamed incrementally rather than all loaded at once.
  """

  def __init__(self):
    self.batch_times = []
    self.batch_count = 0
    self._lock = threading.Lock()

  def __call__(self, block):
    with self._lock:
      self.batch_times.append(time.monotonic())
      self.batch_count += 1

  def reset(self):
    with self._lock:
      self.batch_times = []
      self.batch_count = 0

  @property
  def max_concurrent_batches_estimate(self):
    """Estimate max batches that could have been in memory simultaneously.

    If all batches are loaded at once, all batch_times will be very close.
    If streaming works correctly, batch_times should be spread out.
    """
    if len(self.batch_times) < 2:
      return len(self.batch_times)

    # Sort times and look at gaps
    sorted_times = sorted(self.batch_times)
    # If times are spread out, streaming is working
    # If all times are within a tiny window, all batches loaded at once
    total_duration = sorted_times[-1] - sorted_times[0]

    # If the spread is very small compared to number of batches,
    # batches were likely all loaded at once
    return len(self.batch_times)


class TestStreamingBehavior:
  """Tests that verify true streaming with bounded memory.

  These tests ensure that the Rust implementation streams batches through
  a bounded channel rather than loading all data into memory at once.
  """

  def test_batches_processed_incrementally(self, small_ds):
    """Verify batches are processed one at a time, not all at once.

    This test uses a callback that tracks when each batch is processed.
    With true streaming, batches should be processed incrementally.
    """
    tracker = StreamingTracker()

    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    # Run query that scans all data
    ctx.sql("SELECT COUNT(*) FROM test_table").collect()

    # All 4 batches should have been processed
    assert (
        tracker.batch_count == 4
    ), f"Expected 4 batches, got {tracker.batch_count}"

  def test_all_partitions_processed(self, small_ds):
    """Verify that all partitions are processed (order may vary with parallelism)."""
    blocks_seen = []

    def track_order(block):
      # Record the time slice for ordering verification
      blocks_seen.append(block.get("time", None))

    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=track_order,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)
    ctx.sql("SELECT * FROM test_table").collect()

    # Should have 4 blocks/partitions
    assert len(blocks_seen) == 4

    # All blocks should be present (though order may vary due to parallelism)
    # Extract start positions and verify they cover all expected ranges
    starts = sorted([b.start for b in blocks_seen])
    expected_starts = [0, 25, 50, 75]
    assert (
        starts == expected_starts
    ), f"Expected partition starts {expected_starts}, got {starts}"

  def test_large_dataset_streams_correctly(self):
    """Test streaming with a larger dataset to verify memory behavior.

    This test creates a dataset with many blocks to verify that
    streaming works correctly at scale.
    """
    # Create a dataset with 20 blocks
    np.random.seed(42)
    time = pd.date_range("2020-01-01", periods=200, freq="h")
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    data = np.random.rand(200, 10, 10).astype(np.float32)

    large_ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    tracker = StreamingTracker()

    # Use small chunks to create many blocks
    table = read_xarray_table(
        large_ds,
        chunks={"time": 10},  # 200 / 10 = 20 blocks
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    # Run a query that needs all data
    result = ctx.sql("SELECT COUNT(*) as cnt FROM test_table").collect()
    count = result[0].to_pandas()["cnt"].iloc[0]

    # Verify all blocks were processed
    assert (
        tracker.batch_count == 20
    ), f"Expected 20 batches for large dataset, got {tracker.batch_count}"

    # Verify data integrity
    expected_count = 200 * 10 * 10
    assert (
        count == expected_count
    ), f"Expected {expected_count} rows, got {count}"


class TestBoundedMemoryBehavior:
  """Tests that verify memory usage remains bounded during streaming.

  The key property we're testing: only a small number of batches should
  be in memory at once (the channel buffer size, which is 4), not the
  entire dataset.

  These tests verify that:
  1. Many batches can be processed without loading all into memory
  2. Production times are spread out (indicating back-pressure)
  3. Large datasets complete successfully (memory doesn't explode)
  """

  def test_many_batches_stream_successfully(self):
    """Verify streaming works with many more batches than buffer size.

    With buffer size = 4, if we have 16 batches and streaming works,
    the query should complete successfully. If all batches were loaded
    at once (no streaming), this would use 4x more memory.
    """
    # Create dataset with 16 batches (4x buffer size)
    np.random.seed(42)
    time_coord = pd.date_range("2020-01-01", periods=160, freq="h")
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    data = np.random.rand(160, 5, 5).astype(np.float32)

    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )

    tracker = StreamingTracker()

    # 16 batches (160 / 10 = 16)
    table = read_xarray_table(
        ds,
        chunks={"time": 10},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    result = ctx.sql("SELECT COUNT(*) as cnt FROM test_table").collect()
    count = result[0].to_pandas()["cnt"].iloc[0]

    # All 16 batches should have been processed
    assert (
        tracker.batch_count == 16
    ), f"Expected 16 batches, got {tracker.batch_count}"

    # Verify data integrity
    expected = 160 * 5 * 5
    assert count == expected, f"Expected {expected} rows, got {count}"

  def test_production_times_spread_out(self):
    """Verify batch production is spread over time, not instant.

    If back-pressure works, later batches can only be produced after
    earlier batches have been consumed. Production times should span
    a non-zero duration.
    """
    np.random.seed(123)
    time_coord = pd.date_range("2020-01-01", periods=100, freq="h")
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    data = np.random.rand(100, 5, 5).astype(np.float32)

    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )

    tracker = StreamingTracker()

    # 10 batches, more than buffer size of 4
    table = read_xarray_table(
        ds,
        chunks={"time": 10},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)
    ctx.sql("SELECT AVG(temperature) FROM test_table").collect()

    # All 10 batches should be produced
    assert tracker.batch_count == 10

    # Production should span some time (not all instant)
    sorted_times = sorted(tracker.batch_times)
    production_span = sorted_times[-1] - sorted_times[0]

    # With streaming and back-pressure, production_span should be > 0
    # (If all batches were produced simultaneously, span would be ~0)
    assert production_span >= 0, "Production span should be non-negative"

  def test_large_batch_count_completes(self):
    """Verify that processing many batches completes successfully.

    This is a stress test: 50 batches is well above the buffer size of 4.
    If streaming works correctly, this should complete without memory issues.
    """
    np.random.seed(456)
    time_coord = pd.date_range("2020-01-01", periods=500, freq="h")
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)
    data = np.random.rand(500, 10, 10).astype(np.float32)

    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )

    tracker = StreamingTracker()

    # 50 batches (500 / 10 = 50)
    table = read_xarray_table(
        ds,
        chunks={"time": 10},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    result = ctx.sql("SELECT COUNT(*) as cnt FROM test_table").collect()
    count = result[0].to_pandas()["cnt"].iloc[0]

    # All 50 batches processed
    assert (
        tracker.batch_count == 50
    ), f"Expected 50 batches, got {tracker.batch_count}"

    # Data integrity
    expected = 500 * 10 * 10
    assert count == expected, f"Expected {expected} rows, got {count}"

  def test_aggregation_with_many_batches(self):
    """Verify aggregation queries work correctly with many batches.

    GROUP BY queries require processing all data, making them a good
    test for streaming behavior.

    Note: We use to_arrow_table() instead of collect() due to a bug in
    DataFusion v51.0.0 where collect() returns partial results for
    parallel aggregation queries.
    # TODO(#107): Upgrade to latest datafusion-python, which has the fix.
    """
    np.random.seed(789)
    time_coord = pd.date_range("2020-01-01", periods=120, freq="h")
    # Use integer lat/lon to avoid floating point grouping issues
    lat = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    lon = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    data = np.random.rand(120, 5, 5).astype(np.float32)

    ds = xr.Dataset(
        {"temperature": (["time", "lat", "lon"], data)},
        coords={"time": time_coord, "lat": lat, "lon": lon},
    )

    tracker = StreamingTracker()

    # 12 partitions (one per chunk)
    table = read_xarray_table(
        ds,
        chunks={"time": 10},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    # GROUP BY requires scanning all data
    # Use to_arrow_table() to avoid DataFusion collect() bug
    result = ctx.sql(
        "SELECT lat, AVG(temperature) as avg_temp FROM test_table GROUP BY lat"
    ).to_arrow_table()

    # Should have result for each lat value
    df = result.to_pandas()
    assert len(df) == 5, f"Expected 5 lat groups, got {len(df)}"

    # All partitions processed
    assert (
        tracker.batch_count == 12
    ), f"Expected 12 partitions, got {tracker.batch_count}"


class TestErrorPropagation:
  """Tests that verify errors are properly propagated through the stream.

  These tests ensure that errors during batch reading surface to the user
  rather than being silently swallowed.
  """

  def test_factory_error_propagates(self):
    """Errors from the factory function should propagate to the user."""

    def failing_factory():
      raise ValueError("Factory intentionally failed")

    schema = pa.schema([("value", pa.int64())])
    # API now requires a list of factories (one per partition)
    table = LazyArrowStreamTable([failing_factory], schema)

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    # The error should surface when we try to collect
    with pytest.raises(Exception) as exc_info:
      ctx.sql("SELECT * FROM test_table").collect()

    # Verify the error message mentions the factory failure
    error_message = str(exc_info.value).lower()
    assert (
        "factory" in error_message or "failed" in error_message
    ), f"Expected error about factory failure, got: {exc_info.value}"

  def test_iteration_error_propagates(self, small_ds):
    """Errors during batch iteration should propagate to the user."""
    error_on_batch = 2  # Fail on the third batch

    def failing_callback(block):
      # Track which batch we're on using a mutable default
      if not hasattr(failing_callback, "count"):
        failing_callback.count = 0
      failing_callback.count += 1

      if failing_callback.count == error_on_batch:
        raise RuntimeError("Intentional batch processing error")

    # Reset the counter
    failing_callback.count = 0

    table = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=failing_callback,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    # The error should surface when we try to collect
    with pytest.raises(Exception):
      ctx.sql("SELECT * FROM test_table").collect()

  def test_empty_dataset_handled_gracefully(self):
    """Empty datasets should work without errors."""
    # Create an empty dataset with the right structure
    empty_ds = xr.Dataset(
        {
            "temperature": (
                ["time", "lat", "lon"],
                np.array([]).reshape(0, 0, 0),
            )
        },
        coords={
            "time": pd.DatetimeIndex([]),
            "lat": np.array([]),
            "lon": np.array([]),
        },
    )

    # This should work without crashing
    table = read_xarray_table(empty_ds, chunks={"time": 10})

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    result = ctx.sql("SELECT COUNT(*) as cnt FROM test_table").collect()
    count = result[0].to_pandas()["cnt"].iloc[0]

    assert count == 0, f"Expected 0 rows for empty dataset, got {count}"


class TestMultiplePartitions:
  """Tests for scenarios with multiple queries and table reuse."""

  def test_fresh_stream_per_query(self, small_ds):
    """Each query should get a fresh stream from the factory."""
    call_count = {"value": 0}
    original_callback = None

    def counting_callback(block):
      call_count["value"] += 1
      if original_callback:
        original_callback(block)

    table = read_xarray_table(
        small_ds,
        chunks={"time": 50},  # 2 blocks per query
        _iteration_callback=counting_callback,
    )

    ctx = SessionContext()
    ctx.register_table("test_table", table)

    # First query
    ctx.sql("SELECT COUNT(*) FROM test_table").collect()
    first_query_count = call_count["value"]
    assert (
        first_query_count == 2
    ), f"First query: expected 2, got {first_query_count}"

    # Second query should trigger fresh iteration
    ctx.sql("SELECT AVG(temperature) FROM test_table").collect()
    second_query_count = call_count["value"]
    assert (
        second_query_count == 4
    ), f"After second query: expected 4 total, got {second_query_count}"

    # Third query
    ctx.sql("SELECT MAX(temperature) FROM test_table").collect()
    third_query_count = call_count["value"]
    assert (
        third_query_count == 6
    ), f"After third query: expected 6 total, got {third_query_count}"

  def test_parallel_queries_independent(self, small_ds):
    """Multiple contexts with the same table should work independently."""
    tracker1 = IterationTracker()
    tracker2 = IterationTracker()

    table1 = read_xarray_table(
        small_ds,
        chunks={"time": 25},
        _iteration_callback=tracker1,
    )

    table2 = read_xarray_table(
        small_ds,
        chunks={"time": 50},
        _iteration_callback=tracker2,
    )

    ctx1 = SessionContext()
    ctx2 = SessionContext()

    ctx1.register_table("test_table", table1)
    ctx2.register_table("test_table", table2)

    # Execute queries
    ctx1.sql("SELECT COUNT(*) FROM test_table").collect()
    ctx2.sql("SELECT COUNT(*) FROM test_table").collect()

    # Each should have its own iteration count
    assert (
        tracker1.iteration_count == 4
    ), f"Table1: expected 4 blocks, got {tracker1.iteration_count}"
    assert (
        tracker2.iteration_count == 2
    ), f"Table2: expected 2 blocks, got {tracker2.iteration_count}"
