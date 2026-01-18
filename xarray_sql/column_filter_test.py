"""Tests for query-time column projection pushdown optimization.

These tests verify that xarray-sql can skip processing columns that are
not needed by a SQL query. The optimization happens at QUERY TIME, not
at table registration time:

1. User registers a table with ALL columns (no filtering required)
2. User runs a SQL query that only needs SOME columns
3. At query execution time, DataFusion tells us which columns are needed
4. We only read/pivot those columns from xarray

This is "projection pushdown" - the column selection is pushed down from
the SQL engine to the data source, enabling efficient queries without
requiring users to think about column filtering upfront.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import SessionContext

from .df import _parse_schema, pivot
from .reader import XarrayRecordBatchReader, read_xarray_table


@pytest.fixture
def multi_var_ds():
  """Create a dataset with multiple data variables for testing column filtering."""
  np.random.seed(42)
  time = pd.date_range("2020-01-01", periods=100, freq="h")
  lat = np.linspace(-90, 90, 10)
  lon = np.linspace(-180, 180, 10)

  return xr.Dataset(
      {
          "temperature": (
              ["time", "lat", "lon"],
              np.random.rand(100, 10, 10).astype(np.float32),
          ),
          "humidity": (
              ["time", "lat", "lon"],
              np.random.rand(100, 10, 10).astype(np.float32),
          ),
          "pressure": (
              ["time", "lat", "lon"],
              np.random.rand(100, 10, 10).astype(np.float32),
          ),
          "wind_speed": (
              ["time", "lat", "lon"],
              np.random.rand(100, 10, 10).astype(np.float32),
          ),
      },
      coords={"time": time, "lat": lat, "lon": lon},
  )


@pytest.fixture
def large_multi_var_ds():
  """Create a larger dataset for performance testing of column filtering."""
  np.random.seed(42)
  time = pd.date_range("2020-01-01", periods=500, freq="h")
  lat = np.linspace(-90, 90, 50)
  lon = np.linspace(-180, 180, 50)

  return xr.Dataset(
      {
          "var1": (
              ["time", "lat", "lon"],
              np.random.rand(500, 50, 50).astype(np.float32),
          ),
          "var2": (
              ["time", "lat", "lon"],
              np.random.rand(500, 50, 50).astype(np.float32),
          ),
          "var3": (
              ["time", "lat", "lon"],
              np.random.rand(500, 50, 50).astype(np.float32),
          ),
          "var4": (
              ["time", "lat", "lon"],
              np.random.rand(500, 50, 50).astype(np.float32),
          ),
          "var5": (
              ["time", "lat", "lon"],
              np.random.rand(500, 50, 50).astype(np.float32),
          ),
      },
      coords={"time": time, "lat": lat, "lon": lon},
  )


class ColumnsProcessedTracker:
  """Tracks which columns are actually processed during query execution.

  This tracker is passed to the reader and records the columns present
  in each batch that gets processed. This allows us to verify that
  projection pushdown is working - i.e., that queries for specific
  columns don't process unnecessary columns.
  """

  def __init__(self):
    self.batches_processed = 0
    self.columns_per_batch = []

  def __call__(self, batch_columns: list[str]):
    """Called with the list of column names for each batch processed."""
    self.batches_processed += 1
    self.columns_per_batch.append(list(batch_columns))

  @property
  def all_columns_seen(self) -> set[str]:
    """Return the union of all columns seen across all batches."""
    result = set()
    for cols in self.columns_per_batch:
      result.update(cols)
    return result

  def reset(self):
    self.batches_processed = 0
    self.columns_per_batch = []


class TestQueryTimeProjectionPushdown:
  """Tests for automatic query-time column projection pushdown.

  These tests verify that when a SQL query only needs specific columns,
  only those columns are read from xarray - WITHOUT the user having to
  specify anything at registration time.

  The key insight is that the user registers a table with ALL columns,
  then runs queries that may only need a subset. The optimization should
  happen transparently.
  """

  def test_select_single_column_only_processes_that_column(self, multi_var_ds):
    """Selecting one data_var should only process that column.

    When running `SELECT temperature FROM weather`, we should NOT
    process humidity, pressure, or wind_speed columns at all.
    """
    tracker = ColumnsProcessedTracker()

    # Register table with ALL columns - no filtering
    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _columns_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # Query only temperature
    result = ctx.sql("SELECT temperature FROM weather LIMIT 10").collect()

    # Verify we got results
    assert len(result) > 0
    result_df = result[0].to_pandas()
    assert "temperature" in result_df.columns

    # THE KEY ASSERTION: Only temperature (and dimension coords) were processed
    # We should NOT have processed humidity, pressure, or wind_speed
    processed_cols = tracker.all_columns_seen
    assert "temperature" in processed_cols
    assert "humidity" not in processed_cols, (
        f"humidity should not be processed for SELECT temperature query. "
        f"Processed columns: {processed_cols}"
    )
    assert "pressure" not in processed_cols
    assert "wind_speed" not in processed_cols

  def test_select_multiple_columns_only_processes_those(self, multi_var_ds):
    """Selecting multiple columns should only process those columns."""
    tracker = ColumnsProcessedTracker()

    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _columns_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # Query temperature and pressure only
    result = ctx.sql(
        "SELECT temperature, pressure FROM weather LIMIT 10"
    ).collect()

    processed_cols = tracker.all_columns_seen

    # Should have processed only temperature and pressure (plus dims)
    assert "temperature" in processed_cols
    assert "pressure" in processed_cols
    assert "humidity" not in processed_cols
    assert "wind_speed" not in processed_cols

  def test_aggregation_only_processes_needed_columns(self, multi_var_ds):
    """Aggregation queries should only process columns needed for the aggregation."""
    tracker = ColumnsProcessedTracker()

    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _columns_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # Aggregate only temperature
    result = ctx.sql("SELECT AVG(temperature) as avg_temp FROM weather").collect()

    processed_cols = tracker.all_columns_seen

    # Should have only processed temperature
    assert "temperature" in processed_cols
    assert "humidity" not in processed_cols
    assert "pressure" not in processed_cols
    assert "wind_speed" not in processed_cols

  def test_same_table_different_queries_process_different_columns(
      self, multi_var_ds
  ):
    """Same table can be queried for different columns efficiently.

    This is the key use case: register once, query many times with
    different column needs. Each query should only process what it needs.
    """
    tracker = ColumnsProcessedTracker()

    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _columns_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # First query: only temperature
    ctx.sql("SELECT AVG(temperature) FROM weather").collect()
    first_query_cols = tracker.all_columns_seen.copy()
    tracker.reset()

    # Second query: only humidity
    ctx.sql("SELECT AVG(humidity) FROM weather").collect()
    second_query_cols = tracker.all_columns_seen.copy()

    # First query should have only processed temperature
    assert "temperature" in first_query_cols
    assert "humidity" not in first_query_cols

    # Second query should have only processed humidity
    assert "humidity" in second_query_cols
    assert "temperature" not in second_query_cols

  def test_select_star_processes_all_columns(self, multi_var_ds):
    """SELECT * should process all columns (no optimization possible)."""
    tracker = ColumnsProcessedTracker()

    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _columns_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # SELECT * needs all columns
    ctx.sql("SELECT * FROM weather LIMIT 10").collect()

    processed_cols = tracker.all_columns_seen

    # All data vars should be processed
    assert "temperature" in processed_cols
    assert "humidity" in processed_cols
    assert "pressure" in processed_cols
    assert "wind_speed" in processed_cols

  def test_where_clause_with_projection(self, multi_var_ds):
    """WHERE clause should not require processing unused columns."""
    tracker = ColumnsProcessedTracker()

    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _columns_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # Query temperature with a WHERE clause on lat
    # Should only need temperature and lat (dimension), not other data_vars
    result = ctx.sql(
        "SELECT temperature FROM weather WHERE lat > 0 LIMIT 10"
    ).collect()

    processed_cols = tracker.all_columns_seen

    assert "temperature" in processed_cols
    assert "lat" in processed_cols  # Needed for WHERE clause
    assert "humidity" not in processed_cols
    assert "pressure" not in processed_cols


class TestRegistrationTimeFiltering:
  """Tests for optional registration-time column filtering.

  While query-time pushdown is the primary optimization, users may
  still want to explicitly filter columns at registration time for
  cases where they KNOW they'll never need certain columns.

  This is complementary to query-time pushdown, not a replacement.
  """

  def test_data_vars_parameter_filters_at_registration(self, multi_var_ds):
    """data_vars parameter can still be used for explicit filtering."""
    # This is the existing functionality - users can pre-filter if they want
    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        data_vars=["temperature", "pressure"],
    )

    schema = table.schema()
    column_names = [field.name for field in schema]

    # Schema should only have filtered columns
    assert len(column_names) == 5  # 3 dims + 2 data_vars
    assert "temperature" in column_names
    assert "pressure" in column_names
    assert "humidity" not in column_names

  def test_reader_data_vars_parameter(self, multi_var_ds):
    """XarrayRecordBatchReader accepts data_vars for explicit filtering."""
    reader = XarrayRecordBatchReader(
        multi_var_ds,
        chunks={"time": 25},
        data_vars=["temperature"],
    )

    column_names = [field.name for field in reader.schema]
    assert len(column_names) == 4  # 3 dims + 1 data_var
    assert "temperature" in column_names
    assert "humidity" not in column_names

  def test_invalid_data_vars_raises_error(self, multi_var_ds):
    """Invalid data_vars should raise a helpful error."""
    with pytest.raises(ValueError, match="not found in Dataset"):
      XarrayRecordBatchReader(
          multi_var_ds,
          chunks={"time": 25},
          data_vars=["nonexistent_column"],
      )


class TestCurrentBehavior:
  """Tests documenting the CURRENT behavior before projection pushdown.

  These tests verify that the current implementation processes all
  columns regardless of query needs. They serve as a baseline and
  will help verify that the optimization is working once implemented.
  """

  def test_current_behavior_processes_all_columns(self, multi_var_ds):
    """Currently, all columns are processed even when only some are needed.

    This test documents the current (suboptimal) behavior that we want
    to improve with projection pushdown.
    """
    # Use a simple tracker to count iterations
    class IterationTracker:
      def __init__(self):
        self.call_count = 0

      def __call__(self, block):
        self.call_count += 1

    tracker = IterationTracker()

    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # Query only temperature
    result = ctx.sql("SELECT temperature FROM weather LIMIT 10").collect()

    # Verify query succeeded
    assert len(result) > 0

    # Verify that iteration happened (data was processed)
    assert tracker.call_count > 0

  def test_parse_schema_includes_all_data_vars(self, multi_var_ds):
    """_parse_schema includes ALL data_vars by default."""
    schema = _parse_schema(multi_var_ds)
    column_names = [field.name for field in schema]

    assert len(column_names) == 7  # 3 dims + 4 data_vars
    assert "temperature" in column_names
    assert "humidity" in column_names
    assert "pressure" in column_names
    assert "wind_speed" in column_names

  def test_record_batch_reader_includes_all_columns(self, multi_var_ds):
    """XarrayRecordBatchReader includes ALL columns by default."""
    reader = XarrayRecordBatchReader(multi_var_ds, chunks={"time": 25})

    column_names = [field.name for field in reader.schema]
    assert len(column_names) == 7

    pa_reader = pa.RecordBatchReader.from_stream(reader)
    for batch in pa_reader:
      assert batch.num_columns == 7


class TestPreFilteringWorkaround:
  """Tests demonstrating the pre-filtering workaround.

  Until query-time projection pushdown is implemented, users can
  pre-filter their xarray Dataset before registration to achieve
  similar efficiency benefits.
  """

  def test_prefilter_dataset_before_registration(self, multi_var_ds):
    """Pre-filtering the Dataset is an effective workaround."""
    # Pre-filter to only needed variables
    filtered_ds = multi_var_ds[["temperature"]]

    table = read_xarray_table(filtered_ds, chunks={"time": 25})

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    result = ctx.sql("SELECT AVG(temperature) as avg_temp FROM weather").collect()
    result_df = result[0].to_pandas()

    assert "avg_temp" in result_df.columns
    assert 0.4 < result_df["avg_temp"].iloc[0] < 0.6

  def test_prefilter_allows_reregistration_with_different_columns(
      self, multi_var_ds
  ):
    """Different queries can use different pre-filtered tables.

    This is the workaround for the use case that query-time pushdown
    would handle automatically.
    """
    ctx = SessionContext()

    # Register table with only temperature
    temp_table = read_xarray_table(
        multi_var_ds[["temperature"]], chunks={"time": 25}
    )
    ctx.register_table_provider("weather_temp", temp_table)

    # Register another table with only humidity
    humid_table = read_xarray_table(
        multi_var_ds[["humidity"]], chunks={"time": 25}
    )
    ctx.register_table_provider("weather_humid", humid_table)

    # Query each table
    temp_result = ctx.sql(
        "SELECT AVG(temperature) as avg FROM weather_temp"
    ).collect()
    humid_result = ctx.sql(
        "SELECT AVG(humidity) as avg FROM weather_humid"
    ).collect()

    # Both should work
    assert len(temp_result) > 0
    assert len(humid_result) > 0


class TestPivotFunctionColumnFiltering:
  """Tests for column filtering at the pivot function level."""

  def test_pivot_respects_dataset_variables(self, multi_var_ds):
    """pivot() only includes variables present in the Dataset."""
    filtered_ds = multi_var_ds[["temperature"]]
    small_ds = filtered_ds.isel(
        time=slice(0, 5), lat=slice(0, 3), lon=slice(0, 3)
    )

    df = pivot(small_ds)

    assert "temperature" in df.columns
    assert "humidity" not in df.columns
    assert len(df.columns) == 4  # 3 dims + 1 data_var


class TestMemoryEfficiency:
  """Tests verifying memory efficiency of column filtering."""

  def test_filtered_batch_uses_less_memory(self, large_multi_var_ds):
    """Filtered batches should use less memory than full batches."""
    # Full dataset reader
    full_reader = XarrayRecordBatchReader(
        large_multi_var_ds, chunks={"time": 100}
    )
    full_pa_reader = pa.RecordBatchReader.from_stream(full_reader)
    full_batch = next(iter(full_pa_reader))
    full_size = full_batch.nbytes

    # Filtered dataset reader
    filtered_ds = large_multi_var_ds[["var1"]]
    filtered_reader = XarrayRecordBatchReader(filtered_ds, chunks={"time": 100})
    filtered_pa_reader = pa.RecordBatchReader.from_stream(filtered_reader)
    filtered_batch = next(iter(filtered_pa_reader))
    filtered_size = filtered_batch.nbytes

    # Filtered batch should be smaller
    assert filtered_size < full_size * 0.75
    assert filtered_size < full_size


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
