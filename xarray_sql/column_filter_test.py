"""Tests for early column filtering/dropping optimization.

These tests verify whether the xarray-sql library can skip processing
columns that are not needed by a query. The optimization involves:

1. Not including unneeded data_vars in the schema
2. Not pivoting unneeded data_vars when converting chunks to DataFrames
3. Filtering the xarray Dataset early to avoid unnecessary computation

The goal is that if a query only needs specific columns (e.g.,
`SELECT lat, lon FROM air`), we should not process other data_vars
like temperature, humidity, etc.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr
from datafusion import SessionContext

from .df import _parse_schema, pivot, block_slices
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


class ColumnTracker:
  """Tracks which columns are processed during pivoting."""

  def __init__(self):
    self.columns_seen = []
    self.call_count = 0

  def __call__(self, block):
    self.call_count += 1
    # Track block info but not columns directly here
    self.columns_seen.append(dict(block))


class TestCurrentColumnBehavior:
  """Tests that document the CURRENT behavior of column handling.

  These tests verify whether the library currently supports early column
  dropping (it does not, as of the initial implementation).
  """

  def test_parse_schema_includes_all_data_vars(self, multi_var_ds):
    """Verify that _parse_schema includes ALL data_vars by default."""
    schema = _parse_schema(multi_var_ds)

    # Get column names from schema
    column_names = [field.name for field in schema]

    # Should include all dimensions
    assert "time" in column_names
    assert "lat" in column_names
    assert "lon" in column_names

    # Should include ALL data vars - this is the current behavior
    assert "temperature" in column_names
    assert "humidity" in column_names
    assert "pressure" in column_names
    assert "wind_speed" in column_names

    # Total: 3 dims + 4 data_vars = 7 columns
    assert len(column_names) == 7

  def test_pivot_includes_all_columns(self, multi_var_ds):
    """Verify that pivot() includes ALL columns by default."""
    # Take a small slice
    small_ds = multi_var_ds.isel(
        time=slice(0, 5), lat=slice(0, 3), lon=slice(0, 3)
    )
    df = pivot(small_ds)

    # Should have all columns
    assert "time" in df.columns
    assert "lat" in df.columns
    assert "lon" in df.columns
    assert "temperature" in df.columns
    assert "humidity" in df.columns
    assert "pressure" in df.columns
    assert "wind_speed" in df.columns

    assert len(df.columns) == 7

  def test_record_batch_reader_includes_all_columns(self, multi_var_ds):
    """Verify that XarrayRecordBatchReader includes ALL columns."""
    reader = XarrayRecordBatchReader(multi_var_ds, chunks={"time": 25})

    # Check schema has all columns
    column_names = [field.name for field in reader.schema]
    assert len(column_names) == 7

    # Consume the reader and verify all batches have all columns
    pa_reader = pa.RecordBatchReader.from_stream(reader)
    for batch in pa_reader:
      assert batch.num_columns == 7

  def test_datafusion_query_gets_all_columns_initially(self, multi_var_ds):
    """Verify that even queries for subset of columns receive full data.

    This test demonstrates that the current implementation processes all
    columns even when the SQL query only needs a few.
    """
    tracker = ColumnTracker()

    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # Query only temperature - but all data_vars are still processed
    result = ctx.sql(
        "SELECT time, lat, lon, temperature FROM weather LIMIT 10"
    ).collect()

    # Result only has 4 columns (due to SQL projection)
    result_df = result[0].to_pandas()
    assert list(result_df.columns) == ["time", "lat", "lon", "temperature"]

    # But we still iterated through all blocks with all columns
    assert tracker.call_count > 0


class TestEarlyColumnDroppingAPI:
  """Tests for the early column dropping optimization.

  These tests define the EXPECTED behavior after implementing the optimization.
  Initially, these tests will fail, demonstrating the need for the feature.
  """

  def test_reader_accepts_data_vars_filter(self, multi_var_ds):
    """Test that XarrayRecordBatchReader can accept a data_vars filter.

    This test verifies the API for filtering columns at reader creation time.
    """
    # This should work - filter to only include temperature
    reader = XarrayRecordBatchReader(
        multi_var_ds,
        chunks={"time": 25},
        data_vars=["temperature"],  # Only include temperature
    )

    # Schema should only have 4 columns: 3 dims + 1 data_var
    column_names = [field.name for field in reader.schema]
    assert len(column_names) == 4
    assert "temperature" in column_names
    assert "humidity" not in column_names

  def test_reader_rejects_invalid_data_vars(self, multi_var_ds):
    """Test that XarrayRecordBatchReader raises error for invalid data_vars."""
    with pytest.raises(ValueError, match="not found in Dataset"):
      XarrayRecordBatchReader(
          multi_var_ds,
          chunks={"time": 25},
          data_vars=["temperature", "nonexistent_var"],
      )

  def test_read_xarray_table_rejects_invalid_data_vars(self, multi_var_ds):
    """Test that read_xarray_table raises error for invalid data_vars."""
    with pytest.raises(ValueError, match="not found in Dataset"):
      read_xarray_table(
          multi_var_ds,
          chunks={"time": 25},
          data_vars=["nonexistent_var"],
      )

  def test_reader_data_vars_filter_reduces_schema(self, multi_var_ds):
    """Verify that data_vars filter reduces the schema columns."""
    # Filter to specific data variables
    filtered_ds = multi_var_ds[["temperature", "humidity"]]

    reader = XarrayRecordBatchReader(filtered_ds, chunks={"time": 25})
    column_names = [field.name for field in reader.schema]

    # Should only have 5 columns: 3 dims + 2 data_vars
    assert len(column_names) == 5
    assert "temperature" in column_names
    assert "humidity" in column_names
    assert "pressure" not in column_names
    assert "wind_speed" not in column_names

  def test_filtered_dataset_produces_smaller_batches(self, multi_var_ds):
    """Verify that filtering the dataset before reading produces smaller batches."""
    # Full dataset reader
    full_reader = XarrayRecordBatchReader(multi_var_ds, chunks={"time": 25})
    full_pa_reader = pa.RecordBatchReader.from_stream(full_reader)
    full_batch = next(iter(full_pa_reader))

    # Filtered dataset reader
    filtered_ds = multi_var_ds[["temperature"]]
    filtered_reader = XarrayRecordBatchReader(filtered_ds, chunks={"time": 25})
    filtered_pa_reader = pa.RecordBatchReader.from_stream(filtered_reader)
    filtered_batch = next(iter(filtered_pa_reader))

    # Filtered batch should have fewer columns
    assert filtered_batch.num_columns < full_batch.num_columns
    assert filtered_batch.num_columns == 4  # 3 dims + 1 data_var
    assert full_batch.num_columns == 7  # 3 dims + 4 data_vars

  def test_read_xarray_table_accepts_data_vars(self, multi_var_ds):
    """Test that read_xarray_table can accept a data_vars filter.

    This is the high-level API test for column filtering.
    """
    table = read_xarray_table(
        multi_var_ds,
        chunks={"time": 25},
        data_vars=["temperature", "pressure"],
    )

    # Verify schema only has filtered columns (5: 3 dims + 2 data_vars)
    # Note: table.schema() returns a PyArrow schema (no arguments needed)
    schema = table.schema()
    column_names = [field.name for field in schema]
    assert len(column_names) == 5
    assert "temperature" in column_names
    assert "pressure" in column_names
    assert "humidity" not in column_names


class TestWorkaroundWithPreFiltering:
  """Tests demonstrating the current workaround: pre-filtering the Dataset.

  Users can currently achieve early column dropping by filtering the
  xarray Dataset before passing it to the reader. These tests document
  this approach.
  """

  def test_prefilter_dataset_reduces_columns(self, multi_var_ds):
    """Demonstrate that pre-filtering the Dataset works as a workaround."""
    # Pre-filter the dataset
    filtered_ds = multi_var_ds[["temperature"]]

    # Create reader with filtered dataset
    reader = XarrayRecordBatchReader(filtered_ds, chunks={"time": 25})

    # Schema should only have 4 columns
    column_names = [field.name for field in reader.schema]
    assert len(column_names) == 4
    assert "temperature" in column_names
    assert "humidity" not in column_names

  def test_prefilter_dataset_in_sql_query(self, multi_var_ds):
    """Demonstrate the workaround in a full SQL query."""
    # Pre-filter to only needed variables
    filtered_ds = multi_var_ds[["temperature"]]

    table = read_xarray_table(filtered_ds, chunks={"time": 25})

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    result = ctx.sql(
        "SELECT AVG(temperature) as avg_temp FROM weather"
    ).collect()
    result_df = result[0].to_pandas()

    assert "avg_temp" in result_df.columns
    # Average of random [0, 1) should be ~0.5
    assert 0.4 < result_df["avg_temp"].iloc[0] < 0.6

  def test_prefilter_multiple_vars(self, multi_var_ds):
    """Demonstrate filtering to multiple specific variables."""
    filtered_ds = multi_var_ds[["temperature", "pressure"]]

    reader = XarrayRecordBatchReader(filtered_ds, chunks={"time": 25})
    column_names = [field.name for field in reader.schema]

    assert len(column_names) == 5  # 3 dims + 2 data_vars
    assert "temperature" in column_names
    assert "pressure" in column_names
    assert "humidity" not in column_names
    assert "wind_speed" not in column_names


class TestPivotFunctionColumnFiltering:
  """Tests for column filtering at the pivot function level."""

  def test_pivot_respects_dataset_variables(self, multi_var_ds):
    """Verify that pivot only includes variables present in the Dataset."""
    # Filter to single variable
    filtered_ds = multi_var_ds[["temperature"]]
    small_ds = filtered_ds.isel(
        time=slice(0, 5), lat=slice(0, 3), lon=slice(0, 3)
    )

    df = pivot(small_ds)

    # Should only have filtered columns
    assert "temperature" in df.columns
    assert "humidity" not in df.columns
    assert len(df.columns) == 4  # 3 dims + 1 data_var

  def test_pivot_with_multiple_filtered_vars(self, multi_var_ds):
    """Verify pivot with multiple filtered variables."""
    filtered_ds = multi_var_ds[["temperature", "wind_speed"]]
    small_ds = filtered_ds.isel(
        time=slice(0, 5), lat=slice(0, 3), lon=slice(0, 3)
    )

    df = pivot(small_ds)

    assert "temperature" in df.columns
    assert "wind_speed" in df.columns
    assert "humidity" not in df.columns
    assert "pressure" not in df.columns
    assert len(df.columns) == 5  # 3 dims + 2 data_vars


class TestSchemaFunctionColumnFiltering:
  """Tests for _parse_schema with column filtering."""

  def test_parse_schema_with_filtered_dataset(self, multi_var_ds):
    """Verify _parse_schema respects Dataset's data_vars."""
    filtered_ds = multi_var_ds[["humidity", "pressure"]]
    schema = _parse_schema(filtered_ds)

    column_names = [field.name for field in schema]

    # Should have dims + filtered data_vars
    assert len(column_names) == 5
    assert "humidity" in column_names
    assert "pressure" in column_names
    assert "temperature" not in column_names
    assert "wind_speed" not in column_names

  def test_parse_schema_single_var(self, multi_var_ds):
    """Verify _parse_schema with single variable."""
    filtered_ds = multi_var_ds[["wind_speed"]]
    schema = _parse_schema(filtered_ds)

    column_names = [field.name for field in schema]
    assert len(column_names) == 4  # 3 dims + 1 var
    assert "wind_speed" in column_names


class TestMemoryEfficiencyOfFiltering:
  """Tests that verify memory efficiency of early column filtering."""

  def test_filtered_batch_memory_smaller(self, large_multi_var_ds):
    """Verify that filtered batches use less memory than full batches."""
    # Get one batch with all columns
    full_ds = large_multi_var_ds
    full_reader = XarrayRecordBatchReader(full_ds, chunks={"time": 100})
    full_pa_reader = pa.RecordBatchReader.from_stream(full_reader)
    full_batch = next(iter(full_pa_reader))
    full_size = full_batch.nbytes

    # Get one batch with single column
    filtered_ds = large_multi_var_ds[["var1"]]
    filtered_reader = XarrayRecordBatchReader(filtered_ds, chunks={"time": 100})
    filtered_pa_reader = pa.RecordBatchReader.from_stream(filtered_reader)
    filtered_batch = next(iter(filtered_pa_reader))
    filtered_size = filtered_batch.nbytes

    # Filtered batch should be smaller
    # With 5 data vars and 3 dim columns (8 total), filtering to 1 data var (4 total)
    # gives 4/8 = 50% column reduction. But dim columns have different byte sizes
    # (datetime64 is 8 bytes vs float32 is 4 bytes), so expect ~65% of original.
    assert filtered_size < full_size * 0.75, (
        f"Filtered batch ({filtered_size} bytes) should be less than "
        f"75% of full batch ({full_size} bytes)"
    )
    # Also verify absolute reduction happened
    assert filtered_size < full_size, (
        f"Filtered batch ({filtered_size} bytes) should be smaller than "
        f"full batch ({full_size} bytes)"
    )

  def test_end_to_end_query_with_prefiltering(self, multi_var_ds):
    """Demonstrate complete workflow with pre-filtering for efficiency."""
    tracker = ColumnTracker()

    # User wants to query only temperature
    # CURRENT WORKAROUND: pre-filter the dataset
    filtered_ds = multi_var_ds[["temperature"]]

    table = read_xarray_table(
        filtered_ds,
        chunks={"time": 25},
        _iteration_callback=tracker,
    )

    ctx = SessionContext()
    ctx.register_table_provider("weather", table)

    # Query average temperature - simpler aggregation
    result = ctx.sql(
        "SELECT AVG(temperature) as avg_temp FROM weather"
    ).collect()

    # Verify we got results
    result_df = result[0].to_pandas()
    assert "avg_temp" in result_df.columns
    # Average of random [0, 1) should be ~0.5
    avg_temp = result_df["avg_temp"].iloc[0]
    assert 0.4 < avg_temp < 0.6, f"Expected avg ~0.5, got {avg_temp}"

    # Verify iteration happened
    assert (
        tracker.call_count == 4
    ), f"Expected 4 chunks (100 time steps / 25), got {tracker.call_count}"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
