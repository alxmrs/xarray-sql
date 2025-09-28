"""DataFrame functionality tests for xarray-sql."""

import itertools
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import xarray as xr

from .df import explode, read_xarray, block_slices, from_map, pivot, from_map_batched


# -------------------------
# Helper Dataset Functions
# -------------------------
def rand_wx(start: str, end: str) -> xr.Dataset:
  np.random.seed(42)
  lat = np.linspace(-90, 90, 720)
  lon = np.linspace(-180, 180, 1440)
  time = pd.date_range(start, end, freq="h")
  level = np.array([1000, 500], dtype=np.int32)
  temperature = 15 + 8 * np.random.randn(720, 1440, len(time), len(level))
  precipitation = 10 * np.random.rand(720, 1440, len(time), len(level))

  return xr.Dataset(
      data_vars=dict(
          temperature=(["lat", "lon", "time", "level"], temperature),
          precipitation=(["lat", "lon", "time", "level"], precipitation),
      ),
      coords=dict(lat=lat, lon=lon, time=time, level=level),
      attrs=dict(description="Random weather."),
  )


def create_large_dataset(time_steps=1000, lat_points=100, lon_points=100):
  """Create a large xarray dataset for memory testing."""
  np.random.seed(42)
  time = pd.date_range("2020-01-01", periods=time_steps, freq="h")
  lat = np.linspace(-90, 90, lat_points)
  lon = np.linspace(-180, 180, lon_points)
  temperature = np.random.rand(time_steps, lat_points, lon_points) * 40 - 10
  precipitation = np.random.rand(time_steps, lat_points, lon_points) * 100

  return xr.Dataset(
      {
          "temperature": (["time", "lat", "lon"], temperature),
          "precipitation": (["time", "lat", "lon"], precipitation),
      },
      coords={"time": time, "lat": lat, "lon": lon},
  )


def adding_function(x, y):
  """Simple function returning a DataFrame with sum."""
  return pd.DataFrame({"x": [x], "y": [y], "sum": [x + y]})


# -------------------------
# Pytest Fixtures
# -------------------------
@pytest.fixture
def air_dataset():
  ds = xr.tutorial.open_dataset("air_temperature")
  return ds.chunk({"time": 240})


@pytest.fixture
def air_dataset_small():
  ds = xr.tutorial.open_dataset("air_temperature").chunk({"time": 240})
  return ds.isel(time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10))


@pytest.fixture
def randwx_dataset():
  return rand_wx("1995-01-13T00", "1995-01-13T01")


# -------------------------
# Test Classes
# -------------------------
class TestExplode:

  @pytest.mark.unit
  def test_cardinality(self, air_dataset):
    dss = list(explode(air_dataset))
    expected_chunks = np.prod([len(c) for c in air_dataset.chunks.values()])
    assert len(dss) == expected_chunks

  @pytest.mark.unit
  def test_dim_sizes_one(self, air_dataset):
    ds = next(iter(explode(air_dataset)))
    for dim, size in {"time": 240}.items():
      assert dim in ds.dims
      assert ds.sizes[dim] == size

  @pytest.mark.unit
  def test_data_equal_one_first(self, air_dataset):
    ds = next(iter(explode(air_dataset)))
    iselection = {dim: slice(0, s) for dim, s in ds.sizes.items()}
    assert air_dataset.isel(iselection).equals(ds)

  @pytest.mark.unit
  def test_data_equal_one_last(self, air_dataset):
    dss = list(explode(air_dataset))
    ds = dss[-1]
    iselection = {dim: slice(0, s) for dim, s in ds.sizes.items()}
    assert air_dataset.isel(iselection).equals(ds)


class TestFromMap:

  @pytest.mark.unit
  def test_basic_from_map(self):
    def make_df(x):
      return pd.DataFrame({"value": [x, x * 2], "index": [0, 1]})

    result = from_map(make_df, [1, 2, 3])
    assert isinstance(result, pa.Table)
    assert len(result) == 6
    assert result.column_names == ["value", "index"]

  @pytest.mark.unit
  def test_from_map_with_multiple_iterables(self):
    def add_values(x, y):
      return pd.DataFrame({"sum": [x + y], "x": [x], "y": [y]})

    result = from_map(add_values, [1, 2], [10, 20])
    df = result.to_pandas()
    assert list(df["sum"]) == [11, 22]

  @pytest.mark.unit
  def test_from_map_with_args(self):
    def multiply_and_add(x, multiplier, add_value):
      return pd.DataFrame({"result": [x * multiplier + add_value]})

    result = from_map(multiply_and_add, [1, 2, 3], args=(2, 10))
    df = result.to_pandas()
    assert list(df["result"]) == [12, 14, 16]

  @pytest.mark.unit
  def test_from_map_with_pyarrow_tables(self):
    def make_arrow_table(x):
      df = pd.DataFrame({"value": [x]})
      return pa.Table.from_pandas(df)

    result = from_map(make_arrow_table, [1, 2, 3])
    assert isinstance(result, pa.Table)
    assert len(result) == 3


class TestFromMapBatchedCorrectness:

  @pytest.mark.unit
  def test_basic_functionality(self, air_dataset_small):
    blocks = list(
        block_slices(air_dataset_small, chunks={"time": 4, "lat": 3, "lon": 4})
    )
    expected_schema = pa.Schema.from_pandas(
        pivot(air_dataset_small.isel(blocks[0]))
    )
    reader = from_map_batched(
        pivot,
        [air_dataset_small.isel(block) for block in blocks],
        schema=expected_schema,
    )
    assert isinstance(reader, pa.RecordBatchReader)
    assert reader.schema == expected_schema
    for batch in reader:
      assert batch.schema == expected_schema
      assert len(batch) > 0

  @pytest.mark.unit
  def test_multiple_iterables(self):
    x_values = [1, 2, 3]
    y_values = [10, 20, 30]
    expected_schema = pa.schema(
        [("x", pa.int64()), ("y", pa.int64()), ("sum", pa.int64())]
    )
    reader = from_map_batched(
        adding_function, x_values, y_values, schema=expected_schema
    )
    df = reader.read_all().to_pandas()
    expected_df = pd.DataFrame(
        {
            "x": x_values,
            "y": y_values,
            "sum": [x + y for x, y in zip(x_values, y_values)],
        }
    )
    pd.testing.assert_frame_equal(df, expected_df)


class TestReadXarrayStreaming:

  @pytest.mark.unit
  def test_memory_efficiency(self):
    large_ds = create_large_dataset().chunk({"time": 25})
    tracemalloc.start()
    iterator = read_xarray(large_ds)
    first_size, first_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Note: Additional memory checks can be added here
    assert first_size < large_ds.nbytes * 2
