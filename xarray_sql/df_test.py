import itertools
import tracemalloc
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from .df import explode, read_xarray, block_slices, from_map, pivot, from_map_batched


def rand_wx(start: str, end: str) -> xr.Dataset:
  np.random.seed(42)
  lat = np.linspace(-90, 90, num=720)
  lon = np.linspace(-180, 180, num=1440)
  time = pd.date_range(start, end, freq='h')
  level = np.array([1000, 500], dtype=np.int32)
  reference_time = pd.Timestamp(start)
  temperature = 15 + 8 * np.random.randn(720, 1440, len(time), len(level))
  precipitation = 10 * np.random.rand(720, 1440, len(time), len(level))
  return xr.Dataset(
      data_vars=dict(
          temperature=(['lat', 'lon', 'time', 'level'], temperature),
          precipitation=(['lat', 'lon', 'time', 'level'], precipitation),
      ),
      coords=dict(
          lat=lat,
          lon=lon,
          time=time,
          level=level,
          reference_time=reference_time,
      ),
      attrs=dict(description='Random weather.'),
  )


def create_large_dataset(time_steps=1000, lat_points=100, lon_points=100):
  """Create a large xarray dataset for memory testing."""
  np.random.seed(42)

  # Create coordinates
  time = pd.date_range('2020-01-01', periods=time_steps, freq='h')
  lat = np.linspace(-90, 90, lat_points)
  lon = np.linspace(-180, 180, lon_points)

  # Create large data arrays
  temp_data = np.random.rand(time_steps, lat_points, lon_points) * 40 - 10
  precip_data = np.random.rand(time_steps, lat_points, lon_points) * 100

  return xr.Dataset(
      {
          'temperature': (['time', 'lat', 'lon'], temp_data),
          'precipitation': (['time', 'lat', 'lon'], precip_data),
      },
      coords={
          'time': time,
          'lat': lat,
          'lon': lon,
      },
  )


def adding_function(x, y):
  """Simple function that adds two values and returns a DataFrame."""
  result = pd.DataFrame({'x': [x], 'y': [y], 'sum': [x + y]})
  return result


class DaskTestCase(unittest.TestCase):

  def setUp(self) -> None:
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.chunks = {'time': 240}
    self.air = self.air.chunk(self.chunks)

    self.air_small = self.air.isel(
        time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10)
    ).chunk(self.chunks)
    self.randwx = rand_wx('1995-01-13T00', '1995-01-13T01')


class ExplodeTest(DaskTestCase):

  def test_cardinality(self):
    dss = explode(self.air)
    self.assertEqual(
        len(list(dss)), np.prod([len(c) for c in self.air.chunks.values()])
    )

  def test_dim_sizes__one(self):
    ds = next(iter(explode(self.air)))
    for k, v in self.chunks.items():
      self.assertIn(k, ds.dims)
      self.assertEqual(v, ds.sizes[k])

  def skip_test_dim_sizes__all(self):
    # TODO(alxmrs): Why is this test slow?
    dss = explode(self.air)
    self.assertEqual(
        [tuple(ds.dims.values()) for ds in dss],
        list(itertools.product(*self.air.chunksizes.values())),
    )

  def test_data_equal__one__first(self):
    ds = next(iter(explode(self.air)))
    iselection = {dim: slice(0, s) for dim, s in ds.sizes.items()}
    self.assertEqual(self.air.isel(iselection), ds)

  def test_data_equal__one__last(self):
    dss = list(explode(self.air))
    ds = dss[-1]
    iselection = {dim: slice(0, s) for dim, s in ds.sizes.items()}
    self.assertEqual(self.air.isel(iselection), ds)


class FromMapTest(unittest.TestCase):

  def test_basic_from_map(self):
    """Test basic from_map functionality with pandas DataFrames."""

    def make_df(x):
      return pd.DataFrame({'value': [x, x * 2], 'index': [0, 1]})

    result = from_map(make_df, [1, 2, 3])
    self.assertIsInstance(result, pa.Table)
    self.assertEqual(len(result), 6)  # 3 inputs * 2 rows each
    self.assertEqual(result.column_names, ['value', 'index'])

  def test_from_map_with_multiple_iterables(self):
    """Test from_map with multiple iterables."""

    def add_values(x, y):
      return pd.DataFrame({'sum': [x + y], 'x': [x], 'y': [y]})

    result = from_map(add_values, [1, 2], [10, 20])
    self.assertIsInstance(result, pa.Table)
    self.assertEqual(len(result), 2)

    # Convert to pandas to check values
    df = result.to_pandas()
    self.assertEqual(list(df['sum']), [11, 22])

  def test_from_map_with_args(self):
    """Test from_map with additional arguments."""

    def multiply_and_add(x, multiplier, add_value):
      return pd.DataFrame({'result': [x * multiplier + add_value]})

    result = from_map(multiply_and_add, [1, 2, 3], args=(2, 10))
    self.assertIsInstance(result, pa.Table)
    self.assertEqual(len(result), 3)

    df = result.to_pandas()
    self.assertEqual(
        list(df['result']), [12, 14, 16]
    )  # (1*2+10, 2*2+10, 3*2+10)

  def test_from_map_with_pyarrow_tables(self):
    """Test from_map when function returns PyArrow tables."""

    def make_arrow_table(x):
      df = pd.DataFrame({'value': [x]})
      return pa.Table.from_pandas(df)

    result = from_map(make_arrow_table, [1, 2, 3])
    self.assertIsInstance(result, pa.Table)
    self.assertEqual(len(result), 3)


class TestFromMapBatchedCorrectness(DaskTestCase):
  """Test correctness of from_map_batched function."""

  def test_basic_functionality(self):
    """Test that from_map_batched produces correct RecordBatchReader."""
    blocks = list(
        block_slices(self.air_small, chunks={'time': 4, 'lat': 3, 'lon': 4})
    )

    # Get expected schema
    first_block_df = pivot(self.air_small.isel(blocks[0]))
    expected_schema = pa.Schema.from_pandas(first_block_df)

    # Create RecordBatchReader
    reader = from_map_batched(
        pivot,
        [self.air_small.isel(block) for block in blocks],
        schema=expected_schema,
    )

    # Verify it's a RecordBatchReader
    self.assertIsInstance(reader, pa.RecordBatchReader)

    # Verify schema
    self.assertEqual(reader.schema, expected_schema)

    # Read all batches and verify content
    batches = list(reader)
    self.assertGreater(len(batches), 0)

    # Verify each batch has the correct schema
    for batch in batches:
      self.assertEqual(batch.schema, expected_schema)
      self.assertGreater(len(batch), 0)

  def test_multiple_iterables(self):
    """Test from_map_batched with multiple iterables."""
    x_values = [1, 2, 3, 4, 5]
    y_values = [10, 20, 30, 40, 50]

    expected_schema = pa.schema(
        [('x', pa.int64()), ('y', pa.int64()), ('sum', pa.int64())]
    )

    reader = from_map_batched(
        adding_function, x_values, y_values, schema=expected_schema
    )

    # Read all data
    table = reader.read_all()
    df = table.to_pandas()

    # Verify results
    expected_df = pd.DataFrame(
        {
            'x': x_values,
            'y': y_values,
            'sum': [x + y for x, y in zip(x_values, y_values)],
        }
    )

    pd.testing.assert_frame_equal(df, expected_df)

  def test_with_args_and_kwargs(self):
    """Test from_map_batched with additional args and kwargs."""

    def multiply_and_add(x, multiplier, offset=0):
      result = pd.DataFrame({'x': [x], 'result': [x * multiplier + offset]})
      return result

    values = [1, 2, 3]
    expected_schema = pa.schema([('x', pa.int64()), ('result', pa.int64())])

    reader = from_map_batched(
        multiply_and_add,
        values,
        args=(2,),  # multiplier = 2
        offset=5,  # offset = 5
        schema=expected_schema,
    )

    table = reader.read_all()
    df = table.to_pandas()

    # Verify results: x * 2 + 5
    expected_df = pd.DataFrame(
        {'x': [1, 2, 3], 'result': [7, 9, 11]}  # (1*2+5, 2*2+5, 3*2+5)
    )

    pd.testing.assert_frame_equal(df, expected_df)

  def test_empty_iterables(self):
    """Test from_map_batched with empty iterables."""
    empty_schema = pa.schema([('value', pa.int64())])

    reader = from_map_batched(
        lambda x: pd.DataFrame({'value': [x]}), [], schema=empty_schema
    )

    batches = list(reader)
    self.assertEqual(len(batches), 0)

  def test_consistency_with_regular_map(self):
    """Test that results are consistent with regular mapping."""
    blocks = list(block_slices(self.air_small, chunks={'time': 4, 'lat': 3}))
    datasets = [self.air_small.isel(block) for block in blocks]

    # Get schema from first block
    first_df = pivot(datasets[0])
    schema = pa.Schema.from_pandas(first_df)

    # Use from_map_batched
    reader = from_map_batched(pivot, datasets, schema=schema)
    batched_table = reader.read_all()

    # Regular map approach
    regular_dfs = [pivot(ds) for ds in datasets]
    regular_table = pa.Table.from_pandas(
        pd.concat(regular_dfs, ignore_index=True)
    )

    # Results should be identical
    self.assertEqual(batched_table.schema, regular_table.schema)
    self.assertEqual(len(batched_table), len(regular_table))

    # Compare data (allowing for potential column order differences)
    batched_df = (
        batched_table.to_pandas()
        .sort_values(['time', 'lat', 'lon'])
        .reset_index(drop=True)
    )
    regular_df = (
        regular_table.to_pandas()
        .sort_values(['time', 'lat', 'lon'])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(batched_df, regular_df)

  def test_integration_with_datafusion_via_read_xarray(self):
    """Test integration with the read_xarray function that uses from_map_batched."""

    # Use a small dataset
    air = xr.tutorial.open_dataset('air_temperature')
    air_small = air.isel(time=slice(0, 50), lat=slice(0, 10), lon=slice(0, 15))
    air_chunked = air_small.chunk({'time': 25, 'lat': 5, 'lon': 8})

    # read_xarray uses from_map_batched internally
    arrow_stream = read_xarray(
        air_chunked, chunks={'time': 25, 'lat': 5, 'lon': 8}
    )

    # Verify it returns a proper ArrowStreamExportable (RecordBatchReader)
    self.assertTrue(hasattr(arrow_stream, 'schema'))
    self.assertTrue(hasattr(arrow_stream, '__iter__'))

    # Should be able to read all data
    table = arrow_stream.read_all()
    self.assertGreater(len(table), 0)

    # Should have the expected columns
    expected_columns = {'time', 'lat', 'lon', 'air'}
    actual_columns = set(table.column_names)
    self.assertTrue(expected_columns.issubset(actual_columns))


class ReadXarrayStreamingTest(unittest.TestCase):

  def setUp(self):
    self.large_ds = create_large_dataset().chunk({'time': 25})

  def test_read_xarray__loads_one_chunk_at_a_time(self):
    tracemalloc.start()
    iterable = read_xarray(self.large_ds)
    first_size, first_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    sizes, peaks = [], []

    first_chunk = self.large_ds.isel(next(block_slices(self.large_ds)))
    chunk_size = first_chunk.nbytes

    # Creating the iterator should be inexpensive -- less than one chunk.
    # We multiply by constant factors because chunks have additional overhead
    self.assertLess(first_size, chunk_size * 3)
    self.assertLess(first_peak, chunk_size * 6)

    for it in iterable:
      _ = it
      cur_size, cur_peak = tracemalloc.get_traced_memory()
      tracemalloc.reset_peak()
      sizes.append(cur_size), peaks.append(cur_peak)

    mean_size = np.mean(sizes)
    mean_peak = np.mean(peaks)

    # Provide a bound for each size and peak
    for size in sizes:
      self.assertGreater(mean_size * 1.1, size)
      self.assertGreater(chunk_size * 3, size)
      # malloc size is about 2.66x chunk_size on average
      self.assertLess(chunk_size * 2, size)

    for peak in peaks:
      self.assertGreater(mean_peak * 1.1, peak)
      self.assertGreater(chunk_size * 7, peak)
      # malloc peak is about 6.89x chunk_size on average
      self.assertLess(chunk_size * 4, peak)

    # The peak malloc should never be more than the original dataset!
    self.assertLess(max(peaks), self.large_ds.nbytes)

    tracemalloc.stop()


if __name__ == '__main__':
  unittest.main()
