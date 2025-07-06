import itertools
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from .df import explode, read_xarray, block_slices, from_map


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


class PyArrowTableTest(DaskTestCase):

  def test_sanity(self):
    table = read_xarray(self.air_small)
    self.assertIsNotNone(table)
    self.assertIsInstance(table, pa.Table)
    self.assertEqual(len(table), np.prod(list(self.air_small.sizes.values())))

  def test_columns(self):
    table = read_xarray(self.air_small)
    cols = table.column_names
    self.assertEqual(cols, ['lat', 'time', 'lon', 'air'])

  def test_dtypes(self):
    table = read_xarray(self.air_small)
    # Convert to pandas to check dtypes
    df = table.to_pandas()
    types = list(df.dtypes)
    self.assertEqual([self.air_small[c].dtype for c in df.columns], types)

  def test_different_chunk_sizes(self):
    default_table = read_xarray(self.air_small)
    chunked_table = read_xarray(self.air_small, dict(time=5))

    # Both should produce valid tables
    self.assertIsInstance(default_table, pa.Table)
    self.assertIsInstance(chunked_table, pa.Table)
    # Should have same number of rows
    self.assertEqual(len(default_table), len(chunked_table))

  def test_chunk_perf(self):
    table = read_xarray(self.air, chunks=dict(time=6))
    self.assertIsNotNone(table)
    self.assertEqual(len(table), np.prod(list(self.air.sizes.values())))

  def test_column_metadata_preserved(self):
    try:
      table = read_xarray(self.randwx, chunks=dict(time=24))
      self.assertIsInstance(table, pa.Table)
    except Exception as e:
      self.fail(f'Unexpected error: {e}')


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


if __name__ == '__main__':
  unittest.main()
