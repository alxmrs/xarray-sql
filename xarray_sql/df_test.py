import itertools
import unittest

import dask.dataframe as dd
import numpy as np
import xarray as xr

from .df import explode, read_xarray, block_slices


class DaskTestCase(unittest.TestCase):

  def setUp(self) -> None:
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.chunks = {'time': 240}
    self.air = self.air.chunk(self.chunks)

    self.air_small = self.air.isel(
        time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10)
    ).chunk(self.chunks)


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
      self.assertEqual(v, ds.dims[k])

  def skip_test_dim_sizes__all(self):
    # TODO(alxmrs): Why is this test slow?
    dss = explode(self.air)
    self.assertEqual(
        [tuple(ds.dims.values()) for ds in dss],
        list(itertools.product(*self.air.chunksizes.values())),
    )

  def test_data_equal__one__first(self):
    ds = next(iter(explode(self.air)))
    iselection = {dim: slice(0, s) for dim, s in ds.dims.items()}
    self.assertEqual(self.air.isel(iselection), ds)

  def test_data_equal__one__last(self):
    dss = list(explode(self.air))
    ds = dss[-1]
    iselection = {dim: slice(0, s) for dim, s in ds.dims.items()}
    self.assertEqual(self.air.isel(iselection), ds)


class DaskDataframeTest(DaskTestCase):

  def test_sanity(self):
    df = read_xarray(self.air_small).compute()
    self.assertIsNotNone(df)
    self.assertEqual(len(df), np.prod(list(self.air_small.dims.values())))

  def test_columns(self):
    df = read_xarray(self.air_small).compute()
    cols = list(df.columns)
    self.assertEqual(cols, ['lat', 'time', 'lon', 'air'])

  def test_dtypes(self):
    df: dd.DataFrame = read_xarray(self.air_small).compute()
    types = list(df.dtypes)
    self.assertEqual([self.air_small[c].dtype for c in df.columns], types)

  def test_partitions_dont_match_dataset_chunks(self):
    standard_blocks = list(block_slices(self.air_small))
    default: dd.DataFrame = read_xarray(self.air_small)
    chunked: dd.DataFrame = read_xarray(self.air_small, dict(time=5))

    self.assertEqual(default.npartitions, len(standard_blocks))
    self.assertNotEqual(chunked.npartitions, len(standard_blocks))

  def test_chunk_perf(self):
    df = read_xarray(self.air, chunks=dict(time=6)).compute()
    self.assertIsNotNone(df)
    self.assertEqual(len(df), np.prod(list(self.air.dims.values())))


if __name__ == '__main__':
  unittest.main()
