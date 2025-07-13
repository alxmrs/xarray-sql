import itertools
import unittest

import numpy as np

from tests.conftest import DaskTestCase
from xarray_sql.core import explode


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

  @unittest.skip("Test is slow")
  def test_dim_sizes__all(self):
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
