import unittest

from . import XarrayContext
from .df_test import DaskTestCase


class SqlTestCase(DaskTestCase):

  def test_sanity(self):
    c = XarrayContext()
    c.from_dataset('air', self.air_small)

    query = c.sql('SELECT "lat", "lon", "time", "air" FROM "air" LIMIT 100')

    result = query.to_pandas()
    self.assertIsNotNone(result)
    self.assertLessEqual(len(result), 1320)  # Should be all rows or less
    self.assertGreater(len(result), 0)  # Should have some rows

  def test_agg_small(self):
    c = XarrayContext()
    c.from_dataset('air', self.air_small)

    query = c.sql(
        """
  SELECT
    "lat", "lon", SUM("air") as air_total
  FROM 
    "air" 
  GROUP BY
   "lat", "lon"
  """
    )

    result = query.to_pandas()
    self.assertIsNotNone(result)

    expected = self.air_small.sizes['lat'] * self.air_small.sizes['lon']
    self.assertEqual(len(result), expected)

  def test_agg_regular(self):
    c = XarrayContext()
    c.from_dataset('air', self.air)

    query = c.sql(
        """
    SELECT
      "lat", "lon", AVG("air") as air_total
    FROM 
      "air" 
    GROUP BY
     "lat", "lon"
    """
    )

    result = query.to_pandas()
    self.assertIsNotNone(result)

    expected = self.air.sizes['lat'] * self.air.sizes['lon']
    self.assertEqual(len(result), expected)


if __name__ == '__main__':
  unittest.main()
