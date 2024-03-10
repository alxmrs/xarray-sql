import unittest


from . import Context
from .df_test import DaskTestCase


class SqlTestCase(DaskTestCase):

  def test_sanity(self):
    c = Context()
    c.create_table('air', self.air_small)

    query = c.sql('SELECT "lat", "lon", "time", "air" FROM "air" LIMIT 100')

    result = query.compute()
    self.assertIsNotNone(result)
    self.assertEqual(len(result), 100)

  def test_agg_small(self):
    c = Context()
    c.create_table('air', self.air_small)

    query = c.sql("""
    SELECT
      "lat", "lon", SUM("air") as air_total
    FROM 
      "air" 
    GROUP BY
     "lat", "lon"
    """)

    result = query.compute()
    self.assertIsNotNone(result)

    expected = self.air_small.dims['lat'] * self.air_small.dims['lon']
    self.assertEqual(len(result), expected)

  def test_agg_regular(self):
    c = Context()
    c.create_table('air', self.air)

    query = c.sql("""
    SELECT
      "lat", "lon", AVG("air") as air_total
    FROM 
      "air" 
    GROUP BY
     "lat", "lon"
    """)

    result = query.compute()
    self.assertIsNotNone(result)

    expected = self.air.dims['lat'] * self.air.dims['lon']
    self.assertEqual(len(result), expected)


if __name__ == '__main__':
  unittest.main()
