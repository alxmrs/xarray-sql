import unittest

import xarray as xr
from sqlglot.executor import execute

from . import core


class DatasetTableTest(unittest.TestCase):
  def setUp(self) -> None:
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.query_10 = "select t.lat, t.lon, t.time, t.air from air t limit 10;"

  def test_executes__sanity_check(self):
    table = execute(
      self.query_10,
      tables={'air': core.XarrayDatasetTable(self.air)}
    )
    self.assertIsNotNone(table)
    self.assertEqual(10, len(table))

  def test_executes__is_consistent(self):
    # check for dictionary order...
    table = execute(
      self.query_10,
      tables={'air': core.XarrayDatasetTable(self.air)}
    )
    first_ = table

    for _ in range(100):
      table = execute(
        self.query_10,
        tables={'air': core.XarrayDatasetTable(self.air)}
      )
      self.assertEqual(table.rows, first_.rows)
      self.assertEqual(table.columns, table.columns)


if __name__ == '__main__':
  unittest.main()
