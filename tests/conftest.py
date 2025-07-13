import unittest

import numpy as np
import pandas as pd
import xarray as xr


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
