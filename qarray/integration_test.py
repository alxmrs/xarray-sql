import unittest
import xarray as xr
import qarray as qr
import pandas as pd


class MyTestCase(unittest.TestCase):

  def test_happy_path(self):
    ds = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2/'
    )

    rows = pd.read_sql()


if __name__ == '__main__':
  unittest.main()
