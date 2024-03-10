#!/usr/bin/env python3

import xarray as xr
import xarray_sql as qr

if __name__ == '__main__':
  air = xr.tutorial.open_dataset('air_temperature')
  chunks = {'time': 240, 'lat': 5, 'lon': 7}

  air_small = air.isel(
      time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10)
  ).chunk(chunks)

  df = qr.read_xarray(air_small).compute()

  print(len(df))
