#!/usr/bin/env python3

import xarray as xr
import qarray as qr

if __name__ == '__main__':
  air = xr.tutorial.open_dataset('air_temperature')
  chunks = {'time': 240}
  air = air.chunk(chunks)

  df = qr.to_dd(air).compute()

  print(len(df))