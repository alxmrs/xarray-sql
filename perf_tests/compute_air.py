#!/usr/bin/env python3

import xarray as xr
import xarray_sql as xql

if __name__ == "__main__":
  air = xr.tutorial.open_dataset("air_temperature")
  chunks = {"time": 240}
  air = air.chunk(chunks)

  df = xql.read_xarray(air).read_pandas()

  print(len(df))
