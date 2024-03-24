#!/usr/bin/env python3

import xarray as xr
import xarray_sql as qr
from dask_sql import Context


if __name__ == '__main__':
  air = xr.tutorial.open_dataset('air_temperature')
  chunks = {'time': 240, 'lat': 5, 'lon': 7}
  air = air.chunk(chunks)
  air_small = air.isel(
      time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10)
  ).chunk(chunks)

  df = qr.read_xarray(air_small)

  c = Context()
  c.create_table('air', df)

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

  result = query.compute()

  expected = air_small.dims['lat'] * air_small.dims['lon']
  assert (
      len(result) == expected
  ), f'Length must be {expected}, but was {len(result)}.'
  print(expected)
