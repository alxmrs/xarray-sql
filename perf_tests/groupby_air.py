#!/usr/bin/env python3

import xarray as xr
import qarray as qr
from dask_sql import Context


if __name__ == '__main__':
  air = xr.tutorial.open_dataset('air_temperature')
  chunks = {'time': 240, 'lat': 5, 'lon': 7}
  air = air.chunk(chunks)

  df = qr.to_dd(air)

  c = Context()
  c.create_table('air', df)

  query = c.sql('''
      SELECT
        "lat", "lon", SUM("air") as air_total
      FROM 
        "air" 
      GROUP BY
       "lat", "lon"
      ''')

  result = query.compute()

  expected = air.dims['lat'] * air.dims['lon']
  assert len(result) == expected, f'Length must be {expected}.'
  print(expected)
