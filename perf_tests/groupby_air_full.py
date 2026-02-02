#!/usr/bin/env python3

import xarray as xr
import xarray_sql as xql
from datafusion import SessionContext

if __name__ == "__main__":
  air = xr.tutorial.open_dataset("air_temperature")
  chunks = {"time": 240}
  air = air.chunk(chunks)

  df = xql.read_xarray_table(air)

  ctx = SessionContext()
  ctx.register_table("air", df)

  query = ctx.sql(
      """
      SELECT
        "lat", "lon", SUM("air") as air_total
      FROM
        "air"
      GROUP BY
       "lat", "lon"
      """
  )

  result = query.collect()

  expected = air.sizes["lat"] * air.sizes["lon"]
  actual = sum(len(batch) for batch in result)

  assert actual == expected, f"Length must be {expected}, but was {actual}."
  print(expected)
