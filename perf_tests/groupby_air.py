#!/usr/bin/env python3

from datafusion import SessionContext
import xarray as xr
import xarray_sql as xql


if __name__ == "__main__":
  air = xr.tutorial.open_dataset("air_temperature")
  chunks = {"time": 240, "lat": 5, "lon": 7}
  air = air.chunk(chunks)
  air_small = air.isel(
      time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10)
  ).chunk(chunks)

  df = xql.read_xarray_table(air_small)

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

  expected = air_small.sizes["lat"] * air_small.sizes["lon"]
  actual = sum(len(batch) for batch in result)
  assert actual == expected, f"Length must be {expected}, but was {actual}."
  print(expected)
