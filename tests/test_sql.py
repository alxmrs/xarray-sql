"""SQL functionality tests for xarray-sql using pytest."""

import pytest
import xarray as xr

from xarray_sql import XarrayContext


def test_sanity(air_dataset_small):
  ctx = XarrayContext()
  ctx.from_dataset("air", air_dataset_small)
  result = ctx.sql(
      'SELECT "lat", "lon", "time", "air" FROM "air" LIMIT 100'
  ).to_pandas()
  assert len(result) > 0
  assert len(result) <= 1320
  assert all(col in result.columns for col in ["lat", "lon", "time", "air"])


def test_aggregation_small(air_dataset_small):
  ctx = XarrayContext()
  ctx.from_dataset("air", air_dataset_small)
  query = """
        SELECT lat, lon, SUM(air) AS air_total
        FROM air
        GROUP BY lat, lon
    """
  result = ctx.sql(query).to_pandas()
  expected_rows = (
      air_dataset_small.sizes["lat"] * air_dataset_small.sizes["lon"]
  )
  assert len(result) == expected_rows


def test_aggregation_large(air_dataset_large):
  ctx = XarrayContext()
  ctx.from_dataset("air", air_dataset_large)
  query = """
        SELECT lat, lon, AVG(air) AS air_avg
        FROM air
        GROUP BY lat, lon
    """
  result = ctx.sql(query).to_pandas()
  expected_rows = (
      air_dataset_large.sizes["lat"] * air_dataset_large.sizes["lon"]
  )
  assert len(result) == expected_rows


def test_basic_select_all(air_dataset_small):
  ctx = XarrayContext()
  ctx.from_dataset("air", air_dataset_small)
  result = ctx.sql("SELECT * FROM air LIMIT 10").to_pandas()
  assert len(result) <= 10
  for col in ["lat", "lon", "time", "air"]:
    assert col in result.columns


def test_weather_queries(weather_dataset):
  ctx = XarrayContext()
  ctx.from_dataset("weather", weather_dataset)
  # Selecting specific columns
  result = ctx.sql(
      "SELECT lat, lon, temperature, precipitation FROM weather LIMIT 20"
  ).to_pandas()
  assert "temperature" in result.columns
  assert "precipitation" in result.columns
  # Filtering
  result = ctx.sql(
      "SELECT * FROM weather WHERE temperature > 10 LIMIT 50"
  ).to_pandas()
  assert len(result) > 0
  assert (result["temperature"] > 10).all()


def test_synthetic_aggregations(synthetic_dataset):
  ctx = XarrayContext()
  ctx.from_dataset("synthetic", synthetic_dataset)
  # COUNT aggregation
  result = ctx.sql("SELECT COUNT(*) AS total_count FROM synthetic").to_pandas()
  assert result["total_count"].iloc[0] > 0
  # MIN, MAX, AVG
  query = """
        SELECT MIN(temperature) AS min_temp,
               MAX(temperature) AS max_temp,
               AVG(temperature) AS avg_temp
        FROM synthetic
    """
  result = ctx.sql(query).to_pandas()
  assert result["min_temp"].iloc[0] < result["max_temp"].iloc[0]
  assert (
      result["min_temp"].iloc[0]
      <= result["avg_temp"].iloc[0]
      <= result["max_temp"].iloc[0]
  )


def test_invalid_table_name(air_dataset_small):
  ctx = XarrayContext()
  ctx.from_dataset("air", air_dataset_small)
  with pytest.raises(Exception):
    ctx.sql("SELECT * FROM nonexistent_table")


def test_invalid_column_name(air_dataset_small):
  ctx = XarrayContext()
  ctx.from_dataset("air", air_dataset_small)
  with pytest.raises(Exception):
    ctx.sql("SELECT nonexistent_column FROM air")


def test_sql_syntax_error(air_dataset_small):
  ctx = XarrayContext()
  ctx.from_dataset("air", air_dataset_small)
  with pytest.raises(Exception):
    ctx.sql("SELECT * FORM air")  # Typo: FORM instead of FROM
  with pytest.raises(Exception):
    ctx.sql("SELECT * FROM air WHERE")  # Incomplete WHERE


def test_cross_join(air_and_stations):
  air, stations = air_and_stations
  ctx = XarrayContext()
  ctx.from_dataset("air_data", air)
  ctx.from_dataset("stations", stations)
  result = ctx.sql(
      "SELECT COUNT(*) AS total FROM air_data CROSS JOIN stations"
  ).to_pandas()
  assert result["total"].iloc[0] > 0
