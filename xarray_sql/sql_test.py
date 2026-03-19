"""SQL functionality tests for xarray-sql using pytest."""

import numpy as np
import pytest
import xarray as xr

from . import XarrayContext
from .df_test import create_large_dataset, rand_wx


@pytest.fixture
def air_dataset_small():
  ds = xr.tutorial.open_dataset("air_temperature").chunk({"time": 240})
  return ds.isel(time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10))


@pytest.fixture
def air_dataset_large():
  return xr.tutorial.open_dataset("air_temperature").chunk({"time": 240})


@pytest.fixture
def weather_dataset():
  ds = rand_wx("2023-01-01T00", "2023-01-01T12")
  return ds.isel(time=slice(0, 6), lat=slice(0, 10), lon=slice(0, 10)).chunk(
      {"time": 3}
  )


@pytest.fixture
def synthetic_dataset():
  return create_large_dataset(
      time_steps=50, lat_points=20, lon_points=20
  ).chunk({"time": 25})


@pytest.fixture
def station_dataset():
  return xr.Dataset(
      {
          "station_id": (["station"], [1, 2, 3, 4, 5]),
          "elevation": (["station"], [100, 250, 500, 750, 1000]),
          "name": (
              ["station"],
              ["Station_A", "Station_B", "Station_C", "Station_D", "Station_E"],
          ),
      }
  ).chunk({"station": 5})


@pytest.fixture
def air_and_stations():
  air = (
      xr.tutorial.open_dataset("air_temperature")
      .isel(time=slice(0, 12), lat=slice(0, 5), lon=slice(0, 8))
      .chunk({"time": 6})
  )
  stations = xr.Dataset(
      {
          "station_id": (["station"], [101, 102, 103]),
          "lat": (
              ["station"],
              [air.lat.values[0], air.lat.values[2], air.lat.values[4]],
          ),
          "lon": (
              ["station"],
              [air.lon.values[1], air.lon.values[3], air.lon.values[5]],
          ),
          "elevation": (["station"], [100, 250, 500]),
      }
  ).chunk({"station": 3})
  return air, stations


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


class TestCftimeGregorianLike:
  """Tests for Gregorian-like cftime calendars (noleap, standard, etc.).

  These use pa.timestamp('us') and support string-based SQL filters.
  """

  @pytest.fixture
  def rasm_ds(self):
    """The rasm tutorial dataset uses cftime.DatetimeNoLeap (noleap)."""
    return xr.tutorial.open_dataset("rasm")

  def test_noleap_dataset_registers(self, rasm_ds):
    """A noleap dataset should register without errors."""
    ctx = XarrayContext()
    ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
    result = ctx.sql("SELECT COUNT(*) AS cnt FROM rasm").to_pandas()
    assert result["cnt"].iloc[0] > 0

  def test_select_time_column(self, rasm_ds):
    """Querying the time column should return valid timestamps."""
    ctx = XarrayContext()
    ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
    result = ctx.sql(
        "SELECT DISTINCT time FROM rasm ORDER BY time LIMIT 5"
    ).to_pandas()
    assert len(result) == 5
    times = result["time"].tolist()
    assert times == sorted(times)

  def test_string_filter_works(self, rasm_ds):
    """String-based time filters should work for Gregorian-like calendars."""
    ctx = XarrayContext()
    ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
    result = ctx.sql(
        "SELECT COUNT(*) AS cnt FROM rasm WHERE time >= '1980-10-01'"
    ).to_pandas()
    full = ctx.sql("SELECT COUNT(*) AS cnt FROM rasm").to_pandas()
    assert 0 < result["cnt"].iloc[0] < full["cnt"].iloc[0]

  def test_aggregation(self, rasm_ds):
    """MIN/MAX on timestamp columns should work."""
    ctx = XarrayContext()
    ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
    result = ctx.sql(
        'SELECT MIN(time) AS t_min, MAX(time) AS t_max FROM rasm'
    ).to_pandas()
    assert result["t_min"].iloc[0] < result["t_max"].iloc[0]

  def test_row_count_matches_xarray(self, rasm_ds):
    """Total row count should equal the product of dimension sizes."""
    ctx = XarrayContext()
    ctx.from_dataset("rasm", rasm_ds, chunks={"time": 12})
    result = ctx.sql("SELECT COUNT(*) AS cnt FROM rasm").to_pandas()
    expected = int(np.prod([
        rasm_ds.sizes[d] for d in rasm_ds["Tair"].dims
    ]))
    assert result["cnt"].iloc[0] == expected


class TestCftimeNonGregorian:
  """Tests for non-Gregorian cftime calendars (360_day, julian).

  These use pa.int64() with CF-convention metadata and the cftime() UDF.
  """

  @pytest.fixture
  def ds_360day(self):
    """Synthetic 360-day calendar dataset."""
    import cftime
    times = [cftime.Datetime360Day(2000, m, 1) for m in range(1, 13)]
    return xr.Dataset(
        {"temp": ("time", np.arange(12, dtype=np.float32))},
        coords={"time": times},
    )

  def test_360day_registers(self, ds_360day):
    """A 360-day dataset should register without errors."""
    ctx = XarrayContext()
    ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
    result = ctx.sql("SELECT COUNT(*) AS cnt FROM ds360").to_pandas()
    assert result["cnt"].iloc[0] == 12

  def test_360day_select_ordered(self, ds_360day):
    """Integer offsets should be orderable."""
    ctx = XarrayContext()
    ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
    result = ctx.sql(
        "SELECT DISTINCT time FROM ds360 ORDER BY time"
    ).to_pandas()
    times = result["time"].tolist()
    assert times == sorted(times)
    assert len(times) == 12

  def test_360day_integer_filter(self, ds_360day):
    """Direct integer comparisons should work on int64 time columns."""
    ctx = XarrayContext()
    ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
    # Get all distinct time values to find a midpoint
    all_times = ctx.sql(
        "SELECT DISTINCT time FROM ds360 ORDER BY time"
    ).to_pandas()["time"].tolist()
    mid = all_times[len(all_times) // 2]
    result = ctx.sql(
        f"SELECT COUNT(*) AS cnt FROM ds360 WHERE time >= {mid}"
    ).to_pandas()
    assert 0 < result["cnt"].iloc[0] < 12

  def test_360day_cftime_udf_registered(self, ds_360day):
    """from_dataset should auto-register a cftime() UDF for 360-day calendars."""
    ctx = XarrayContext()
    ctx.from_dataset("ds360", ds_360day, chunks={"time": 6})
    # The cftime() UDF should convert a date string to the int64 offset,
    # enabling ergonomic filtering.
    result = ctx.sql(
        "SELECT COUNT(*) AS cnt FROM ds360 "
        "WHERE time >= cftime('2000-07-01')"
    ).to_pandas()
    # July through December = 6 months
    assert result["cnt"].iloc[0] == 6

  def test_gregorian_like_no_cftime_udf(self):
    """Gregorian-like calendars should NOT register a cftime() UDF."""
    ds = xr.tutorial.open_dataset("rasm")
    ctx = XarrayContext()
    ctx.from_dataset("rasm", ds, chunks={"time": 12})
    # Using cftime() should fail since it's not registered for noleap.
    with pytest.raises(Exception):
      ctx.sql(
          "SELECT COUNT(*) FROM rasm WHERE time >= cftime('1980-01-01')"
      ).collect()
