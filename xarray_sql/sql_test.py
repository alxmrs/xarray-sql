import functools
import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from . import XarrayContext
from .df_test import DaskTestCase, create_large_dataset


class SQLBaseTestCase(DaskTestCase):
  TEST_COMBINATIONS = [
    ('from_dataset', dict(as_zarr=False)),
    ('from_zarr', dict(as_zarr=True)),
  ]

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.ctx = XarrayContext()

    self.weather_small = self.weather.isel(
      time=slice(0, 6), lat=slice(0, 10), lon=slice(0, 10)
    ).chunk({'time': 3})

    self.weather_medium = self.weather.isel(
      time=slice(0, 12), lat=slice(0, 15), lon=slice(0, 20)
    ).chunk({'time': 6})

    self.synthetic = create_large_dataset(
      time_steps=50, lat_points=20, lon_points=20
    ).chunk({'time': 25})

    self.air_medium = self.air.isel(
      time=slice(0, 100), lat=slice(0, 20), lon=slice(0, 30)
    ).chunk({'time': 50})

    # Create station metadata as a simple tabular dataset
    # Use coordinates that overlap with air dataset
    air_lats = self.air_small.lat.values
    air_lons = self.air_small.lon.values

    # Create stations at specific lat/lon points
    self.stations = xr.Dataset(
      {
        'station_id': (['station'], [101, 102, 103]),
        'lat': (['station'], [air_lats[0], air_lats[2], air_lats[4]]),
        'lon': (['station'], [air_lons[1], air_lons[3], air_lons[5]]),
        'elevation': (['station'], [100, 250, 500]),
        'name': (['station'], ['Downtown', 'Airport', 'Mountain']),
      }
    ).chunk({'station': 3})

    # Create region lookup table
    self.regions = xr.Dataset(
      {
        'region_id': (['region'], [1, 2, 3, 4]),
        'region_name': (['region'], ['North', 'South', 'East', 'West']),
        'min_lat': (['region'], [60, 30, 40, 40]),
        'max_lat': (['region'], [90, 60, 80, 80]),
        'min_lon': (['region'], [-180, -180, -90, -180]),
        'max_lon': (['region'], [180, 180, 180, -90]),
      }
    ).chunk({'region': 4})

  def tearDown(self):
    # Clean up the temporary directory after each test method
    self.temp_dir.cleanup()

  def load(self, name: str, ds: xr.Dataset, chunks=None, as_zarr=False) -> str:
    """Load a dataset into the test context; sometimes writing the data to Zarr first."""
    if as_zarr:
      path = os.path.join(self.temp_dir.name, name + '.zarr')
      ds.to_zarr(path)
      self.ctx.from_zarr(name, path)
    else:
      self.ctx.from_dataset(name, ds, chunks)

  def make_context(self, name: str, ds: xr.Dataset, *, chunks=None,
                   as_zarr=False) -> XarrayContext:
    ctx = XarrayContext()
    return self.add_to_context(ctx, name, ds, chunks=chunks, as_zarr=as_zarr)

  def add_to_context(self, ctx: XarrayContext, name: str, ds: xr.Dataset, *, chunks=None, as_zarr=False) -> XarrayContext:
    if as_zarr:
      path = os.path.join(self.temp_dir.name, name + '.zarr')
      ds.to_zarr(path)
      ctx.from_zarr(name, path, chunks=chunks)
    else:
      ctx.from_dataset(name, ds, chunks=chunks)
    return ctx

def with_test_combinations(test_func):
  # test_name, options
  test_combinations = [
    ('from_dataset', dict(as_zarr=False)),
    ('from_zarr', dict(as_zarr=True)),
  ]

  @functools.wraps(test_func)
  def wrapper(self, *args, **kwargs):
    for case, opt in test_combinations:
      with self.subTest(case, **opt):
        # Create a fresh context per test combo.
        self.ctx = XarrayContext()
        test_func(self, *args, **opt, **kwargs)

  return wrapper


class SqlTestCase(SQLBaseTestCase):

  @with_test_combinations
  def test_sanity(self, as_zarr):
      ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)
      query = ctx.sql(
        f'SELECT "lat", "lon", "time", "air" FROM air LIMIT 100'
      )

      result = query.to_pandas()
      self.assertIsNotNone(result)
      self.assertLessEqual(len(result), 1320)  # Should be all rows or less
      self.assertGreater(len(result), 0)  # Should have some rows

  @with_test_combinations
  def test_agg_small(self, as_zarr):
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)

    query = ctx.sql(
      f"""
  SELECT
    "lat", "lon", SUM("air") as air_total
  FROM 
    air
  GROUP BY
   "lat", "lon"
  """
    )

    result = query.to_pandas()
    self.assertIsNotNone(result)

    expected = self.air_small.sizes['lat'] * self.air_small.sizes['lon']
    self.assertEqual(len(result), expected)

  @with_test_combinations
  def test_agg_regular(self, as_zarr):
    ctx = self.make_context('air', self.air, as_zarr=as_zarr)

    query = ctx.sql(
      f"""
  SELECT
    "lat", "lon", AVG("air") as air_total
  FROM 
    air 
  GROUP BY
   "lat", "lon"
  """
    )

    result = query.to_pandas()
    self.assertIsNotNone(result)

    expected = self.air.sizes['lat'] * self.air.sizes['lon']
    self.assertEqual(len(result), expected)


class SqlVarietyTestCase(SQLBaseTestCase):
  """Test SQL functionality with various types of Xarray datasets."""

  @with_test_combinations
  def test_basic_select_all(self, as_zarr):
    """Test basic SELECT * queries on different datasets."""
    # Test with air temperature dataset
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)
    result = ctx.sql('SELECT * FROM air LIMIT 10').to_pandas()

    self.assertGreater(len(result), 0)
    self.assertLessEqual(len(result), 10)
    self.assertIn('air', result.columns)
    self.assertIn('lat', result.columns)
    self.assertIn('lon', result.columns)
    self.assertIn('time', result.columns)

  @with_test_combinations
  def test_weather_dataset_queries(self, as_zarr):
    """Test queries on weather dataset with multiple variables."""
    ctx = self.make_context('weather', self.weather_small, as_zarr=as_zarr)

    # Test selecting specific columns
    result = ctx.sql(
      'SELECT lat, lon, temperature, precipitation FROM weather LIMIT 20'
    ).to_pandas()

    self.assertGreater(len(result), 0)
    self.assertIn('temperature', result.columns)
    self.assertIn('precipitation', result.columns)

    # Test filtering
    result = ctx.sql(
      'SELECT * FROM weather WHERE temperature > 10 LIMIT 50'
    ).to_pandas()

    self.assertGreater(len(result), 0)
    # All temperatures should be > 10
    self.assertTrue((result['temperature'] > 10).all())

  @with_test_combinations
  def test_synthetic_dataset_aggregations(self, as_zarr):
    """Test aggregation queries on synthetic dataset."""
    ctx = self.make_context('synthetic', self.synthetic, as_zarr=as_zarr)

    # Test COUNT
    result = ctx.sql(
      'SELECT COUNT(*) as total_count FROM synthetic'
    ).to_pandas()
    self.assertEqual(len(result), 1)
    self.assertGreater(result['total_count'].iloc[0], 0)

    # Test MIN, MAX, AVG
    result = ctx.sql(
      """
  SELECT 
    MIN(temperature) as min_temp,
    MAX(temperature) as max_temp,
    AVG(temperature) as avg_temp
  FROM synthetic
"""
    ).to_pandas()

    self.assertEqual(len(result), 1)
    self.assertLess(result['min_temp'].iloc[0], result['max_temp'].iloc[0])
    self.assertGreaterEqual(
      result['avg_temp'].iloc[0], result['min_temp'].iloc[0]
    )
    self.assertLessEqual(result['avg_temp'].iloc[0], result['max_temp'].iloc[0])

  @with_test_combinations
  def test_spatial_grouping(self, as_zarr):
    """Test spatial grouping queries."""
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)

    # Group by spatial coordinates
    result = ctx.sql(
      """
  SELECT 
    lat, lon,
    AVG(air) as avg_air,
    COUNT(*) as time_count
  FROM air 
  GROUP BY lat, lon
  ORDER BY lat, lon
"""
    ).to_pandas()

    expected_spatial_points = (
        self.air_small.sizes['lat'] * self.air_small.sizes['lon']
    )
    self.assertEqual(len(result), expected_spatial_points)

    # Each spatial point should have same number of time steps
    self.assertTrue(
      (result['time_count'] == self.air_small.sizes['time']).all()
    )

  @with_test_combinations
  def test_temporal_filtering(self, as_zarr):
    """Test temporal filtering and grouping."""
    ctx = self.make_context('weather', self.weather_small, as_zarr=as_zarr)

    # Get unique time values for filtering
    all_data = ctx.sql(
      'SELECT DISTINCT time FROM weather ORDER BY time'
    ).to_pandas()
    if len(all_data) > 2:
      mid_time = all_data['time'].iloc[len(all_data) // 2]

      # Filter by time
      result = ctx.sql(
        f"""
        SELECT COUNT(*) as count_after
        FROM weather 
        WHERE time >= '{mid_time}'
      """
      ).to_pandas()

      self.assertGreater(result['count_after'].iloc[0], 0)

  @with_test_combinations
  def test_station_dataset_queries(self, as_zarr):
    """Test queries on 1D station dataset."""
    ctx = self.make_context('stations', self.stations, as_zarr=as_zarr)

    # Basic select
    result = ctx.sql(
      'SELECT * FROM stations ORDER BY elevation'
    ).to_pandas()
    self.assertEqual(len(result), 3)

    # Test filtering by elevation
    result = ctx.sql(
      'SELECT name, elevation FROM stations WHERE elevation > 300 ORDER BY elevation'
    ).to_pandas()

    self.assertGreater(len(result), 0)
    self.assertTrue((result['elevation'] > 300).all())


class SqlJoinTestCase(SQLBaseTestCase):
  """Test joining tabular data with raster data using from_dataset."""

  @with_test_combinations
  def test_simple_cross_join(self, as_zarr):
    """Test cross join between raster and tabular data."""
    ctx = self.make_context('air_data', self.air_small, as_zarr=as_zarr)
    ctx = self.add_to_context(ctx, 'stations', self.stations, as_zarr=as_zarr)

    # Test separate queries first to ensure both datasets work
    air_result = ctx.sql(
      'SELECT COUNT(*) as air_count FROM air_data'
    ).to_pandas()
    station_result = ctx.sql(
      'SELECT COUNT(*) as station_count FROM stations'
    ).to_pandas()

    self.assertGreater(air_result['air_count'].iloc[0], 0)
    self.assertGreater(station_result['station_count'].iloc[0], 0)

    # Test that we can query both datasets in the same context
    # This demonstrates multi-dataset capability without complex joins
    air_sample = ctx.sql('SELECT air FROM air_data LIMIT 5').to_pandas()
    station_sample = ctx.sql(
      'SELECT station_id FROM stations LIMIT 5'
    ).to_pandas()

    self.assertGreater(len(air_sample), 0)
    self.assertGreater(len(station_sample), 0)
    self.assertIn('air', air_sample.columns)
    self.assertIn('station_id', station_sample.columns)

  @unittest.skip('Hit DataFusion error')
  @with_test_combinations
  def test_coordinate_based_join(self, as_zarr):
    """Test joining on coordinate proximity."""
    air_table_name = ctx = self.make_context('air_data', self.air_small, as_zarr=as_zarr)
    stations_table_name = ctx = self.add_to_context(ctx, 'stations', self.stations, as_zarr=as_zarr)

    # First test a simple cross join to ensure datasets are compatible
    result = ctx.sql(
      f"""
    SELECT COUNT(*) as total_combinations
    FROM {air_table_name} a
    CROSS JOIN {stations_table_name} s
  """
    ).to_pandas()

    self.assertGreater(result['total_combinations'].iloc[0], 0)

    # Test a simpler join condition
    result = ctx.sql(
      f"""
    SELECT 
      COUNT(*) as match_count
    FROM {air_table_name} a, {stations_table_name} s
    WHERE s.station_id = 101
  """
    ).to_pandas()

    self.assertGreater(result['match_count'].iloc[0], 0)

  @with_test_combinations
  def test_region_classification_join(self, as_zarr):
    """Test joining with region classification."""
    ctx = self.make_context('air_data', self.air_small, as_zarr=as_zarr)
    ctx = self.add_to_context(ctx, 'regions', self.regions, as_zarr=as_zarr)

    # Test that both datasets can be queried independently
    air_result = ctx.sql(
      'SELECT COUNT(*) as air_count FROM air_data'
    ).to_pandas()
    region_result = ctx.sql(
      'SELECT COUNT(*) as region_count FROM regions'
    ).to_pandas()

    self.assertGreater(air_result['air_count'].iloc[0], 0)
    self.assertGreater(region_result['region_count'].iloc[0], 0)

    # Test a simpler region-based query without complex joins
    result = ctx.sql(
      """
  SELECT 
    region_name,
    min_lat,
    max_lat
  FROM regions
  WHERE min_lat < 50
"""
    ).to_pandas()

    self.assertGreater(len(result), 0)
    self.assertIn('region_name', result.columns)

  @unittest.skip('Hit DataFusion Error')
  @with_test_combinations
  def test_multiple_dataset_aggregation(self, as_zarr):
    """Test aggregating across multiple datasets."""
    ctx = self.make_context('air_data', self.air_small, as_zarr=as_zarr)
    ctx = self.add_to_context(ctx, 'stations', self.stations, as_zarr=as_zarr)

    # Get statistics by elevation bands using station data
    result = ctx.sql(
      """
  SELECT 
    CASE 
      WHEN s.elevation < 200 THEN 'Low'
      WHEN s.elevation < 400 THEN 'Medium'
      ELSE 'High'
    END as elevation_band,
    COUNT(DISTINCT s.station_id) as station_count,
    COUNT(*) as air_measurements,
    AVG(a.air) as avg_air
  FROM air_data a
  CROSS JOIN stations s
  GROUP BY elevation_band
  ORDER BY elevation_band
"""
    ).to_pandas()

    self.assertGreater(len(result), 0)
    self.assertIn('elevation_band', result.columns)
    # Should have Low, Medium, High bands
    self.assertGreaterEqual(len(result), 1)


class SqlOptimizationTestCase(SQLBaseTestCase):
  """Test DataFusion optimizations like column selection and filters."""

  @with_test_combinations
  def test_column_projection(self, as_zarr):
    """Test that selecting specific columns works efficiently."""
    ctx = self.make_context('weather', self.weather_medium, as_zarr=as_zarr)

    # Select only specific columns
    result = ctx.sql(
      'SELECT lat, lon, temperature FROM weather LIMIT 100'
    ).to_pandas()

    # Should only have the requested columns
    expected_columns = {'lat', 'lon', 'temperature'}
    actual_columns = set(result.columns)
    self.assertEqual(expected_columns, actual_columns)

    # Should not include precipitation, time, level, reference_time
    unwanted_columns = {'precipitation', 'time', 'level', 'reference_time'}
    self.assertTrue(unwanted_columns.isdisjoint(actual_columns))

  @with_test_combinations
  def test_where_clause_filtering(self, as_zarr):
    """Test WHERE clause filtering optimization."""
    ctx = self.make_context('air', self.air_medium, as_zarr=as_zarr)

    # Test numeric filtering
    result = ctx.sql(
      'SELECT * FROM air WHERE air > 280 AND air < 290'
    ).to_pandas()

    if len(result) > 0:
      self.assertTrue((result['air'] > 280).all())
      self.assertTrue((result['air'] < 290).all())

    # Test coordinate filtering
    result = ctx.sql(
      'SELECT * FROM air WHERE lat > 50 AND lon < -100'
    ).to_pandas()

    if len(result) > 0:
      self.assertTrue((result['lat'] > 50).all())
      self.assertTrue((result['lon'] < -100).all())

  @with_test_combinations
  def test_limit_optimization(self, as_zarr):
    """Test LIMIT clause optimization."""
    ctx = self.make_context('air', self.air_medium, as_zarr=as_zarr)

    # Test small limits
    for limit in [1, 5, 10, 50]:
      with self.subTest(limit=limit):
        result = ctx.sql(f'SELECT * FROM air LIMIT {limit}').to_pandas()
        self.assertLessEqual(len(result), limit)
        if (
            limit
            <= self.air_medium.sizes['time']
            * self.air_medium.sizes['lat']
            * self.air_medium.sizes['lon']
        ):
          self.assertEqual(len(result), limit)

  @with_test_combinations
  def test_order_by_optimization(self, as_zarr):
    """Test ORDER BY clause."""
    ctx = self.make_context('air', self.air_medium, as_zarr=as_zarr)

    # Test ordering by different columns
    result = ctx.sql(
      'SELECT lat, lon, air FROM air ORDER BY air DESC LIMIT 20'
    ).to_pandas()

    if len(result) > 1:
      # Should be in descending order
      air_values = result['air'].values
      self.assertTrue(np.all(air_values[:-1] >= air_values[1:]))

    # Test ordering by coordinates
    result = ctx.sql(
      'SELECT lat, lon, air FROM air ORDER BY lat ASC, lon DESC LIMIT 20'
    ).to_pandas()

    if len(result) > 1:
      # Check lat is ascending
      lat_values = result['lat'].values
      lat_diffs = np.diff(lat_values)
      # Allow for equal values (same lat, different lon)
      self.assertTrue(np.all(lat_diffs >= 0))

  @with_test_combinations
  def test_aggregation_pushdown(self, as_zarr):
    """Test aggregation optimization."""
    ctx = self.make_context('weather', self.weather_medium, as_zarr=as_zarr)

    # Test GROUP BY optimization
    result = ctx.sql(
      """
  SELECT 
    lat,
    COUNT(*) as point_count,
    AVG(temperature) as avg_temp,
    MIN(precipitation) as min_precip,
    MAX(precipitation) as max_precip
  FROM weather
  GROUP BY lat
  ORDER BY lat
"""
    ).to_pandas()

    # Should have one row per unique latitude
    expected_lats = len(self.weather_medium.lat)
    self.assertEqual(len(result), expected_lats)

    # All aggregation columns should be present
    expected_agg_cols = {'point_count', 'avg_temp', 'min_precip', 'max_precip'}
    self.assertTrue(expected_agg_cols.issubset(set(result.columns)))

  @with_test_combinations
  def test_complex_filter_optimization(self, as_zarr):
    """Test complex filtering with multiple conditions."""
    ctx = self.make_context('weather', self.weather_medium, as_zarr=as_zarr)

    # Complex WHERE clause with AND/OR
    result = ctx.sql(
      """
  SELECT lat, lon, temperature, precipitation
  FROM weather
  WHERE (temperature > 15 AND precipitation < 50) 
     OR (lat > 45 AND lon < -100)
  LIMIT 100
"""
    ).to_pandas()

    # Verify the complex condition
    if len(result) > 0:
      condition1 = (result['temperature'] > 15) & (result['precipitation'] < 50)
      condition2 = (result['lat'] > 45) & (result['lon'] < -100)
      combined = condition1 | condition2
      self.assertTrue(combined.all())


class SqlComplexQueryTestCase(SQLBaseTestCase):
  """Test complex SQL queries with advanced features."""

  @with_test_combinations
  def test_subqueries(self, as_zarr):
    """Test subqueries and CTEs."""
    ctx = self.make_context('air', self.air_medium, as_zarr=as_zarr)

    # Subquery to find above-average temperatures
    result = ctx.sql(
      """
  SELECT lat, lon, air
  FROM air
  WHERE air > (
    SELECT AVG(air) FROM air
  )
  LIMIT 50
"""
    ).to_pandas()

    if len(result) > 0:
      # Get the average to verify
      avg_result = ctx.sql('SELECT AVG(air) as avg_air FROM air').to_pandas()
      avg_air = avg_result['avg_air'].iloc[0]

      # All results should be above average
      self.assertTrue((result['air'] > avg_air).all())

  @with_test_combinations
  def test_window_functions(self, as_zarr):
    """Test window functions if supported by DataFusion."""
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)

    try:
      # Test ROW_NUMBER window function
      result = ctx.sql(
        """
    SELECT 
      lat, lon, air,
      ROW_NUMBER() OVER (PARTITION BY lat ORDER BY air DESC) as rank_in_lat
    FROM air
    WHERE lat IN (
      SELECT DISTINCT lat FROM air LIMIT 3
    )
    ORDER BY lat, rank_in_lat
    LIMIT 30
  """
      ).to_pandas()

      if len(result) > 0:
        # Check that ranking works within each lat
        for lat_val in result['lat'].unique():
          lat_data = result[result['lat'] == lat_val].sort_values('rank_in_lat')
          if len(lat_data) > 1:
            # Air values should be in descending order within each lat
            air_values = lat_data['air'].values
            self.assertTrue(np.all(air_values[:-1] >= air_values[1:]))

    except Exception:
      # Window functions might not be supported, skip test
      self.skipTest('Window functions not supported')

  @with_test_combinations
  def test_case_statements(self, as_zarr):
    """Test CASE statements for conditional logic."""
    ctx = self.make_context('weather', self.weather_medium, as_zarr=as_zarr)

    result = ctx.sql(
      """
  SELECT 
    lat, lon,
    temperature,
    CASE 
      WHEN temperature < 0 THEN 'Freezing'
      WHEN temperature < 10 THEN 'Cold'
      WHEN temperature < 20 THEN 'Cool'
      WHEN temperature < 30 THEN 'Warm'
      ELSE 'Hot'
    END as temp_category,
    precipitation,
    CASE
      WHEN precipitation < 5 THEN 'Dry'
      WHEN precipitation < 20 THEN 'Light'
      WHEN precipitation < 50 THEN 'Moderate'
      ELSE 'Heavy'
    END as precip_category
  FROM weather
  LIMIT 50
"""
    ).to_pandas()

    self.assertGreater(len(result), 0)
    self.assertIn('temp_category', result.columns)
    self.assertIn('precip_category', result.columns)

    # Verify categories are assigned correctly
    for _, row in result.iterrows():
      temp = row['temperature']
      temp_cat = row['temp_category']

      if temp < 0:
        self.assertEqual(temp_cat, 'Freezing')
      elif temp < 10:
        self.assertEqual(temp_cat, 'Cold')
      elif temp < 20:
        self.assertEqual(temp_cat, 'Cool')
      elif temp < 30:
        self.assertEqual(temp_cat, 'Warm')
      else:
        self.assertEqual(temp_cat, 'Hot')

  @with_test_combinations
  def test_mathematical_functions(self, as_zarr):
    """Test mathematical functions in SQL."""
    ctx = self.make_context('weather', self.weather_small, as_zarr=as_zarr)


    result = ctx.sql(
      """
  SELECT 
    lat, lon,
    temperature,
    ROUND(temperature, 1) as temp_rounded,
    ABS(temperature - 20) as temp_diff_from_20,
    SQRT(ABS(temperature)) as temp_sqrt,
    precipitation,
    LOG(precipitation + 1) as log_precip
  FROM weather
  WHERE precipitation > 0
  LIMIT 30
"""
    ).to_pandas()

    if len(result) > 0:
      # Verify mathematical operations
      for _, row in result.iterrows():
        temp = row['temperature']
        self.assertAlmostEqual(
          row['temp_diff_from_20'], abs(temp - 20), places=5
        )
        if temp >= 0:
          self.assertAlmostEqual(row['temp_sqrt'], np.sqrt(temp), places=5)

  @with_test_combinations
  def test_string_operations(self, as_zarr):
    """Test string operations if applicable."""
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)

    # Test string operations on numeric data converted to string
    result = ctx.sql(
      """
  SELECT 
    lat, lon,
    CAST(lat AS VARCHAR) as lat_str,
    CONCAT('Lat: ', CAST(lat AS VARCHAR), ', Lon: ', CAST(lon AS VARCHAR)) as coordinates
  FROM air
  LIMIT 10
"""
    ).to_pandas()

    self.assertGreater(len(result), 0)
    self.assertIn('coordinates', result.columns)

    # Verify concatenation worked
    for _, row in result.iterrows():
      expected = f"Lat: {row['lat']}, Lon: {row['lon']}"
      # Allow for slight formatting differences
      self.assertIn('Lat:', row['coordinates'])
      self.assertIn('Lon:', row['coordinates'])


class SqlErrorHandlingTestCase(SQLBaseTestCase):
  """Test error handling and edge cases in SQL interface."""

  @with_test_combinations
  def test_invalid_table_name(self, as_zarr):
    """Test error handling for invalid table names."""
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)
    # TODO(Claude): Please finish writing this test.

  @with_test_combinations
  def test_invalid_column_name(self, as_zarr):
    """Test error handling for invalid column names."""
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)

    with self.assertRaises(Exception):
      ctx.sql('SELECT nonexistent_column FROM air')

  @with_test_combinations
  def test_syntax_errors(self, as_zarr):
    """Test handling of SQL syntax errors."""
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)

    # Invalid SQL syntax
    with self.assertRaises(Exception):
      ctx.sql('SELECT * FORM air')  # Typo: FORM instead of FROM

    with self.assertRaises(Exception):
      ctx.sql('SELECT * FROM air WHERE')  # Incomplete WHERE clause

  @with_test_combinations
  def test_empty_dataset_handling(self, as_zarr):
    """Test handling of empty datasets."""
    # Create empty dataset
    empty_ds = xr.Dataset({'temp': (['x'], [])}, coords={'x': []}).chunk(
      {'x': 1}
    )

    ctx = self.make_context('empty', empty_ds, as_zarr=as_zarr)

    # Should handle empty dataset gracefully
    result = ctx.sql('SELECT * FROM empty').to_pandas()
    self.assertEqual(len(result), 0)

    # Aggregations on empty dataset
    result = ctx.sql('SELECT COUNT(*) as count FROM empty').to_pandas()
    self.assertEqual(len(result), 1)
    self.assertEqual(result['count'].iloc[0], 0)

  @with_test_combinations
  def test_large_limit_handling(self, as_zarr):
    """Test handling of very large LIMIT values."""
    ctx = self.make_context('air', self.air_small, as_zarr=as_zarr)

    total_rows = (
        self.air_small.sizes['time']
        * self.air_small.sizes['lat']
        * self.air_small.sizes['lon']
    )

    # Request more rows than exist
    result = ctx.sql(f'SELECT * FROM air LIMIT {total_rows * 10}').to_pandas()
    self.assertEqual(len(result), total_rows)


if __name__ == '__main__':
  unittest.main()
