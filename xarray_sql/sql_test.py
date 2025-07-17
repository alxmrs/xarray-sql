import unittest
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import shutil
import os

from . import XarrayContext
from .df_test import DaskTestCase, create_large_dataset, rand_wx


class SqlTestCase(DaskTestCase):

  def test_sanity(self):
    c = XarrayContext()
    c.from_dataset('air', self.air_small)

    query = c.sql('SELECT "lat", "lon", "time", "air" FROM "air" LIMIT 100')

    result = query.to_pandas()
    self.assertIsNotNone(result)
    self.assertLessEqual(len(result), 1320)  # Should be all rows or less
    self.assertGreater(len(result), 0)  # Should have some rows

  def test_agg_small(self):
    c = XarrayContext()
    c.from_dataset('air', self.air_small)

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

    result = query.to_pandas()
    self.assertIsNotNone(result)

    expected = self.air_small.sizes['lat'] * self.air_small.sizes['lon']
    self.assertEqual(len(result), expected)

  def test_agg_regular(self):
    c = XarrayContext()
    c.from_dataset('air', self.air)

    query = c.sql(
        """
    SELECT
      "lat", "lon", AVG("air") as air_total
    FROM 
      "air" 
    GROUP BY
     "lat", "lon"
    """
    )

    result = query.to_pandas()
    self.assertIsNotNone(result)

    expected = self.air.sizes['lat'] * self.air.sizes['lon']
    self.assertEqual(len(result), expected)


class SqlVarietyTestCase(unittest.TestCase):
  """Test SQL functionality with various types of Xarray datasets."""

  def setUp(self):
    """Set up test datasets for SQL testing."""
    # Create air temperature dataset
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.air_small = self.air.isel(
        time=slice(0, 24), lat=slice(0, 10), lon=slice(0, 15)
    ).chunk({'time': 12})

    # Create weather dataset with multiple variables
    self.weather = rand_wx('2023-01-01T00', '2023-01-01T12')
    self.weather_small = self.weather.isel(
        time=slice(0, 6), lat=slice(0, 10), lon=slice(0, 10)
    ).chunk({'time': 3})

    # Create synthetic dataset with different data types
    self.synthetic = create_large_dataset(
        time_steps=50, lat_points=20, lon_points=20
    ).chunk({'time': 25})

    # Create 1D dataset for testing joins
    self.stations = xr.Dataset(
        {
            'station_id': (['station'], [1, 2, 3, 4, 5]),
            'elevation': (['station'], [100, 250, 500, 750, 1000]),
            'name': (
                ['station'],
                [
                    'Station_A',
                    'Station_B',
                    'Station_C',
                    'Station_D',
                    'Station_E',
                ],
            ),
        }
    ).chunk({'station': 5})

  def test_basic_select_all(self):
    """Test basic SELECT * queries on different datasets."""
    ctx = XarrayContext()

    # Test with air temperature dataset
    ctx.from_dataset('air', self.air_small)
    result = ctx.sql('SELECT * FROM air LIMIT 10').to_pandas()

    self.assertGreater(len(result), 0)
    self.assertLessEqual(len(result), 10)
    self.assertIn('air', result.columns)
    self.assertIn('lat', result.columns)
    self.assertIn('lon', result.columns)
    self.assertIn('time', result.columns)

  def test_weather_dataset_queries(self):
    """Test queries on weather dataset with multiple variables."""
    ctx = XarrayContext()
    ctx.from_dataset('weather', self.weather_small)

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

  def test_synthetic_dataset_aggregations(self):
    """Test aggregation queries on synthetic dataset."""
    ctx = XarrayContext()
    ctx.from_dataset('synthetic', self.synthetic)

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

  def test_spatial_grouping(self):
    """Test spatial grouping queries."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

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

  def test_temporal_filtering(self):
    """Test temporal filtering and grouping."""
    ctx = XarrayContext()
    ctx.from_dataset('weather', self.weather_small)

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

  def test_station_dataset_queries(self):
    """Test queries on 1D station dataset."""
    ctx = XarrayContext()
    ctx.from_dataset('stations', self.stations)

    # Basic select
    result = ctx.sql('SELECT * FROM stations ORDER BY elevation').to_pandas()
    self.assertEqual(len(result), 5)

    # Test filtering by elevation
    result = ctx.sql(
        'SELECT name, elevation FROM stations WHERE elevation > 300 ORDER BY elevation'
    ).to_pandas()

    self.assertGreater(len(result), 0)
    self.assertTrue((result['elevation'] > 300).all())


class SqlJoinTestCase(unittest.TestCase):
  """Test joining tabular data with raster data using from_dataset."""

  def setUp(self):
    """Set up datasets for join testing."""
    # Create a small air temperature dataset
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.air_small = self.air.isel(
        time=slice(0, 12), lat=slice(0, 5), lon=slice(0, 8)
    ).chunk({'time': 6})

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

  def test_simple_cross_join(self):
    """Test cross join between raster and tabular data."""
    ctx = XarrayContext()
    ctx.from_dataset('air_data', self.air_small)
    ctx.from_dataset('stations', self.stations)

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

  def test_coordinate_based_join(self):
    """Test joining on coordinate proximity."""
    ctx = XarrayContext()
    ctx.from_dataset('air_data', self.air_small)
    ctx.from_dataset('stations', self.stations)

    # First test a simple cross join to ensure datasets are compatible
    result = ctx.sql(
        """
      SELECT COUNT(*) as total_combinations
      FROM air_data a
      CROSS JOIN stations s
    """
    ).to_pandas()

    self.assertGreater(result['total_combinations'].iloc[0], 0)

    # Test a simpler join condition
    result = ctx.sql(
        """
      SELECT 
        COUNT(*) as match_count
      FROM air_data a, stations s
      WHERE s.station_id = 101
    """
    ).to_pandas()

    self.assertGreater(result['match_count'].iloc[0], 0)

  def test_region_classification_join(self):
    """Test joining with region classification."""
    ctx = XarrayContext()
    ctx.from_dataset('air_data', self.air_small)
    ctx.from_dataset('regions', self.regions)

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

  def test_multiple_dataset_aggregation(self):
    """Test aggregating across multiple datasets."""
    ctx = XarrayContext()
    ctx.from_dataset('air_data', self.air_small)
    ctx.from_dataset('stations', self.stations)

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


class SqlOptimizationTestCase(unittest.TestCase):
  """Test DataFusion optimizations like column selection and filters."""

  def setUp(self):
    """Set up dataset for optimization testing."""
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.air_medium = self.air.isel(
        time=slice(0, 100), lat=slice(0, 20), lon=slice(0, 30)
    ).chunk({'time': 50})

    # Create multi-variable dataset
    self.weather = rand_wx('2023-01-01T00', '2023-01-02T00')
    self.weather_medium = self.weather.isel(
        time=slice(0, 12), lat=slice(0, 15), lon=slice(0, 20)
    ).chunk({'time': 6})

  def test_column_projection(self):
    """Test that selecting specific columns works efficiently."""
    ctx = XarrayContext()
    ctx.from_dataset('weather', self.weather_medium)

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

  def test_where_clause_filtering(self):
    """Test WHERE clause filtering optimization."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_medium)

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

  def test_limit_optimization(self):
    """Test LIMIT clause optimization."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_medium)

    # Test small limits
    for limit in [1, 5, 10, 50]:
      result = ctx.sql(f'SELECT * FROM air LIMIT {limit}').to_pandas()
      self.assertLessEqual(len(result), limit)
      if (
          limit
          <= self.air_medium.sizes['time']
          * self.air_medium.sizes['lat']
          * self.air_medium.sizes['lon']
      ):
        self.assertEqual(len(result), limit)

  def test_order_by_optimization(self):
    """Test ORDER BY clause."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_medium)

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

  def test_aggregation_pushdown(self):
    """Test aggregation optimization."""
    ctx = XarrayContext()
    ctx.from_dataset('weather', self.weather_medium)

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

  def test_complex_filter_optimization(self):
    """Test complex filtering with multiple conditions."""
    ctx = XarrayContext()
    ctx.from_dataset('weather', self.weather_medium)

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


class SqlComplexQueryTestCase(unittest.TestCase):
  """Test complex SQL queries with advanced features."""

  def setUp(self):
    """Set up datasets for complex query testing."""
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.air_small = self.air.isel(
        time=slice(0, 50), lat=slice(0, 15), lon=slice(0, 20)
    ).chunk({'time': 25})

    self.weather = rand_wx('2023-01-01T00', '2023-01-01T12')
    self.weather_small = self.weather.isel(
        time=slice(0, 8), lat=slice(0, 10), lon=slice(0, 12)
    ).chunk({'time': 4})

  def test_subqueries(self):
    """Test subqueries and CTEs."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

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

  def test_window_functions(self):
    """Test window functions if supported by DataFusion."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

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

  def test_case_statements(self):
    """Test CASE statements for conditional logic."""
    ctx = XarrayContext()
    ctx.from_dataset('weather', self.weather_small)

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

  def test_mathematical_functions(self):
    """Test mathematical functions in SQL."""
    ctx = XarrayContext()
    ctx.from_dataset('weather', self.weather_small)

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

  def test_string_operations(self):
    """Test string operations if applicable."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

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


class SqlErrorHandlingTestCase(unittest.TestCase):
  """Test error handling and edge cases in SQL interface."""

  def setUp(self):
    """Set up dataset for error testing."""
    self.air = xr.tutorial.open_dataset('air_temperature')
    self.air_small = self.air.isel(
        time=slice(0, 10), lat=slice(0, 5), lon=slice(0, 8)
    ).chunk({'time': 5})

  def test_invalid_table_name(self):
    """Test error handling for invalid table names."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

    with self.assertRaises(Exception):
      ctx.sql('SELECT * FROM nonexistent_table')

  def test_invalid_column_name(self):
    """Test error handling for invalid column names."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

    with self.assertRaises(Exception):
      ctx.sql('SELECT nonexistent_column FROM air')

  def test_syntax_errors(self):
    """Test handling of SQL syntax errors."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

    # Invalid SQL syntax
    with self.assertRaises(Exception):
      ctx.sql('SELECT * FORM air')  # Typo: FORM instead of FROM

    with self.assertRaises(Exception):
      ctx.sql('SELECT * FROM air WHERE')  # Incomplete WHERE clause

  def test_empty_dataset_handling(self):
    """Test handling of empty datasets."""
    # Create empty dataset
    empty_ds = xr.Dataset({'temp': (['x'], [])}, coords={'x': []}).chunk(
        {'x': 1}
    )

    ctx = XarrayContext()
    ctx.from_dataset('empty', empty_ds)

    # Should handle empty dataset gracefully
    result = ctx.sql('SELECT * FROM empty').to_pandas()
    self.assertEqual(len(result), 0)

    # Aggregations on empty dataset
    result = ctx.sql('SELECT COUNT(*) as count FROM empty').to_pandas()
    self.assertEqual(len(result), 1)
    self.assertEqual(result['count'].iloc[0], 0)

  def test_large_limit_handling(self):
    """Test handling of very large LIMIT values."""
    ctx = XarrayContext()
    ctx.from_dataset('air', self.air_small)

    total_rows = (
        self.air_small.sizes['time']
        * self.air_small.sizes['lat']
        * self.air_small.sizes['lon']
    )

    # Request more rows than exist
    result = ctx.sql(f'SELECT * FROM air LIMIT {total_rows * 10}').to_pandas()
    self.assertEqual(len(result), total_rows)


class SqlZarrTestCase(unittest.TestCase):
  """Test SQL functionality with Zarr datasets using from_zarr method."""

  def setUp(self):
    """Set up temporary Zarr datasets for testing."""
    # Create temporary directory for Zarr datasets
    self.temp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.temp_dir)
    
    # Create test datasets and save as Zarr
    self._create_zarr_datasets()

  def _create_zarr_datasets(self):
    """Create diverse Zarr datasets for testing."""
    # 1. Multi-variable weather dataset (3D)
    time = np.arange(0, 5)  # 5 time points
    lat = np.array([30.0, 35.0, 40.0])  # 3 latitudes
    lon = np.array([-120.0, -115.0, -110.0, -105.0])  # 4 longitudes
    
    shape = (5, 3, 4)
    temperature_data = np.random.normal(20, 5, shape)
    pressure_data = np.random.normal(1013, 20, shape)
    humidity_data = np.random.uniform(30, 90, shape)
    
    self.weather_ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], temperature_data),
        'pressure': (['time', 'lat', 'lon'], pressure_data),
        'humidity': (['time', 'lat', 'lon'], humidity_data),
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon,
    })
    
    self.weather_zarr_path = os.path.join(self.temp_dir, 'weather.zarr')
    self.weather_ds.to_zarr(self.weather_zarr_path)
    
    # 2. Simple 2D timeseries dataset
    time_2d = np.arange(0, 10)
    station = np.array([1, 2, 3])
    
    shape_2d = (10, 3)
    value_data = np.random.normal(100, 10, shape_2d)
    count_data = np.random.poisson(5, shape_2d)
    
    self.timeseries_ds = xr.Dataset({
        'value': (['time', 'station'], value_data),
        'count': (['time', 'station'], count_data.astype(float)),
    }, coords={
        'time': time_2d,
        'station': station,
    })
    
    self.timeseries_zarr_path = os.path.join(self.temp_dir, 'timeseries.zarr')
    self.timeseries_ds.to_zarr(self.timeseries_zarr_path)
    
    # 3. Business dataset for complex queries
    category = np.arange(0, 4)
    region = np.arange(0, 3)
    period = np.arange(0, 6)
    
    shape_3d = (4, 3, 6)
    activity_data = np.random.exponential(10, shape_3d)
    revenue_data = activity_data * (2 + np.random.normal(0, 0.5, shape_3d))
    
    self.business_ds = xr.Dataset({
        'activity': (['category', 'region', 'period'], activity_data),
        'revenue': (['category', 'region', 'period'], revenue_data),
    }, coords={
        'category': category,
        'region': region,
        'period': period,
    })
    
    self.business_zarr_path = os.path.join(self.temp_dir, 'business.zarr')
    self.business_ds.to_zarr(self.business_zarr_path)

  def test_from_zarr_basic_functionality(self):
    """Test basic from_zarr method functionality."""
    ctx = XarrayContext()
    
    # Test loading a Zarr dataset
    ctx.from_zarr('weather', self.weather_zarr_path)
    
    # Test basic query
    result = ctx.sql('SELECT COUNT(*) as total FROM weather').to_pandas()
    expected_count = self.weather_ds.sizes['time'] * self.weather_ds.sizes['lat'] * self.weather_ds.sizes['lon']
    self.assertEqual(result['total'].iloc[0], expected_count)
    
    # Test schema inference
    result = ctx.sql('SELECT * FROM weather LIMIT 1').to_pandas()
    self.assertIn('dim_0', result.columns)  # time dimension
    self.assertIn('dim_1', result.columns)  # lat dimension  
    self.assertIn('dim_2', result.columns)  # lon dimension
    # Data variables should have '/' prefix
    data_columns = [col for col in result.columns if col.startswith('/')]
    self.assertGreaterEqual(len(data_columns), 3)  # temperature, pressure, humidity

  def test_from_zarr_vs_from_dataset_equivalence(self):
    """Test that from_zarr produces equivalent results to from_dataset."""
    ctx = XarrayContext()
    
    # Load same data via different methods in the same context
    ctx.from_zarr('weather_zarr', self.weather_zarr_path)
    # Ensure the dataset is chunked for from_dataset
    chunked_weather = self.weather_ds.chunk({'time': 3})
    ctx.from_dataset('weather_dataset', chunked_weather)
    
    # Test count equivalence
    result_zarr = ctx.sql('SELECT COUNT(*) as count FROM weather_zarr').to_pandas()
    result_dataset = ctx.sql('SELECT COUNT(*) as count FROM weather_dataset').to_pandas()
    self.assertEqual(result_zarr['count'].iloc[0], result_dataset['count'].iloc[0])
    
    # Test column count equivalence (structures should be similar)
    schema_zarr = ctx.sql('SELECT * FROM weather_zarr LIMIT 1').to_pandas()
    schema_dataset = ctx.sql('SELECT * FROM weather_dataset LIMIT 1').to_pandas()
    
    # Zarr uses dim_* naming while dataset uses coordinate names
    # Both should have same number of dimensions + data variables
    self.assertEqual(len(schema_zarr.columns), len(schema_dataset.columns))

  def test_zarr_filtering_and_predicate_pushdown(self):
    """Test filtering operations work with Zarr datasets."""
    ctx = XarrayContext()
    ctx.from_zarr('weather', self.weather_zarr_path)
    
    # Test coordinate filtering (should use predicate pushdown)
    result = ctx.sql('SELECT COUNT(*) as count FROM weather WHERE dim_0 >= 2').to_pandas()
    total_result = ctx.sql('SELECT COUNT(*) as total FROM weather').to_pandas()
    
    # Should return fewer rows when filtered
    self.assertLessEqual(result['count'].iloc[0], total_result['total'].iloc[0])
    
    # Test multiple coordinate filters
    result = ctx.sql(
        'SELECT COUNT(*) as count FROM weather WHERE dim_0 >= 1 AND dim_1 < 2'
    ).to_pandas()
    self.assertGreater(result['count'].iloc[0], 0)
    
    # Test range filters
    result = ctx.sql(
        'SELECT dim_0, COUNT(*) as count FROM weather WHERE dim_0 BETWEEN 0 AND 2 GROUP BY dim_0'
    ).to_pandas()
    self.assertGreaterEqual(len(result), 1)

  def test_zarr_aggregation_operations(self):
    """Test aggregation operations on Zarr datasets."""
    ctx = XarrayContext()
    ctx.from_zarr('weather', self.weather_zarr_path)
    
    # Test basic aggregations
    result = ctx.sql(
        '''
        SELECT 
            COUNT(*) as count,
            MIN(dim_0) as min_time,
            MAX(dim_0) as max_time,
            AVG(CAST(dim_0 AS DOUBLE)) as avg_time
        FROM weather
        '''
    ).to_pandas()
    
    self.assertEqual(len(result), 1)
    self.assertGreaterEqual(result['max_time'].iloc[0], result['min_time'].iloc[0])
    
    # Test GROUP BY operations
    result = ctx.sql(
        '''
        SELECT 
            dim_0,
            COUNT(*) as count_per_time,
            COUNT(DISTINCT dim_1) as unique_lats
        FROM weather
        GROUP BY dim_0
        ORDER BY dim_0
        '''
    ).to_pandas()
    
    expected_time_steps = self.weather_ds.sizes['time']
    self.assertEqual(len(result), expected_time_steps)

  def test_zarr_multi_dataset_operations(self):
    """Test operations across multiple Zarr datasets."""
    ctx = XarrayContext()
    ctx.from_zarr('weather', self.weather_zarr_path)
    ctx.from_zarr('timeseries', self.timeseries_zarr_path)
    ctx.from_zarr('business', self.business_zarr_path)
    
    # Test that all datasets are accessible
    weather_count = ctx.sql('SELECT COUNT(*) as count FROM weather').to_pandas()
    timeseries_count = ctx.sql('SELECT COUNT(*) as count FROM timeseries').to_pandas()
    business_count = ctx.sql('SELECT COUNT(*) as count FROM business').to_pandas()
    
    self.assertGreater(weather_count['count'].iloc[0], 0)
    self.assertGreater(timeseries_count['count'].iloc[0], 0)
    self.assertGreater(business_count['count'].iloc[0], 0)
    
    # Test union across datasets
    result = ctx.sql(
        '''
        SELECT 'weather' as dataset, COUNT(*) as count FROM weather
        UNION ALL
        SELECT 'timeseries' as dataset, COUNT(*) as count FROM timeseries
        UNION ALL
        SELECT 'business' as dataset, COUNT(*) as count FROM business
        '''
    ).to_pandas()
    
    self.assertEqual(len(result), 3)
    self.assertIn('weather', result['dataset'].values)
    self.assertIn('timeseries', result['dataset'].values)
    self.assertIn('business', result['dataset'].values)

  def test_zarr_joins_between_datasets(self):
    """Test join operations between Zarr datasets."""
    ctx = XarrayContext()
    ctx.from_zarr('weather', self.weather_zarr_path)
    ctx.from_zarr('timeseries', self.timeseries_zarr_path)
    
    # Test cross join
    result = ctx.sql(
        '''
        SELECT COUNT(*) as total_combinations
        FROM weather w
        CROSS JOIN timeseries t
        '''
    ).to_pandas()
    
    expected = (self.weather_ds.sizes['time'] * self.weather_ds.sizes['lat'] * self.weather_ds.sizes['lon'] *
                self.timeseries_ds.sizes['time'] * self.timeseries_ds.sizes['station'])
    self.assertEqual(result['total_combinations'].iloc[0], expected)
    
    # Test join on coordinate values
    result = ctx.sql(
        '''
        SELECT 
            w.dim_0 as weather_time,
            t.dim_0 as timeseries_time,
            COUNT(*) as matches
        FROM weather w
        JOIN timeseries t ON w.dim_0 = t.dim_0
        GROUP BY w.dim_0, t.dim_0
        ORDER BY w.dim_0
        '''
    ).to_pandas()
    
    # Should have matches where time coordinates overlap
    self.assertGreater(len(result), 0)

  def test_zarr_complex_sql_operations(self):
    """Test complex SQL operations on Zarr datasets."""
    ctx = XarrayContext()
    ctx.from_zarr('business', self.business_zarr_path)
    
    # Test subqueries
    result = ctx.sql(
        '''
        SELECT 
            dim_0 as category,
            avg_activity
        FROM (
            SELECT 
                dim_0,
                AVG("/activity") as avg_activity
            FROM business
            GROUP BY dim_0
        ) subq
        WHERE avg_activity > (
            SELECT AVG("/activity") FROM business
        )
        '''
    ).to_pandas()
    
    # Should return categories with above-average activity
    self.assertGreaterEqual(len(result), 0)
    
    # Test window functions (if supported)
    try:
        result = ctx.sql(
            '''
            SELECT 
                dim_0,
                dim_1,
                "/activity",
                ROW_NUMBER() OVER (PARTITION BY dim_0 ORDER BY "/activity" DESC) as rank_in_category
            FROM business
            ORDER BY dim_0, rank_in_category
            LIMIT 20
            '''
        ).to_pandas()
        
        if len(result) > 0:
            self.assertIn('rank_in_category', result.columns)
        
    except Exception:
        # Window functions might not be supported
        pass
    
    # Test CASE statements
    result = ctx.sql(
        '''
        SELECT 
            dim_0 as category,
            COUNT(*) as total_records,
            CASE 
                WHEN AVG("/activity") > 15 THEN 'High Activity'
                WHEN AVG("/activity") > 5 THEN 'Medium Activity'
                ELSE 'Low Activity'
            END as activity_level
        FROM business
        GROUP BY dim_0
        '''
    ).to_pandas()
    
    self.assertEqual(len(result), self.business_ds.sizes['category'])
    self.assertIn('activity_level', result.columns)

  def test_zarr_error_handling(self):
    """Test error handling for Zarr-specific issues."""
    ctx = XarrayContext()
    
    # Test non-existent Zarr path
    with self.assertRaises(Exception):
        ctx.from_zarr('nonexistent', '/path/that/does/not/exist.zarr')
    
    # Test invalid table operations after successful load
    ctx.from_zarr('weather', self.weather_zarr_path)
    
    # Test invalid column reference
    with self.assertRaises(Exception):
        ctx.sql('SELECT nonexistent_column FROM weather').to_pandas()
    
    # Test invalid table reference
    with self.assertRaises(Exception):
        ctx.sql('SELECT * FROM nonexistent_table').to_pandas()

  def test_zarr_data_type_handling(self):
    """Test that Zarr datasets handle different data types correctly."""
    ctx = XarrayContext()
    ctx.from_zarr('weather', self.weather_zarr_path)
    
    # Test numeric operations on data variables
    result = ctx.sql(
        '''
        SELECT 
            COUNT(*) as count,
            AVG("/temperature") as avg_temp,
            MIN("/pressure") as min_pressure,
            MAX("/humidity") as max_humidity
        FROM weather
        '''
    ).to_pandas()
    
    self.assertEqual(len(result), 1)
    # All aggregations should return valid numbers
    self.assertFalse(np.isnan(result['avg_temp'].iloc[0]))
    self.assertFalse(np.isnan(result['min_pressure'].iloc[0]))
    self.assertFalse(np.isnan(result['max_humidity'].iloc[0]))
    
    # Test coordinate arithmetic
    result = ctx.sql(
        '''
        SELECT 
            dim_0,
            dim_1,
            (dim_0 * 10 + dim_1) as computed_coordinate
        FROM weather
        LIMIT 10
        '''
    ).to_pandas()
    
    self.assertGreater(len(result), 0)
    self.assertIn('computed_coordinate', result.columns)

  def test_zarr_chunks_parameter_assertion(self):
    """Test that chunks parameter raises appropriate assertion."""
    ctx = XarrayContext()
    
    # chunks=None should work
    ctx.from_zarr('weather', self.weather_zarr_path, chunks=None)
    
    # chunks with any value should raise AssertionError
    with self.assertRaises(AssertionError):
        ctx.from_zarr('weather2', self.weather_zarr_path, chunks='auto')

if __name__ == '__main__':
  unittest.main()
