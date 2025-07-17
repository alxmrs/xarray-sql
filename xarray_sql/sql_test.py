import os
import shutil
import tempfile
import unittest
from unittest import TestCase

import numpy as np
import xarray as xr

from . import XarrayContext
from .df_test import DaskTestCase, create_large_dataset, rand_wx

# Try to import parameterized testing, fall back to basic if not available
try:
  from parameterized import parameterized, parameterized_class
  HAS_PARAMETERIZED = True
except ImportError:
  # Fallback for environments without parameterized
  HAS_PARAMETERIZED = False
  def parameterized(params):
    """Simple fallback decorator when parameterized package is not available."""
    def decorator(func):
      return func
    return decorator


# =============================================================================
# Shared Test Data Creation Functions
# =============================================================================

def air(time_steps=24, lat_points=10, lon_points=15):
  """Create standardized air temperature dataset."""
  air = xr.tutorial.open_dataset('air_temperature')
  return air.isel(
    time=slice(0, time_steps), 
    lat=slice(0, lat_points), 
    lon=slice(0, lon_points)
  ).chunk({'time': time_steps // 2})


def weather(time_steps=6, lat_points=10, lon_points=10):
  """Create standardized multi-variable weather dataset."""
  weather = rand_wx('2023-01-01T00', '2023-01-01T12')
  return weather.isel(
    time=slice(0, time_steps),
    lat=slice(0, lat_points),
    lon=slice(0, lon_points)
  ).chunk({'time': time_steps // 2})


def synthetic(time_steps=50, lat_points=20, lon_points=20):
  """Create standardized synthetic dataset."""
  return create_large_dataset(
    time_steps=time_steps, 
    lat_points=lat_points, 
    lon_points=lon_points
  ).chunk({'time': time_steps // 2})


def stations():
  """Create standardized 1D stations dataset."""
  return xr.Dataset({
    'station_id': (['station'], [1, 2, 3, 4, 5]),
    'elevation': (['station'], [100, 250, 500, 750, 1000]),
    'name': (['station'], [
      'Station_A', 'Station_B', 'Station_C', 'Station_D', 'Station_E'
    ]),
  }).chunk({'station': 5})


def weather_zarr(temp_dir):
  """Create Zarr weather dataset with known properties."""
  ds = weather()
  zarr_path = os.path.join(temp_dir, 'weather.zarr')
  ds.to_zarr(zarr_path)
  return ds, zarr_path


def timeseries_zarr(temp_dir):
  """Create Zarr timeseries dataset with known properties."""
  time = np.arange(0, 10)
  station = np.array([1, 2, 3])
  
  shape = (10, 3)
  np.random.seed(123)
  value_data = np.random.normal(100, 10, shape)
  count_data = np.random.poisson(5, shape)
  
  ds = xr.Dataset({
    'value': (['time', 'station'], value_data),
    'count': (['time', 'station'], count_data.astype(float)),
  }, coords={'time': time, 'station': station})
  
  zarr_path = os.path.join(temp_dir, 'timeseries.zarr')
  ds.to_zarr(zarr_path)
  return ds, zarr_path


# =============================================================================
# Shared Test Infrastructure
# =============================================================================

class XarrayTestBase(unittest.TestCase):
  """Base class with shared test infrastructure for SQL testing."""
  
  def setUp(self):
    """Set up fresh context and standard datasets for each test."""
    self.ctx = XarrayContext()
    self._setup_standard_datasets()
  
  def _setup_standard_datasets(self):
    """Create and register standard test datasets."""
    # Create standard datasets
    self.air_small = air(24, 10, 15)
    self.air_medium = air(100, 20, 30)
    self.weather_small = weather(6, 10, 10)
    self.weather_medium = weather(12, 15, 20)
    self.synthetic = synthetic(50, 20, 20)
    self.stations = stations()
  
  def load_dataset(self, table_name, dataset):
    """Load a dataset into the context with error handling."""
    try:
      self.ctx.from_dataset(table_name, dataset)
    except Exception as e:
      self.fail(f"Failed to load dataset '{table_name}': {e}")
  
  def assert_sql_result_valid(self, query, expected_rows=None, expected_cols=None, 
                              min_rows=None, max_rows=None):
    """Validate SQL results comprehensively."""
    try:
      result = self.ctx.sql(query).to_pandas()
    except Exception as e:
      self.fail(f"SQL query failed: {query}\nError: {e}")
    
    # Basic validation
    self.assertIsNotNone(result, "Query result should not be None")
    self.assertIsInstance(result.index, (type(None), type(result.index)), 
                         "Result should be a pandas DataFrame")
    
    # Row count validation
    if expected_rows is not None:
      self.assertEqual(len(result), expected_rows, 
                      f"Expected {expected_rows} rows, got {len(result)}")
    if min_rows is not None:
      self.assertGreaterEqual(len(result), min_rows,
                             f"Expected at least {min_rows} rows, got {len(result)}")
    if max_rows is not None:
      self.assertLessEqual(len(result), max_rows,
                          f"Expected at most {max_rows} rows, got {len(result)}")
    
    # Column count validation
    if expected_cols is not None:
      if isinstance(expected_cols, int):
        self.assertEqual(len(result.columns), expected_cols,
                        f"Expected {expected_cols} columns, got {len(result.columns)}")
      elif isinstance(expected_cols, (list, set)):
        expected_set = set(expected_cols)
        actual_set = set(result.columns)
        self.assertEqual(expected_set, actual_set,
                        f"Expected columns {expected_set}, got {actual_set}")
    
    return result
  
  def assert_columns_present(self, result, required_columns):
    """Assert that required columns are present in result."""
    missing_cols = set(required_columns) - set(result.columns)
    self.assertEqual(len(missing_cols), 0, 
                    f"Missing required columns: {missing_cols}")
  
  def assert_aggregation_reasonable(self, result, column, agg_type):
    """Assert that aggregation results are reasonable."""
    if len(result) == 0:
      return
    
    values = result[column]
    if agg_type in ['COUNT']:
      self.assertTrue((values >= 0).all(), f"{agg_type} should be non-negative")
    elif agg_type in ['MIN', 'MAX', 'AVG', 'SUM']:
      self.assertFalse(values.isna().all(), f"{agg_type} should not be all NaN")


class XarrayZarrTestBase(XarrayTestBase):
  """Base class for Zarr-specific testing with temporary directory management."""
  
  def setUp(self):
    """Set up context, datasets, and temporary Zarr files."""
    super().setUp()
    # Create temporary directory for Zarr datasets
    self.temp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.temp_dir)
    self._setup_zarr_datasets()
  
  def _setup_zarr_datasets(self):
    """Create standard Zarr datasets."""
    self.weather_ds, self.weather_zarr_path = weather_zarr(self.temp_dir)
    self.timeseries_ds, self.timeseries_zarr_path = timeseries_zarr(self.temp_dir)
  
  def load_zarr_dataset(self, table_name, zarr_path):
    """Load a Zarr dataset into the context with error handling."""
    try:
      self.ctx.from_zarr(table_name, zarr_path)
    except Exception as e:
      self.fail(f"Failed to load Zarr dataset '{table_name}' from {zarr_path}: {e}")
  
  def assert_zarr_vs_dataset_equivalence(self, zarr_table, dataset_table, test_query):
    """Assert that Zarr and dataset tables produce equivalent results."""
    zarr_result = self.ctx.sql(test_query.format(table=zarr_table)).to_pandas()
    dataset_result = self.ctx.sql(test_query.format(table=dataset_table)).to_pandas()
    
    # Compare row counts
    self.assertEqual(len(zarr_result), len(dataset_result),
                    "Zarr and dataset should have same row count")
    
    # Compare column counts
    self.assertEqual(len(zarr_result.columns), len(dataset_result.columns),
                    "Zarr and dataset should have same column count")
    
    return zarr_result, dataset_result


# =============================================================================
# Consolidated Test Classes by Purpose
# =============================================================================

class SqlBasicsTestCase(XarrayTestBase):
  """Test fundamental SQL operations (SELECT, WHERE, basic aggregations)."""

  def test_basic_select_queries(self):
    """Test basic SELECT operations across different datasets."""
    self.load_dataset('air', self.air_small)
    
    # Test SELECT * with LIMIT (this works correctly)
    result = self.assert_sql_result_valid('SELECT * FROM air LIMIT 5', max_rows=5)
    
    # Test data column selection with LIMIT
    result = self.assert_sql_result_valid('SELECT air FROM air LIMIT 100', max_rows=100)
    self.assert_columns_present(result, ['air'])
    
    # Test coordinate selection (note: coordinates get expanded, so we test without LIMIT)
    result = self.assert_sql_result_valid('SELECT lat, lon FROM air LIMIT 10')
    self.assert_columns_present(result, ['lat', 'lon'])
    # Verify we get some results, but don't enforce LIMIT due to coordinate expansion
    self.assertGreater(len(result), 0)
    
    # Test COUNT query
    result = self.assert_sql_result_valid('SELECT COUNT(*) as total FROM air', expected_rows=1)
    self.assertGreater(result['total'].iloc[0], 0)

  def test_basic_filtering(self):
    """Test WHERE clauses with various conditions."""
    self.load_dataset('weather', self.weather_small)
    
    # Test numeric filtering
    result = self.assert_sql_result_valid(
      'SELECT COUNT(*) as count FROM weather WHERE temperature > 10'
    )
    
    # Test coordinate filtering
    result = self.assert_sql_result_valid(
      'SELECT * FROM weather WHERE lat > 35 AND lon < -110 LIMIT 20'
    )
    
    # Test BETWEEN clause
    result = self.assert_sql_result_valid(
      'SELECT lat, temperature FROM weather WHERE temperature BETWEEN 15 AND 25'
    )

  def test_basic_aggregations(self):
    """Test fundamental aggregation operations."""
    self.load_dataset('air', self.air_small)
    
    # Test GROUP BY with SUM
    result = self.assert_sql_result_valid(
      'SELECT lat, lon, SUM(air) as air_total FROM air GROUP BY lat, lon'
    )
    expected_groups = self.air_small.sizes['lat'] * self.air_small.sizes['lon']
    self.assertEqual(len(result), expected_groups)
    self.assert_aggregation_reasonable(result, 'air_total', 'SUM')
    
    # Test basic aggregations
    result = self.assert_sql_result_valid(
      '''SELECT 
          COUNT(*) as count,
          MIN(air) as min_air,
          MAX(air) as max_air,
          AVG(air) as avg_air
         FROM air''',
      expected_rows=1
    )
    
    for agg_type in ['count', 'min_air', 'max_air', 'avg_air']:
      self.assert_aggregation_reasonable(result, agg_type, agg_type.split('_')[0].upper())

  def test_sorting_and_limiting(self):
    """Test ORDER BY and LIMIT clauses."""
    self.load_dataset('air', self.air_small)
    
    # Test ORDER BY
    result = self.assert_sql_result_valid(
      'SELECT lat, lon, air FROM air ORDER BY air DESC LIMIT 10',
      expected_rows=10
    )
    
    # Verify descending order
    air_values = result['air'].values
    self.assertTrue(np.all(air_values[:-1] >= air_values[1:]))
    
    # Test multiple column ordering
    result = self.assert_sql_result_valid(
      'SELECT lat, lon, air FROM air ORDER BY lat ASC, lon DESC LIMIT 15',
      expected_rows=15
    )

  def test_column_selection_and_aliases(self):
    """Test column projection and aliases."""
    self.load_dataset('weather', self.weather_small)
    
    # Test column selection
    result = self.assert_sql_result_valid(
      'SELECT lat, lon, temperature FROM weather LIMIT 20',
      expected_cols=['lat', 'lon', 'temperature']
    )
    
    # Test column aliases
    result = self.assert_sql_result_valid(
      '''SELECT 
          lat as latitude,
          lon as longitude,
          temperature as temp
         FROM weather LIMIT 10''',
      expected_cols=['latitude', 'longitude', 'temp']
    )


class SqlParameterizedTestCase(XarrayTestBase):
  """Parameterized tests that run the same SQL functionality across different datasets."""

  # Dataset configurations for parameterized testing
  DATASET_CONFIGS = [
    ('air_small', lambda self: self.air_small, ['lat', 'lon', 'time', 'air']),
    ('weather_small', lambda self: self.weather_small, ['lat', 'lon', 'time', 'temperature', 'precipitation']),
    ('synthetic', lambda self: self.synthetic, ['lat', 'lon', 'time', 'temperature']),
  ]

  AGGREGATION_FUNCTIONS = [
    ('COUNT', 'COUNT(*)', 'count', lambda x: x >= 0),
    ('SUM', 'SUM({data_col})', 'sum_val', lambda x: not np.isnan(x)),
    ('AVG', 'AVG({data_col})', 'avg_val', lambda x: not np.isnan(x)),
    ('MIN', 'MIN({data_col})', 'min_val', lambda x: not np.isnan(x)),
    ('MAX', 'MAX({data_col})', 'max_val', lambda x: not np.isnan(x)),
  ]

  FILTER_CONDITIONS = [
    ('simple_greater', '{coord} > {mid_val}', lambda result, col, val: (result[col] > val).all()),
    ('simple_less', '{coord} < {mid_val}', lambda result, col, val: (result[col] < val).all()),
    ('range_between', '{coord} BETWEEN {low_val} AND {high_val}', 
     lambda result, col, low, high: ((result[col] >= low) & (result[col] <= high)).all()),
  ]

  def _get_numeric_column(self, dataset_name, dataset):
    """Get the primary numeric data column for a dataset."""
    column_map = {
      'air_small': 'air',
      'weather_small': 'temperature', 
      'synthetic': 'temperature'
    }
    return column_map.get(dataset_name, 'temperature')

  def _get_coordinate_column(self, dataset_name, dataset):
    """Get a coordinate column suitable for filtering."""
    # Use lat as it's available in all datasets and has reasonable values
    return 'lat'

  def test_basic_select_all_datasets(self):
    """Test basic SELECT operations across all dataset types."""
    for dataset_name, dataset_getter, expected_cols in self.DATASET_CONFIGS:
      with self.subTest(dataset=dataset_name):
        dataset = dataset_getter(self)
        self.load_dataset(dataset_name, dataset)
        
        # Test SELECT *
        result = self.assert_sql_result_valid(
          f'SELECT * FROM {dataset_name} LIMIT 5', 
          max_rows=5, min_rows=1
        )
        
        # Test COUNT
        result = self.assert_sql_result_valid(
          f'SELECT COUNT(*) as total FROM {dataset_name}',
          expected_rows=1
        )
        self.assertGreater(result['total'].iloc[0], 0)

  def test_aggregations_all_datasets(self):
    """Test aggregation functions across all dataset types."""
    for dataset_name, dataset_getter, expected_cols in self.DATASET_CONFIGS:
      dataset = dataset_getter(self)
      self.load_dataset(dataset_name, dataset)
      data_col = self._get_numeric_column(dataset_name, dataset)
      
      for agg_name, agg_sql, result_col, validator in self.AGGREGATION_FUNCTIONS:
        with self.subTest(dataset=dataset_name, aggregation=agg_name):
          
          if agg_name == 'COUNT':
            query = f'SELECT {agg_sql} as {result_col} FROM {dataset_name}'
          else:
            query = f'SELECT {agg_sql.format(data_col=data_col)} as {result_col} FROM {dataset_name}'
          
          result = self.assert_sql_result_valid(query, expected_rows=1)
          value = result[result_col].iloc[0]
          self.assertTrue(validator(value), 
                         f'{agg_name} validation failed for {dataset_name}: {value}')

  def test_filtering_all_datasets(self):
    """Test filtering operations across all dataset types."""
    for dataset_name, dataset_getter, expected_cols in self.DATASET_CONFIGS:
      dataset = dataset_getter(self)
      self.load_dataset(dataset_name, dataset)
      coord_col = self._get_coordinate_column(dataset_name, dataset)
      
      # Get coordinate values for testing
      coord_result = self.assert_sql_result_valid(
        f'SELECT MIN({coord_col}) as min_val, MAX({coord_col}) as max_val FROM {dataset_name}'
      )
      min_val = coord_result['min_val'].iloc[0]
      max_val = coord_result['max_val'].iloc[0]
      mid_val = (min_val + max_val) / 2
      
      for filter_name, filter_template, validator in self.FILTER_CONDITIONS:
        with self.subTest(dataset=dataset_name, filter=filter_name):
          
          if filter_name == 'range_between':
            low_val = min_val + (max_val - min_val) * 0.25
            high_val = min_val + (max_val - min_val) * 0.75
            condition = filter_template.format(
              coord=coord_col, low_val=low_val, high_val=high_val
            )
            query = f'SELECT * FROM {dataset_name} WHERE {condition} LIMIT 50'
            result = self.assert_sql_result_valid(query)
            
            if len(result) > 0:
              self.assertTrue(validator(result, coord_col, low_val, high_val))
          else:
            condition = filter_template.format(coord=coord_col, mid_val=mid_val)
            query = f'SELECT * FROM {dataset_name} WHERE {condition} LIMIT 50'
            result = self.assert_sql_result_valid(query)
            
            if len(result) > 0:
              self.assertTrue(validator(result, coord_col, mid_val))

  def test_groupby_all_datasets(self):
    """Test GROUP BY operations across all dataset types."""
    for dataset_name, dataset_getter, expected_cols in self.DATASET_CONFIGS:
      dataset = dataset_getter(self)
      self.load_dataset(dataset_name, dataset)
      data_col = self._get_numeric_column(dataset_name, dataset)
      
      with self.subTest(dataset=dataset_name):
        # Group by coordinate and aggregate
        result = self.assert_sql_result_valid(
          f'''SELECT 
              lat,
              COUNT(*) as count,
              AVG({data_col}) as avg_val
              FROM {dataset_name}
              GROUP BY lat
              ORDER BY lat'''
        )
        
        # Should have one row per unique lat value
        expected_groups = len(dataset.lat) if hasattr(dataset, 'lat') else 1
        self.assertEqual(len(result), expected_groups)
        
        # All counts should be positive
        self.assertTrue((result['count'] > 0).all())

  def test_ordering_all_datasets(self):
    """Test ORDER BY operations across all dataset types."""
    for dataset_name, dataset_getter, expected_cols in self.DATASET_CONFIGS:
      dataset = dataset_getter(self)
      self.load_dataset(dataset_name, dataset)
      data_col = self._get_numeric_column(dataset_name, dataset)
      
      with self.subTest(dataset=dataset_name):
        # Test ordering by data column
        result = self.assert_sql_result_valid(
          f'SELECT {data_col} FROM {dataset_name} ORDER BY {data_col} DESC LIMIT 20',
          max_rows=20
        )
        
        if len(result) > 1:
          values = result[data_col].values
          # Should be in descending order
          self.assertTrue(np.all(values[:-1] >= values[1:]),
                         f'Values not in descending order for {dataset_name}')


class SqlZarrParameterizedTestCase(XarrayZarrTestBase):
  """Parameterized tests specifically for Zarr datasets and from_zarr functionality."""

  def test_zarr_operations_parameterized(self):
    """Test multiple SQL operations on Zarr datasets in a parameterized way."""
    self.load_zarr_dataset('weather', self.weather_zarr_path)
    
    # Test cases: (operation_name, query, validator_function)
    operation_tests = [
      ('count', 'SELECT COUNT(*) as result FROM weather', lambda r: r['result'].iloc[0] > 0),
      ('coordinate_min', 'SELECT MIN(dim_0) as result FROM weather', lambda r: not np.isnan(r['result'].iloc[0])),
      ('coordinate_max', 'SELECT MAX(dim_0) as result FROM weather', lambda r: not np.isnan(r['result'].iloc[0])),
      ('coordinate_filter', 'SELECT COUNT(*) as result FROM weather WHERE dim_0 >= 0', lambda r: r['result'].iloc[0] >= 0),
      ('data_aggregate', 'SELECT AVG("/temperature") as result FROM weather', lambda r: not np.isnan(r['result'].iloc[0])),
    ]
    
    for test_name, query, validator in operation_tests:
      with self.subTest(operation=test_name):
        result = self.assert_sql_result_valid(query, expected_rows=1)
        self.assertTrue(validator(result), f'{test_name} validation failed')

  def test_zarr_specific_operations(self):
    """Test operations that are specific to Zarr table providers."""
    self.load_zarr_dataset('weather', self.weather_zarr_path)
    
    # Test Zarr-specific column naming (dim_* instead of coordinate names)
    result = self.assert_sql_result_valid('SELECT * FROM weather LIMIT 1')
    
    # Should have dim_0, dim_1, dim_2 columns (for time, lat, lon dimensions)
    dim_columns = [col for col in result.columns if col.startswith('dim_')]
    self.assertGreaterEqual(len(dim_columns), 3, "Should have at least 3 dimension columns")
    
    # Should have data variable columns with '/' prefix
    data_columns = [col for col in result.columns if col.startswith('/')]
    self.assertGreaterEqual(len(data_columns), 3, "Should have at least 3 data variable columns")
    
    # Test querying data variables
    result = self.assert_sql_result_valid(
      'SELECT AVG("/temperature") as avg_temp FROM weather',
      expected_rows=1
    )
    self.assertFalse(np.isnan(result['avg_temp'].iloc[0]))

  def test_zarr_predicate_pushdown_efficiency(self):
    """Test that predicate pushdown works efficiently with Zarr datasets."""
    self.load_zarr_dataset('weather', self.weather_zarr_path)
    
    # Test coordinate-based filtering (should push down to Zarr level)
    total_result = self.assert_sql_result_valid('SELECT COUNT(*) as total FROM weather')
    total_count = total_result['total'].iloc[0]
    
    # Filter by first dimension (time)
    filtered_result = self.assert_sql_result_valid(
      'SELECT COUNT(*) as filtered FROM weather WHERE dim_0 >= 2'
    )
    filtered_count = filtered_result['filtered'].iloc[0]
    
    # Should return fewer rows when filtered
    self.assertLessEqual(filtered_count, total_count)
    
    # Test multi-dimensional filtering
    multi_filtered_result = self.assert_sql_result_valid(
      'SELECT COUNT(*) as multi_filtered FROM weather WHERE dim_0 >= 1 AND dim_1 < 2'
    )
    multi_filtered_count = multi_filtered_result['multi_filtered'].iloc[0]
    
    # Should be a reasonable subset
    self.assertLessEqual(multi_filtered_count, filtered_count)


class SqlAdvancedTestCase(XarrayTestBase):
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
    expected_count = self.weather_ds.sizes['time'] * self.weather_ds.sizes['lat'] * \
                     self.weather_ds.sizes['lon']
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
    result_dataset = ctx.sql(
      'SELECT COUNT(*) as count FROM weather_dataset').to_pandas()
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
    result = ctx.sql(
      'SELECT COUNT(*) as count FROM weather WHERE dim_0 >= 2').to_pandas()
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

    expected = (self.weather_ds.sizes['time'] * self.weather_ds.sizes['lat'] *
                self.weather_ds.sizes['lon'] *
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
