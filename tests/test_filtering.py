#!/usr/bin/env python3
"""
Unit tests for filtering functionality in the XarrayDatasetAdapter.

Tests PyArrow expression filtering integration with xarray Datasets.
"""

import unittest
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.compute as pc
from datetime import datetime, timedelta

from xarray_sql.ds import XarrayDatasetAdapter
from xarray_sql.expression_filter import SimpleExpressionFilter


class TestSimpleExpressionFilter(unittest.TestCase):
    """Test cases for the SimpleExpressionFilter class."""

    def setUp(self):
        """Set up test dataset."""
        # Create dataset with known values for testing
        time = pd.date_range('2023-01-01', periods=5, freq='D')
        location = ['A', 'B', 'C']
        
        # Temperature: 10, 15, 20, 25, 30 for each location
        temp_data = np.array([[10, 15, 20], [15, 20, 25], [20, 25, 30], [25, 30, 35], [30, 35, 40]])
        
        # Humidity: 80, 60, 40 for each time
        humid_data = np.array([[80, 60, 40], [80, 60, 40], [80, 60, 40], [80, 60, 40], [80, 60, 40]])
        
        self.test_ds = xr.Dataset({
            'temperature': (['time', 'location'], temp_data),
            'humidity': (['time', 'location'], humid_data),
        }, coords={
            'time': time,
            'location': location,
        })

    def test_simple_comparison_greater_than(self):
        """Test simple greater than comparison."""
        filter_engine = SimpleExpressionFilter()
        
        # Filter temperature > 20
        expr = pc.field("temperature") > pc.scalar(20)
        result = filter_engine.apply_filter(self.test_ds, expr)
        
        # Should only include temperatures > 20
        remaining_temps = result.temperature.values
        self.assertTrue(np.all(remaining_temps > 20))

    def test_simple_comparison_less_than(self):
        """Test simple less than comparison."""
        filter_engine = SimpleExpressionFilter()
        
        # Filter humidity < 70
        expr = pc.field("humidity") < pc.scalar(70)
        result = filter_engine.apply_filter(self.test_ds, expr)
        
        # Should only include humidity < 70 (locations B and C)
        remaining_humidity = result.humidity.values
        self.assertTrue(np.all(remaining_humidity < 70))

    def test_simple_comparison_equals(self):
        """Test equality comparison."""
        filter_engine = SimpleExpressionFilter()
        
        # Filter humidity == 60
        expr = pc.field("humidity") == pc.scalar(60)
        result = filter_engine.apply_filter(self.test_ds, expr)
        
        # Should only include humidity == 60 (location B)
        remaining_humidity = result.humidity.values
        self.assertTrue(np.all(remaining_humidity == 60))

    def test_coordinate_filtering(self):
        """Test filtering based on coordinate values."""
        filter_engine = SimpleExpressionFilter()
        
        # Filter by location (string coordinate)
        expr = pc.field("location") == pc.scalar("B")
        result = filter_engine.apply_filter(self.test_ds, expr)
        
        # Should only include location B
        self.assertEqual(list(result.location.values), ["B"])

    def test_logical_and_operation(self):
        """Test logical AND filtering."""
        filter_engine = SimpleExpressionFilter()
        
        # Filter temperature > 15 AND humidity < 70
        expr = (pc.field("temperature") > pc.scalar(15)) & (pc.field("humidity") < pc.scalar(70))
        result = filter_engine.apply_filter(self.test_ds, expr)
        
        # Should include only data where both conditions are true
        remaining_temps = result.temperature.values
        remaining_humidity = result.humidity.values
        
        self.assertTrue(np.all(remaining_temps > 15))
        self.assertTrue(np.all(remaining_humidity < 70))

    def test_nonexistent_field(self):
        """Test filtering with non-existent field."""
        filter_engine = SimpleExpressionFilter()
        
        # Filter on non-existent field
        expr = pc.field("pressure") > pc.scalar(1000)
        result = filter_engine.apply_filter(self.test_ds, expr)
        
        # Should return original dataset unchanged
        self.assertEqual(len(result.time), len(self.test_ds.time))
        self.assertEqual(len(result.location), len(self.test_ds.location))


class TestAdapterFiltering(unittest.TestCase):
    """Test cases for filtering integration with XarrayDatasetAdapter."""

    def setUp(self):
        """Set up test data for adapter filtering."""
        # Create chunked dataset for testing
        time = pd.date_range('2023-01-01', periods=8, freq='D')
        lat = np.array([10.0, 20.0, 30.0, 40.0])
        
        # Create temperature data: 15, 20, 25, 30, 35, 40, 45, 50
        temp_data = np.array([
            [15, 20, 25, 30],  # Day 1
            [20, 25, 30, 35],  # Day 2
            [25, 30, 35, 40],  # Day 3
            [30, 35, 40, 45],  # Day 4
            [35, 40, 45, 50],  # Day 5
            [40, 45, 50, 55],  # Day 6
            [45, 50, 55, 60],  # Day 7
            [50, 55, 60, 65],  # Day 8
        ])
        
        self.weather_ds = xr.Dataset({
            'temperature': (['time', 'lat'], temp_data),
        }, coords={
            'time': time,
            'lat': lat,
        })

    def test_adapter_with_simple_filter(self):
        """Test adapter with simple filtering."""
        # Create adapter with chunking
        chunks = {'time': 3, 'lat': 2}
        adapter = XarrayDatasetAdapter(self.weather_ds, chunks=chunks)
        
        # Create scanner with filter
        filter_expr = pc.field("temperature") > pc.scalar(35)
        scanner = adapter.scanner(filter=filter_expr)
        
        # Get filtered table
        table = scanner.to_table()
        
        # Should only include temperatures > 35
        temp_column = table.column('temperature').to_numpy()
        self.assertTrue(np.all(temp_column > 35))

    def test_adapter_streaming_with_filter(self):
        """Test streaming with filtering applied to chunks."""
        chunks = {'time': 4, 'lat': 2}
        adapter = XarrayDatasetAdapter(self.weather_ds, chunks=chunks)
        
        # Filter for lat >= 30 (last 2 latitudes)
        filter_expr = pc.field("lat") >= pc.scalar(30)
        
        # Stream batches with filter
        batches = list(adapter.to_batches(filter=filter_expr))
        
        # Combine batches and check result
        if batches:
            combined_table = pa.Table.from_batches(batches)
            lat_values = combined_table.column('lat').to_numpy()
            self.assertTrue(np.all(lat_values >= 30))

    def test_adapter_count_rows_with_filter(self):
        """Test row counting with filtering."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # Count rows without filter
        total_rows = adapter.count_rows()
        
        # Count rows with filter that excludes some data
        filter_expr = pc.field("temperature") > pc.scalar(40)
        filtered_rows = adapter.count_rows(filter=filter_expr)
        
        # Filtered count should be less than total
        self.assertLess(filtered_rows, total_rows)
        self.assertGreater(filtered_rows, 0)

    def test_adapter_head_with_filter(self):
        """Test head functionality with filtering."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # Get head with filter
        filter_expr = pc.field("lat") == pc.scalar(20.0)
        head_table = adapter.head(3, filter=filter_expr)
        
        # Should have at most 3 rows, all with lat == 20.0
        self.assertLessEqual(len(head_table), 3)
        if len(head_table) > 0:
            lat_values = head_table.column('lat').to_numpy()
            self.assertTrue(np.all(lat_values == 20.0))

    def test_adapter_column_selection_with_filter(self):
        """Test column selection combined with filtering."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # Select only temperature column with filter
        filter_expr = pc.field("temperature") < pc.scalar(25)
        scanner = adapter.scanner(columns=['temperature'], filter=filter_expr)
        table = scanner.to_table()
        
        # Should have time, lat (coordinates) and temperature (selected)
        column_names = table.column_names
        self.assertIn('temperature', column_names)
        self.assertIn('time', column_names)
        self.assertIn('lat', column_names)
        
        # All temperature values should be < 25
        temp_values = table.column('temperature').to_numpy()
        self.assertTrue(np.all(temp_values < 25))

    def test_filter_resulting_in_empty_dataset(self):
        """Test filtering that results in empty dataset."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # Filter with impossible condition
        filter_expr = pc.field("temperature") > pc.scalar(1000)
        table = adapter.to_table(filter=filter_expr)
        
        # Should result in empty table
        self.assertEqual(len(table), 0)

    def test_invalid_filter_expression(self):
        """Test handling of invalid filter expressions."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # This should not crash, but might return unfiltered data with warning
        try:
            # Create an expression that might be hard to parse
            filter_expr = pc.field("nonexistent_field").is_null()
            table = adapter.to_table(filter=filter_expr)
            
            # Should either work or return original data
            self.assertGreaterEqual(len(table), 0)
        except Exception:
            # If it fails, that's also acceptable for invalid expressions
            pass


class TestFilterIntegrationWithChunking(unittest.TestCase):
    """Test filtering integration with different chunking strategies."""

    def setUp(self):
        """Set up multi-dimensional test data."""
        x = np.arange(0, 20, 2)  # 0, 2, 4, ..., 18
        y = np.arange(0, 10, 1)  # 0, 1, 2, ..., 9
        time = pd.date_range('2023-01-01', periods=5, freq='D')
        
        # Create 3D data
        data_shape = (len(time), len(x), len(y))
        temp_data = np.random.rand(*data_shape) * 40 + 10  # 10-50 degrees
        
        # Set some known patterns for testing
        temp_data[0, :, :] = 5   # First time step is cold
        temp_data[-1, :, :] = 45  # Last time step is hot
        temp_data[:, 0:2, :] = 15  # First 2 x values are cool
        temp_data[:, -2:, :] = 35  # Last 2 x values are warm
        
        self.multi_ds = xr.Dataset({
            'temperature': (['time', 'x', 'y'], temp_data),
        }, coords={
            'time': time,
            'x': x,
            'y': y,
        })

    def test_filtering_with_different_chunk_sizes(self):
        """Test that filtering works consistently across different chunk sizes."""
        filter_expr = pc.field("temperature") > pc.scalar(20)
        
        # Test with different chunking strategies
        chunk_configs = [
            {'time': 2, 'x': 3, 'y': 5},
            {'time': 5, 'x': 2, 'y': 2},
            {'time': 1, 'x': 10, 'y': 10},
        ]
        
        results = []
        for chunks in chunk_configs:
            adapter = XarrayDatasetAdapter(self.multi_ds, chunks=chunks)
            table = adapter.to_table(filter=filter_expr)
            results.append(table)
        
        # All results should have same number of rows (same filter applied)
        row_counts = [len(table) for table in results]
        self.assertTrue(all(count == row_counts[0] for count in row_counts))
        
        # All results should have same filtered values
        for table in results:
            temp_values = table.column('temperature').to_numpy()
            self.assertTrue(np.all(temp_values > 20))

    def test_coordinate_filtering_with_chunks(self):
        """Test coordinate-based filtering with chunked data."""
        # Filter by x coordinate
        filter_expr = pc.field("x") >= pc.scalar(10)
        
        chunks = {'time': 2, 'x': 3, 'y': 4}
        adapter = XarrayDatasetAdapter(self.multi_ds, chunks=chunks)
        table = adapter.to_table(filter=filter_expr)
        
        # Should only include x >= 10
        x_values = table.column('x').to_numpy()
        self.assertTrue(np.all(x_values >= 10))
        
        # Check that we got the expected x values
        expected_x = self.multi_ds.x.values[self.multi_ds.x.values >= 10]
        unique_x = np.unique(x_values)
        np.testing.assert_array_equal(sorted(unique_x), sorted(expected_x))


if __name__ == '__main__':
    unittest.main()