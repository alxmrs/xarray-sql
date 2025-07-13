#!/usr/bin/env python3
"""
Unit tests for the XarrayDatasetAdapter and related classes.

Tests the PyArrow Dataset-compatible interface for xarray.Dataset
with lazy chunking and streaming capabilities.
"""

import unittest
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
from datetime import datetime, timedelta

from xarray_sql.ds import XarrayDatasetAdapter, SchemaMapper, XarrayScanner


class TestSchemaMapper(unittest.TestCase):
    """Test cases for schema translation between xarray and PyArrow."""

    def setUp(self):
        """Set up test datasets."""
        # Simple dataset with various data types
        self.simple_ds = xr.Dataset({
            'temperature': (['time', 'location'], 
                           np.random.rand(5, 3).astype(np.float32)),
            'humidity': (['time', 'location'], 
                        np.random.rand(5, 3).astype(np.float64)),
            'active': (['location'], [True, False, True]),
            'count': (['time'], np.arange(5, dtype=np.int32)),
        }, coords={
            'time': pd.date_range('2023-01-01', periods=5, freq='D'),
            'location': ['A', 'B', 'C'],
        })

    def test_schema_creation_basic_types(self):
        """Test schema creation with basic data types."""
        schema = SchemaMapper.from_xarray(self.simple_ds)
        
        # Check that all coordinates and data vars are present
        field_names = [field.name for field in schema]
        expected_fields = ['time', 'location', 'temperature', 'humidity', 'active', 'count']
        
        for field in expected_fields:
            self.assertIn(field, field_names)
        
        # Check specific type mappings
        field_dict = {field.name: field.type for field in schema}
        
        # Coordinates
        self.assertIsInstance(field_dict['time'], pa.TimestampType)
        self.assertEqual(field_dict['location'], pa.string())
        
        # Data variables
        self.assertEqual(field_dict['temperature'], pa.float32())
        self.assertEqual(field_dict['humidity'], pa.float64())
        self.assertEqual(field_dict['active'], pa.bool_())
        self.assertEqual(field_dict['count'], pa.int32())

    def test_numpy_to_arrow_type_conversion(self):
        """Test individual numpy to arrow type conversions."""
        # Integer types
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.int8), pa.int8())
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.int16), pa.int16())
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.int32), pa.int32())
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.int64), pa.int64())
        
        # Float types
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.float32), pa.float32())
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.float64), pa.float64())
        
        # Boolean
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.bool_), pa.bool_())
        
        # Datetime
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.datetime64), pa.timestamp('ns'))
        
        # String types
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.dtype('U10')), pa.string())
        self.assertEqual(SchemaMapper._numpy_to_arrow_type(np.dtype('S10')), pa.string())


class TestXarrayDatasetAdapter(unittest.TestCase):
    """Test cases for the XarrayDatasetAdapter class."""

    def setUp(self):
        """Set up test datasets with different chunking patterns."""
        # 2D dataset for chunking tests
        time = pd.date_range('2023-01-01', periods=10, freq='D')
        location = ['A', 'B', 'C', 'D']
        
        temp_data = np.random.rand(10, 4) * 30 + 10  # 10-40 degrees
        pressure_data = np.random.rand(10, 4) * 100 + 1000  # 1000-1100 hPa
        
        self.weather_ds = xr.Dataset({
            'temperature': (['time', 'location'], temp_data),
            'pressure': (['time', 'location'], pressure_data),
        }, coords={
            'time': time,
            'location': location,
        })
        
        # Add explicit chunking
        self.chunked_ds = self.weather_ds.chunk({'time': 3, 'location': 2})
        
        # Small dataset for testing edge cases
        self.tiny_ds = xr.Dataset({
            'value': (['x'], [1, 2, 3]),
        }, coords={
            'x': [10, 20, 30],
        })

    def test_adapter_initialization(self):
        """Test basic adapter initialization."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # Check basic properties
        self.assertIsInstance(adapter.schema, pa.Schema)
        self.assertEqual(len(adapter._blocks), 1)  # Single chunk for non-chunked data
        
        # Test with explicit chunking
        chunks = {'time': 5, 'location': 2}
        chunked_adapter = XarrayDatasetAdapter(self.weather_ds, chunks=chunks)
        self.assertGreater(len(chunked_adapter._blocks), 1)

    def test_schema_property(self):
        """Test schema property returns correct PyArrow schema."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        schema = adapter.schema
        
        field_names = [field.name for field in schema]
        expected_fields = ['time', 'location', 'temperature', 'pressure']
        
        for field in expected_fields:
            self.assertIn(field, field_names)

    def test_scanner_creation(self):
        """Test scanner creation with various options."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # Basic scanner
        scanner = adapter.scanner()
        self.assertIsInstance(scanner, XarrayScanner)
        self.assertIsNone(scanner._filter)
        self.assertIsNone(scanner._columns)
        
        # Scanner with column selection
        scanner_cols = adapter.scanner(columns=['temperature'])
        self.assertEqual(scanner_cols._columns, ['temperature'])
        
        # Scanner with batch size
        scanner_batch = adapter.scanner(batch_size=500)
        self.assertEqual(scanner_batch._batch_size, 500)

    def test_get_fragments(self):
        """Test fragment enumeration."""
        chunks = {'time': 4, 'location': 2}
        adapter = XarrayDatasetAdapter(self.weather_ds, chunks=chunks)
        
        fragments = list(adapter.get_fragments())
        self.assertGreater(len(fragments), 1)
        
        # Check fragment structure
        for frag in fragments:
            self.assertIn('fragment_id', frag)
            self.assertIn('block', frag)
            self.assertIn('estimated_rows', frag)
            self.assertIsInstance(frag['estimated_rows'], int)
            self.assertGreater(frag['estimated_rows'], 0)

    def test_block_size_estimation(self):
        """Test block size estimation without loading data."""
        adapter = XarrayDatasetAdapter(self.tiny_ds)
        
        # Should estimate correctly for simple case
        for block in adapter._blocks:
            estimated_size = adapter._estimate_block_size(block)
            self.assertEqual(estimated_size, 3)  # 3 points in tiny dataset

    def test_direct_table_conversion(self):
        """Test direct conversion to PyArrow table."""
        adapter = XarrayDatasetAdapter(self.tiny_ds)
        table = adapter.to_table()
        
        self.assertIsInstance(table, pa.Table)
        self.assertEqual(len(table), 3)  # 3 rows
        self.assertEqual(len(table.column_names), 2)  # x coord + value data var
        
        # Check column names
        self.assertIn('x', table.column_names)
        self.assertIn('value', table.column_names)

    def test_streaming_batches(self):
        """Test streaming conversion to batches."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        batches = list(adapter.to_batches())
        
        self.assertGreater(len(batches), 0)
        
        for batch in batches:
            self.assertIsInstance(batch, pa.RecordBatch)
            self.assertGreater(len(batch), 0)
        
        # Combine all batches should equal full table
        combined_table = pa.Table.from_batches(batches)
        direct_table = adapter.to_table()
        
        self.assertEqual(len(combined_table), len(direct_table))
        self.assertEqual(combined_table.column_names, direct_table.column_names)

    def test_count_rows(self):
        """Test row counting functionality."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        row_count = adapter.count_rows()
        
        # Should equal total size of dataset after pivot
        expected_rows = self.weather_ds.sizes['time'] * self.weather_ds.sizes['location']
        self.assertEqual(row_count, expected_rows)

    def test_head_functionality(self):
        """Test head() method for sampling."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        
        # Test default head
        head_table = adapter.head()
        self.assertLessEqual(len(head_table), 5)
        
        # Test custom head size
        head_10 = adapter.head(10)
        self.assertLessEqual(len(head_10), 10)
        
        # Test larger than dataset
        head_large = adapter.head(1000)
        full_table = adapter.to_table()
        self.assertEqual(len(head_large), len(full_table))

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        empty_ds = xr.Dataset({}, coords={})
        adapter = XarrayDatasetAdapter(empty_ds)
        
        table = adapter.to_table()
        self.assertEqual(len(table), 0)
        
        batches = list(adapter.to_batches())
        self.assertEqual(len(batches), 0)
        
        count = adapter.count_rows()
        self.assertEqual(count, 0)


class TestXarrayScanner(unittest.TestCase):
    """Test cases for the XarrayScanner class."""

    def setUp(self):
        """Set up test data."""
        self.ds = xr.Dataset({
            'temp': (['time', 'loc'], np.random.rand(6, 2)),
            'humid': (['time', 'loc'], np.random.rand(6, 2)),
        }, coords={
            'time': pd.date_range('2023-01-01', periods=6, freq='D'),
            'loc': ['X', 'Y'],
        })
        
        # Create chunked version
        chunks = {'time': 2, 'loc': 1}
        self.adapter = XarrayDatasetAdapter(self.ds, chunks=chunks)

    def test_column_selection(self):
        """Test column selection in scanner."""
        # Select only temperature
        scanner = self.adapter.scanner(columns=['temp'])
        table = scanner.to_table()
        
        # Should have time, loc (coordinates) + temp (selected data var)
        self.assertIn('temp', table.column_names)
        self.assertIn('time', table.column_names)  # Coordinate preserved
        self.assertIn('loc', table.column_names)   # Coordinate preserved
        
        # Should NOT have humidity
        self.assertNotIn('humid', table.column_names)

    def test_batch_streaming(self):
        """Test streaming batches with chunked data."""
        scanner = self.adapter.scanner(batch_size=100)
        batches = list(scanner.to_batches())
        
        # Should have multiple batches due to chunking
        self.assertGreater(len(batches), 1)
        
        # Each batch should be a RecordBatch
        for batch in batches:
            self.assertIsInstance(batch, pa.RecordBatch)
            self.assertGreater(len(batch), 0)

    def test_scanner_to_table_conversion(self):
        """Test scanner table conversion."""
        scanner = self.adapter.scanner()
        table = scanner.to_table()
        
        # Should match direct adapter conversion
        direct_table = self.adapter.to_table()
        self.assertEqual(len(table), len(direct_table))
        self.assertEqual(table.column_names, direct_table.column_names)

    def test_scanner_head_with_chunks(self):
        """Test head functionality with chunked data."""
        scanner = self.adapter.scanner()
        head_table = scanner.head(3)
        
        self.assertLessEqual(len(head_table), 3)
        self.assertEqual(head_table.column_names, self.adapter.schema.names)

    def test_scanner_count_rows(self):
        """Test row counting through scanner."""
        scanner = self.adapter.scanner()
        count = scanner.count_rows()
        
        expected_count = self.ds.sizes['time'] * self.ds.sizes['loc']
        self.assertEqual(count, expected_count)


class TestIntegrationWithExistingCode(unittest.TestCase):
    """Test integration with existing xarray-sql components."""

    def setUp(self):
        """Set up test data using patterns from existing tests."""
        # Use similar pattern as df_test.py
        time = pd.date_range('2023-01-01', periods=8, freq='D')
        lat = np.linspace(-90, 90, 4)
        
        temp_data = np.random.rand(8, 4) * 40 - 10
        
        self.weather_ds = xr.Dataset({
            'temperature': (['time', 'lat'], temp_data),
        }, coords={
            'time': time,
            'lat': lat,
        })

    def test_pivot_integration(self):
        """Test that adapter properly uses existing pivot function."""
        adapter = XarrayDatasetAdapter(self.weather_ds)
        table = adapter.to_table()
        
        # Compare with direct pivot usage
        from xarray_sql.df import pivot
        direct_df = pivot(self.weather_ds)
        direct_table = pa.Table.from_pandas(direct_df)
        
        # Should have same structure
        self.assertEqual(len(table), len(direct_table))
        self.assertEqual(table.column_names, direct_table.column_names)

    def test_block_slices_integration(self):
        """Test integration with existing block_slices function."""
        from xarray_sql.core import block_slices
        
        chunks = {'time': 3, 'lat': 2}
        adapter = XarrayDatasetAdapter(self.weather_ds, chunks=chunks)
        
        # Should use same blocks as direct block_slices call
        direct_blocks = list(block_slices(self.weather_ds, chunks))
        self.assertEqual(len(adapter._blocks), len(direct_blocks))
        
        # Blocks should be equivalent
        for adapter_block, direct_block in zip(adapter._blocks, direct_blocks):
            self.assertEqual(adapter_block, direct_block)

    def test_chunked_dataset_compatibility(self):
        """Test compatibility with pre-chunked xarray datasets."""
        # Create pre-chunked dataset
        chunked_ds = self.weather_ds.chunk({'time': 4, 'lat': 2})
        
        # Adapter should work with pre-chunked data
        adapter = XarrayDatasetAdapter(chunked_ds)
        table = adapter.to_table()
        
        self.assertIsInstance(table, pa.Table)
        self.assertGreater(len(table), 0)


if __name__ == '__main__':
    unittest.main()