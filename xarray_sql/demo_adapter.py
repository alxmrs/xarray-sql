#!/usr/bin/env python3
"""
Demonstration of the XarrayDatasetAdapter functionality.

Shows how to use the PyArrow Dataset-compatible adapter for xarray.Dataset
with lazy chunking and filtering capabilities.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pyarrow.compute as pc

from xarray_sql.ds import XarrayDatasetAdapter


def create_sample_weather_data():
    """Create a sample weather dataset for demonstration."""
    # Create coordinates
    time = pd.date_range('2023-01-01', periods=10, freq='D')
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 8)
    
    # Create data with patterns for testing
    temp_data = np.random.rand(10, 5, 8) * 40 - 10  # Range: -10 to 30
    temp_data[0:3, :, :] = 25  # First 3 days are warm
    temp_data[7:, :, :] = -5   # Last 3 days are cold
    
    humidity_data = np.random.rand(10, 5, 8) * 100  # Range: 0 to 100
    humidity_data[:, 0:2, :] = 85  # High humidity in first 2 latitudes
    
    pressure_data = np.random.rand(10, 5, 8) * 100 + 1000  # 1000-1100 hPa
    
    return xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], temp_data),
        'humidity': (['time', 'lat', 'lon'], humidity_data),
        'pressure': (['time', 'lat', 'lon'], pressure_data),
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon,
    })


def demo_basic_adapter():
    """Demonstrate basic adapter functionality."""
    print("=== Basic Adapter Functionality ===")
    
    # Create sample data
    weather_ds = create_sample_weather_data()
    print(f"Original dataset shape: {weather_ds.dims}")
    
    # Create adapter
    adapter = XarrayDatasetAdapter(weather_ds)
    print(f"Adapter schema: {adapter.schema}")
    
    # Basic operations
    print(f"Total rows: {adapter.count_rows()}")
    
    # Get first few rows
    head_table = adapter.head(5)
    print(f"Head table shape: {len(head_table)} rows x {len(head_table.column_names)} columns")
    print(f"Column names: {head_table.column_names}")
    
    print()


def demo_chunking():
    """Demonstrate chunking functionality."""
    print("=== Chunking and Streaming ===")
    
    weather_ds = create_sample_weather_data()
    
    # Create adapter with chunking
    chunks = {'time': 3, 'lat': 2, 'lon': 4}
    adapter = XarrayDatasetAdapter(weather_ds, chunks=chunks)
    
    # Show fragment information
    fragments = list(adapter.get_fragments())
    print(f"Number of chunks: {len(fragments)}")
    for i, frag in enumerate(fragments[:3]):  # Show first 3
        print(f"  Chunk {i}: {frag['estimated_rows']} estimated rows")
    
    # Stream as batches
    batch_count = 0
    total_rows = 0
    for batch in adapter.to_batches():
        batch_count += 1
        total_rows += len(batch)
    
    print(f"Streaming: {batch_count} batches, {total_rows} total rows")
    print()


def demo_filtering():
    """Demonstrate filtering functionality."""
    print("=== Filtering with PyArrow Expressions ===")
    
    weather_ds = create_sample_weather_data()
    adapter = XarrayDatasetAdapter(weather_ds)
    
    # Total rows without filtering
    total_rows = adapter.count_rows()
    print(f"Total rows (no filter): {total_rows}")
    
    # Filter by temperature
    temp_filter = pc.field("temperature") > pc.scalar(20)
    temp_filtered_rows = adapter.count_rows(filter=temp_filter)
    print(f"Rows with temperature > 20: {temp_filtered_rows}")
    
    # Filter by coordinate (latitude)
    lat_filter = pc.field("lat") > pc.scalar(0)
    lat_filtered_rows = adapter.count_rows(filter=lat_filter)
    print(f"Rows with lat > 0: {lat_filtered_rows}")
    
    # Combined filter
    combined_filter = (pc.field("temperature") > pc.scalar(0)) & (pc.field("humidity") < pc.scalar(90))
    combined_filtered_rows = adapter.count_rows(filter=combined_filter)
    print(f"Rows with temp > 0 AND humidity < 90: {combined_filtered_rows}")
    
    # Get actual filtered data
    filtered_table = adapter.to_table(filter=temp_filter)
    if len(filtered_table) > 0:
        temp_values = filtered_table.column('temperature').to_numpy()
        print(f"Filtered temperature range: {temp_values.min():.1f} to {temp_values.max():.1f}")
        print(f"All temperatures > 20: {np.all(temp_values > 20)}")
    
    print()


def demo_column_selection():
    """Demonstrate column selection."""
    print("=== Column Selection ===")
    
    weather_ds = create_sample_weather_data()
    adapter = XarrayDatasetAdapter(weather_ds)
    
    # Select only temperature
    temp_table = adapter.to_table(columns=['temperature'])
    print(f"Temperature-only table columns: {temp_table.column_names}")
    
    # Select temperature and humidity with filtering
    filter_expr = pc.field("lat") >= pc.scalar(30)
    selected_table = adapter.to_table(
        columns=['temperature', 'humidity'], 
        filter=filter_expr
    )
    print(f"Selected columns with filter: {selected_table.column_names}")
    print(f"Filtered table rows: {len(selected_table)}")
    
    print()


def demo_integration_with_pyarrow():
    """Demonstrate integration with PyArrow ecosystem."""
    print("=== Integration with PyArrow Ecosystem ===")
    
    weather_ds = create_sample_weather_data()
    adapter = XarrayDatasetAdapter(weather_ds, chunks={'time': 4, 'lat': 2})
    
    # Get PyArrow table
    table = adapter.to_table(filter=pc.field("temperature") > pc.scalar(10))
    
    # Use PyArrow compute functions
    temp_column = table.column('temperature')
    mean_temp = pc.mean(temp_column).as_py()
    max_temp = pc.max(temp_column).as_py()
    
    print(f"Filtered data statistics:")
    print(f"  Mean temperature: {mean_temp:.2f}")
    print(f"  Max temperature: {max_temp:.2f}")
    
    # Group by operations (example)
    try:
        # This would work if we had categorical data
        unique_times = pc.unique(table.column('time'))
        print(f"  Unique time points: {len(unique_times)}")
    except Exception as e:
        print(f"  Time analysis: {len(table)} data points")
    
    print()


def main():
    """Run all demonstrations."""
    print("XarrayDatasetAdapter Demonstration")
    print("=" * 50)
    print()
    
    demo_basic_adapter()
    demo_chunking()
    demo_filtering()
    demo_column_selection()
    demo_integration_with_pyarrow()
    
    print("=== Summary ===")
    print("✅ PyArrow Dataset-compatible interface")
    print("✅ Lazy chunking and streaming")
    print("✅ PyArrow expression filtering")
    print("✅ Column selection")
    print("✅ Integration with PyArrow ecosystem")
    print("✅ Maintains xarray chunking benefits")


if __name__ == "__main__":
    main()