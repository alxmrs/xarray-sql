#!/usr/bin/env python3
"""
Create test Zarr datasets for testing the multi-variable implementation.
"""

import numpy as np
import xarray as xr
import os
import shutil

def create_multi_variable_zarr():
    """Create a simple multi-variable Zarr dataset for testing."""
    
    print("Creating test Zarr dataset...")
    
    # Create coordinate arrays
    time = np.arange(0, 3)  # 3 time points
    lat = np.arange(0, 2)   # 2 lat points  
    lon = np.arange(0, 2)   # 2 lon points
    
    # Create data variables with known patterns
    # Temperature: simple incremental values
    temperature_data = np.arange(1.0, 13.0).reshape(3, 2, 2)
    
    # Pressure: different pattern (multiples of 100)
    pressure_data = np.arange(100.0, 1300.0, 100.0).reshape(3, 2, 2)
    
    print("Data shapes:")
    print(f"  time: {time.shape}")
    print(f"  lat: {lat.shape}")  
    print(f"  lon: {lon.shape}")
    print(f"  temperature: {temperature_data.shape}")
    print(f"  pressure: {pressure_data.shape}")
    
    print("\nSample data values:")
    print(f"  temperature[0,0,:] = {temperature_data[0,0,:]}")
    print(f"  pressure[0,0,:] = {pressure_data[0,0,:]}")
    
    # Create xarray dataset
    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], temperature_data),
        'pressure': (['time', 'lat', 'lon'], pressure_data),
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon,
    })
    
    print(f"\nDataset structure:")
    print(ds)
    
    # Save as Zarr
    zarr_path = './test_data/multi_var.zarr'
    
    # Remove existing directory if it exists
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    
    # Create parent directory
    os.makedirs('./test_data', exist_ok=True)
    
    # Save to Zarr format
    ds.to_zarr(zarr_path)
    
    print(f"\n✅ Created Zarr dataset at: {zarr_path}")
    
    # Print expected output for verification
    print("\n🎯 Expected table structure (3×2×2 = 12 rows):")
    print("   Columns: [dim_0, dim_1, dim_2, temperature, pressure]")
    print("   Sample rows:")
    
    for t in range(3):
        for lat_idx in range(2):
            for lon_idx in range(2):
                temp_val = temperature_data[t, lat_idx, lon_idx]
                press_val = pressure_data[t, lat_idx, lon_idx]
                print(f"     Row: [{t}, {lat_idx}, {lon_idx}, {temp_val:.1f}, {press_val:.1f}]")
    
    return zarr_path

def create_inconsistent_zarr():
    """Create a Zarr dataset with inconsistent dimensions for testing error handling."""
    
    print("\nCreating inconsistent dimension test dataset...")
    
    # Different shapes - this should trigger our validation error
    time = np.arange(0, 3)
    lat = np.arange(0, 2)
    lon = np.arange(0, 2)
    
    # Temperature: shape (3, 2, 2)
    temperature_data = np.random.rand(3, 2, 2)
    
    # Pressure: different shape (3, 2) - missing lon dimension
    pressure_data = np.random.rand(3, 2)
    
    # Create individual arrays (can't use xarray Dataset due to shape mismatch)
    temp_ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], temperature_data),
    }, coords={'time': time, 'lat': lat, 'lon': lon})
    
    press_ds = xr.Dataset({
        'pressure': (['time', 'lat'], pressure_data),
    }, coords={'time': time, 'lat': lat})
    
    # Save separately and then manually combine directory structure
    zarr_path = './test_data/inconsistent.zarr'
    
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    
    # This is a bit hacky, but we'll create the structure manually
    temp_ds.temperature.to_zarr(zarr_path + '/temperature')
    press_ds.pressure.to_zarr(zarr_path + '/pressure')
    
    print(f"✅ Created inconsistent Zarr dataset at: {zarr_path}")
    print("   This should trigger dimension consistency errors")
    
    return zarr_path

if __name__ == "__main__":
    try:
        # Create test datasets
        consistent_path = create_multi_variable_zarr()
        inconsistent_path = create_inconsistent_zarr()
        
        print(f"\n🎉 Test datasets created successfully!")
        print(f"   Consistent: {consistent_path}")
        print(f"   Inconsistent: {inconsistent_path}")
        print(f"\nNow run: cargo run --example test_real_zarr")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install xarray numpy")
    except Exception as e:
        print(f"❌ Error creating test data: {e}")