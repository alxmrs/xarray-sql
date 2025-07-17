#!/usr/bin/env python3
"""
Create diverse test Zarr datasets for comprehensive SQL testing.
"""

import numpy as np
import xarray as xr
import os
import shutil

def create_weather_dataset():
    """Create a weather dataset with temperature, pressure, humidity."""
    print("Creating weather dataset...")
    
    # 4D dataset: time (5), lat (3), lon (4), altitude (2)
    time = np.arange(0, 5)  # 5 time points
    lat = np.array([30.0, 35.0, 40.0])  # 3 latitudes
    lon = np.array([-120.0, -115.0, -110.0, -105.0])  # 4 longitudes  
    altitude = np.array([0, 1000])  # 2 altitude levels (0m, 1000m)
    
    # Create 4D data arrays
    shape = (5, 3, 4, 2)  # 120 total points
    
    # Temperature: varies by lat, decreases with altitude
    temperature_data = np.random.normal(20, 5, shape)
    for alt_idx in range(2):
        for lat_idx in range(3):
            temperature_data[:, lat_idx, :, alt_idx] += (lat[lat_idx] - 35) * 0.5 - alt_idx * 10
    
    # Pressure: decreases with altitude, varies by location
    pressure_data = np.random.normal(1013, 20, shape)
    for alt_idx in range(2):
        pressure_data[:, :, :, alt_idx] -= alt_idx * 100
    
    # Humidity: random but realistic
    humidity_data = np.random.uniform(30, 90, shape)
    
    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon', 'altitude'], temperature_data),
        'pressure': (['time', 'lat', 'lon', 'altitude'], pressure_data),
        'humidity': (['time', 'lat', 'lon', 'altitude'], humidity_data),
    }, coords={
        'time': time,
        'lat': lat,
        'lon': lon,
        'altitude': altitude,
    })
    
    zarr_path = '../test_data/weather.zarr'
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    ds.to_zarr(zarr_path)
    
    print(f"✅ Created weather dataset: {zarr_path}")
    print(f"   Shape: {shape} = {np.prod(shape)} rows")
    print(f"   Variables: temperature, pressure, humidity")
    return zarr_path

def create_ocean_dataset():
    """Create an ocean dataset with different dimensions (3D)."""
    print("\nCreating ocean dataset...")
    
    # 3D dataset: depth (4), lat (5), lon (6) 
    depth = np.array([0, 10, 50, 100])  # 4 depth levels
    lat = np.array([25.0, 30.0, 35.0, 40.0, 45.0])  # 5 latitudes
    lon = np.array([-130.0, -125.0, -120.0, -115.0, -110.0, -105.0])  # 6 longitudes
    
    shape = (4, 5, 6)  # 120 total points
    
    # Sea temperature: decreases with depth and varies by latitude
    sea_temp_data = np.zeros(shape)
    for depth_idx in range(4):
        for lat_idx in range(5):
            sea_temp_data[depth_idx, lat_idx, :] = 25 + (lat[lat_idx] - 35) * 0.3 - depth[depth_idx] * 0.1
    
    # Salinity: varies by location and depth
    salinity_data = np.random.normal(35, 1, shape)
    for depth_idx in range(4):
        salinity_data[depth_idx, :, :] += depth_idx * 0.2
    
    ds = xr.Dataset({
        'sea_temperature': (['depth', 'lat', 'lon'], sea_temp_data),
        'salinity': (['depth', 'lat', 'lon'], salinity_data),
    }, coords={
        'depth': depth,
        'lat': lat,  # Same lat coordinates as weather for potential joins
        'lon': lon,
    })
    
    zarr_path = '../test_data/ocean.zarr'
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    ds.to_zarr(zarr_path)
    
    print(f"✅ Created ocean dataset: {zarr_path}")
    print(f"   Shape: {shape} = {np.prod(shape)} rows")
    print(f"   Variables: sea_temperature, salinity")
    return zarr_path

def create_simple_timeseries():
    """Create a simple 2D time series for basic testing."""
    print("\nCreating simple timeseries dataset...")
    
    # 2D dataset: time (10), station (3)
    time = np.arange(0, 10)  # 10 time points
    station = np.array([1, 2, 3])  # 3 stations
    
    shape = (10, 3)  # 30 total points
    
    # Simple metrics
    value_data = np.random.normal(100, 10, shape)
    count_data = np.random.poisson(5, shape)
    
    ds = xr.Dataset({
        'value': (['time', 'station'], value_data),
        'count': (['time', 'station'], count_data.astype(float)),
    }, coords={
        'time': time,
        'station': station,
    })
    
    zarr_path = '../test_data/timeseries.zarr'
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    ds.to_zarr(zarr_path)
    
    print(f"✅ Created timeseries dataset: {zarr_path}")
    print(f"   Shape: {shape} = {np.prod(shape)} rows")
    print(f"   Variables: value, count")
    return zarr_path

def create_single_dimension_dataset():
    """Create a 1D dataset for testing edge cases."""
    print("\nCreating single dimension dataset...")
    
    # 1D dataset: just index (8)
    index = np.arange(0, 8)
    
    shape = (8,)  # 8 total points
    
    # Single variable
    measurement_data = np.array([10.5, 15.2, 20.1, 18.7, 12.3, 8.9, 14.6, 22.1])
    
    ds = xr.Dataset({
        'measurement': (['index'], measurement_data),
    }, coords={
        'index': index,
    })
    
    zarr_path = '../test_data/single_dim.zarr'
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    ds.to_zarr(zarr_path)
    
    print(f"✅ Created single dimension dataset: {zarr_path}")
    print(f"   Shape: {shape} = {np.prod(shape)} rows")
    print(f"   Variables: measurement")
    return zarr_path

def create_large_sparse_dataset():
    """Create a larger dataset with some interesting patterns for aggregation testing."""
    print("\nCreating large sparse dataset...")
    
    # 3D dataset: category (4), region (6), period (8)
    category = np.array([0, 1, 2, 3])  # 4 categories
    region = np.arange(0, 6)  # 6 regions
    period = np.arange(0, 8)  # 8 periods
    
    shape = (4, 6, 8)  # 192 total points
    
    # Create pattern: some categories are more active in certain regions/periods
    activity_data = np.zeros(shape)
    revenue_data = np.zeros(shape)
    
    for cat in range(4):
        for reg in range(6):
            for per in range(8):
                # Category patterns
                if cat == 0:  # Category 0 active in first half
                    activity_data[cat, reg, per] = max(0, 100 - per * 10 + np.random.normal(0, 5))
                elif cat == 1:  # Category 1 active in certain regions
                    activity_data[cat, reg, per] = max(0, reg * 15 + np.random.normal(0, 8))
                elif cat == 2:  # Category 2 has seasonal pattern
                    activity_data[cat, reg, per] = max(0, 50 + 30 * np.sin(per * np.pi / 4) + np.random.normal(0, 10))
                else:  # Category 3 is sparse
                    activity_data[cat, reg, per] = max(0, np.random.exponential(5) if np.random.random() > 0.6 else 0)
                
                # Revenue correlated with activity
                revenue_data[cat, reg, per] = activity_data[cat, reg, per] * (2 + np.random.normal(0, 0.5))
    
    ds = xr.Dataset({
        'activity': (['category', 'region', 'period'], activity_data),
        'revenue': (['category', 'region', 'period'], revenue_data),
    }, coords={
        'category': category,
        'region': region,
        'period': period,
    })
    
    zarr_path = '../test_data/business.zarr'
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    ds.to_zarr(zarr_path)
    
    print(f"✅ Created business dataset: {zarr_path}")
    print(f"   Shape: {shape} = {np.prod(shape)} rows")
    print(f"   Variables: activity, revenue")
    return zarr_path

if __name__ == "__main__":
    try:
        # Create test data directory
        os.makedirs('../test_data', exist_ok=True)
        
        print("🏗️  Creating diverse test datasets for SQL integration tests...\n")
        
        # Create all test datasets
        datasets = []
        datasets.append(create_weather_dataset())
        datasets.append(create_ocean_dataset())
        datasets.append(create_simple_timeseries())
        datasets.append(create_single_dimension_dataset())
        datasets.append(create_large_sparse_dataset())
        
        print(f"\n🎉 Successfully created {len(datasets)} test datasets!")
        print("\n📊 Dataset Summary:")
        print("   1. weather.zarr     - 4D (time×lat×lon×altitude) - temperature, pressure, humidity")
        print("   2. ocean.zarr       - 3D (depth×lat×lon) - sea_temperature, salinity")
        print("   3. timeseries.zarr  - 2D (time×station) - value, count")
        print("   4. single_dim.zarr  - 1D (index) - measurement")
        print("   5. business.zarr    - 3D (category×region×period) - activity, revenue")
        
        print("\n🔗 Join Testing Opportunities:")
        print("   • Weather ⋈ Ocean: matching lat coordinates")
        print("   • Different dimensionalities: 4D ⋈ 3D ⋈ 2D ⋈ 1D")
        print("   • Time-based joins: weather.time ⋈ timeseries.time")
        print("   • Categorical joins: various coordinate-based relationships")
        
        print(f"\n💡 Ready for SQL integration tests!")
        print("   Run: cargo run --example sql_integration_tests")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install xarray numpy")
    except Exception as e:
        print(f"❌ Error creating test datasets: {e}")