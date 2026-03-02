import numpy as np
import xarray as xr
from context import XarrayContext


# create a fake era5 dataset for testing
times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64")
lats = np.array([0.0, 1.0, 2.0])
lons = np.array([0.0, 1.0, 2.0])
levels = np.array([500, 850])

ds = xr.Dataset(
    {
        "temperature_2m": (["time", "lat", "lon"], np.random.rand(3, 3, 3)),
        "wind_speed":     (["time", "lat", "lon"], np.random.rand(3, 3, 3)),
        "pressure":       (["time", "lat", "lon", "level"], np.random.rand(3, 3, 3, 2)),
        "humidity":       (["time", "lat", "lon", "level"], np.random.rand(3, 3, 3, 2)),
    },
    coords={
        "time":  times,
        "lat":   lats,
        "lon":   lons,
        "level": levels,
    }
).chunk({"time": 1})

print("Variables:", list(ds.data_vars))
print("Dimensions:", list(ds.dims))

ctx = XarrayContext()
ctx.register_catalog_from_dataset(ds)

print("\nCatalogs:", ctx.catalog_names())
print("Schemas:", ctx.catalog("xarray").schema_names())
print("Tables:", ctx.catalog("xarray").schema("data").table_names())

print("\n--- Surface variables (time, lat, lon) ---")
result = ctx.sql("SELECT * FROM xarray.data.time_lat_lon LIMIT 5").collect()
for batch in result:
    print(batch.to_pandas())


print("\n--- Atmospheric variables (time, lat, lon, level) ---")
result = ctx.sql("SELECT * FROM xarray.data.time_lat_lon_level LIMIT 5").collect()
for batch in result:
    print(batch.to_pandas())

print("\n--- Joined surface + atmospheric on shared dims ---")
result = ctx.sql("""
    SELECT
        s.time, s.lat, s.lon,
        s.temperature_2m,
        a.level,
        a.pressure
    FROM xarray.data.time_lat_lon s
    JOIN xarray.data.time_lat_lon_level a
        ON s.time = a.time
        AND s.lat = a.lat
        AND s.lon = a.lon
    LIMIT 10
""").collect()
for batch in result:
    print(batch.to_pandas())