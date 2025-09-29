import itertools
import tracemalloc

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from .df import explode, read_xarray, block_slices, from_map, pivot, from_map_batched


def rand_wx(start: str, end: str) -> xr.Dataset:
    np.random.seed(42)
    lat = np.linspace(-90, 90, num=720)
    lon = np.linspace(-180, 180, num=1440)
    time = pd.date_range(start, end, freq="h")
    level = np.array([1000, 500], dtype=np.int32)
    reference_time = pd.Timestamp(start)
    temperature = 15 + 8 * np.random.randn(720, 1440, len(time), len(level))
    precipitation = 10 * np.random.rand(720, 1440, len(time), len(level))
    return xr.Dataset(
        data_vars=dict(
            temperature=(["lat", "lon", "time", "level"], temperature),
            precipitation=(["lat", "lon", "time", "level"], precipitation),
        ),
        coords=dict(
            lat=lat,
            lon=lon,
            time=time,
            level=level,
            reference_time=reference_time,
        ),
        attrs=dict(description="Random weather."),
    )


def create_large_dataset(time_steps=1000, lat_points=100, lon_points=100):
    """Create a large xarray dataset for memory testing."""
    np.random.seed(42)

    time = pd.date_range("2020-01-01", periods=time_steps, freq="h")
    lat = np.linspace(-90, 90, lat_points)
    lon = np.linspace(-180, 180, lon_points)

    temp_data = np.random.rand(time_steps, lat_points, lon_points) * 40 - 10
    precip_data = np.random.rand(time_steps, lat_points, lon_points) * 100

    return xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp_data),
            "precipitation": (["time", "lat", "lon"], precip_data),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def adding_function(x, y):
    """Simple function that adds two values and returns a DataFrame."""
    result = pd.DataFrame({"x": [x], "y": [y], "sum": [x + y]})
    return result



@pytest.fixture
def air():
    ds = xr.tutorial.open_dataset("air_temperature")
    chunks = {"time": 240}
    return ds.chunk(chunks)


@pytest.fixture
def air_small(air):
    return air.isel(time=slice(0, 12), lat=slice(0, 11), lon=slice(0, 10)).chunk({"time": 240})


@pytest.fixture
def randwx():
    return rand_wx("1995-01-13T00", "1995-01-13T01")


@pytest.fixture
def large_ds():
    return create_large_dataset().chunk({"time": 25})



def test_explode_cardinality(air):
    dss = explode(air)
    assert len(list(dss)) == np.prod([len(c) for c in air.chunks.values()])


def test_explode_dim_sizes_one(air):
    chunks = {"time": 240}
    ds = next(iter(explode(air)))
    for k, v in chunks.items():
        assert k in ds.dims
        assert v == ds.sizes[k]


@pytest.mark.skip(reason="TODO(alxmrs): Why is this test slow?") # this was the original comment
def test_explode_dim_sizes_all(air):
    dss = explode(air)
    assert [tuple(ds.dims.values()) for ds in dss] == list(itertools.product(*air.chunksizes.values()))


def test_explode_data_equal_one_first(air):
    ds = next(iter(explode(air)))
    iselection = {dim: slice(0, s) for dim, s in ds.sizes.items()}
    assert air.isel(iselection).equals(ds)


def test_explode_data_equal_one_last(air):
    dss = list(explode(air))
    ds = dss[-1]
    iselection = {dim: slice(0, s) for dim, s in ds.sizes.items()}
    assert air.isel(iselection).equals(ds)


def test_from_map_basic():
    def make_df(x):
        return pd.DataFrame({"value": [x, x * 2], "index": [0, 1]})

    result = from_map(make_df, [1, 2, 3])
    assert isinstance(result, pa.Table)
    assert len(result) == 6
    assert result.column_names == ["value", "index"]


def test_from_map_multiple_iterables():
    def add_values(x, y):
        return pd.DataFrame({"sum": [x + y], "x": [x], "y": [y]})

    result = from_map(add_values, [1, 2], [10, 20])
    assert isinstance(result, pa.Table)
    assert len(result) == 2

    df = result.to_pandas()
    assert list(df["sum"]) == [11, 22]


def test_from_map_with_args():
    def multiply_and_add(x, multiplier, add_value):
        return pd.DataFrame({"result": [x * multiplier + add_value]})

    result = from_map(multiply_and_add, [1, 2, 3], args=(2, 10))
    assert isinstance(result, pa.Table)
    assert len(result) == 3

    df = result.to_pandas()
    assert list(df["result"]) == [12, 14, 16]


def test_from_map_with_pyarrow_tables():
    def make_arrow_table(x):
        df = pd.DataFrame({"value": [x]})
        return pa.Table.from_pandas(df)

    result = from_map(make_arrow_table, [1, 2, 3])
    assert isinstance(result, pa.Table)
    assert len(result) == 3



def test_from_map_batched_basic_functionality(air_small):
    blocks = list(block_slices(air_small, chunks={"time": 4, "lat": 3, "lon": 4}))

    first_block_df = pivot(air_small.isel(blocks[0]))
    expected_schema = pa.Schema.from_pandas(first_block_df)

    reader = from_map_batched(pivot, [air_small.isel(block) for block in blocks], schema=expected_schema)

    assert isinstance(reader, pa.RecordBatchReader)
    assert reader.schema == expected_schema

    batches = list(reader)
    assert len(batches) > 0
    for batch in batches:
        assert batch.schema == expected_schema
        assert len(batch) > 0


def test_from_map_batched_multiple_iterables():
    x_values = [1, 2, 3, 4, 5]
    y_values = [10, 20, 30, 40, 50]

    expected_schema = pa.schema([("x", pa.int64()), ("y", pa.int64()), ("sum", pa.int64())])

    reader = from_map_batched(adding_function, x_values, y_values, schema=expected_schema)
    table = reader.read_all()
    df = table.to_pandas()

    expected_df = pd.DataFrame({"x": x_values, "y": y_values, "sum": [x + y for x, y in zip(x_values, y_values)]})
    pd.testing.assert_frame_equal(df, expected_df)


def test_from_map_batched_with_args_and_kwargs():
    def multiply_and_add(x, multiplier, offset=0):
        return pd.DataFrame({"x": [x], "result": [x * multiplier + offset]})

    values = [1, 2, 3]
    expected_schema = pa.schema([("x", pa.int64()), ("result", pa.int64())])

    reader = from_map_batched(multiply_and_add, values, args=(2,), offset=5, schema=expected_schema)
    table = reader.read_all()
    df = table.to_pandas()

    expected_df = pd.DataFrame({"x": [1, 2, 3], "result": [7, 9, 11]})
    pd.testing.assert_frame_equal(df, expected_df)


def test_from_map_batched_empty_iterables():
    empty_schema = pa.schema([("value", pa.int64())])

    reader = from_map_batched(lambda x: pd.DataFrame({"value": [x]}), [], schema=empty_schema)
    batches = list(reader)
    assert len(batches) == 0


def test_from_map_batched_consistency_with_regular_map(air_small):
    blocks = list(block_slices(air_small, chunks={"time": 4, "lat": 3}))
    datasets = [air_small.isel(block) for block in blocks]

    first_df = pivot(datasets[0])
    schema = pa.Schema.from_pandas(first_df)

    reader = from_map_batched(pivot, datasets, schema=schema)
    batched_table = reader.read_all()

    regular_dfs = [pivot(ds) for ds in datasets]
    regular_table = pa.Table.from_pandas(pd.concat(regular_dfs, ignore_index=True))

    assert batched_table.schema == regular_table.schema
    assert len(batched_table) == len(regular_table)

    batched_df = batched_table.to_pandas().sort_values(["time", "lat", "lon"]).reset_index(drop=True)
    regular_df = regular_table.to_pandas().sort_values(["time", "lat", "lon"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(batched_df, regular_df)


def test_from_map_batched_integration_with_datafusion_via_read_xarray():
    air = xr.tutorial.open_dataset("air_temperature")
    air_small = air.isel(time=slice(0, 50), lat=slice(0, 10), lon=slice(0, 15))
    air_chunked = air_small.chunk({"time": 25, "lat": 5, "lon": 8})

    arrow_stream = read_xarray(air_chunked, chunks={"time": 25, "lat": 5, "lon": 8})

    assert hasattr(arrow_stream, "schema")
    assert hasattr(arrow_stream, "__iter__")

    table = arrow_stream.read_all()
    assert len(table) > 0

    expected_columns = {"time", "lat", "lon", "air"}
    actual_columns = set(table.column_names)
    assert expected_columns.issubset(actual_columns)



def test_read_xarray_loads_one_chunk_at_a_time(large_ds):
    tracemalloc.start()
    iterable = read_xarray(large_ds)
    first_size, first_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    sizes, peaks = [], []

    first_chunk = large_ds.isel(next(block_slices(large_ds)))
    chunk_size = first_chunk.nbytes

    # Creating the iterator should be inexpensive -- less than one chunk.
    # We multiply by constant factors because chunks have additional overhead
    assert first_size < chunk_size * 3
    assert first_peak < chunk_size * 6

    for it in iterable:
        _ = it
        cur_size, cur_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        sizes.append(cur_size)
        peaks.append(cur_peak)

    mean_size = np.mean(sizes)
    mean_peak = np.mean(peaks)

    for size in sizes:
        assert mean_size * 1.1 > size
        assert chunk_size * 3 > size
        assert chunk_size * 2 < size

    for peak in peaks:
        assert mean_peak * 1.1 > peak
        assert chunk_size * 7 > peak
        assert chunk_size * 4 < peak

    assert max(peaks) < large_ds.nbytes

    tracemalloc.stop()
