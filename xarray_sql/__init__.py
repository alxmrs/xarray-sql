from .df import read_xarray, from_map
from .reader import XarrayRecordBatchReader, read_xarray_lazy, read_xarray_table
from .sql import XarrayContext
from ._native import LazyArrowStreamTable

__all__ = [
    # High-level API (recommended)
    "read_xarray_table",
    "XarrayContext",
    # Lower-level building blocks
    "read_xarray",
    "read_xarray_lazy",
    "from_map",
    "XarrayRecordBatchReader",
    "LazyArrowStreamTable",
]
