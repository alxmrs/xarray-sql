"""Lazy, streaming adapter over the native (non-FFI) engine's DataFrame.

:class:`NativeFrame` wraps the Rust ``NativeDataFrame`` and exposes just the
slice of the ``datafusion-python`` ``DataFrame`` interface the xarray
round-trip consumes — ``schema()``, ``execute_stream()``, ``to_pandas()`` — plus
structured column projection and coordinate filtering for the chunked
reconstruction path. Every consumer streams: nothing is collected up front, so a
reduction (or a chunked scan) over a store larger than memory never holds the
whole input or output at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa


def coord_dtype_tag(values: np.ndarray) -> str | None:
    """Map a coordinate array's numpy dtype to a native filter dtype tag.

    Returns ``None`` for dtypes the native filter can't represent (e.g.
    strings), so the caller can skip pushing a predicate on that dimension.
    """
    kind = values.dtype.kind
    if kind == "M":  # datetime64
        return "timestamp_ns"
    if kind == "f":
        return "float64"
    if kind in ("i", "u"):
        return "int64"
    return None


def _coord_values_for_filter(values: np.ndarray, dtype_tag: str) -> list:
    """Convert coordinate values to the Python scalars the native filter wants."""
    if dtype_tag == "timestamp_ns":
        as_ns = values.astype("datetime64[ns]").astype(np.int64)
        return [int(v) for v in as_ns]
    if dtype_tag == "float64":
        return [float(v) for v in values]
    return [int(v) for v in values]


class _Batch:
    """Adapt a PyArrow RecordBatch to datafusion-python's stream-item API."""

    __slots__ = ("_batch",)

    def __init__(self, batch: pa.RecordBatch) -> None:
        self._batch = batch

    def to_pyarrow(self) -> pa.RecordBatch:
        return self._batch


class NativeFrame:
    """Streaming, lazy stand-in for a ``datafusion-python`` DataFrame."""

    def __init__(self, native_df: Any) -> None:
        self._df = native_df

    # -- interface the round-trip consumes -----------------------------------

    def schema(self) -> pa.Schema:
        return self._df.schema()

    def execute_stream(self):
        """Yield result batches lazily, wrapped so ``.to_pyarrow()`` works."""
        return (_Batch(b) for b in self._df.execute_stream())

    def to_pandas(self):
        batches = list(self._df.execute_stream())
        if batches:
            return pa.Table.from_batches(batches).to_pandas()
        return pa.Table.from_batches([], schema=self._df.schema()).to_pandas()

    # -- chunked round-trip helpers ------------------------------------------

    def select_columns(self, columns: list[str]) -> "NativeFrame":
        return NativeFrame(self._df.select_columns(list(columns)))

    def filter_coord(self, column: str, values: np.ndarray) -> "NativeFrame":
        """Keep rows whose ``column`` is one of ``values`` (pushes into the scan).

        Coordinate ranges pushed here prune source partitions, so a single
        output chunk reads only the partitions it overlaps.
        """
        tag = coord_dtype_tag(np.asarray(values))
        if tag is None:
            return self
        native_values = _coord_values_for_filter(np.asarray(values), tag)
        return NativeFrame(self._df.filter_in(column, native_values, tag))

    def distinct_sorted_values(self, column: str) -> np.ndarray:
        """Ascending distinct values of ``column`` (coordinate discovery)."""
        frame = self._df.distinct_sorted(column)
        batches = list(frame.execute_stream())
        if not batches:
            return np.asarray([])
        return np.concatenate(
            [b.column(0).to_numpy(zero_copy_only=False) for b in batches]
        )
