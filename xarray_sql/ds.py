"""
PyArrow Dataset adapter for xarray.Dataset with lazy chunking support.

This module provides a PyArrow Dataset-compatible interface for xarray.Dataset
that maintains lazy evaluation and leverages xarray's chunking capabilities.
"""

import typing as t
from typing import Optional, Iterator, Union, Dict, Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import xarray as xr
import pandas as pd
import numpy as np

from xarray_sql.core import Chunks, block_slices, Block
from xarray_sql.df import pivot



class SchemaMapper:
    """Handles conversion between xarray Dataset structure and PyArrow schemas."""
    
    @staticmethod
    def from_xarray(ds: xr.Dataset) -> pa.Schema:
        """Create PyArrow schema from xarray Dataset."""
        fields = []
        
        # Add coordinate fields
        for coord_name, coord in ds.coords.items():
            pa_type = SchemaMapper._numpy_to_arrow_type(coord.dtype)
            fields.append(pa.field(coord_name, pa_type))
        
        # Add data variable fields
        for var_name, var in ds.data_vars.items():
            pa_type = SchemaMapper._numpy_to_arrow_type(var.dtype)
            fields.append(pa.field(var_name, pa_type))
        
        return pa.schema(fields)
    
    @staticmethod
    def _numpy_to_arrow_type(numpy_dtype) -> pa.DataType:
        """Convert numpy dtype to PyArrow type."""
        if np.issubdtype(numpy_dtype, np.integer):
            if numpy_dtype == np.int8:
                return pa.int8()
            elif numpy_dtype == np.int16:
                return pa.int16()
            elif numpy_dtype == np.int32:
                return pa.int32()
            elif numpy_dtype == np.int64:
                return pa.int64()
            else:
                return pa.int64()  # Default for integer types
        elif np.issubdtype(numpy_dtype, np.floating):
            if numpy_dtype == np.float32:
                return pa.float32()
            elif numpy_dtype == np.float64:
                return pa.float64()
            else:
                return pa.float64()  # Default for float types
        elif np.issubdtype(numpy_dtype, np.bool_):
            return pa.bool_()
        elif np.issubdtype(numpy_dtype, np.datetime64):
            return pa.timestamp('ns')
        elif numpy_dtype.kind in ['U', 'S', 'O']:  # String types
            return pa.string()
        else:
            # Fallback to string for unknown types
            return pa.string()


class XarrayScanner:
    """Scanner for lazy evaluation and streaming of xarray Dataset chunks."""
    
    def __init__(
        self,
        adapter: 'XarrayDatasetAdapter',
        filter_expression: Optional[pc.Expression] = None,
        columns: Optional[t.List[str]] = None,
        batch_size: int = 1_000_000
    ):
        self._adapter = adapter
        self._filter = filter_expression
        self._columns = columns
        self._batch_size = batch_size
    
    def to_batches(self) -> Iterator[pa.RecordBatch]:
        """Convert xarray chunks to PyArrow RecordBatches lazily."""
        for block in self._adapter._blocks:
            chunk_ds = self._adapter.ds.isel(block)
            
            # Skip empty datasets
            if len(chunk_ds.dims) == 0 and len(chunk_ds.data_vars) == 0:
                continue
            
            # Apply column selection if specified
            if self._columns:
                chunk_ds = chunk_ds[self._columns]

            # Handle datasets with no dimensions (0-dimensional)
            if len(chunk_ds.dims) == 0:
                # Create a simple DataFrame for 0-dimensional data
                data = {}
                for var_name, var in chunk_ds.data_vars.items():
                    data[var_name] = [var.values.item()]
                df = pd.DataFrame(data)
            else:
                # Convert chunk to DataFrame using existing pivot function
                df = pivot(chunk_ds)
            
            # Skip empty DataFrames
            if len(df) == 0:
                continue
                
            # Convert to PyArrow table and apply filter in one step
            table = pa.Table.from_pandas(df)
            
            # Apply filtering directly on PyArrow table (no extra conversions)
            if self._filter is not None:
                try:
                    table = table.filter(self._filter)
                except Exception as e:
                    # If filtering fails, log warning and continue with unfiltered data
                    import warnings
                    warnings.warn(
                        f"Failed to apply filter expression: {e}. "
                        "Continuing with unfiltered data.",
                        UserWarning
                    )
            
            # Yield batches from the (potentially filtered) table
            if len(table) > 0:
                for batch in table.to_batches(max_chunksize=self._batch_size):
                    yield batch
    
    
    def _is_empty_dataset(self, ds: xr.Dataset) -> bool:
        """Check if dataset is effectively empty after filtering."""
        # Check if any dimension has size 0
        for dim_size in ds.sizes.values():
            if dim_size == 0:
                return True
        return False
    
    def to_table(self) -> pa.Table:
        """Load all chunks into a single PyArrow Table."""
        batches = list(self.to_batches())
        if not batches:
            # Return empty table with correct schema
            # Create empty arrays for each field in schema
            arrays = []
            schema = self._adapter.schema
            for field in schema:
                arrays.append(pa.array([], type=field.type))
            return pa.Table.from_arrays(arrays, schema=schema)
        return pa.Table.from_batches(batches)
    
    def count_rows(self) -> int:
        """Count total rows efficiently from dataset structure."""
        # If no filter, calculate directly from dataset dimensions
        if self._filter is None:
            return self._count_rows_unfiltered()
        
        # With filter, we need to process chunks but can still optimize
        return self._count_rows_filtered()
    
    def _count_rows_unfiltered(self) -> int:
        """Fast row count without filtering - calculate from dataset structure."""
        ds = self._adapter.ds
        
        # Apply column selection if specified (affects available coordinates)
        if self._columns:
            # Column selection doesn't change row count, just column count
            # Row count is still determined by coordinate combinations
            pass
        
        # Calculate total number of coordinate combinations (rows after pivot)
        # Each combination of coordinate values becomes one row
        total_rows = 1
        for dim_name, dim_size in ds.sizes.items():
            total_rows *= dim_size
            
        return total_rows
    
    def _count_rows_filtered(self) -> int:
        """Count rows with filtering - reuse to_batches logic."""
        # Simply count rows from all batches without materializing the full data
        total_count = 0
        for batch in self.to_batches():
            total_count += len(batch)
        return total_count
    
    def head(self, num_rows: int = 5) -> pa.Table:
        """Return first num_rows from the dataset."""
        collected_rows = 0
        batches = []
        
        for batch in self.to_batches():
            if collected_rows >= num_rows:
                break
            
            needed_rows = min(num_rows - collected_rows, len(batch))
            if needed_rows < len(batch):
                batch = batch.slice(0, needed_rows)
            
            batches.append(batch)
            collected_rows += len(batch)
        
        if not batches:
            # Return empty table with correct schema
            arrays = []
            schema = self._adapter.schema
            for field in schema:
                arrays.append(pa.array([], type=field.type))
            return pa.Table.from_arrays(arrays, schema=schema)
        return pa.Table.from_batches(batches)


class XarrayDatasetAdapter:
    """
    PyArrow Dataset-compatible adapter for xarray.Dataset.
    
    Provides lazy evaluation and streaming capabilities by mapping xarray chunks
    to PyArrow batches, converting data just-in-time using the pivot operation.
    """
    
    def __init__(self, ds: xr.Dataset, chunks: Optional[Chunks] = None):
        """
        Initialize adapter for xarray Dataset.
        
        Args:
            ds: xarray Dataset to adapt
            chunks: Optional chunking specification. If None, uses existing chunks
                   or treats entire dataset as single chunk.
        """
        self.ds = ds
        self.chunks = chunks
        
        # Pre-compute blocks for efficient iteration
        self._blocks = self._compute_blocks()
        
        # Build PyArrow schema
        self._schema = SchemaMapper.from_xarray(ds)
    
    def _compute_blocks(self) -> list[Block]:
        """Compute blocks, handling non-chunked datasets gracefully."""
        # If chunks specified, use them
        if self.chunks is not None:
            return list(block_slices(self.ds, self.chunks))
        
        # If dataset is already chunked, use existing chunks
        if hasattr(self.ds, 'chunks') and self.ds.chunks:
            return list(block_slices(self.ds, None))
        
        # For non-chunked datasets, create a single block spanning entire dataset
        single_block = {dim: slice(None) for dim in self.ds.dims}
        return [single_block]
    
    @property
    def schema(self) -> pa.Schema:
        """Get PyArrow schema for this dataset."""
        return self._schema
    
    def scanner(
        self,
        filter: Optional[pc.Expression] = None,
        columns: Optional[t.List[str]] = None,
        batch_size: int = 1_000_000,
        **kwargs
    ) -> XarrayScanner:
        """
        Create a scanner for lazy evaluation.
        
        Args:
            filter: PyArrow expression for filtering (not yet implemented)
            columns: List of columns to select
            batch_size: Maximum number of rows per batch
            **kwargs: Additional scanner options (for future compatibility)
        
        Returns:
            XarrayScanner for lazy evaluation
        """
        return XarrayScanner(self, filter, columns, batch_size)
    
    def to_batches(self, **scanner_kwargs) -> Iterator[pa.RecordBatch]:
        """Stream dataset as PyArrow RecordBatches."""
        return self.scanner(**scanner_kwargs).to_batches()
    
    def to_table(self, **scanner_kwargs) -> pa.Table:
        """Load entire dataset into a single PyArrow Table."""
        return self.scanner(**scanner_kwargs).to_table()
    
    def count_rows(self, **scanner_kwargs) -> int:
        """Count total rows in dataset efficiently."""
        # For simple cases (no filter, no complex scanner args), compute directly
        if not scanner_kwargs or (len(scanner_kwargs) == 1 and 'columns' in scanner_kwargs):
            # Fast path: calculate from dataset structure without creating scanner
            total_rows = 1
            for dim_size in self.ds.sizes.values():
                total_rows *= dim_size
            return total_rows
        
        # For complex cases (with filters), delegate to scanner
        return self.scanner(**scanner_kwargs).count_rows()
    
    def head(self, num_rows: int = 5, **scanner_kwargs) -> pa.Table:
        """Get first num_rows from dataset."""
        return self.scanner(**scanner_kwargs).head(num_rows)
    
    def get_fragments(self) -> Iterator[Dict[str, Any]]:
        """
        Get information about dataset fragments (chunks).
        
        Returns iterator of fragment metadata.
        """
        for i, block in enumerate(self._blocks):
            yield {
                'fragment_id': i,
                'block': block,
                'estimated_rows': self._estimate_block_size(block)
            }
    
    def _estimate_block_size(self, block: Block) -> int:
        """Estimate number of rows in a block without loading data."""
        size = 1
        for dim, slice_obj in block.items():
            if isinstance(slice_obj, slice):
                start = slice_obj.start or 0
                stop = slice_obj.stop or self.ds.sizes[dim]
                size *= stop - start
            else:
                size *= 1  # Single index
        return int(size)  # Ensure we return Python int, not numpy int
