import xarray as xr
from datafusion import SessionContext
from zarrquet import ZarrTableProvider
import sys

from .df import read_xarray, Chunks


class XarrayContext(SessionContext):
  """A datafusion `SessionContext` that also supports `xarray.Dataset`s."""

  def from_dataset(
      self,
      table_name: str,
      input_table: xr.Dataset,
      chunks: Chunks = None,
  ):
    arrow_table = read_xarray(input_table, chunks)
    return self.from_arrow(arrow_table, table_name)

  def from_zarr(
      self,
      table_name: str,
      zarr_path: str,
      chunks: Chunks = None,
  ):
    if sys.version_info < (3, 11):
      raise ValueError(
          f'method not supported below Python 3.11. {sys.version} found.'
      )
    
    # Load the Zarr dataset to get schema information
    ds = xr.open_zarr(zarr_path, chunks=chunks or {})
    
    # Apply auto-chunking if needed to fix chunking errors
    if not ds.chunks:
      # Auto-chunk with moderate sizes
      chunk_spec = {}
      for dim, size in ds.sizes.items():
        if 'time' in dim.lower():
          chunk_spec[dim] = min(24, size)
        else:
          chunk_spec[dim] = min(10, size)
      ds = ds.chunk(chunk_spec)
    
    # Convert to Arrow schema via read_xarray
    arrow_reader = read_xarray(ds, chunks)
    arrow_schema = arrow_reader.schema
    
    # Create ZarrTableProvider with the schema
    zarr_provider = ZarrTableProvider(zarr_path, arrow_schema)
    return self.register_table_provider(table_name, zarr_provider)
