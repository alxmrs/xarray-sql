import xarray as xr
from datafusion import SessionContext
from zarrquet import ZarrTableProvider
import sys
import asyncio
import pyarrow as pa

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
  ):
    if sys.version_info < (3, 11):
      raise ValueError(
          f'method not supported below Python 3.11. {sys.version} found.'
      )

    if not zarr_path.endswith('/'):
      zarr_path += '/'
    
    # Use the new non-FFI approach
    async def read_zarr_data():
      zarr_provider = ZarrTableProvider(zarr_path)
      batches = await zarr_provider.read_to_arrow()
      # Convert list of RecordBatches to a single Arrow Table
      if batches:
        return pa.Table.from_batches(batches)
      else:
        # Handle empty case - get schema and create empty table
        schema = await zarr_provider.get_schema()
        return pa.Table.from_arrays([], schema=schema)
    
    # Run the async operation
    arrow_table = asyncio.run(read_zarr_data())
    return self.from_arrow(arrow_table, table_name)
