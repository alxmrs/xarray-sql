import xarray as xr
import zarr.storage
from datafusion import SessionContext
from zarrquet import ZarrTableProvider

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
    assert chunks is None, 'chunks not supported (at the moment).'
    zarr_provider = ZarrTableProvider(zarr_path)
    return self.register_table_provider(table_name, zarr_provider)