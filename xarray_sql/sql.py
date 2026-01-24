import xarray as xr
from datafusion import SessionContext

from .df import Chunks
from .reader import read_xarray_table


class XarrayContext(SessionContext):
  """A datafusion `SessionContext` that also supports `xarray.Dataset`s."""

  def from_dataset(
      self,
      table_name: str,
      input_table: xr.Dataset,
      chunks: Chunks = None,
  ):
    table = read_xarray_table(input_table, chunks)
    self.register_table_provider(table_name, table)
