import xarray as xr
from datafusion import SessionContext
from datafusion.catalog import Table
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
    table = Table.from_dataset(arrow_table)
    return self.register_table(table_name, arrow_table )
