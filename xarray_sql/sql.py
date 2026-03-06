import xarray as xr
from datafusion import SessionContext

from .df import Chunks
from .reader import read_xarray_table, register_catalog_from_dataset


class XarrayContext(SessionContext):
  """
  A regular DataFusion SessionContext but with an extra method
  for registering xarray datasets.
  """

  def from_dataset(
      self,
      table_name: str,
      input_table: xr.Dataset,
      chunks: Chunks = None,
  ):
    table = read_xarray_table(input_table, chunks)
    self.register_table(table_name, table)

  def register_catalog_from_dataset(
      self, ds, catalog_name="xarray", schema_name="data", chunks=None
  ):
    register_catalog_from_dataset(self, ds, catalog_name, schema_name, chunks)
