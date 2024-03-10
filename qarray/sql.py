import xarray as xr
import dask_sql.input_utils

from .df import read_xarray, Chunks


class Context(dask_sql.Context):
  """See the `dask_sql.Context` docs."""

  def create_table(
      self,
      table_name: str,
      input_table: dask_sql.input_utils.InputType,
      chunks: Chunks = None,
      *args,
      **kwargs,
  ):
    if isinstance(input_table, xr.Dataset):
      input_table = read_xarray(input_table, chunks)
    super().create_table(table_name, input_table, *args, **kwargs)
