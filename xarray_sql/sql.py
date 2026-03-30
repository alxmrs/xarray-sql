import xarray as xr
from datafusion import SessionContext

from . import cft
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
    self.register_table(table_name, table)

    # Auto-register a cftime() UDF for non-Gregorian cftime coordinates
    # so users can write: WHERE time > cftime('0500-01-01')
    for coord_name in input_table.dims:
      if cft.is_cftime_index(input_table, coord_name):
        units, cal = cft.encoding(input_table, coord_name)
        if not cft.is_gregorian_like(cal):
          self.register_udf(cft.make_cftime_udf(units, cal))
          break  # One UDF per context is enough.

    return self
