import pyarrow as pa
import xarray as xr
from datafusion import SessionContext, udf

from . import cft
from .df import Chunks
from .reader import read_xarray_table


def _make_cftime_udf(units: str, calendar: str):
  """Create a DataFusion scalar UDF that converts date strings to int64 offsets.

  This enables ergonomic SQL filtering on non-Gregorian cftime columns::

      SELECT * FROM ds360 WHERE time > cftime('0500-01-01')

  The UDF parses the input string as a cftime datetime in the given
  calendar system and returns the corresponding int64 offset in the
  specified units.
  """
  import cftime as _cftime

  def _cftime_scalar(date_strings: pa.Array) -> pa.Array:
    results = []
    for s in date_strings.to_pylist():
      if s is None:
        results.append(None)
        continue
      dt = _cftime.datetime.strptime(s, '%Y-%m-%d', calendar=calendar)
      val = _cftime.date2num(dt, units=units, calendar=calendar)
      results.append(int(val))
    return pa.array(results, type=pa.int64())

  return udf(
      _cftime_scalar,
      [pa.utf8()],
      pa.int64(),
      'immutable',
      'cftime',
  )


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
          self.register_udf(_make_cftime_udf(units, cal))
          break  # One UDF per context is enough.

    return self
