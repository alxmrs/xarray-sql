import functools
import itertools
import operator
import typing as t

import sqlglot.executor.table
import xarray as xr

Row = t.List[t.Any]


# TODO(alxmrs): Could this be vectorized somehow? Maybe with map_block?
def unravel(ds: xr.Dataset) -> t.Iterator[Row]:
  dim_keys, dim_vals = zip(*ds.dims.items())

  for idx in itertools.product(dim_vals):
    coord_idx = dict(zip(dim_keys, idx))
    data = ds.isel(coord_idx)
    coord_data = [ds.coords[v][coord_idx[v]] for v in dim_keys]
    row = [_unbox(v.values) for v in coord_data + list(data.data_vars.values())]
    yield row


# Borrowed from google/weather-tools
def ichunked(iterable: t.Iterable, chunk_size: int) -> t.Iterator[t.Iterable]:
  input_ = iter(iterable)
  try:
    while True:
      it = itertools.islice(input_, chunk_size)
      # Peek to check if `it` has next item
      first = next(it)
      yield itertools.chain([first], it)
  except StopIteration:
    pass


def _index_to_position(index: int, dimensions: t.List[int]) -> t.List[int]:
  """Converts a table index into a position within an nd-array."""
  # Authored by ChatGPT from the following prompt:
  #   """
  #   Lets say I have an integer “index”. It represents the index of an array
  #   where the last index value is the product of the dimensions of the array.
  #   Each index value corresponds to a cell in the array. The array can be D
  #   dimensions, and usually D is 3 or 4. In Python, how do I convert the
  #   integer “index” into the position of the cell in the array (i.e. a list
  #   of integers of length D where each value represents the position along
  #   the dimension)?
  #   """
  position = []
  for dim in reversed(dimensions):
    position.insert(0, index % dim)
    index //= dim
  return position


def _unbox(array):
  """When a numpy array is a scalar, it extracts the value."""
  try:
    return array.item()
  except ValueError:
    return array


class XarrayDatasetTable(sqlglot.executor.table.Table):
  """Translates an Xarray Dataset into a flat Table for SQL query execution."""

  # TODO(alxmrs): Does this need to take a column_range?
  def __init__(self, ds: xr.Dataset, column_range=None) -> None:
    self.ds = ds

    # Collect the these up front to guarantee an order for the columns and data.
    self._dkeys, self._dvals = zip(*ds.dims.items())

    columns = tuple(list(self._dkeys) + list(ds.data_vars.keys()))

    super().__init__(columns, None, column_range)

  def __len__(self):
    # math.prod is not available in all python versions, so we quickly implement
    # it here.
    return functools.reduce(operator.mul, self.ds.dims.values(), 1)

  def __getitem__(self, index):
    """Translates a flat table index into a nd-array lookup."""
    positions = _index_to_position(index, self._dvals)
    coord_idx = dict(zip(self._dkeys, positions))
    item = self.ds.isel(coord_idx)
    coord_vals = [self.ds.coords[v][coord_idx[v]] for v in self._dkeys]
    row = [_unbox(v.values) for v in coord_vals + list(item.data_vars.values())]
    self.reader.row = row
    return self.reader

