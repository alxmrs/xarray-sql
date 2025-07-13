import typing as t

import pandas as pd
import pyarrow as pa
import xarray as xr

from xarray_sql.core import Block, Chunks, block_slices


def from_map(
    func: t.Callable, *iterables, args: t.Optional[t.Tuple] = None, **kwargs
) -> pa.Table:
  """Create a PyArrow Table by mapping a function over iterables.

  This is equivalent to dask's from_map but returns a PyArrow Table
  that can be used with DataFusion instead of a Dask DataFrame.

  Args:
    func: Function to apply to each element of the iterables.
    *iterables: Iterable objects to map the function over.
    args: Additional positional arguments to pass to func.
    **kwargs: Additional keyword arguments to pass to func.

  Returns:
    A PyArrow Table containing the concatenated results.
  """
  if args is None:
    args = ()

  # Apply the function to each combination of iterable elements
  results = []
  for items in zip(*iterables) if len(iterables) > 1 else iterables[0]:
    if isinstance(items, tuple):
      result = func(*items, *args, **kwargs)
    else:
      result = func(items, *args, **kwargs)

    # Convert result to PyArrow Table
    if isinstance(result, pd.DataFrame):
      pa_table = pa.Table.from_pandas(result)
    elif isinstance(result, pa.Table):
      pa_table = result
    else:
      # Try to convert to pandas first, then to PyArrow
      try:
        df = pd.DataFrame(result)
        pa_table = pa.Table.from_pandas(df)
      except Exception as e:
        raise ValueError(
            f"Cannot convert function result to PyArrow Table: {e}"
        )

    results.append(pa_table)

  # Concatenate all results
  if not results:
    raise ValueError("No results to concatenate")

  return pa.concat_tables(results)


def pivot(ds: xr.Dataset) -> pd.DataFrame:
  """Convert an Xarray Dataset into a Pandas DataFrame."""
  return ds.to_dataframe().reset_index()


def read_xarray(ds: xr.Dataset, chunks: Chunks = None) -> pa.Table:
  """Pivots an Xarray Dataset into a PyArrow Table, partitioned by chunks.

  Args:
    ds: An Xarray Dataset. All `data_vars` mush share the same dimensions.
    chunks: Xarray-like chunks. If not provided, will default to the Dataset's
     chunks. The product of the chunk sizes becomes the standard length of each
     dataframe partition.

  Returns:
    A PyArrow Table, which is a table representation of the input Dataset.
  """
  fst = next(iter(ds.values())).dims
  assert all(
      da.dims == fst for da in ds.values()
  ), "All dimensions must be equal. Please filter data_vars in the Dataset."

  blocks = list(block_slices(ds, chunks))

  def pivot_block(b: Block) -> pd.DataFrame:
    return pivot(ds.isel(b))

  return from_map(pivot_block, blocks)



