"""Dataset accessor for the DataFrame interchange protocol."""

from __future__ import annotations

import typing as t

import xarray as xr

from . import core


AccessorReturn = t.TypeVar("AccessorReturn")


@xr.register_dataset_accessor("df_protocol")
class DataFrameProtocolAccessor:
  """Access DataFrame interchange utilities from an xarray Dataset."""

  def __init__(self, dataset: xr.Dataset) -> None:
    self._dataset = dataset

  def __dataframe__(self, allow_copy: bool = False):
    """Expose the dataset via the DataFrame interchange protocol."""
    return core.dataset_to_protocol(self._dataset, allow_copy=allow_copy)


def register_dataset_dataframe_accessor() -> None:
  """Ensure the dataset accessor is registered (import side-effect)."""
  # Accessing the class ensures module import registers the accessor.
  _ = DataFrameProtocolAccessor


