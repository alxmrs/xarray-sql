"""Core DataFrame interchange protocol objects for xarray datasets."""

from __future__ import annotations

import enum
import typing as t
from abc import ABC, abstractmethod

import numpy as np
import xarray as xr


class DlpackDeviceType(enum.IntEnum):
  """Integer enum for device type codes matching DLPack."""

  CPU = 1
  CUDA = 2
  CPU_PINNED = 3
  OPENCL = 4
  VULKAN = 7
  METAL = 8
  VPI = 9
  ROCM = 10


class DtypeKind(enum.IntEnum):
  """
  Integer enum for data types.

  Attributes
  ----------
  INT : int
      Matches to signed integer data type.
  UINT : int
      Matches to unsigned integer data type.
  FLOAT : int
      Matches to floating point data type.
  BOOL : int
      Matches to boolean data type.
  STRING : int
      Matches to string data type (UTF-8 encoded).
  DATETIME : int
      Matches to datetime data type.
  CATEGORICAL : int
      Matches to categorical data type.
  """

  INT = 0
  UINT = 1
  FLOAT = 2
  BOOL = 20
  STRING = 21  # UTF-8
  DATETIME = 22
  CATEGORICAL = 23


Dtype = t.Tuple["DtypeKind", int, str, str]  # see Column.dtype


class ColumnNullType(enum.IntEnum):
  """
  Integer enum for null type representation.

  Attributes
  ----------
  NON_NULLABLE : int
      Non-nullable column.
  USE_NAN : int
      Use explicit float NaN value.
  USE_SENTINEL : int
      Sentinel value besides NaN.
  USE_BITMASK : int
      The bit is set/unset representing a null on a certain position.
  USE_BYTEMASK : int
      The byte is set/unset representing a null on a certain position.
  """

  NON_NULLABLE = 0
  USE_NAN = 1
  USE_SENTINEL = 2
  USE_BITMASK = 3
  USE_BYTEMASK = 4


class ColumnBuffers(t.TypedDict):
  # first element is a buffer containing the column data;
  # second element is the data buffer's associated dtype
  data: t.Tuple["Buffer", Dtype]

  # first element is a buffer containing mask values indicating missing data;
  # second element is the mask value buffer's associated dtype.
  # None if the null representation is not a bit or byte mask
  validity: t.Optional[t.Tuple["Buffer", Dtype]]

  # first element is a buffer containing the offset values for
  # variable-size binary data (e.g., variable-length strings);
  # second element is the offsets buffer's associated dtype.
  # None if the data buffer does not have an associated offsets buffer
  offsets: t.Optional[t.Tuple["Buffer", Dtype]]


class CategoricalDescription(t.TypedDict):
  # whether the ordering of dictionary indices is semantically meaningful
  is_ordered: bool
  # whether a dictionary-style mapping of categorical values to other objects exists
  is_dictionary: bool
  # Python-level only (e.g. ``{int: str}``).
  # None if not a dictionary-style categorical.
  categories: t.Optional["Column"]


class Buffer(ABC):
  """
  Data in the buffer is guaranteed to be contiguous in memory.

  Note that there is no dtype attribute present, a buffer can be thought of
  as simply a block of memory. However, if the column that the buffer is
  attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
  implemented, then that dtype information will be contained in the return
  value from ``__dlpack__``.

  This distinction is useful to support both data exchange via DLPack on a
  buffer and (b) dtypes like variable-length strings which do not have a
  fixed number of bytes per element.
  """

  @property
  @abstractmethod
  def bufsize(self) -> int:
    """
    Buffer size in bytes.
    """
    ...

  @property
  @abstractmethod
  def ptr(self) -> int:
    """
    Pointer to start of the buffer as an integer.
    """
    ...

  @abstractmethod
  def __dlpack__(self):
    """
    Produce DLPack capsule (see array API standard).

    Raises:

        - TypeError : if the buffer contains unsupported dtypes.
        - NotImplementedError : if DLPack support is not implemented

    Useful to have to connect to array libraries. Support optional because
    it's not completely trivial to implement for a Python-only library.
    """
    raise NotImplementedError("__dlpack__")

  @abstractmethod
  def __dlpack_device__(self) -> t.Tuple[DlpackDeviceType, t.Optional[int]]:
    """
    Device type and device ID for where the data in the buffer resides.
    Uses device type codes matching DLPack.
    Note: must be implemented even if ``__dlpack__`` is not.
    """
    ...


class XarrayBuffer(Buffer):
  """Buffer implementation wrapping a NumPy ndarray without copying."""

  def __init__(self, array: np.ndarray) -> None:
    if not array.flags["C_CONTIGUOUS"]:
      raise ValueError(
          "Dataset backing array must be C-contiguous for zero-copy exchange."
      )
    self._array = array

  @property
  def bufsize(self) -> int:
    return int(self._array.nbytes)

  @property
  def ptr(self) -> int:
    return int(self._array.__array_interface__["data"][0])

  def __dlpack__(self):
    if hasattr(self._array, "__dlpack__"):
      return self._array.__dlpack__()
    raise NotImplementedError("__dlpack__")

  def __dlpack_device__(self) -> t.Tuple[DlpackDeviceType, t.Optional[int]]:
    return (DlpackDeviceType.CPU, 0)



def dataset_to_protocol(
    dataset: xr.Dataset, *, allow_copy: bool
) -> t.Any:  # pragma: no cover - placeholder
  """Return a DataFrame Interchange protocol object for the dataset.

  This is a stub that will be expanded with the actual implementation.
  """
  raise NotImplementedError


