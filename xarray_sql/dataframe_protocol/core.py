"""Core DataFrame interchange protocol objects for xarray datasets.

This module implements the DataFrame Interchange Protocol specification
as defined at: https://data-apis.org/dataframe-protocol/latest/API.html

Protocol version: 0

Note: These interfaces are reimplemented based on the specification.
They are not pip-installable from a separate package, but follow the
standard protocol definition to ensure interoperability with other
dataframe libraries.
"""

from __future__ import annotations

# Protocol version matching the DataFrame Interchange Protocol specification
# See: https://data-apis.org/dataframe-protocol/latest/API.html
__dataframe_version__ = 0

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


class Column(ABC):
  """
  A column object, with only the methods and properties required by the
  interchange protocol defined.

  A column can contain one or more chunks. Each chunk can contain up to three
  buffers - a data buffer, a mask buffer (depending on null representation),
  and an offsets buffer (if variable-size binary; e.g., variable-length
  strings).

  Note: this Column object can only be produced by ``__dataframe__``, so
        doesn't need its own version or ``__column__`` protocol.
  """

  @abstractmethod
  def size(self) -> int:
    """
    Size of the column, in elements.

    Corresponds to DataFrame.num_rows() if column is a single chunk;
    equal to size of this current chunk otherwise.

    Is a method rather than a property because it may cause a (potentially
    expensive) computation for some dataframe implementations.
    """
    ...

  @property
  @abstractmethod
  def offset(self) -> int:
    """
    Offset of first element.

    May be > 0 if using chunks; for example for a column with N chunks of
    equal size M (only the last chunk may be shorter),
    ``offset = n * M``, ``n = 0 .. N-1``.
    """
    ...

  @property
  @abstractmethod
  def dtype(self) -> Dtype:
    """
    Dtype description as a tuple ``(kind, bit-width, format string, endianness)``.

    Bit-width : the number of bits as an integer
    Format string : data type description format string in Apache Arrow C
                    Data Interface format.
    Endianness : current only native endianness (``=``) is supported
    """
    ...

  @property
  @abstractmethod
  def describe_categorical(self) -> CategoricalDescription:
    """
    If the dtype is categorical, there are two options:
    - There are only values in the data buffer.
    - There is a separate non-categorical Column encoding categorical values.

    Raises TypeError if the dtype is not categorical

    Returns the dictionary with description on how to interpret the data buffer:
        - "is_ordered" : bool, whether the ordering of dictionary indices is
                         semantically meaningful.
        - "is_dictionary" : bool, whether a mapping of
                            categorical values to other objects exists
        - "categories" : Column representing the (implicit) mapping of indices to
                         category values (e.g. an array of cat1, cat2, ...).
                         None if not a dictionary-style categorical.
    """
    ...

  @property
  @abstractmethod
  def describe_null(self) -> t.Tuple[ColumnNullType, t.Any]:
    """
    Return the missing value (or "null") representation the column dtype
    uses, as a tuple ``(kind, value)``.

    Value : if kind is "sentinel value", the actual value. If kind is a bit
    mask or a byte mask, the value (0 or 1) indicating a missing value. None
    otherwise.
    """
    ...

  @property
  @abstractmethod
  def null_count(self) -> t.Optional[int]:
    """
    Number of null elements, if known.
    """
    ...

  @property
  @abstractmethod
  def metadata(self) -> t.Dict[str, t.Any]:
    """
    The metadata for the column. See `DataFrame.metadata` for more details.
    """
    ...

  @abstractmethod
  def num_chunks(self) -> int:
    """
    Return the number of chunks the column consists of.
    """
    ...

  @abstractmethod
  def get_chunks(
      self, n_chunks: t.Optional[int] = None
  ) -> t.Iterable["Column"]:
    """
    Return an iterator yielding the chunks.

    See `DataFrame.get_chunks` for details on ``n_chunks``.
    """
    ...

  @abstractmethod
  def get_buffers(self) -> ColumnBuffers:
    """
    Return a dictionary containing the underlying buffers.

    The returned dictionary has the following contents:

        - "data": a two-element tuple whose first element is a buffer
                  containing the data and whose second element is the data
                  buffer's associated dtype.
        - "validity": a two-element tuple whose first element is a buffer
                      containing mask values indicating missing data and
                      whose second element is the mask value buffer's
                      associated dtype. None if the null representation is
                      not a bit or byte mask.
        - "offsets": a two-element tuple whose first element is a buffer
                     containing the offset values for variable-size binary
                     data (e.g., variable-length strings) and whose second
                     element is the offsets buffer's associated dtype. None
                     if the data buffer does not have an associated offsets
                     buffer.
    """
    ...


class XarrayColumn(Column):
  """
  Minimal concrete Column implementation backed by an xarray.DataArray.

  Accepts a 1-D or multi-dimensional DataArray representing a single logical column.
  Multi-dimensional arrays are automatically raveled to 1D in the constructor.
  If the underlying array is chunked (e.g., a dask-backed DataArray), this class
  can expose protocol chunks via num_chunks()/get_chunks().

  Notes:
    - Multi-dimensional arrays are raveled to 1D, preserving chunking structure.
      For dask arrays, raveling automatically computes the correct 1D chunks
      (product of chunks across all dimensions).
    - For lazy (dask) arrays, materializing buffers requires compute. This can
      be controlled via the allow_compute flag.
    - Advanced features (categoricals, string offsets, explicit masks) are not
      implemented here.
  """

  def __init__(
      self,
      dataarray: xr.DataArray,
      *,
      allow_compute: bool = True,
      base_offset: int = 0,
  ) -> None:
    # Ravel multi-dimensional arrays to 1D for the column protocol
    if dataarray.ndim > 1:
      # Ravel the data while preserving chunking structure
      # For dask arrays, raveling automatically computes correct 1D chunks
      dataarray = dataarray.ravel()
    
    
    self._da = dataarray
    self._allow_compute = bool(allow_compute)
    self._offset = int(base_offset)

  def size(self) -> int:
    return int(self._da.shape[0])

  @property
  def offset(self) -> int:
    return self._offset

  @property
  def dtype(self) -> Dtype:
    np_dtype = self._da.dtype
    endianness = "="

    if np_dtype.kind == "b":
      kind = DtypeKind.BOOL
      bits = 8
      fmt = ""
    elif np_dtype.kind == "i":
      kind = DtypeKind.INT
      bits = int(np_dtype.itemsize * 8)
      fmt = ""
    elif np_dtype.kind == "u":
      kind = DtypeKind.UINT
      bits = int(np_dtype.itemsize * 8)
      fmt = ""
    elif np_dtype.kind == "f":
      kind = DtypeKind.FLOAT
      bits = int(np_dtype.itemsize * 8)
      fmt = ""
    elif np_dtype.kind == "M":
      kind = DtypeKind.DATETIME
      bits = int(np_dtype.itemsize * 8)
      fmt = "tsn"
    elif np_dtype.kind in {"U", "S", "O"}:
      kind = DtypeKind.STRING
      bits = 0
      fmt = "u"
    else:
      kind = DtypeKind.UINT
      bits = int(np_dtype.itemsize * 8)
      fmt = ""

    return (kind, bits, fmt, endianness)

  @property
  def describe_categorical(self) -> CategoricalDescription:
    raise TypeError("Not a categorical column")

  @property
  def describe_null(self) -> t.Tuple[ColumnNullType, t.Any]:
    if self._da.dtype.kind == "f":
      return (ColumnNullType.USE_NAN, np.nan)
    return (ColumnNullType.NON_NULLABLE, None)

  @property
  def null_count(self) -> t.Optional[int]:
    return None

  @property
  def metadata(self) -> t.Dict[str, t.Any]:
    return dict(self._da.attrs)

  def num_chunks(self) -> int:
    # After raveling in __init__, self._da is guaranteed to be 1D
    # So if chunks exist, they must be 1D (tuple of length 1)
    data = self._da.data
    if hasattr(data, "chunks"):
      chunks = getattr(data, "chunks")
      if isinstance(chunks, tuple):
        # After raveling, chunks should be 1D, so chunks[0] contains chunk sizes
        return len(chunks[0])
    return 1

  def get_chunks(
      self, n_chunks: t.Optional[int] = None
  ) -> t.Iterable["Column"]:
    # Get current chunk sizes
    # After raveling in __init__, self._da is guaranteed to be 1D
    # So if chunks exist, they must be 1D (tuple of length 1)
    data = self._da.data
    if hasattr(data, "chunks") and isinstance(data.chunks, tuple):
      # After raveling, chunks should be 1D, so chunks[0] contains chunk sizes
      sizes = [int(s) for s in data.chunks[0]]
    else:
      sizes = [self._da.shape[0]]

    total_chunks = len(sizes)

    # After ravel, self._da is 1D, so dims[0] is the single dimension
    # Safe to access since ravel() always produces a 1D array with at least one dim
    dim_name = self._da.dims[0]

    # If no subdivision requested, yield each original chunk
    if n_chunks is None:
      start = 0
      for sz in sizes:
        end = start + sz
        yield XarrayColumn(
            self._da.isel({dim_name: slice(start, end)}),
            allow_compute=self._allow_compute,
            base_offset=self._offset + start,
        )
        start = end
      return

    # Ensure n_chunks is a multiple of original number of chunks
    if n_chunks % total_chunks != 0:
      raise ValueError(
          f"n_chunks={n_chunks} must be a multiple of num_chunks={total_chunks}"
      )

    factor = n_chunks // total_chunks

    start_chunk = 0
    for sz in sizes:
      base_size = sz // factor
      remainder = sz % factor
      current = 0
      for i in range(factor):
        extra = remainder if i == factor - 1 else 0
        sub_size = base_size + extra
        yield XarrayColumn(
            self._da.isel(
                {
                    dim_name: slice(
                        start_chunk + current, start_chunk + current + sub_size
                    )
                }
            ),
            allow_compute=self._allow_compute,
            base_offset=self._offset + start_chunk + current,
        )
        current += sub_size
      start_chunk += sz

  def get_buffers(self) -> ColumnBuffers:
    array_like = self._da.data
    if hasattr(array_like, "compute"):
      if not self._allow_compute:
        raise RuntimeError(
            "Buffer materialization requires compute but allow_compute=False"
        )
      array_like = array_like.compute()
    np_array = np.asarray(array_like)
    data_buf: Buffer = XarrayBuffer(np_array)
    return t.cast(
        ColumnBuffers,
        {
            "data": (data_buf, self.dtype),
            "validity": None,
            "offsets": None,
        },
    )


