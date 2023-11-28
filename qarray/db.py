import dataclasses
from typing import Any, Optional, Sequence, Type, Union

import numpy as np
import pep249
import xarray as xr
from pep249 import type_constructors as tc

apilevel = '2.0'
threadsafety = 0  # Assume thread-unsafe until proven otherwise.
paramstyle = 'named'


@dataclasses.dataclass(frozen=True, order=True)
class ColumnDescription:
  name: str
  type_code: Any
  display_size: Optional[int] = None
  internal_size: Optional[int] = None
  precision: Optional[float] = None
  scale: Optional[float] = None
  null_ok: Optional[bool] = None

  @classmethod
  def from_dataarray(cls, da: xr.DataArray) -> 'ColumnDescription':
    return cls(
        da.name,
        tc.NUMBER,  # TODO(alxmrs): Parse numpy dtypes from the array.
    )


class Cursor(pep249.Cursor):

  def __init__(self, ds: xr.Dataset):
    self.ds = ds
    self._rowcount = -1

  @property
  def description(self) -> list[tuple[str, Type]]:
    # This should be a property so it can be read-only.
    return [
      (da.name, tc.NUMBER)
      for da in self.ds.values()
    ]

  @property
  def rowcount(self) -> int:
    return self._rowcount

  def close(self) -> None:
    pass

  def fetchone(self) -> Optional[pep249.ResultRow]:
    pass

  def fetchmany(self, size: Optional[int] = None) -> pep249.ResultSet:
    pass

  def fetchall(self) -> pep249.ResultSet:
    pass

  def nextset(self) -> Optional[bool]:
    pass

  def execute(
      self,
      operation: pep249.SQLQuery,
      parameters: Optional[pep249.QueryParameters] = None,
  ) -> 'Cursor':
    pass

  def executemany(
      self,
      operation: pep249.SQLQuery,
      seq_of_parameters: Sequence[pep249.QueryParameters],
  ) -> 'Cursor':
    pass

  def callproc(self, procname: str, parameters: ... = None):
    raise pep249.NotSupportedError('Procedures not supported.')

  def setinputsizes(self, sizes: Sequence[Optional[Union[int, Type]]]) -> None:
    pass

  def setoutputsize(self, size: int, column: Optional[int]) -> None:
    pass


class Connection(pep249.Connection):

  def __init__(self, ds: xr.Dataset):
    self.ds = ds.copy(deep=False)

  def commit(self):
    raise pep249.NotSupportedError('transactions not supported.')

  def rollback(self):
    raise pep249.NotSupportedError('transactions not supported.')

  def close(self) -> None:
    self.ds.close()

  def cursor(self) -> Cursor:
    return Cursor(self.ds)


class Qarray:

  @classmethod
  def connect(cls, ds: xr.Dataset) -> Connection:
    return Connection(ds)
