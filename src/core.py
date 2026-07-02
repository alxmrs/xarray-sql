import xarray as xr
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple, Union

class XarrayContext:
    """
    Contexto para manejar datasets de Xarray como tablas SQL.
    """
    def __init__(self):
        self._tables: Dict[str, Any] = {}
        self._chunks: Dict[str, Optional[Dict]] = {}
        self._table_names: Dict[str, Tuple[str, ...]] = {}

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, name: Optional[str] = None, chunks: Optional[Dict] = None) -> 'XarrayContext':
        """
        Crea un nuevo contexto a partir de un dataset de Xarray.
        
        Args:
            ds: Dataset de Xarray.
            name: Nombre de la tabla (opcional). Si no se proporciona, se usa 'default'.
            chunks: Tamaño de los chunks (opcional).
        
        Returns:
            XarrayContext: Nuevo contexto con el dataset registrado.
        """
        ctx = cls()
        table_name = name if name is not None else 'default'
        ctx.from_dataset(table_name, ds, chunks)
        return ctx

    def from_dataset(self, name: str, ds: xr.Dataset, chunks: Optional[Dict] = None, table_names: Optional[Dict[Union[str, Tuple[str, ...]], str]] = None):
        """
        Registra un dataset de Xarray como una tabla SQL.
        
        Args:
            name: Nombre de la tabla.
            ds: Dataset de Xarray.
            chunks: Tamaño de los chunks para la tabla.
            table_names: Diccionario para mapear nombres de tablas a dimensiones.
                         Formato: {'nombre_tabla': ('dim1', 'dim2', ...)}
                         Si se pasa con tuplas como clave, se invierte automáticamente.
        """
        self._tables[name] = ds
        self._chunks[name] = chunks
        
        # Procesar table_names
        if table_names is not None:
            inverted = {}
            for key, value in table_names.items():
                if isinstance(key, tuple):
                    # Si la clave es una tupla, intercambiar
                    inverted[value] = key
                else:
                    # Si la clave es un string, usarlo directamente
                    inverted[key] = value
            self._table_names.update(inverted)

    @classmethod
    def read_xarray(cls, ds: xr.Dataset, chunks: Optional[Dict] = None, table_names: Optional[Dict[str, Tuple[str, ...]]] = None) -> 'XarrayContext':
        """
        Lee un dataset de Xarray y lo registra en un nuevo contexto.
        Esta función es un alias de `from_dataset` con una API más limpia.
        
        Args:
            ds: Dataset de Xarray.
            chunks: Tamaño de los chunks.
            table_names: Diccionario para mapear nombres de tablas a dimensiones.
                         Formato: {'nombre_tabla': ('dim1', 'dim2', ...)}
        
        Returns:
            XarrayContext: Nuevo contexto con el dataset registrado.
        """
        ctx = cls()
        ctx.from_dataset('default', ds, chunks, table_names)
        return ctx

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, chunks: Optional[Dict] = None, table_names: Optional[Dict[str, Tuple[str, ...]]] = None) -> 'XarrayContext':
        """
        Alias de `read_xarray` para mantener consistencia con la nomenclatura de Xarray.
        
        Args:
            ds: Dataset de Xarray.
            chunks: Tamaño de los chunks.
            table_names: Diccionario para mapear nombres de tablas a dimensiones.
                         Formato: {'nombre_tabla': ('dim1', 'dim2', ...)}
        
        Returns:
            XarrayContext: Nuevo contexto con el dataset registrado.
        """
        return cls.read_xarray(ds, chunks, table_names)

    def __getitem__(self, key: str) -> Any:
        """
        Permite acceso a tablas mediante subíndices: ctx['table']
        """
        return self.table(key)

    def __setitem__(self, key: str, value: Any):
        """
        Permite registrar tablas mediante asignación: ctx['table'] = ds
        """
        if isinstance(value, xr.Dataset):
            self.from_dataset(key, value)
        else:
            raise TypeError(f"Expected xr.Dataset, got {type(value)}")

    def table(self, name: str) -> Any:
        """
        Obtiene una tabla registrada.
        """
        if name not in self._tables:
            raise KeyError(f"Table '{name}' not found. Available tables: {list(self._tables.keys())}")
        return self._tables[name]

    @property
    def chunks(self) -> Dict[str, Optional[Dict]]:
        """
        Obtiene los chunks de las tablas registradas.
        """
        return self._chunks

    def get_chunks(self, table_name: str) -> Optional[Dict]:
        """
        Obtiene los chunks de una tabla específica.
        """
        return self._chunks.get(table_name)

    def deregister_table(self, name: str):
        """
        Elimina una tabla registrada.
        """
        if name in self._tables:
            del self._tables[name]
            del self._chunks[name]
            if name in self._table_names:
                del self._table_names[name]

    def list_tables(self) -> list:
        """
        Lista todas las tablas registradas.
        """
        return list(self._tables.keys())

    def get_table_names(self) -> Dict[str, Tuple[str, ...]]:
        """
        Obtiene el mapeo de nombres de tablas a dimensiones.
        """
        return self._table_names

    def to_dataset(self, df: pd.DataFrame, dims: Optional[List[str]] = None) -> xr.Dataset:
        """
        Convierte un DataFrame a un Dataset de Xarray con inferencia automática de dimensiones.
        
        Args:
            df: DataFrame a convertir.
            dims: Dimensiones para el Dataset (opcional). Si no se proporcionan, se infieren automáticamente.
        
        Returns:
            xr.Dataset: Dataset convertido.
        """
        if dims is None:
            dims = self._infer_dims(df)
        
        missing_dims = [d for d in dims if d not in df.columns]
        if missing_dims:
            raise ValueError(f"Columns not found in DataFrame: {missing_dims}")
        
        for d in dims:
            if len(df[d].unique()) != len(df):
                raise ValueError(f"Column '{d}' does not have unique values. Cannot use as dimension.")
        
        if len(dims) == 1:
            df_indexed = df.set_index(dims[0])
        else:
            df_indexed = df.set_index(dims)
        
        return df_indexed.to_xarray()

    def _infer_dims(self, df: pd.DataFrame) -> List[str]:
        """
        Infiere automáticamente las dimensiones a partir de las columnas del DataFrame.
        
        Args:
            df: DataFrame de entrada.
        
        Returns:
            List[str]: Lista de dimensiones inferidas.
        """
        possible_dims = ['time', 'sample', 'step', 'epoch', 'batch', 'id', 'index']
        all_cols = df.columns.tolist()
        dims = []
        for col in possible_dims:
            if col in all_cols and len(df[col].unique()) > 1:
                dims.append(col)
                all_cols.remove(col)
        
        if not dims:
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                dims = [non_numeric_cols[0]]
            else:
                dims = [all_cols[0]]
        
        return dims

    def sql(self, query: str) -> pd.DataFrame:
        """
        Ejecuta una consulta SQL sobre las tablas registradas.
        """
        # Placeholder para SQL real
        return pd.DataFrame()

    def __repr__(self) -> str:
        """
        Representación legible del contexto.
        """
        tables_info = "\n".join([
            f"  - {name}: {type(table).__name__} (chunks: {self._chunks.get(name)})"
            for name, table in self._tables.items()
        ])
        if not tables_info:
            tables_info = "  (No tables registered)"
        return f"XarrayContext(tables={len(self._tables)}):\n{tables_info}"