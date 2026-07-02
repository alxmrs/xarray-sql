import xarray as xr
import pandas as pd
from typing import Dict, Optional, Any, List

class XarrayContext:
    """
    Contexto para manejar datasets de Xarray como tablas SQL.
    """
    def __init__(self):
        self._tables: Dict[str, Any] = {}
        self._chunks: Dict[str, Optional[Dict]] = {}

    def from_dataset(self, name: str, ds: xr.Dataset, chunks: Optional[Dict] = None):
        """
        Registra un dataset de Xarray como una tabla SQL.
        
        Args:
            name: Nombre de la tabla.
            ds: Dataset de Xarray.
            chunks: Tamaño de los chunks para la tabla.
        """
        self._tables[name] = ds
        self._chunks[name] = chunks

    def __getitem__(self, key: str) -> Any:
        """
        Permite acceso a tablas mediante subíndices: ctx['table']
        """
        return self.table(key)

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

    def list_tables(self) -> list:
        """
        Lista todas las tablas registradas.
        """
        return list(self._tables.keys())

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
        
        # Verificar que las dimensiones existen en el DataFrame
        missing_dims = [d for d in dims if d not in df.columns]
        if missing_dims:
            raise ValueError(f"Columns not found in DataFrame: {missing_dims}")
        
        # Verificar que las dimensiones tengan valores únicos
        for d in dims:
            if len(df[d].unique()) != len(df):
                raise ValueError(f"Column '{d}' does not have unique values. Cannot use as dimension.")
        
        # Crear índice multi-dimensional si hay más de una dimensión
        if len(dims) == 1:
            df_indexed = df.set_index(dims[0])
        else:
            df_indexed = df.set_index(dims)
        
        # Convertir a Dataset
        return df_indexed.to_xarray()

    def _infer_dims(self, df: pd.DataFrame) -> List[str]:
        """
        Infiere automáticamente las dimensiones a partir de las columnas del DataFrame.
        
        Args:
            df: DataFrame de entrada.
        
        Returns:
            List[str]: Lista de dimensiones inferidas.
        """
        # Lista de columnas que típicamente son dimensiones
        possible_dims = ['time', 'sample', 'step', 'epoch', 'batch', 'id', 'index']
        
        # Obtener todas las columnas
        all_cols = df.columns.tolist()
        
        # Intentar encontrar dimensiones potenciales
        dims = []
        for col in possible_dims:
            if col in all_cols and len(df[col].unique()) > 1:
                dims.append(col)
                all_cols.remove(col)
        
        # Si no se encontraron dimensiones, usar la primera columna no numérica
        if not dims:
            non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                dims = [non_numeric_cols[0]]
            else:
                # Si todas son numéricas, usar la primera columna
                dims = [all_cols[0]]
        
        return dims

    def sql(self, query: str) -> pd.DataFrame:
        """
        Ejecuta una consulta SQL sobre las tablas registradas.
        """
        # Lógica para ejecutar consultas SQL
        # ...
        return pd.DataFrame()