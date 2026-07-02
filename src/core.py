import xarray as xr
import pandas as pd
from typing import Dict, Optional, Any

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

    def to_dataset(self, table_name: str, **kwargs) -> xr.Dataset:
        """
        Convierte una tabla SQL a un dataset de Xarray.
        """
        # Lógica para convertir tabla a dataset
        # ...
        return xr.Dataset()

    def sql(self, query: str) -> pd.DataFrame:
        """
        Ejecuta una consulta SQL sobre las tablas registradas.
        """
        # Lógica para ejecutar consultas SQL
        # ...
        return pd.DataFrame()