import xarray as xr
from datafusion import SessionContext
from catalog import register_catalog_from_dataset


class XarrayContext(SessionContext):
    """
    A regular DataFusion SessionContext but with an extra method
    for registering xarray datasets.
    """

    def register_catalog_from_dataset(self, ds, catalog_name="xarray", schema_name="data", chunks=None):
        register_catalog_from_dataset(self, ds, catalog_name, schema_name, chunks)