import xarray as xr
import datafusion as dfn
from xarray_sql.reader import read_xarray_table


def group_vars_by_dims(ds):
    """
    Group variables in the dataset based on shared dims

    ("time", "lat", "lon"):          ["temperature_2m", "wind_speed"],
    ("time", "lat", "lon", "level"): ["pressure", "humidity"]
    """
    groups = {}

    for var_name, var in ds.data_vars.items():
        dims = var.dims

        if dims not in groups:
            groups[dims] = []

        groups[dims].append(var_name)

    return groups


def dims_to_table_name(dims):
    """
    "time", "lat", "lon" -> "time_lat_lon"
    """
    return "_".join(dims)

    
class XarraySchemaProvider(dfn.catalog.SchemaProvider):
    """
    Custom datafusion schema that holds the tables
    """

    def __init__(self, ds, groups, chunks):
        # dictionary to store the tables 
        self.tables = {}

        # create a table for for each group of vars
        for dims, var_names in groups.items():
            table_name = dims_to_table_name(dims)
            subset = ds[var_names]
            self.tables[table_name] = read_xarray_table(subset, chunks)

    def table_names(self):
        return set(self.tables.keys())

    def table(self, name):
        return self.tables.get(name)

    def table_exist(self, name):
        return name in self.tables

    def register_table(self, name, table):
        self.tables[name] = table

    def deregister_table(self, name, cascade=True):
        del self.tables[name]


class XarrayCatalogProvider(dfn.catalog.CatalogProvider):
    """
    Custom datafusion catalog that holds the schemas
    """
    #Constructor
    def __init__(self, ds, schema_name, chunks):
        groups = group_vars_by_dims(ds)

        # dictionary to store schemas using previous schema class
        """
        "data": {
        "time_lat_lon":       [temperature_2m, wind_speed],
        "time_lat_lon_level": [pressure, humidity]
        }
        """
        self.schemas = {
            schema_name: XarraySchemaProvider(ds, groups, chunks)
        }

    """
    Other methods from test_catalog.py
    """
    def schema_names(self):
        return set(self.schemas.keys())

    def schema(self, name):
        return self.schemas.get(name)

    def register_schema(self, name, schema):
        self.schemas[name] = schema

    def deregister_schema(self, name, cascade=True):
        del self.schemas[name]


def register_catalog_from_dataset(ctx, ds, catalog_name="xarray", schema_name="data", chunks=None):
    """
    Main function. Takes an xarray dataset and registers it
    with DataFusion so you can query it with SQL.
    """
    catalog = XarrayCatalogProvider(ds, schema_name, chunks)
    ctx.register_catalog_provider(catalog_name, catalog)