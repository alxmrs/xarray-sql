import xarray as xr
from datafusion import SessionContext
from collections import defaultdict

from . import cftime as cft
from .df import Chunks
from .reader import read_xarray_table


class XarrayContext(SessionContext):
    """A datafusion `SessionContext` that also supports `xarray.Dataset`s."""

    def from_dataset(
        self,
        table_name: str,
        input_table: xr.Dataset,
        chunks: Chunks = None,
    ):
        """Register an xarray Dataset as a queryable SQL table.

        For datasets with non-Gregorian cftime coordinates (e.g. 360_day,
        julian), a ``cftime()`` scalar UDF is automatically registered so
        you can write ergonomic SQL filters::

            ctx.from_dataset("ds360", ds, chunks={"time": 6})
            ctx.sql("SELECT * FROM ds360 WHERE time >= cftime('2000-07-01')")

        The UDF converts a date string to the int64 offset used to store
        that calendar's time axis.

        .. note::

            Only one ``cftime()`` UDF is registered per context, using the
            units and calendar of the *first* non-Gregorian coordinate
            encountered. If you register multiple datasets with *different*
            non-Gregorian calendars (e.g. one 360_day and one julian), the
            UDF from the first registration will be used for all subsequent
            ``cftime()`` calls and may produce incorrect offsets for the
            other dataset. In that case, create a separate ``XarrayContext``
            for each calendar.

        Args:
            table_name: The SQL table name to register the dataset under.
            input_table: An xarray Dataset. All data_vars must share the
                same dimensions.
            chunks: Xarray-like chunks specification. If not provided, uses
                the Dataset's existing chunks.

        Returns:
            self, to allow chaining.
        """

        table = read_xarray_table(input_table, chunks)
        self.register_table(table_name, table)

        # Auto-register a cftime() UDF for non-Gregorian cftime coordinates
        # so users can write: WHERE time > cftime('0500-01-01')
        for coord_name in input_table.dims:
            if cft.is_cftime_index(input_table, coord_name):
                units, cal = cft.encoding(input_table, coord_name)
                if not cft.is_gregorian_like(cal):
                    self.register_udf(cft.make_cftime_udf(units, cal))
                    break  # One UDF per context is enough.

        return self

    def from_dataset_multi(
        self,
        dataset_name: str,
        input_table: xr.Dataset,
        dim_aliases: dict[tuple[str, ...], str] | None = None,
        chunks: Chunks = None,
    ):
        """Register an xarray Dataset with mixed dimensions as multiple SQL tables.

        When a Dataset contains variables with different dimensions (e.g. some
        variables on a 3D grid and others on a 4D grid), this method splits them
        into separate tables and registers each one. Each table is named
        ``<dataset_name>.<dim1>_<dim2>_...`` by default, or using the override
        provided in ``dim_aliases``::

            ctx.from_dataset_multi('era5', ds, chunks={'time': 24})
            # registers: 'era5.time_lat_lon' and 'era5.time_lat_lon_level'
            ctx.sql('SELECT AVG(temperature_2m) FROM "era5.time_lat_lon"')

        Args:
        dataset_name: A name for the dataset, used as a prefix for all
            registered table names.
        input_table: An xarray Dataset. Variables may have differing dimensions.
        dim_aliases: Optional mapping from dimension tuples to custom table
            name suffixes. For example,
            ``{('time', 'lat', 'lon'): 'surface'}`` registers the table as
            ``era5.surface`` instead of ``era5.time_lat_lon``.
        chunks: Xarray-like chunks specification. If not provided, uses
            the Dataset's existing chunks.

        Returns:
            self, to allow chaining.
        """

        dim_aliases = dim_aliases or {}
        groups = _group_vars_by_dims(input_table)

        for dims, var_names in groups.items():
            suffix = dim_aliases.get(dims, "_".join(dims))
            table_name = f"{dataset_name}_{suffix}"
            sub_ds = input_table[var_names]
            self.from_dataset(table_name, sub_ds, chunks)

        return self


def _group_vars_by_dims(ds: xr.Dataset) -> dict[tuple[str, ...], list[str]]:
    """Group variables in the dataset based on shared dims.

    ("time", "lat", "lon"):          ["temperature_2m", "wind_speed"],
    ("time", "lat", "lon", "level"): ["pressure", "humidity"]
    """
    groups = defaultdict(list)
    for var_name, var in ds.data_vars.items():
        dims = var.dims
        groups[dims].append(var_name)
    return groups
