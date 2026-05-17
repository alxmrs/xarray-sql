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
        *,
        dim_group_aliases: dict[tuple[str, ...], str] | None = None,
        chunks: Chunks = None,
    ):
        """Register an xarray Dataset as one or more queryable SQL tables.

        When all data variables share the same dimensions, the dataset is
        registered as a single table named ``table_name``. When variables
        have differing dimensions (e.g. some on a 3D grid and others on a
        4D grid), the dataset is split into one table per dimension group,
        each named ``<table_name>_<dim1>_<dim2>_...`` by default::

            ctx.from_dataset('era5', ds, chunks={'time': 24})
            # registers: 'era5_time_lat_lon' and 'era5_time_lat_lon_level'
            ctx.sql('SELECT AVG(temperature_2m) FROM "era5_time_lat_lon"')

        Use ``dim_group_aliases`` to override the suffix for specific
        dimension tuples::

            ctx.from_dataset(
                'era5', ds,
                dim_group_aliases={('time', 'lat', 'lon'): 'surface'},
            )
            # registers: 'era5_surface' and 'era5_time_lat_lon_level'

        For datasets with non-Gregorian cftime coordinates (e.g. 360_day,
        julian), a ``cftime()`` scalar UDF is automatically registered so
        you can write ergonomic SQL filters::

            ctx.from_dataset("ds360", ds, chunks={"time": 6})
            ctx.sql("SELECT * FROM ds360 WHERE time >= cftime('2000-07-01')")

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
            table_name: The SQL table name (or table-name prefix, when the
                dataset has mixed dimensions) to register the dataset under.
            input_table: An xarray Dataset.
            dim_group_aliases: Optional mapping from dimension tuples to
                custom table-name suffixes, used when the dataset has
                variables with differing dimensions.
            chunks: Xarray-like chunks specification. If not provided, uses
                the Dataset's existing chunks.

        Returns:
            self, to allow chaining.
        """
        groups = _group_vars_by_dims(input_table)

        if len(groups) <= 1:
            return self._from_dataset(table_name, input_table, chunks)

        dim_group_aliases = dim_group_aliases or {}
        for dims, var_names in groups.items():
            suffix = dim_group_aliases.get(dims, "_".join(dims))
            sub_name = f"{table_name}_{suffix}"
            self._from_dataset(sub_name, input_table[var_names], chunks)

        return self

    def _from_dataset(
        self,
        table_name: str,
        input_table: xr.Dataset,
        chunks: Chunks = None,
    ):
        """Register a uniform-dimension Dataset as a single SQL table."""

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
