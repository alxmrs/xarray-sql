import xarray as xr
from datafusion import SessionContext
from datafusion.catalog import Schema
from collections import defaultdict

from . import cftime as cft
from .df import Chunks
from .ds import XarrayDataFrame
from .reader import read_xarray_table


class XarrayContext(SessionContext):
    """A datafusion `SessionContext` that also supports `xarray.Dataset`s."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track registered xarray Datasets so XarrayDataFrame can recover
        # defaults (dimension_columns) and metadata (var/dataset attrs,
        # non-dim coords, dim-coord dtype) that the forward pivot drops.
        # Keys are the fully-qualified table names users will reference
        # in SQL (e.g. ``"air"`` for a uniform-dim Dataset, or
        # ``"era5.surface"`` for one entry from a multi-dim-group split).
        self._registered_datasets: dict[str, xr.Dataset] = {}

    def from_dataset(
        self,
        name: str,
        input_table: xr.Dataset,
        *,
        table_names: dict[tuple[str, ...], str] | None = None,
        chunks: Chunks = None,
    ):
        """Register an xarray Dataset as one or more queryable SQL tables.

        When all data variables share the same dimensions, the dataset is
        registered as a single table named ``name``. When variables have
        differing dimensions (e.g. some on a 3D grid and others on a 4D
        grid), the dataset is split into one table per dimension group.
        The tables are registered under a SQL schema (namespace) named
        ``name`` and named ``<dim1>_<dim2>_...`` by default::

            ctx.from_dataset('era5', ds, chunks={'time': 24})
            # registers tables: 'era5.time_lat_lon' and
            #                   'era5.time_lat_lon_level'
            ctx.sql('SELECT AVG(temperature_2m) FROM era5.time_lat_lon')

        Use ``table_names`` to override the name for specific dimension
        tuples::

            ctx.from_dataset(
                'era5', ds,
                table_names={('time', 'lat', 'lon'): 'surface'},
            )
            ctx.sql('SELECT * FROM era5.surface')

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
            name: The SQL identifier under which the dataset is registered.
                For datasets with uniform dimensions, this is the table
                name. For datasets with mixed dimensions, this is the name
                of a SQL schema (namespace) containing one table per
                dimension group.
            input_table: An xarray Dataset.
            table_names: Optional mapping from dimension tuples to custom
                table names within the schema, used when the dataset has
                variables with differing dimensions.
            chunks: Xarray-like chunks specification. If not provided, uses
                the Dataset's existing chunks.

        Returns:
            self, to allow chaining.
        """
        groups = _group_vars_by_dims(input_table)

        if len(groups) <= 1:
            return self._from_dataset(name, input_table, chunks)

        table_names = table_names or {}
        schema = Schema.memory_schema(self)
        self.catalog().register_schema(name, schema)

        for dims, var_names in groups.items():
            sub_name = table_names.get(dims, "_".join(dims))
            sub_ds = input_table[var_names]
            schema.register_table(sub_name, read_xarray_table(sub_ds, chunks))
            # Track the fully-qualified name for XarrayDataFrame metadata
            # recovery on round-trip.
            self._registered_datasets[f"{name}.{sub_name}"] = sub_ds
            self._maybe_register_cftime_udf(sub_ds)

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
        self._registered_datasets[table_name] = input_table
        self._maybe_register_cftime_udf(input_table)
        return self

    def _maybe_register_cftime_udf(self, ds: xr.Dataset) -> None:
        """Auto-register a cftime() UDF for non-Gregorian cftime coordinates."""
        for coord_name in ds.dims:
            if cft.is_cftime_index(ds, coord_name):
                units, cal = cft.encoding(ds, coord_name)
                if not cft.is_gregorian_like(cal):
                    self.register_udf(cft.make_cftime_udf(units, cal))
                    break  # One UDF per context is enough.

    def sql(self, query: str, *args, **kwargs) -> XarrayDataFrame:
        """Run a SQL query, returning an :class:`XarrayDataFrame` wrapper.

        Identical to ``datafusion.SessionContext.sql`` except the returned
        object wraps the DataFusion DataFrame. The wrapper exposes
        ``.to_pandas()`` (unchanged), forwards every other DataFusion
        method via ``__getattr__``, and adds
        ``.to_dataset(dimension_columns=[...])`` for round-tripping the
        result back to an ``xr.Dataset``.

        Args:
            query: A SQL query string.
            *args: Forwarded to ``SessionContext.sql``.
            **kwargs: Forwarded to ``SessionContext.sql``.

        Returns:
            An :class:`XarrayDataFrame` wrapping the DataFusion DataFrame.
        """
        inner = super().sql(query, *args, **kwargs)
        return XarrayDataFrame(inner, templates=self._registered_datasets)


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
