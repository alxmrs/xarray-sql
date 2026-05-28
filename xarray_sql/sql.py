import xarray as xr
from datafusion import SessionContext

from . import cftime as cft
from .df import Chunks
from .ds import XarrayDataFrame, _RegistryView
from .reader import read_xarray_table


class XarrayContext(SessionContext):
    """A datafusion `SessionContext` that also supports `xarray.Dataset`s."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track registered xarray Datasets so XarrayDataFrame can recover
        # defaults (dimension_columns) and metadata (var/dataset attrs, non-dim
        # coords, dim-coord dtype) the forward pivot strips.
        self._registered_datasets: dict[str, xr.Dataset] = {}

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
        self._registered_datasets[table_name] = input_table

        # Auto-register a cftime() UDF for non-Gregorian cftime coordinates
        # so users can write: WHERE time > cftime('0500-01-01')
        for coord_name in input_table.dims:
            if cft.is_cftime_index(input_table, coord_name):
                units, cal = cft.encoding(input_table, coord_name)
                if not cft.is_gregorian_like(cal):
                    self.register_udf(cft.make_cftime_udf(units, cal))
                    break  # One UDF per context is enough.

        return self

    def sql(self, query: str, *args, **kwargs) -> XarrayDataFrame:
        """Run a SQL query, returning an :class:`XarrayDataFrame` wrapper.

        Identical to ``datafusion.SessionContext.sql`` except the returned
        object wraps the DataFusion DataFrame. The wrapper exposes
        ``.to_pandas()`` (unchanged), forwards every other DataFusion
        method via ``__getattr__``, and adds
        ``.to_dataset(dimension_columns=[...])`` for round-tripping the result
        back to an ``xr.Dataset``.

        Args:
            query: A SQL query string.
            *args: Forwarded to ``SessionContext.sql``.
            **kwargs: Forwarded to ``SessionContext.sql``.

        Returns:
            An :class:`XarrayDataFrame` wrapping the DataFusion DataFrame.
        """
        inner = super().sql(query, *args, **kwargs)
        registry = _RegistryView(templates=dict(self._registered_datasets))
        return XarrayDataFrame(inner, registry=registry)
