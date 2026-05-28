"""Reconstruct xarray Datasets from SQL query results.

The inverse of the forward Dataset-to-table pivot done by
:func:`xarray_sql.df.pivot`. Exposes :class:`XarrayDataFrame`, a thin
wrapper around the DataFusion ``DataFrame`` returned by
:meth:`XarrayContext.sql`, with a :meth:`XarrayDataFrame.to_dataset`
method that round-trips a query result back to ``xr.Dataset``.

``.to_dataset()`` is lazy by default for ``SELECT *``-style queries:
data variables are backed by :class:`SQLBackendArray` wrapped in
``xarray.core.indexing.LazilyIndexedArray``. Slicing and ``.sel`` are
translated into DataFusion ``filter`` expressions and consumed via
``execute_stream``, so only the requested slab is materialized as Arrow
``RecordBatch`` es and scattered directly into numpy. Aggregation
queries (whose result has columns not in the registered template) fall
back to an eager Arrow-native materialize-and-scatter path that also
avoids pandas. ``.compute()`` always returns an in-memory Dataset.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from datafusion import col, literal

# ``sparse_extent`` selects the dim-coord range of the output for a
# filtered query:
#   - "result"   : keep only the dim values present in the query result
#                  (sparse output equal to whatever rows came back).
#   - "template" : reindex to the registered Dataset's full coord ranges,
#                  filling absent cells with ``fill_value``.
SparseExtent = Literal["result", "template"]


# ---------------------------------------------------------------------------
# Registry view (shared between XarrayContext and the wrapper)
# ---------------------------------------------------------------------------


@dataclass
class _RegistryView:
    """Snapshot of the registered Datasets handed to a wrapper.

    Maps each ``ctx.from_dataset(name, ds)`` registration to its source
    ``xr.Dataset``. Held privately by :class:`XarrayDataFrame` so
    :meth:`XarrayDataFrame.to_dataset` can recover metadata that the
    forward pivot drops. Not part of the public API.
    """

    templates: dict[str, xr.Dataset] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ds_var_dims(ds: xr.Dataset) -> list[str]:
    """Return a Dataset's data-variable dim order.

    The forward path validates that all data variables share the same dims
    tuple, so the first var's dim order is canonical. Falls back to
    ``ds.dims`` keys for empty Datasets. Always use this rather than
    ``list(ds.dims)`` when round-tripping, since the latter is in
    canonical name order and may not match the variable's axis order.
    """
    if ds.data_vars:
        return list(next(iter(ds.data_vars.values())).dims)
    return list(ds.dims)


def _to_arrow_table(result: Any) -> pa.Table:
    """Convert any supported tabular result to a ``pyarrow.Table``.

    Accepts ``pa.Table``, ``pd.DataFrame``, ``XarrayDataFrame``, or any
    duck-typed object exposing ``execute_stream`` (DataFusion DataFrame).
    The XarrayDataFrame / DataFusion paths consume Arrow ``RecordBatch``
    es directly via ``execute_stream`` -- no pandas copy.
    """
    if isinstance(result, pa.Table):
        return result
    if isinstance(result, XarrayDataFrame):
        batches = [b.to_pyarrow() for b in result._inner.execute_stream()]
        return pa.Table.from_batches(batches)
    if isinstance(result, pd.DataFrame):
        return pa.Table.from_pandas(result, preserve_index=False)
    if hasattr(result, "execute_stream"):
        batches = [b.to_pyarrow() for b in result.execute_stream()]
        return pa.Table.from_batches(batches)
    if hasattr(result, "to_pandas"):
        return pa.Table.from_pandas(result.to_pandas(), preserve_index=False)
    raise TypeError(
        f"Unsupported result type {type(result).__name__!r}; expected "
        "pa.Table, pd.DataFrame, datafusion.DataFrame, or XarrayDataFrame"
    )


def _drop_null_dim_rows_arrow(
    table: pa.Table, dimension_columns: list[str]
) -> pa.Table:
    """Drop rows with null dim coords from a ``pa.Table``. Warns once."""
    import pyarrow.compute as pc

    if table.num_rows == 0:
        return table
    # Combine per-dim null masks: keep rows where NO dim column is null.
    keep = None
    for d in dimension_columns:
        col_arr = table.column(d)
        is_null = pc.is_null(col_arr)
        not_null = pc.invert(is_null)
        keep = not_null if keep is None else pc.and_(keep, not_null)
    if keep is None:
        return table
    n_dropped = table.num_rows - int(pc.sum(keep).as_py())
    if n_dropped == 0:
        return table
    null_cols = [
        d
        for d in dimension_columns
        if int(pc.sum(pc.is_null(table.column(d))).as_py()) > 0
    ]
    warnings.warn(
        f"Dropping {n_dropped} row(s) with null dim values in "
        f"columns {null_cols} before reshape",
        stacklevel=3,
    )
    return table.filter(keep)


def _apply_template(ds: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    """Recover metadata that the forward SQL pivot strips.

    Adds back, where unambiguous:

    * Data-variable ``attrs`` and ``encoding`` for vars present in
      ``template`` (aggregation aliases like ``air_avg`` get nothing).
      Dtype-bound encoding keys (``dtype``, ``_FillValue``,
      ``missing_value``) are intentionally dropped: SQL may have
      changed the column's dtype (e.g. ``int16`` -> ``float64`` after
      ``AVG`` or a null-introducing filter), and reattaching the
      source's packing would make a later ``ds.to_netcdf()`` write
      corrupt values.
    * Dim-coordinate dtype, where SQL upcasted (datetime is the
      canonical case).
    * Non-dim coordinates whose dims are all present in ``ds`` (scalar
      coords attach as-is; vector coords use ``.sel``).
    * Dataset-level ``attrs``.

    Skipped coords are warned about once per call.
    """
    out = ds.copy()

    # 1. Data-var attrs / encoding for vars present in the template.
    #    Aggregation aliases absent from template intentionally inherit nothing.
    for name in list(out.data_vars):
        if name in template.data_vars:
            out[name].attrs = dict(template[name].attrs)
            # Drop dtype-bound encoding keys; SQL may have changed dtype.
            enc = {
                k: v
                for k, v in template[name].encoding.items()
                if k not in {"dtype", "_FillValue", "missing_value"}
            }
            out[name].encoding = enc

    # 2. Restore dim-coordinate dtype when SQL changed it (e.g. datetime
    #    upcast through pyarrow / pandas) and copy the source's dim-coord
    #    attrs (``standard_name``, ``long_name``, ``units``, etc.).
    for d in list(out.dims):
        if d in template.coords:
            tdt = template.coords[d].dtype
            if out.coords[d].dtype != tdt:
                try:
                    out = out.assign_coords({d: out.coords[d].astype(tdt)})
                except (ValueError, TypeError):
                    pass  # incompatible cast; leave as-is
            out[d].attrs = dict(template.coords[d].attrs)

    # 3. Non-dim coordinates whose dims are all present in the result.
    out_dims = set(out.dims)
    skipped: list[str] = []
    for cname, coord in template.coords.items():
        if cname in template.dims:
            continue  # dim coord; already in out
        if not set(coord.dims) <= out_dims:
            continue  # spans dims the result lacks
        try:
            if not coord.dims:
                # Scalar coord (e.g. weather_dataset.reference_time).
                out = out.assign_coords({cname: coord})
            else:
                sel = {d: out.coords[d] for d in coord.dims}
                out = out.assign_coords({cname: coord.sel(sel)})
        except (KeyError, ValueError, TypeError):
            skipped.append(cname)

    # 4. Dataset-level attrs.
    out.attrs = dict(template.attrs)

    if skipped:
        warnings.warn(
            f"Could not re-attach non-dim coordinates from template: {skipped}",
            stacklevel=3,
        )
    return out


def _scatter_batches_to_ndarray(
    batches: list[pa.RecordBatch],
    dimension_columns: list[str],
    requested: dict[str, np.ndarray],
    var_name: str,
    out_shape: tuple[int, ...],
    dtype: np.dtype,
    drop_axes: list[int],
) -> np.ndarray:
    """Scatter Arrow ``RecordBatch`` rows into an N-D numpy buffer.

    Each batch carries dim columns and a single value column for
    ``var_name``. For every row we look up the position of its dim
    coordinate values in the caller's requested coord arrays via
    ``np.searchsorted`` and write the value at that N-D index. NaN-fill
    initialization handles missing combinations (sparse results).
    """
    # NaN fill for float outputs; default for int/datetime falls through
    # to ``np.empty``-style undefined values (but every output cell is
    # written below for non-sparse cases).
    out = (
        np.full(out_shape, np.nan, dtype=dtype)
        if np.issubdtype(dtype, np.floating)
        else np.empty(out_shape, dtype=dtype)
    )

    # ``requested[d]`` may be in any order (callers can iselect arbitrary
    # positions, and template coords like air_temperature.lat are descending).
    # ``np.searchsorted`` requires ascending input, so we sort each requested
    # array once, search there, and remap back to the original positions.
    sorted_idx = {d: np.argsort(requested[d]) for d in dimension_columns}
    sorted_req = {d: requested[d][sorted_idx[d]] for d in dimension_columns}

    for batch in batches:
        if batch.num_rows == 0:
            continue
        schema_names = batch.schema.names
        # Build per-dim position arrays for this batch (positions within
        # the caller's requested coord order).
        positions = []
        for d in dimension_columns:
            col_arr = batch.column(schema_names.index(d))
            vals = col_arr.to_numpy(zero_copy_only=False)
            pos_in_sorted = np.searchsorted(sorted_req[d], vals)
            positions.append(sorted_idx[d][pos_in_sorted])
        value_arr = batch.column(schema_names.index(var_name)).to_numpy(
            zero_copy_only=False
        )
        out[tuple(positions)] = value_arr.astype(dtype, copy=False)

    if drop_axes:
        out = np.squeeze(out, axis=tuple(drop_axes))
    return cast(np.ndarray, out)


class SQLBackendArray(xr.backends.BackendArray):
    """Lazy N-D array backed by DataFusion's native filter pushdown.

    On each ``__getitem__`` call, the requested xarray indexer is
    translated into a DataFusion filter expression (``df.filter(expr)``)
    and a column projection (``df.select(*cols)``). The filtered
    DataFrame is consumed via ``execute_stream`` as a sequence of Arrow
    ``RecordBatch`` es and scattered into a preallocated numpy buffer,
    so only the requested slab is materialized. No pandas hop and no
    SQL string synthesis.

    Constructed by :func:`_lazy_to_xarray`; users should not instantiate
    this class directly.
    """

    def __init__(
        self,
        inner_df: Any,
        var_name: str,
        dimension_columns: list[str],
        coord_arrays: dict[str, np.ndarray],
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self._inner_df = inner_df
        self._var_name = var_name
        self._dimension_columns = list(dimension_columns)
        self._coord_arrays = coord_arrays
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        return cast(
            np.ndarray,
            xr.core.indexing.explicit_indexing_adapter(
                key,
                self.shape,
                xr.core.indexing.IndexingSupport.OUTER,
                self._raw_getitem,
            ),
        )

    def __copy__(self) -> "SQLBackendArray":
        # The backend is read-only; the underlying DataFusion DataFrame
        # holds a non-picklable SessionContext reference, so sharing the
        # same backend across a copy is both safe and necessary.
        return self

    def __deepcopy__(self, memo: dict) -> "SQLBackendArray":
        return self

    # ------------------------------------------------------------------

    def _raw_getitem(self, key: tuple) -> np.ndarray:
        """Materialize the slab described by *key* via DataFusion + Arrow.

        ``key`` is a tuple of ``int``/``slice``/1-D integer-array, one per
        dim, in :attr:`_dimension_columns` order.
        """
        requested: dict[str, np.ndarray] = {}
        # Dims whose indexer covers the full extent (slice(None) or
        # equivalent). For these we omit the filter predicate entirely
        # so DataFusion doesn't have to evaluate a tautology.
        full_dims: set[str] = set()
        drop_axes: list[int] = []
        for axis, (dim, k) in enumerate(zip(self._dimension_columns, key)):
            coord = self._coord_arrays[dim]
            if isinstance(k, slice):
                start = 0 if k.start is None else k.start
                stop = len(coord) if k.stop is None else k.stop
                step = 1 if k.step is None else k.step
                requested[dim] = np.asarray(coord[start:stop:step])
                if start == 0 and stop >= len(coord) and step == 1:
                    full_dims.add(dim)
            elif isinstance(k, (int, np.integer)):
                requested[dim] = np.asarray([coord[int(k)]])
                drop_axes.append(axis)
            else:
                arr = np.asarray(k)
                requested[dim] = np.asarray(coord[arr])
                if (
                    len(arr) == len(coord)
                    and (arr == np.arange(len(coord))).all()
                ):
                    full_dims.add(dim)

        out_shape = tuple(len(requested[d]) for d in self._dimension_columns)
        if any(n == 0 for n in out_shape):
            empty = np.empty(out_shape, dtype=self.dtype)
            squeezed = (
                np.squeeze(empty, axis=tuple(drop_axes)) if drop_axes else empty
            )
            return cast(np.ndarray, squeezed)

        # Build a single DataFusion filter expression as the AND of per-dim
        # predicates. For a single requested value: equality. For multiple:
        # OR-chain of equalities (DataFusion 52.0.0 does not expose a clean
        # ``Expr.in_list`` from Python; OR-chained equalities constant-fold
        # equivalently and stay typed).
        predicates = []
        for dim in self._dimension_columns:
            if dim in full_dims:
                continue
            vals = requested[dim]
            if len(vals) == 1:
                predicates.append(col(f'"{dim}"') == literal(vals[0]))
            else:
                eq = col(f'"{dim}"') == literal(vals[0])
                for v in vals[1:]:
                    eq = eq | (col(f'"{dim}"') == literal(v))
                predicates.append(eq)

        filtered = self._inner_df
        if predicates:
            combined = predicates[0]
            for p in predicates[1:]:
                combined = combined & p
            filtered = filtered.filter(combined)
        projected = filtered.select(
            *(col(f'"{c}"') for c in self._dimension_columns + [self._var_name])
        )

        # Consume the projected DataFrame as Arrow RecordBatches. The
        # DataFusion wrapper exposes ``.to_pyarrow()`` to convert each
        # batch into a true ``pyarrow.RecordBatch``.
        batches = [b.to_pyarrow() for b in projected.execute_stream()]
        return _scatter_batches_to_ndarray(
            batches=batches,
            dimension_columns=self._dimension_columns,
            requested=requested,
            var_name=self._var_name,
            out_shape=out_shape,
            dtype=self.dtype,
            drop_axes=drop_axes,
        )


def _is_pushdownable(
    template: xr.Dataset | None,
    dimension_columns: list[str],
    result_cols: list[str],
) -> bool:
    """Decide whether a query can be lazily pushed down.

    True when there is a registered template, every requested ``dimension_columns``
    appears in the result, and the result has no columns beyond the
    template's dim coords and data vars (i.e. no aggregation aliases).
    """
    if template is None:
        return False
    allowed = set(template.dims) | set(template.data_vars)
    rcols = set(result_cols)
    if not rcols <= allowed:
        return False
    if not set(dimension_columns) <= rcols:
        return False
    return True


def _lazy_to_xarray(
    inner_df: Any,
    dimension_columns: list[str],
    template: xr.Dataset | None,
    sparse_extent: SparseExtent,
    fill_value: Any,
) -> xr.Dataset:
    """Build a lazy ``xr.Dataset`` whose data vars are :class:`SQLBackendArray`.

    Coord arrays are discovered per-dim via ``inner_df.select(col(d))
    .distinct().sort(...).execute_stream()`` (one Arrow-native DataFrame
    chain per dim). No SQL strings, no regexes; coord discovery cost is
    bounded by N single-column scans where N is the number of dims.
    """
    if sparse_extent not in ("result", "template"):
        raise ValueError(
            "sparse_extent must be 'result' or 'template', got "
            f"{sparse_extent!r}"
        )
    if sparse_extent == "template" and template is None:
        raise ValueError(
            "sparse_extent='template' requires template= to be supplied"
        )

    schema = inner_df.schema()
    field_names = [f.name for f in schema]
    field_types = {f.name: f.type for f in schema}

    coord_arrays: dict[str, np.ndarray] = {}
    for d in dimension_columns:
        dim_only = (
            inner_df.select(col(f'"{d}"')).distinct().sort(col(f'"{d}"').sort())
        )
        chunks = [b.to_pyarrow() for b in dim_only.execute_stream()]
        if not chunks:
            coord_arrays[d] = np.asarray([])
            continue
        coord_arrays[d] = np.concatenate(
            [c.column(0).to_numpy(zero_copy_only=False) for c in chunks]
        )
    shape = tuple(len(coord_arrays[d]) for d in dimension_columns)

    data_vars: dict[str, xr.Variable] = {}
    for name in field_names:
        if name in dimension_columns:
            continue
        np_dtype = field_types[name].to_pandas_dtype()
        backend = SQLBackendArray(
            inner_df=inner_df,
            var_name=name,
            dimension_columns=dimension_columns,
            coord_arrays=coord_arrays,
            shape=shape,
            dtype=np_dtype,
        )
        lazy = xr.core.indexing.LazilyIndexedArray(backend)
        data_vars[name] = xr.Variable(dimension_columns, lazy)

    coords_arg = {d: coord_arrays[d] for d in dimension_columns}
    ds = xr.Dataset(data_vars=data_vars, coords=coords_arg)

    if sparse_extent == "template":
        assert template is not None
        indexers = {
            d: template.coords[d].values
            for d in dimension_columns
            if d in template.coords and d in template.dims
        }
        if indexers:
            ds = ds.reindex(indexers, fill_value=fill_value)

    if template is not None:
        ds = _apply_template(ds, template)
    return ds


def _eager_to_xarray(
    result: Any,
    dimension_columns: list[str],
    template: xr.Dataset | None = None,
    sparse_extent: SparseExtent = "result",
    fill_value: Any = np.nan,
) -> xr.Dataset:
    """Convert a tabular result to an ``xr.Dataset`` eagerly.

    Used by :meth:`XarrayDataFrame.to_dataset` for queries that cannot be
    lazily pushed down (aggregations where the result has columns not in
    the registered template). The data path is Arrow-native: input is
    materialized as a single ``pa.Table`` and each data variable is
    scattered into a numpy buffer via :func:`_scatter_batches_to_ndarray`.
    No pandas hop in the common case (the XarrayDataFrame input goes
    straight through ``execute_stream`` to Arrow).
    """
    import pyarrow.compute as pc

    if not dimension_columns:
        raise ValueError("dimension_columns must be non-empty")
    if sparse_extent not in ("result", "template"):
        raise ValueError(
            "sparse_extent must be 'result' or 'template', got "
            f"{sparse_extent!r}"
        )
    if sparse_extent == "template" and template is None:
        raise ValueError(
            "sparse_extent='template' requires template= to be supplied"
        )

    table = _to_arrow_table(result)
    missing = [c for c in dimension_columns if c not in table.schema.names]
    if missing:
        raise ValueError(
            f"dimension_columns not found in result columns: {missing}; "
            f"available columns: {table.schema.names}"
        )

    table = _drop_null_dim_rows_arrow(table, dimension_columns)

    # Duplicate detection. ``pyarrow.compute.unique`` lacks a struct
    # kernel in pyarrow 23, so we scan once with a Python set over dim
    # tuples. The cost is O(N) on the aggregation result (typically
    # small) and only triggers a second pass on the error path.
    if table.num_rows > 0:
        dim_np = {
            d: table.column(d).to_numpy(zero_copy_only=False)
            for d in dimension_columns
        }
        seen: set[tuple] = set()
        first: dict[str, Any] | None = None
        for i in range(table.num_rows):
            tup = tuple(dim_np[d][i] for d in dimension_columns)
            if tup in seen:
                first = dict(zip(dimension_columns, tup))
                break
            seen.add(tup)
        if first is not None:
            raise ValueError(
                f"Result has duplicate dim tuples (e.g. {first}); cannot "
                "uniquely reshape into an xr.Dataset. Aggregate or de-dup "
                "the result first."
            )

    # Discover per-dim coord arrays from the result (sorted unique values).
    coord_arrays: dict[str, np.ndarray] = {}
    for d in dimension_columns:
        unique = pc.unique(table.column(d))
        sorted_unique = pc.array_sort_indices(unique)
        coord_arrays[d] = unique.take(sorted_unique).to_numpy(
            zero_copy_only=False
        )
    out_shape = tuple(len(coord_arrays[d]) for d in dimension_columns)
    requested = coord_arrays  # caller's order == sorted coord order

    # Scatter each non-dim column into its own ndarray, then assemble a
    # Dataset.
    data_vars: dict[str, xr.Variable] = {}
    for name in table.schema.names:
        if name in dimension_columns:
            continue
        np_dtype = table.schema.field(name).type.to_pandas_dtype()
        # ``_scatter_batches_to_ndarray`` accepts a list of RecordBatches,
        # so feed it the Table's batches directly.
        arr = _scatter_batches_to_ndarray(
            batches=table.to_batches(),
            dimension_columns=dimension_columns,
            requested=requested,
            var_name=name,
            out_shape=out_shape,
            dtype=np_dtype,
            drop_axes=[],
        )
        data_vars[name] = xr.Variable(dimension_columns, arr)

    coords_arg = {d: coord_arrays[d] for d in dimension_columns}
    ds = xr.Dataset(data_vars=data_vars, coords=coords_arg)

    if sparse_extent == "template":
        assert template is not None  # validated above
        indexers = {
            d: template.coords[d].values
            for d in dimension_columns
            if d in template.coords and d in template.dims
        }
        if indexers:
            ds = ds.reindex(indexers, fill_value=fill_value)

    if template is not None:
        ds = _apply_template(ds, template)
    return ds


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------


class XarrayDataFrame:
    """Wrapper around a DataFusion ``DataFrame`` with xarray-aware helpers.

    Returned by :meth:`xarray_sql.XarrayContext.sql`. Forwards every
    attribute it does not define itself to the wrapped DataFrame, so
    ``.collect()``, ``.schema()``, ``.show()``, ``.count()`` all work
    unchanged.

    Carries a private snapshot of the context's registered Datasets so
    :meth:`to_dataset` can default ``dimension_columns`` and recover metadata
    dropped by the forward pivot.

    Users should not construct this class directly; let
    :meth:`XarrayContext.sql` produce it.
    """

    def __init__(
        self,
        inner: Any,
        registry: _RegistryView | None = None,
    ) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_registry", registry or _RegistryView())

    def to_pandas(self) -> pd.DataFrame:
        """Materialize the result as a ``pd.DataFrame`` (DataFusion API)."""
        return self._inner.to_pandas()

    def to_dataset(
        self,
        dimension_columns: list[str] | None = None,
        template: xr.Dataset | None = None,
        template_table: str | None = None,
        sparse_extent: SparseExtent = "result",
        fill_value: Any = np.nan,
    ) -> xr.Dataset:
        """Convert the result to an ``xr.Dataset``.

        Args:
            dimension_columns: Result columns to use as Dataset dimensions. When
                ``None``, defaults to the dims of the registered Dataset
                referenced by the SQL ``FROM`` clause (if exactly one
                matches), or any single registered Dataset whose dims are
                all present in the result columns.
            template: Source ``xr.Dataset`` to recover metadata from.
                Overrides any auto-resolved template.
            template_table: Name of a registered table to use as the
                template. Mutually exclusive with ``template``.
            sparse_extent: ``"result"`` (default) keeps only dim values
                present in the result. ``"template"`` reindexes to the
                template's full coord ranges, filling absent cells with
                ``fill_value``; requires a template.
            fill_value: Used when ``sparse_extent="template"`` reindexes
                to a wider extent. Defaults to ``np.nan``.

        Returns:
            An ``xr.Dataset`` with ``dimension_columns`` as dimensions and the
            remaining result columns as data variables.

        Raises:
            ValueError: ``dimension_columns`` cannot be inferred, names a missing
                column, or the result has duplicate dim tuples;
                ``template_table`` is unknown; both ``template`` and
                ``template_table`` are passed; or
                ``sparse_extent="template"`` is requested without a
                resolvable template.
        """
        if template is not None and template_table is not None:
            raise ValueError("Pass at most one of template= or template_table=")
        if template is None:
            template = self._resolve_template(template_table)
        if dimension_columns is None:
            dimension_columns = self._infer_dimension_columns(
                preferred_template=template
            )
        if _is_pushdownable(
            template, dimension_columns, self._result_columns()
        ):
            return _lazy_to_xarray(
                inner_df=self._inner,
                dimension_columns=dimension_columns,
                template=template,
                sparse_extent=sparse_extent,
                fill_value=fill_value,
            )
        return _eager_to_xarray(
            self,
            dimension_columns=dimension_columns,
            template=template,
            sparse_extent=sparse_extent,
            fill_value=fill_value,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_template(
        self, template_table: str | None
    ) -> xr.Dataset | None:
        """Pick a template Dataset for metadata recovery.

        Priority:
          1. Explicit ``template_table`` argument.
          2. If exactly one Dataset is registered on the context, use it.
          3. None.

        No SQL parsing is involved: option 2 reads only the registry's
        contents. If multiple Datasets are registered, the caller must
        pass ``template=`` or ``template_table=`` explicitly.
        """
        templates = self._registry.templates
        if template_table is not None:
            if template_table not in templates:
                raise ValueError(
                    f"template_table={template_table!r} is not a "
                    "registered table on this context. Registered: "
                    f"{list(templates)}"
                )
            return templates[template_table]
        if len(templates) == 1:
            return next(iter(templates.values()))
        return None

    def _infer_dimension_columns(
        self, preferred_template: xr.Dataset | None = None
    ) -> list[str]:
        """Pick a default ``dimension_columns`` from the registry, or raise.

        Uses the data variable's dim order (via :func:`_ds_var_dims`) so
        the round-trip preserves the original axis order.
        """
        result_cols = set(self._result_columns())
        if (
            preferred_template is not None
            and set(preferred_template.dims) <= result_cols
        ):
            return _ds_var_dims(preferred_template)
        if not self._registry.templates:
            raise ValueError(
                "dimension_columns cannot be inferred (no registered Dataset on "
                "this result); pass dimension_columns=[...] explicitly."
            )
        candidates = [
            _ds_var_dims(t)
            for t in self._registry.templates.values()
            if set(t.dims) <= result_cols
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "dimension_columns cannot be inferred: no registered Dataset has "
                "all of its dims present in the result columns. Pass "
                "dimension_columns=[...] explicitly."
            )
        raise ValueError(
            "dimension_columns cannot be inferred unambiguously: multiple "
            "registered Datasets are compatible with the result. Pass "
            "dimension_columns=[...] explicitly."
        )

    def _result_columns(self) -> list[str]:
        """Return the result's column names without materializing rows."""
        try:
            schema = self._inner.schema()
        except Exception:
            return list(self._inner.to_pandas().columns)
        return [field.name for field in schema]

    def __getattr__(self, name: str) -> Any:
        # Runs only when ``name`` is not found via normal lookup, so this
        # safely forwards anything we have not overridden.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return repr(self._inner)
