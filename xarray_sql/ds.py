"""Reconstruct xarray Datasets from SQL query results.

The inverse of the forward Dataset-to-table pivot done by
:func:`xarray_sql.df.pivot`. Exposes :class:`XarrayDataFrame`, a thin
wrapper around the DataFusion ``DataFrame`` returned by
:meth:`XarrayContext.sql`, with a :meth:`XarrayDataFrame.to_dataset`
method that round-trips a query result back to ``xr.Dataset``.

Every ``.to_dataset()`` result is lazy: data variables are backed by
:class:`SQLBackendArray` wrapped in
``xarray.core.indexing.LazilyIndexedArray``. xarray indexing operations
(``isel``, ``sel``, slicing) translate to DataFusion ``filter``
expressions and consume the filtered DataFrame via ``execute_stream``,
so only the requested slab is materialized as Arrow ``RecordBatch`` es
and scattered into numpy. Pushdown and laziness are orthogonal: queries
whose filters cannot be pushed down (e.g. aggregations) still stream
their result lazily on first access. ``.compute()`` materializes the
whole Dataset in memory.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from datafusion import col, literal

# ``sparsity`` selects the dim-coord range of the output for a
# filtered query:
#   - "result"   : keep only the dim values present in the query result
#                  (sparse output equal to whatever rows came back).
#   - "template" : reindex to the registered Dataset's full coord ranges,
#                  filling absent cells with ``fill_value``.
Sparsity = Literal["result", "template"]


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


def apply_template(ds: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
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
    """Convert filtered Arrow ``RecordBatch`` rows into a dense N-D numpy slab.

    SQL query results arrive as flat rows; xarray expects N-D arrays.
    This bridges the two: each row carries the dim-coord values that
    identify its cell in the output cube plus the value to write there.
    We look up the row's N-D position by binary-searching its coord
    values within the caller's requested coord arrays
    (``np.searchsorted``), then scatter-write the value at that index.

    Missing combinations (sparse results from filtered queries) stay as
    ``NaN`` for floating-point outputs by pre-filling the buffer; integer
    outputs leave them as ``np.empty``-style undefined values.
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
    """Read-only lazy N-D array view over a DataFusion DataFrame.

    Bridges xarray's lazy-indexing interface
    (:class:`xarray.backends.BackendArray`) to a DataFusion query result,
    so an xarray ``Dataset`` can present a SQL query as if it were a
    materialized N-D array without actually loading any data until the
    caller asks for it. This is the workhorse that lets
    :meth:`XarrayDataFrame.to_dataset` return a Dataset cheaply.

    On each ``__getitem__`` call, the requested xarray indexer is
    translated into a DataFusion filter expression (``df.filter(expr)``)
    and a column projection (``df.select(*cols)``). The filtered
    DataFrame is consumed via ``execute_stream`` as a sequence of Arrow
    ``RecordBatch`` es and scattered into a preallocated numpy buffer,
    so only the requested slab is materialized.

    Constraints and caveats:

    - Read-only: there is no write path; the backend exists to surface
      query results, not to round-trip writes into a SQL store.
    - The underlying DataFusion ``DataFrame`` holds a reference to its
      originating ``SessionContext``, which is not picklable. The class
      therefore overrides ``__copy__`` and ``__deepcopy__`` to return
      ``self`` -- this is safe because the backend is read-only.
    - ``IndexingSupport.OUTER``: ``BasicIndexer`` and ``OuterIndexer``
      are translated to filter predicates directly; ``VectorizedIndexer``
      paths through xarray's adapter to outer-then-gather and so still
      works, just less efficiently.

    Raises:
        ValueError, datafusion exceptions: propagated from the
            underlying ``df.filter().select().execute_stream()`` chain
            if a predicate refers to a missing column, the dtype of a
            literal is incompatible, or the execution itself fails.
        AssertionError: from ``np.searchsorted`` mis-alignment, which
            indicates the result contains coordinate values not present
            in the wrapper's pre-computed coord arrays -- usually a
            symptom of a filtered query whose coord discovery missed a
            value.

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
        for axis, (dim, k) in enumerate(
            zip(self._dimension_columns, key, strict=True)
        ):
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


def _lazy_to_xarray(
    inner_df: Any,
    dimension_columns: list[str],
    template: xr.Dataset | None,
    sparsity: Sparsity,
    fill_value: Any,
) -> xr.Dataset:
    """Build a lazy ``xr.Dataset`` whose data vars are :class:`SQLBackendArray`.

    Coord arrays are discovered per-dim via ``inner_df.select(col(d))
    .distinct().sort(...).execute_stream()`` (one Arrow-native DataFrame
    chain per dim). Coord discovery cost is bounded by N single-column
    scans where N is the number of dims.
    """
    if sparsity not in ("result", "template"):
        raise ValueError(
            f"sparsity must be 'result' or 'template', got {sparsity!r}"
        )
    if sparsity == "template" and template is None:
        raise ValueError(
            "sparsity='template' requires template= to be supplied"
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

    if sparsity == "template":
        assert template is not None
        indexers = {
            d: template.coords[d].values
            for d in dimension_columns
            if d in template.coords and d in template.dims
        }
        if indexers:
            ds = ds.reindex(indexers, fill_value=fill_value)

    if template is not None:
        ds = apply_template(ds, template)
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
        templates: dict[str, xr.Dataset] | None = None,
    ) -> None:
        """Construct a wrapper.

        Args:
            inner: The underlying ``datafusion.DataFrame`` returned by
                :meth:`XarrayContext.sql`.
            templates: Snapshot of the registered Datasets on the producing
                context, keyed by the SQL identifier each was registered
                under. Used by :meth:`to_dataset` to recover metadata that
                the forward pivot strips. ``None`` means no metadata
                recovery is possible from registrations alone; callers may
                still pass ``template=`` to :meth:`to_dataset` explicitly.
        """
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_templates", dict(templates or {}))

    def to_pandas(self) -> pd.DataFrame:
        """Materialize the result as a ``pd.DataFrame`` (DataFusion API)."""
        return self._inner.to_pandas()

    def to_dataset(
        self,
        dimension_columns: list[str] | None = None,
        template: xr.Dataset | None = None,
        template_table: str | None = None,
        sparsity: Sparsity = "result",
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
            sparsity: ``"result"`` (default) keeps only dim values
                present in the result. ``"template"`` reindexes to the
                template's full coord ranges, filling absent cells with
                ``fill_value``; requires a template.
            fill_value: Used when ``sparsity="template"`` reindexes
                to a wider extent. Defaults to ``np.nan``.

        Returns:
            An ``xr.Dataset`` with ``dimension_columns`` as dimensions and the
            remaining result columns as data variables.

        Raises:
            ValueError: ``dimension_columns`` cannot be inferred, names a missing
                column, or the result has duplicate dim tuples;
                ``template_table`` is unknown; both ``template`` and
                ``template_table`` are passed; or
                ``sparsity="template"`` is requested without a
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
        # Always go through the lazy path: pushdown is an optimization
        # for queries whose result columns map cleanly onto the template's
        # dim+var set, but laziness itself is orthogonal -- aggregation
        # results still benefit from streaming via execute_stream.
        return _lazy_to_xarray(
            inner_df=self._inner,
            dimension_columns=dimension_columns,
            template=template,
            sparsity=sparsity,
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
        """
        templates = self._templates
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
        if not self._templates:
            raise ValueError(
                "dimension_columns cannot be inferred (no registered "
                "Dataset on this result); pass dimension_columns=[...] "
                "explicitly."
            )
        candidates = [
            _ds_var_dims(t)
            for t in self._templates.values()
            if set(t.dims) <= result_cols
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "dimension_columns cannot be inferred: no registered "
                "Dataset has all of its dims present in the result "
                "columns. Pass dimension_columns=[...] explicitly."
            )
        raise ValueError(
            "dimension_columns cannot be inferred unambiguously: multiple "
            "registered Datasets are compatible with the result. Pass "
            "dimension_columns=[...] explicitly."
        )

    def _result_columns(self) -> list[str]:
        """Return the result's column names without materializing rows."""
        return [field.name for field in self._inner.schema()]

    def __getattr__(self, name: str) -> Any:
        # Runs only when ``name`` is not found via normal lookup, so this
        # safely forwards anything we have not overridden.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return repr(self._inner)
