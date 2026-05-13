"""Reverse path: tabular SQL results back to xarray Datasets.

Inverse of the forward "raster -> table" pivot done by
:func:`xarray_sql.df.pivot`. Exposes:

* :class:`XarrayDataFrame` -- thin wrapper around the DataFusion
  ``DataFrame`` returned by :meth:`XarrayContext.sql`. Adds
  :meth:`XarrayDataFrame.to_dataset` for converting query results back to
  ``xr.Dataset`` while keeping every other DataFusion method available.

Phase 1 is eager: ``to_dataset`` materializes through pandas, then
reshapes back to an N-D Dataset. The structure is shaped so the lazy
``BackendArray`` swap in Phase 2 is a localized change in
``_eager_to_xarray``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

SparseExtent = Literal["result", "template"]


# ---------------------------------------------------------------------------
# Registry view (shared between XarrayContext and the wrapper)
# ---------------------------------------------------------------------------


@dataclass
class _RegistryView:
    """Snapshot of ``XarrayContext`` registrations handed to a wrapper.

    Held privately by :class:`XarrayDataFrame` so it can derive defaults
    (``dim_cols``, ``template``), recover metadata that the forward
    pivot drops, and re-execute sub-queries for lazy pushdown. Not part
    of the public API.

    Attributes:
        templates: Map of registered table name -> source ``xr.Dataset``.
        query: SQL string this wrapper was produced from. Used for
            FROM-clause matching and as the base query for lazy
            sub-queries.
        ctx: Reference to the ``XarrayContext`` that produced this
            wrapper, used to execute sub-queries via the parent
            ``SessionContext.sql`` (avoids wrapper recursion).
    """

    templates: dict[str, xr.Dataset] = field(default_factory=dict)
    query: str = ""
    ctx: Any = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# Matches "FROM ident" / "JOIN ident" with bare or double-quoted identifiers.
# Subqueries (``FROM (``) and CTE bodies do not match because ``(`` is not a
# valid identifier start.
_FROM_OR_JOIN_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_]*))',
    re.IGNORECASE,
)

# Matches a bare unfiltered SELECT *, e.g. ``SELECT * FROM "air"``. Used by
# ``_lazy_to_xarray`` to skip the DISTINCT-per-dim coord scan when the
# query is known to cover the full registered Dataset.
_UNFILTERED_SELECT_STAR_RE = re.compile(
    r'^\s*SELECT\s+\*\s+FROM\s+"?([A-Za-z_][A-Za-z0-9_]*)"?\s*;?\s*$',
    re.IGNORECASE | re.DOTALL,
)


def _extract_from_tables(query: str) -> list[str]:
    """Return identifiers appearing after FROM/JOIN in *query*.

    Conservative: only matches plain or double-quoted identifiers.
    Aliases (``FROM air a`` or ``FROM air AS a``) are unaffected because
    the alias is not captured. Subqueries do not match; CTE aliases match
    but typically do not collide with registered table names.
    """
    seen: list[str] = []
    for quoted, bare in _FROM_OR_JOIN_RE.findall(query):
        name = quoted or bare
        if name and name not in seen:
            seen.append(name)
    return seen


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


def _normalize_to_pandas(result: Any) -> pd.DataFrame:
    """Convert any supported tabular result to a ``pd.DataFrame``.

    Accepts ``pa.Table``, ``pd.DataFrame``, ``XarrayDataFrame``, or any
    duck-typed object exposing ``.to_pandas()`` (e.g. the DataFusion
    ``DataFrame``).
    """
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pa.Table):
        return result.to_pandas()
    if isinstance(result, XarrayDataFrame):
        return result.to_pandas()
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    raise TypeError(
        f"Unsupported result type {type(result).__name__!r}; expected "
        "pa.Table, pd.DataFrame, datafusion.DataFrame, or XarrayDataFrame"
    )


def _drop_null_dim_rows(df: pd.DataFrame, dim_cols: list[str]) -> pd.DataFrame:
    """Drop rows with null dim coords. Emits a single warning if any."""
    null_mask = df[list(dim_cols)].isna().any(axis=1)
    n_dropped = int(null_mask.sum())
    if n_dropped == 0:
        return df
    null_cols = [c for c in dim_cols if df[c].isna().any()]
    warnings.warn(
        f"Dropping {n_dropped} row(s) with null dim values in "
        f"columns {null_cols} before reshape",
        stacklevel=3,
    )
    return df.loc[~null_mask].reset_index(drop=True)


def _apply_template(ds: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    """Recover metadata that the forward SQL pivot strips.

    Adds back, where unambiguous:

    * Data-variable ``attrs`` and ``encoding`` for vars present in
      ``template`` (aggregation aliases like ``air_avg`` get nothing).
    * Dim-coordinate dtype, where SQL upcasted (datetime is the canonical
      case).
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
    #    upcast through pyarrow / pandas).
    for d in list(out.dims):
        if d in template.coords:
            tdt = template.coords[d].dtype
            if out.coords[d].dtype != tdt:
                try:
                    out = out.assign_coords({d: out.coords[d].astype(tdt)})
                except (ValueError, TypeError):
                    pass  # incompatible cast; leave as-is

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


def _raw_sql(ctx: Any, query: str) -> Any:
    """Run a SQL query through the parent ``SessionContext.sql``.

    Bypasses :meth:`XarrayContext.sql` to avoid wrapping the result, used
    internally by :class:`SQLBackendArray` and :func:`_lazy_to_xarray`
    for sub-queries.
    """
    from datafusion import SessionContext

    return SessionContext.sql(ctx, query)


def _sql_literal(value: Any) -> str:
    """Format a Python/numpy scalar as a SQL literal accepted by DataFusion."""
    if isinstance(value, np.datetime64) or isinstance(value, pd.Timestamp):
        ts = pd.Timestamp(value)
        return f"TIMESTAMP '{ts.strftime('%Y-%m-%d %H:%M:%S.%f')}'"
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (str, bytes)):
        s = value.decode() if isinstance(value, bytes) else value
        escaped = s.replace("'", "''")
        return f"'{escaped}'"
    raise TypeError(
        f"Cannot build SQL literal for value of type {type(value).__name__}"
    )


class SQLBackendArray(xr.backends.BackendArray):
    """Lazy N-D array backed by re-executing a wrapped SQL query.

    Translates xarray indexers (BasicIndexer, OuterIndexer; Vectorized
    falls back to materialize) into SQL ``WHERE`` clauses pushed down
    into a sub-query of the original SQL. Materializes only the
    requested slice on each ``__getitem__`` call.

    Constructed by :func:`_lazy_to_xarray`; users should not instantiate
    this class directly.
    """

    def __init__(
        self,
        ctx: Any,
        base_query: str,
        var_name: str,
        dim_cols: list[str],
        coord_arrays: dict[str, np.ndarray],
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        self._ctx = ctx
        self._base_query = base_query
        self._var_name = var_name
        self._dim_cols = list(dim_cols)
        self._coord_arrays = coord_arrays
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._full_cache: np.ndarray | None = None

    def __getitem__(self, key: Any) -> np.ndarray:
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.OUTER,
            self._raw_getitem,
        )

    def __copy__(self) -> "SQLBackendArray":
        # The backend is read-only and re-executes the same SQL on every
        # access, so returning self is semantically equivalent to a copy.
        return self

    def __deepcopy__(self, memo: dict) -> "SQLBackendArray":
        # SessionContext is not picklable; sharing the same backend across
        # the copy is safe because every access re-runs SQL from scratch.
        return self

    # ------------------------------------------------------------------

    def _raw_getitem(self, key: tuple) -> np.ndarray:
        """Materialize the slab described by *key* via a wrapped SQL query.

        ``key`` is a tuple of ``int``/``slice``/1-D integer-array, one per
        dim, in :attr:`_dim_cols` order.
        """
        requested: dict[str, list[Any]] = {}
        # Dims whose indexer covers the full extent (slice(None) or
        # equivalent). For these we omit the WHERE clause entirely to avoid
        # generating huge ``dim IN (...)`` clauses that DataFusion would
        # then have to parse only to constant-fold to TRUE.
        full_dims: set[str] = set()
        drop_axes: list[int] = []
        for axis, (dim, k) in enumerate(zip(self._dim_cols, key)):
            coord = self._coord_arrays[dim]
            if isinstance(k, slice):
                if k.step not in (None, 1):
                    return self._fallback_materialize_and_index(key)
                start = 0 if k.start is None else k.start
                stop = len(coord) if k.stop is None else k.stop
                covers_all = start == 0 and stop >= len(coord)
                requested[dim] = list(coord[k])
                if covers_all:
                    full_dims.add(dim)
            elif isinstance(k, (int, np.integer)):
                requested[dim] = [coord[int(k)]]
                drop_axes.append(axis)
            else:
                arr = np.asarray(k)
                if arr.ndim != 1:
                    return self._fallback_materialize_and_index(key)
                requested[dim] = list(coord[arr])
                if (
                    len(arr) == len(coord)
                    and (arr == np.arange(len(coord))).all()
                ):
                    full_dims.add(dim)

        out_shape = tuple(len(requested[d]) for d in self._dim_cols)
        if any(n == 0 for n in out_shape):
            empty = np.empty(out_shape, dtype=self.dtype)
            return (
                np.squeeze(empty, axis=tuple(drop_axes)) if drop_axes else empty
            )

        try:
            conds = [
                self._build_cond(dim, requested[dim])
                for dim in self._dim_cols
                if dim not in full_dims
            ]
        except TypeError:
            return self._fallback_materialize_and_index(key)

        cols = ", ".join(f'"{c}"' for c in self._dim_cols + [self._var_name])
        where_clause = f" WHERE {' AND '.join(conds)}" if conds else ""
        wrapped = (
            f"SELECT {cols} FROM ({self._base_query}) AS _xql_base"
            f"{where_clause}"
        )
        try:
            raw_df = _raw_sql(self._ctx, wrapped).to_pandas()
        except Exception:
            return self._fallback_materialize_and_index(key)

        if len(raw_df) == 0:
            # WHERE matched no rows: caller asked for empty slab.
            return np.empty(out_shape, dtype=self.dtype)

        da = raw_df.set_index(list(self._dim_cols)).to_xarray()[self._var_name]
        # Reorder to match the caller's requested coord order per dim.
        try:
            da = da.sel({d: requested[d] for d in self._dim_cols})
        except (KeyError, ValueError):
            return self._fallback_materialize_and_index(key)
        arr = np.asarray(da.values)
        if drop_axes:
            arr = np.squeeze(arr, axis=tuple(drop_axes))
        return arr

    def _build_cond(self, dim: str, vals: list[Any]) -> str:
        if len(vals) == 1:
            return f'"{dim}" = {_sql_literal(vals[0])}'
        in_list = ", ".join(_sql_literal(v) for v in vals)
        return f'"{dim}" IN ({in_list})'

    def _fallback_materialize_and_index(self, key: tuple) -> np.ndarray:
        """Materialize the full base query once, cache it, then numpy-index."""
        if self._full_cache is None:
            all_df = _raw_sql(self._ctx, self._base_query).to_pandas()
            sorted_df = all_df.sort_values(self._dim_cols).reset_index(
                drop=True
            )
            full = sorted_df.set_index(self._dim_cols).to_xarray()[
                self._var_name
            ]
            full = full.sel({d: self._coord_arrays[d] for d in self._dim_cols})
            self._full_cache = np.asarray(full.values)
        return self._full_cache[key]


def _is_pushdownable(
    template: xr.Dataset | None,
    dim_cols: list[str],
    result_cols: list[str],
) -> bool:
    """Decide whether a query can be lazily pushed down.

    True when there is a registered template, every requested ``dim_cols``
    appears in the result, and the result has no columns beyond the
    template's dim coords and data vars (i.e. no aggregation aliases).
    """
    if template is None:
        return False
    allowed = set(template.dims) | set(template.data_vars)
    rcols = set(result_cols)
    if not rcols <= allowed:
        return False
    if not set(dim_cols) <= rcols:
        return False
    return True


def _lazy_to_xarray(
    ctx: Any,
    base_query: str,
    dim_cols: list[str],
    template: xr.Dataset | None,
    sparse_extent: SparseExtent,
    fill_value: Any,
) -> xr.Dataset:
    """Build a lazy ``xr.Dataset`` whose data vars are SQLBackendArrays."""
    if sparse_extent not in ("result", "template"):
        raise ValueError(
            "sparse_extent must be 'result' or 'template', got "
            f"{sparse_extent!r}"
        )
    if sparse_extent == "template" and template is None:
        raise ValueError(
            "sparse_extent='template' requires template= to be supplied"
        )

    schema = _raw_sql(ctx, base_query).schema()
    field_names = [f.name for f in schema]
    field_types = {f.name: f.type for f in schema}

    # Coord-array source. For a bare ``SELECT * FROM <table>`` query where
    # the table is registered, the template's full coord arrays are correct
    # and avoid a per-dim DISTINCT scan that would otherwise read the whole
    # dataset 3-4 times at construction. For anything else (WHERE / JOIN /
    # CTE) the template may mismatch the actual filtered result, so fall
    # back to DISTINCT-per-dim (correct but more expensive).
    is_bare_select_star = (
        _UNFILTERED_SELECT_STAR_RE.match(base_query) is not None
    )
    can_skip_distinct = (
        is_bare_select_star
        and template is not None
        and all(d in template.coords for d in dim_cols)
    )
    coord_arrays: dict[str, np.ndarray] = {}
    for d in dim_cols:
        if can_skip_distinct:
            assert template is not None
            coord_arrays[d] = np.asarray(template.coords[d].values)
        else:
            distinct_q = (
                f'SELECT DISTINCT "{d}" FROM ({base_query}) AS _xql_base '
                f'ORDER BY "{d}"'
            )
            df = _raw_sql(ctx, distinct_q).to_pandas()
            coord_arrays[d] = np.asarray(df[d].values)
    shape = tuple(len(coord_arrays[d]) for d in dim_cols)

    data_vars: dict[str, xr.Variable] = {}
    for name in field_names:
        if name in dim_cols:
            continue
        np_dtype = field_types[name].to_pandas_dtype()
        backend = SQLBackendArray(
            ctx=ctx,
            base_query=base_query,
            var_name=name,
            dim_cols=dim_cols,
            coord_arrays=coord_arrays,
            shape=shape,
            dtype=np_dtype,
        )
        lazy = xr.core.indexing.LazilyIndexedArray(backend)
        data_vars[name] = xr.Variable(dim_cols, lazy)

    coords_arg = {d: coord_arrays[d] for d in dim_cols}
    ds = xr.Dataset(data_vars=data_vars, coords=coords_arg)

    if sparse_extent == "template":
        assert template is not None
        indexers = {
            d: template.coords[d].values
            for d in dim_cols
            if d in template.coords and d in template.dims
        }
        if indexers:
            ds = ds.reindex(indexers, fill_value=fill_value)

    if template is not None:
        ds = _apply_template(ds, template)
    return ds


def _eager_to_xarray(
    result: Any,
    dim_cols: list[str],
    template: xr.Dataset | None = None,
    sparse_extent: SparseExtent = "result",
    fill_value: Any = np.nan,
) -> xr.Dataset:
    """Convert a tabular result to an ``xr.Dataset`` eagerly.

    Internal helper. The wrapper :meth:`XarrayDataFrame.to_dataset`
    dispatches here for queries that cannot be lazily pushed down
    (aggregations) and -- in Phase 1 -- for all queries. Phase 2 will
    introduce a lazy alternative for ``SELECT *``-style queries.
    """
    if not dim_cols:
        raise ValueError("dim_cols must be non-empty")
    if sparse_extent not in ("result", "template"):
        raise ValueError(
            "sparse_extent must be 'result' or 'template', got "
            f"{sparse_extent!r}"
        )
    if sparse_extent == "template" and template is None:
        raise ValueError(
            "sparse_extent='template' requires template= to be supplied"
        )
    df = _normalize_to_pandas(result)
    missing = [c for c in dim_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"dim_cols not found in result columns: {missing}; "
            f"available columns: {list(df.columns)}"
        )
    df = _drop_null_dim_rows(df, dim_cols)
    dup_mask = df.duplicated(subset=dim_cols)
    if dup_mask.any():
        first = df.loc[dup_mask.idxmax(), dim_cols].to_dict()
        raise ValueError(
            f"Result has duplicate dim tuples (e.g. {first}); cannot "
            "uniquely reshape into an xr.Dataset. Aggregate or de-dup "
            "the result first."
        )
    df = df.sort_values(list(dim_cols)).reset_index(drop=True)
    ds = df.set_index(list(dim_cols)).to_xarray()

    if sparse_extent == "template":
        assert template is not None  # validated above
        indexers = {
            d: template.coords[d].values
            for d in dim_cols
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
    :meth:`to_dataset` can default ``dim_cols`` and recover metadata
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
        dim_cols: list[str] | None = None,
        template: xr.Dataset | None = None,
        template_table: str | None = None,
        sparse_extent: SparseExtent = "result",
        fill_value: Any = np.nan,
    ) -> xr.Dataset:
        """Convert the result to an ``xr.Dataset``.

        Args:
            dim_cols: Result columns to use as Dataset dimensions. When
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
            An ``xr.Dataset`` with ``dim_cols`` as dimensions and the
            remaining result columns as data variables.

        Raises:
            ValueError: ``dim_cols`` cannot be inferred, names a missing
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
        if dim_cols is None:
            dim_cols = self._infer_dim_cols(preferred_template=template)
        if _is_pushdownable(template, dim_cols, self._result_columns()):
            ctx = self._registry.ctx
            assert ctx is not None  # set by XarrayContext.sql
            return _lazy_to_xarray(
                ctx=ctx,
                base_query=self._registry.query,
                dim_cols=dim_cols,
                template=template,
                sparse_extent=sparse_extent,
                fill_value=fill_value,
            )
        return _eager_to_xarray(
            self,
            dim_cols=dim_cols,
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
        """Pick a template Dataset for metadata recovery, or return None.

        Priority:
          1. Explicit ``template_table`` argument.
          2. The registered table whose name uniquely appears in the
             FROM/JOIN clauses of the SQL query.
          3. None.
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
        referenced = _extract_from_tables(self._registry.query)
        matched = [n for n in referenced if n in templates]
        if len(matched) == 1:
            return templates[matched[0]]
        return None

    def _infer_dim_cols(
        self, preferred_template: xr.Dataset | None = None
    ) -> list[str]:
        """Pick a default ``dim_cols`` from the registry, or raise.

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
                "dim_cols cannot be inferred (no registered Dataset on "
                "this result); pass dim_cols=[...] explicitly."
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
                "dim_cols cannot be inferred: no registered Dataset has "
                "all of its dims present in the result columns. Pass "
                "dim_cols=[...] explicitly."
            )
        raise ValueError(
            "dim_cols cannot be inferred unambiguously: multiple "
            "registered Datasets are compatible with the result. Pass "
            "dim_cols=[...] explicitly."
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
