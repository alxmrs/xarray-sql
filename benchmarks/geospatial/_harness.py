"""Shared harness for the geospatial SQL benchmarks.

The suite is *expressibility-first*: each case states a geospatial operation we
normally reach for an array library to perform, expresses it in SQL against
``xarray-sql``, and proves the SQL answer matches an xarray/array reference
implementation. Wall-clock and peak memory are reported too, but the headline
is correctness + clarity of the SQL.

These helpers keep each case script short and uniform:

* :func:`banner` / :func:`show_sql` — readable section headers and SQL echo.
* :func:`timed` — a context manager that reports elapsed time and peak memory.
* :func:`assert_grid_close` — assert a SQL result (round-tripped to an
  ``xr.DataArray``) matches an xarray reference, aligned by coordinate label.
  Raises ``AssertionError`` on mismatch (so a broken case fails loudly rather
  than silently "passing").
* :func:`run_case` — run a case's ``main()``, turning a raised
  :class:`CaseSkipped` (e.g. an offline dataset) into a clean skip.
"""

from __future__ import annotations

import contextlib
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator

import xarray as xr

_WIDTH = 72


class CaseSkipped(Exception):
    """Raised by a case when it cannot run in this environment (e.g. offline)."""


def banner(text: str) -> None:
    """Print a titled section divider."""
    print(f"\n{'─' * _WIDTH}")
    print(f"  {text}")
    print(f"{'─' * _WIDTH}")


def show_sql(sql: str, *, label: str = "SQL") -> None:
    """Echo a SQL statement so the reader sees exactly what ran."""
    print(f"\n  {label}:")
    for line in sql.strip("\n").splitlines():
        print(f"    │ {line}")
    print()


@contextlib.contextmanager
def timed(label: str) -> Iterator[None]:
    """Time a block and report elapsed wall-clock and peak memory.

    Peak memory is the Python-allocator peak during the block (via
    ``tracemalloc``); it captures the materialized result and intermediate
    buffers, which is what we care about for "did this blow up memory".
    """
    tracemalloc.start()
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  ⏱  {label}: {elapsed:.3f}s  (peak {peak / 1e6:.1f} MB)")


def assert_grid_close(
    name: str,
    got: xr.DataArray,
    ref: xr.DataArray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Assert two gridded ``DataArray`` results match, then print PASS.

    For cases whose SQL result is round-tripped back to an ``xr.DataArray``
    (via ``XarrayDataFrame.to_dataset``), compare it to the array reference the
    xarray way: align ``ref`` onto ``got``'s coordinates and dimension order,
    then ``xr.testing.assert_allclose``. This aligns by *label*, so neither side
    needs an explicit sort, and NaNs in matching cells compare equal.

    Helper coordinates xarray attaches along the way (e.g. the ``hour`` label a
    ``groupby("time.hour")`` leaves behind) are dropped before comparing.
    """
    aligned = ref.reindex_like(got).transpose(*got.dims)
    extra = [c for c in aligned.coords if c not in got.coords]
    aligned = aligned.drop_vars(extra)
    xr.testing.assert_allclose(got, aligned, rtol=rtol, atol=atol)
    print(
        f"  ✅ {name}: SQL matches xarray reference "
        f"(n={got.size:,}, coordinate-aligned)"
    )


def run_case(main: Callable[[], None], title: str) -> int:
    """Run a case ``main()``; turn :class:`CaseSkipped` into a clean skip.

    Returns a process exit code: 0 on success or skip, 1 on failure. Use as
    ``if __name__ == '__main__': raise SystemExit(run_case(main, '...'))``.
    """
    banner(title)
    try:
        main()
    except CaseSkipped as exc:
        print(f"\n  ⏭  SKIPPED: {exc}")
        return 0
    except Exception as exc:  # noqa: BLE001 — surface any failure as exit 1
        print(f"\n  ❌ FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print(f"\n  🎉 {title}: done.")
    return 0
