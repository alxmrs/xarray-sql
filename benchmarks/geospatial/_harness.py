"""Shared harness for the geospatial SQL benchmarks.

The suite is *expressibility-first*: each case states a geospatial operation we
normally reach for an array library to perform, expresses it in SQL against
``xarray-sql``, and proves the SQL answer matches an xarray/array reference
implementation. Wall-clock and peak memory are reported too, but the headline
is correctness + clarity of the SQL.

These helpers keep each case script short and uniform:

* :func:`banner` / :func:`show_sql` — readable section headers and SQL echo.
* :func:`timed` — a context manager that reports elapsed time and peak memory.
* :func:`check_close` — assert a SQL result matches an array reference, then
  print a PASS line. Raises ``AssertionError`` on mismatch (so a broken case
  fails loudly rather than silently "passing").
* :func:`run_case` — run a case's ``main()``, turning a raised
  :class:`CaseSkipped` (e.g. an offline dataset) into a clean skip.
"""

from __future__ import annotations

import contextlib
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

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


def check_close(
    name: str,
    got: Any,
    expected: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    equal_nan: bool = True,
) -> None:
    """Assert ``got`` matches the array reference ``expected``, then print PASS.

    Both arguments are coerced to ``float64`` numpy arrays and flattened, so a
    SQL result column (pandas Series / pyarrow) can be compared directly to an
    xarray ``DataArray`` reference. Order-sensitive: sort both sides on a shared
    key before calling if the SQL result order is not guaranteed.
    """
    g = np.asarray(getattr(got, "values", got), dtype=np.float64).ravel()
    e = np.asarray(
        getattr(expected, "values", expected), dtype=np.float64
    ).ravel()
    if g.shape != e.shape:
        raise AssertionError(
            f"{name}: shape mismatch — SQL {g.shape} vs reference {e.shape}"
        )
    np.testing.assert_allclose(g, e, rtol=rtol, atol=atol, equal_nan=equal_nan)
    print(
        f"  ✅ {name}: SQL matches array reference "
        f"(n={g.size}, rtol={rtol:g}, atol={atol:g})"
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
