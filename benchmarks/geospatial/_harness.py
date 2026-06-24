"""Shared harness for the geospatial SQL benchmarks.

The suite is *expressibility-first*: each case states a geospatial operation we
normally reach for an array library to perform, expresses it in SQL against
``xarray-sql``, and proves the SQL answer matches an xarray/array reference
implementation. Wall-clock and peak memory are reported too, but the headline
is correctness + clarity of the SQL.

These helpers keep each case script short and uniform:

* :func:`banner` / :func:`show_sql` — readable section headers and SQL echo.
* :func:`timed` — a context manager that reports elapsed time and peak memory,
  for one-time steps (opening data, registering tables).
* :func:`measured` — a loop wrapper (``for _ in measured(label): ...``) for a
  repeatable step (a query, a computation). It runs the body once normally, or —
  under ``GEOBENCH_PROFILE`` — a warmup plus ``GEOBENCH_REPS`` timed repetitions,
  writing a statistical summary to the ``GEOBENCH_CSV`` perf table.
* :func:`assert_grid_close` — assert a SQL result (round-tripped to an
  ``xr.DataArray``) matches an xarray reference, aligned by coordinate label.
  Raises ``AssertionError`` on mismatch (so a broken case fails loudly rather
  than silently "passing").
* :func:`run_case` — run a case's ``main()``, turning a raised
  :class:`CaseSkipped` (e.g. an offline dataset) into a clean skip.
"""

from __future__ import annotations

import contextlib
import csv
import os
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from typing import Any

import xarray as xr

_WIDTH = 72

# Performance profiling, opt-in via environment variables. With GEOBENCH_PROFILE
# set, a ``for _ in measured(label):`` block runs GEOBENCH_WARMUP + GEOBENCH_REPS
# times instead of once; GEOBENCH_CSV=<path> collects one summary row per such
# block into a shared CSV — the perf table. Without the flag, runs are unchanged.
_CSV_HEADER = [
    "case",
    "title",
    "step",
    "reps",
    "t_min_s",
    "t_median_s",
    "t_mean_s",
    "t_stdev_s",
    "t_max_s",
    "peak_mb",
]
_current_case = ""
_current_title = ""

_EE_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
]


class CaseSkipped(Exception):
    """Raised by a case when it cannot run in this environment (e.g. offline)."""


def initialize_earth_engine() -> Any:
    """Initialize Earth Engine from Application Default Credentials, or skip.

    Uses the credentials from ``gcloud auth application-default login`` (with the
    Earth Engine scope) and the ADC project — so no separate ``earthengine
    authenticate`` OAuth flow is needed, which also sidesteps the "this app is
    blocked" error some org policies raise. Override the project with the
    ``EARTHENGINE_PROJECT`` environment variable. Returns the initialized ``ee``
    module; raises :class:`CaseSkipped` if EE is unavailable or unauthenticated.
    """
    try:
        import ee
        import google.auth
    except ImportError as exc:  # pragma: no cover
        raise CaseSkipped(
            "Earth Engine support needs `pip install earthengine-api`"
        ) from exc
    try:
        credentials, adc_project = google.auth.default(scopes=_EE_SCOPES)
        ee.Initialize(
            credentials,
            project=os.environ.get("EARTHENGINE_PROJECT") or adc_project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
    except Exception as exc:  # noqa: BLE001 — not authenticated → skip
        raise CaseSkipped(
            f"Earth Engine not initialized ({exc}); run "
            "`gcloud auth application-default login` (or set EARTHENGINE_PROJECT)"
        ) from exc
    return ee


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


def _append_csv(step: str, times: list[float], peak_bytes: int) -> None:
    """Append one step's summary stats to the GEOBENCH_CSV perf table, if set."""
    path = os.environ.get("GEOBENCH_CSV", "")
    if not path:
        return
    row = {
        "case": _current_case,
        "title": _current_title,
        "step": step,
        "reps": len(times),
        "t_min_s": round(min(times), 6),
        "t_median_s": round(statistics.median(times), 6),
        "t_mean_s": round(statistics.fmean(times), 6),
        "t_stdev_s": round(statistics.stdev(times), 6)
        if len(times) > 1
        else 0.0,
        "t_max_s": round(max(times), 6),
        "peak_mb": round(peak_bytes / 1e6, 1),
    }
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_HEADER)
        if fresh:
            writer.writeheader()
        writer.writerow(row)


def measured(label: str) -> Iterator[None]:
    """Time a repeatable block, optionally repeating it for a perf profile.

    Use it as a loop — ``for _ in measured("SQL …"): got = <query>``. Without
    profiling it runs the body once and prints a ``⏱`` line, exactly like
    :func:`timed`. Under ``GEOBENCH_PROFILE`` it runs a warmup pass plus
    ``GEOBENCH_REPS`` measured passes, times each, and appends one row of summary
    statistics to the ``GEOBENCH_CSV`` perf table. The body must be safe to
    repeat — a query or pure computation, not one-time setup such as table
    registration (which stays in :func:`timed`).
    """
    if not os.environ.get("GEOBENCH_PROFILE"):
        with timed(label):
            yield
        return
    reps = max(1, int(os.environ.get("GEOBENCH_REPS", "5")))
    warmup = max(0, int(os.environ.get("GEOBENCH_WARMUP", "1")))
    times: list[float] = []
    peak_max = 0
    for i in range(warmup + reps):
        tracemalloc.start()
        tracemalloc.reset_peak()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        if i >= warmup:
            times.append(elapsed)
            peak_max = max(peak_max, peak)
    _append_csv(label, times, peak_max)
    print(
        f"  📊 {label}: median {statistics.median(times):.3f}s "
        f"[min {min(times):.3f}, max {max(times):.3f}, "
        f"n={len(times)}, peak {peak_max / 1e6:.0f} MB]"
    )


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
    short = {
        d: (got.sizes[d], ref.sizes[d])
        for d in ref.dims
        if d in got.sizes and got.sizes[d] != ref.sizes[d]
    }
    if short:
        raise AssertionError(
            f"{name}: SQL result does not cover the reference grid "
            f"(dim: got vs ref = {short}); the comparison would be partial"
        )
    aligned = ref.reindex_like(got).transpose(*got.dims)
    extra = [c for c in aligned.coords if c not in got.coords]
    aligned = aligned.drop_vars(extra)
    xr.testing.assert_allclose(got, aligned, rtol=rtol, atol=atol)
    print(
        f"  ✅ {name}: SQL matches xarray reference "
        f"(n={got.size:,}, coordinate-aligned)"
    )


def show_result(
    result: xr.DataArray | xr.Dataset, *, label: str = "Result (SQL → xarray)"
) -> None:
    """Print the SQL result as an xarray object, using its standard repr.

    Called after the match is verified, so a run shows *what* it computed — the
    gridded answer round-tripped back out of SQL as an ``xarray`` object.
    """
    print(f"\n  {label}:\n")
    print(result)


def run_case(main: Callable[[], None], title: str) -> int:
    """Run a case ``main()``; turn :class:`CaseSkipped` into a clean skip.

    Returns a process exit code: 0 on success or skip, 1 on failure. Use as
    ``if __name__ == '__main__': raise SystemExit(run_case(main, '...'))``.
    """
    global _current_case, _current_title
    _current_title = title
    _current_case = os.path.splitext(os.path.basename(sys.argv[0]))[0]
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
