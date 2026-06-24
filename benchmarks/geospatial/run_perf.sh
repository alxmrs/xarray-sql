#!/usr/bin/env bash
# Cold-vs-cold performance benchmark.
#
# Runs each case once per fresh process, with no warmup, repeated GEOBENCH_REPS
# times. A fresh process per repetition is deliberate: it makes the SQL operation
# AND the xarray reference each pay a *cold* read on every measurement. An
# in-process warm loop is unfair here — `xr.open_zarr(chunks=None)` caches each
# variable in memory after the first read, so the xarray reference would serve
# later reps from RAM while the SQL side re-reads the store. One process per rep
# defeats that (and the OS/connection reuse), so both sides are measured cold.
#
# Each run appends one row per step to a raw CSV; perf_summary.py then reports the
# median/spread across the independent cold runs.
#
#   GEOBENCH_REPS=5 benchmarks/geospatial/run_perf.sh [summary.csv]
#
# For representative numbers use a release build of xarray-sql and run close to
# the data (a VM in the bucket's region). Override the launcher with
# GEOBENCH_PYRUN (e.g. `GEOBENCH_PYRUN="python"` to use an already-built venv
# instead of the default `uv run`, which builds an unoptimized editable install).
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPS="${GEOBENCH_REPS:-5}"
SUMMARY="${1:-$DIR/perf.csv}"
RAW="$(mktemp)"
read -r -a PYRUN <<<"${GEOBENCH_PYRUN:-uv run}"

for f in "$DIR"/0[1-8]_*.py; do
  name="$(basename "$f")"
  for i in $(seq 1 "$REPS"); do
    if GEOBENCH_PROFILE=1 GEOBENCH_WARMUP=0 GEOBENCH_REPS=1 GEOBENCH_CSV="$RAW" \
        "${PYRUN[@]}" "$f" >/dev/null 2>&1; then
      echo "  $name  rep $i/$REPS  ok"
    else
      echo "  $name  rep $i/$REPS  skip/fail"
    fi
  done
done

python3 "$DIR/perf_summary.py" "$RAW" "$SUMMARY"
echo "wrote $SUMMARY"
