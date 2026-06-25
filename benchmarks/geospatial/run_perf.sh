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
# Each run appends one row per step to a raw CSV; this script then aggregates the
# median/spread across the independent cold runs into a summary CSV + markdown.
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

# Aggregate the per-process cold runs (one row each) into a per-step summary and
# a markdown table. Each raw row is one cold sample (reps=1), so we collect the
# samples per (case, title, step) and report median/spread/peak across them.
python3 - "$RAW" "$SUMMARY" <<'PY'
import csv, statistics, sys

raw, summary = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(raw, newline="")))
groups: dict = {}
for r in rows:
    groups.setdefault((r["case"], r["title"], r["step"]), []).append(
        (float(r["t_median_s"]), float(r["peak_mb"]))
    )

header = ["case", "title", "step", "reps", "t_min_s", "t_median_s",
          "t_mean_s", "t_stdev_s", "t_max_s", "peak_mb"]
out = []
for (case, title, step), vals in groups.items():
    ts = [t for t, _ in vals]
    out.append({
        "case": case, "title": title, "step": step, "reps": len(ts),
        "t_min_s": round(min(ts), 6),
        "t_median_s": round(statistics.median(ts), 6),
        "t_mean_s": round(statistics.fmean(ts), 6),
        "t_stdev_s": round(statistics.stdev(ts), 6) if len(ts) > 1 else 0.0,
        "t_max_s": round(max(ts), 6),
        "peak_mb": round(max(p for _, p in vals), 1),
    })
out.sort(key=lambda r: (r["case"], 0 if r["step"].upper().startswith("SQL") else 1))

with open(summary, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=header)
    w.writeheader()
    w.writerows(out)

print("| Case | Step | reps | median (s) | stdev (s) | min (s) | max (s) | peak (MB) |")
print("|---|---|--:|--:|--:|--:|--:|--:|")
seen = set()
for r in out:
    cell = r["title"] if r["case"] not in seen else ""
    seen.add(r["case"])
    step = "SQL" if r["step"].upper().startswith("SQL") else "xarray reference"
    print(f"| {cell} | {step} | {r['reps']} | {r['t_median_s']:.3f} | "
          f"{r['t_stdev_s']:.3f} | {r['t_min_s']:.3f} | {r['t_max_s']:.3f} | "
          f"{r['peak_mb']:.1f} |")
PY
echo "wrote $SUMMARY"
