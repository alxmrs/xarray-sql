#!/usr/bin/env python3
"""Aggregate the cold-run perf CSV into a per-step summary and a markdown table.

``run_perf.sh`` runs each case once per fresh process and appends one row per
measured step (the SQL operation, the xarray reference) to a raw CSV — so every
row is an *independent cold measurement*. This reads those rows and reports, per
(case, step), the median and spread across the cold runs, writes a summary CSV,
and prints a markdown table.

    perf_summary.py RAW.csv [SUMMARY.csv]
"""

from __future__ import annotations

import csv
import statistics
import sys

_HEADER = [
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


def main() -> None:
    raw_path = sys.argv[1]
    summary_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(raw_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Each raw row is one cold run (reps=1), so its t_median_s == the sample.
    groups: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for r in rows:
        key = (r["case"], r["title"], r["step"])
        groups.setdefault(key, []).append(
            (float(r["t_median_s"]), float(r["peak_mb"]))
        )

    summary = []
    for (case, title, step), vals in groups.items():
        times = [t for t, _ in vals]
        summary.append(
            {
                "case": case,
                "title": title,
                "step": step,
                "reps": len(times),
                "t_min_s": round(min(times), 6),
                "t_median_s": round(statistics.median(times), 6),
                "t_mean_s": round(statistics.fmean(times), 6),
                "t_stdev_s": round(statistics.stdev(times), 6)
                if len(times) > 1
                else 0.0,
                "t_max_s": round(max(times), 6),
                "peak_mb": round(max(p for _, p in vals), 1),
            }
        )

    summary.sort(
        key=lambda r: (
            str(r["case"]),
            0 if str(r["step"]).upper().startswith("SQL") else 1,
        )
    )

    if summary_path:
        with open(summary_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_HEADER)
            writer.writeheader()
            writer.writerows(summary)

    print(
        "| Case | Step | reps | median (s) | stdev (s) | min (s) | max (s) | peak (MB) |"
    )
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    seen: set[str] = set()
    for r in summary:
        case = str(r["case"])
        cell = str(r["title"]) if case not in seen else ""
        seen.add(case)
        step = (
            "SQL"
            if str(r["step"]).upper().startswith("SQL")
            else "xarray reference"
        )
        print(
            f"| {cell} | {step} | {r['reps']} | {r['t_median_s']:.3f} | "
            f"{r['t_stdev_s']:.3f} | {r['t_min_s']:.3f} | {r['t_max_s']:.3f} | "
            f"{r['peak_mb']:.1f} |"
        )


if __name__ == "__main__":
    main()
