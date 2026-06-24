#!/usr/bin/env bash
#
# Run every geospatial benchmark case with `uv run` (each script declares its
# own dependencies via PEP 723 inline metadata). Works from any directory: it
# resolves its own location, so the cases are found and the paths handed to
# `uv run` are absolute.
#
#   ./run_all.sh                 # from anywhere
#   bash benchmarks/geospatial/run_all.sh
#
# Each script's metadata points xarray-sql at this local checkout
# ([tool.uv.sources] path = "../../"), so uv uses the in-repo build (which has
# features newer than the latest PyPI release) — relative to the script, so it
# resolves no matter the working directory.
#
# Network/credential-gated cases (ERA5, WeatherBench2, Earth Engine) skip
# cleanly when their data is unavailable. Exits non-zero if any case fails
# (a skip is not a failure).

set -uo pipefail

# Directory this script lives in, regardless of the caller's working directory.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

status=0
for script in "$DIR"/[0-9][0-9]_*.py; do
  name="$(basename "$script")"
  echo "════════════════════════════════════════ ${name}"
  if uv run "$script"; then
    echo "✅ ${name}"
  else
    echo "❌ ${name} (exit $?)"
    status=1
  fi
done

exit "$status"
