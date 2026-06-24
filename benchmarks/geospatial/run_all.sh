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
# These cases use xarray-sql features (e.g. XarrayDataFrame.to_dataset) that may
# be newer than the latest PyPI release, so we hand `uv` the *local* checkout
# with --with-editable rather than letting it resolve xarray-sql from PyPI. Once
# a release ships those features, plain `uv run <script>` will work too.
#
# Network/credential-gated cases (ERA5, WeatherBench2, Earth Engine) skip
# cleanly when their data is unavailable. Exits non-zero if any case fails
# (a skip is not a failure).

set -uo pipefail

# Directory this script lives in, and the repo root, regardless of the caller's
# working directory.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

status=0
for script in "$DIR"/[0-9][0-9]_*.py; do
  name="$(basename "$script")"
  echo "════════════════════════════════════════ ${name}"
  if uv run --with-editable "$REPO_ROOT" "$script"; then
    echo "✅ ${name}"
  else
    echo "❌ ${name} (exit $?)"
    status=1
  fi
done

exit "$status"
