"""Simple SQL REPL for xarray-sql.

Run with: python -m xarray_sql.repl

Starts with a demo "air" table (xarray tutorial dataset). Type SQL and see
results. Commands: .quit or .exit to exit.
"""

import sys

# Enable up/down arrow history for input() (Unix/Mac built-in; Windows: pip install pyreadline3)
try:
  import readline  # noqa: F401
except ImportError:
  pass

import xarray as xr

from .sql import XarrayContext

MAX_DISPLAY_ROWS = 100


def main():
  ctx = XarrayContext()
  # Demo table: streaming path (no read_all); requires _native to be built
  print("Loading demo table 'air' (xarray tutorial air_temperature)...")
  air = xr.tutorial.open_dataset("air_temperature").chunk({"time": 240})
  ctx.from_dataset("air", air)
  print("Ready. Type SQL or .quit / .exit to exit.\n")

  while True:
    try:
      line = input("xarray-sql> ").strip()
    except EOFError:
      print()
      break

    if not line:
      continue
    if line in (".quit", ".exit"):
      break

    try:
      result = ctx.sql(line).to_pandas()
      display = result.head(MAX_DISPLAY_ROWS)
      print(display.to_string())
      if len(result) > MAX_DISPLAY_ROWS:
        print(f"... ({len(result) - MAX_DISPLAY_ROWS} more rows)")
    except Exception as e:
      print(f"Error: {e}", file=sys.stderr)
    print()

  sys.exit(0)


if __name__ == "__main__":
  main()
