# Performance testing & profiling

So far, this includes statistical profiles via py-spy. 

## Dev Process

1. Run a profile test with the `profile.sh` script as so:

   ```shell
   # PROFILE_CASE_PY=groupby_air.py
   sudo ./profile.sh $PROFILE_CASE_PY
   ```

   This will open a flame graph in the browser.

2. After tuning code in xarray-sql, run another profile to generate a SVG.

3. Please commit the "after" profile SVG along with the performance improvements.

