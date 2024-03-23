#!/usr/bin/env python3
"""Demo of calculating global average sea surface temperature (SST) with SQL.

Please run the following to set up cloud resources:
```
gcloud auth application-default login
coiled setup
```
"""
import argparse
import xarray as xr
import xarray_sql as qr

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=str, default='2020-01-01', help='start time ISO string')
parser.add_argument('--end', type=str, default='2020-01-02', help='end time ISO string')
parser.add_argument('--cluster', action='store_true', help='deploy on coiled cluster')

args = parser.parse_args()

if args.cluster:
  from coiled import Cluster

  cluster = Cluster(
    region='us-central1',
    worker_memory='16 GiB',
    spot_policy='spot_with_fallback',
    arm=True,
  )
  client = cluster.get_client()
  cluster.adapt(minimum=1, maximum=100)
else:
  from dask.distributed import LocalCluster
  cluster = LocalCluster(processes=False)
  client = cluster.get_client()

era5_ds = xr.open_zarr(
  'gs://gcp-public-data-arco-era5/ar/'
  '1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
  chunks={'time': 240, 'level': 1}
)
print('dataset opened.')
era5_sst_ds = era5_ds[['sea_surface_temperature']].sel(
  time=slice(args.start, args.end),
  level=1000,  # surface level only.
)

c = qr.Context()
# chunk sizes determined from VM memory limit of 16 GiB.
c.create_table('era5', era5_sst_ds, chunks=dict(time=24))

print('beginning query.')
df = c.sql("""
SELECT
  DATE("time") as date,
  AVG("sea_surface_temperature") as daily_avg_sst
FROM
  "era5"
GROUP BY
  DATE("time")
""")

df.to_csv(f'global_avg_sst_{args.start}-{args.end}_*.cvs')
