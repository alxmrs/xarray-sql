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


def rand_wx(start: str, end: str) -> xr.Dataset:
  import numpy as np
  import pandas as pd

  np.random.seed(42)

  lat = np.linspace(-90, 90, num=720)
  lon = np.linspace(-180, 180, num=1440)
  time = pd.date_range(start, end, freq='H')
  level = np.array([1000, 500], dtype=np.int32)
  reference_time = pd.Timestamp(start)

  temperature = 15 + 8 * np.random.randn(720, 1440, len(time), len(level))
  precipitation = 10 * np.random.rand(720, 1440, len(time), len(level))

  return xr.Dataset(
      data_vars=dict(
          sea_surface_temperature=(
              ['lat', 'lon', 'time', 'level'],
              temperature,
          ),
          precipitation=(['lat', 'lon', 'time', 'level'], precipitation),
      ),
      coords=dict(
          lat=lat,
          lon=lon,
          time=time,
          level=level,
          reference_time=reference_time,
      ),
      attrs=dict(description='Random weather.'),
  )


parser = argparse.ArgumentParser()
parser.add_argument(
    '--start', type=str, default='2020-01-01', help='start time ISO string'
)
parser.add_argument(
    '--end', type=str, default='2020-01-02', help='end time ISO string'
)
parser.add_argument(
    '--cluster',
    action='store_true',
    help='deploy on coiled cluster, default: local cluster',
)
parser.add_argument(
    '--memory-opt-cluster',
    action='store_true',
    help='deploy on memory-optimized coiled cluster, default: local cluster',
)
parser.add_argument(
    '--fake',
    action='store_true',
    help='use local dummy data, default: ARCO-ERA5 data',
)

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
elif args.mem_opt_cluster:
    from coiled import Cluster

    cluster = Cluster(
        region='us-central1',
        spot_policy='spot_with_fallback',
        worker_vm_types=['m3-ultramem-32'],
        arm=True,
    )

    client = cluster.get_client()
    cluster.adapt(minimum=1, maximum=50)
else:
  from dask.distributed import LocalCluster

  cluster = LocalCluster(processes=False)
  client = cluster.get_client()

if args.fake:
  era5_ds = rand_wx(args.start, args.end).chunk({'time': 240, 'level': 1})
else:
  era5_ds = xr.open_zarr(
      'gs://gcp-public-data-arco-era5/ar/'
      '1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
      chunks={'time': 240, 'level': 1},
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
# TODO(alxmrs): `DATE` function is not supported in Apache Calcite out-of-the-box.
df = c.sql(
    """
SELECT
  "time",
  AVG("sea_surface_temperature") as daily_avg_sst
FROM
  "era5"
GROUP BY
  "time"
"""
)

df.to_csv(f'global_avg_sst_{args.start}_to_{args.end}_*.cvs')
