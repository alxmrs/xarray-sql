#!/usr/bin/env python3
"""Demo of calculating global average sea surface temperature (SST) with SQL.

Please run the following to set up cloud resources:
```
gcloud auth application-default login
coiled setup
```
"""
import argparse

import numpy as np
import xarray as xr
import xarray_sql as qr


def rand_wx(start_time: str, end_time: str) -> xr.Dataset:
  """Produce a random ARCO-ERA5-like weather dataset."""
  np.random.seed(42)

  lat = np.linspace(-90, 90, num=720)
  lon = np.linspace(-180, 180, num=1440)
  time = xr.date_range(start_time, end_time, freq='H')
  level = np.array([1000, 500], dtype=np.int32)

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
      ),
      attrs=dict(description='Random weather.'),
  )


def tfmt(time: np.datetime64, unit='h') -> str:
  """Returns a bucket-friendly date string from a numpy datetime."""
  return np.datetime_as_string(time, unit=unit).replace(':', '')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--start', type=str, default='1940-01-01', help='start time ISO string'
)
parser.add_argument(
    '--end', type=str, default='1940-01-02', help='end time ISO string'
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
      spot_policy='spot_with_fallback',
      worker_mv_types=['t2a-standard-16'],  # 4 GiBs RAM per CPU, ARM.
  )

  client = cluster.get_client()
  cluster.adapt(minimum=1, maximum=100)
elif args.memory_opt_cluster:
  from coiled import Cluster

  cluster = Cluster(
      region='us-central1',
      spot_policy='spot_with_fallback',
      worker_vm_types=['m3-ultramem-32'],  # 30.5 GiBs RAM per CPU, x86.
  )

  client = cluster.get_client()
  cluster.adapt(minimum=1, maximum=25)
else:
  from dask.distributed import LocalCluster

  cluster = LocalCluster(processes=False)
  client = cluster.get_client()

if args.fake:
  era5_ds = rand_wx(args.start, args.end).chunk({'time': 240, 'level': 1})
else:
  era5_ds = xr.open_zarr(
      'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/',
      chunks={'time': 240, 'level': 1},
  )

  assert np.datetime64(args.start) >= np.datetime64(
      '1940-01-01'
  ), 'ARCO-ERA5 does not go back before 1940-01-01!'

  assert (
      np.datetime64(args.end) <= era5_ds.time[-1].values
  ), f'ARCO-ERA5 does not run until {args.end}!'

print('dataset opened.')

era5_sst_ds = era5_ds[['sea_surface_temperature']].sel(
    time=slice(args.start, args.end),
    level=1000,  # surface level only.
)

print(f'sst_size={era5_sst_ds.nbytes / 2**40}TiBs')

c = qr.Context()
# `time=48` produces 190 MiB chunks
# `time=96` produces 380 MiB chunks
# `time=192` produces 760 MiB chunks
# `time=240` produces 950 MiB chunks
# `time=720` produces 2851 MiB chunks --> utilizes 30 GiBs memory per CPU.
time_chunks = 96  # four day chunks.
if args.memory_opt_cluster:
  time_chunks = 720  # one month chunks.
c.create_table('era5', era5_sst_ds, chunks=dict(time=time_chunks))

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

# Store the results for visualization later on.
start, end = tfmt(era5_sst_ds.time[0].values), tfmt(era5_sst_ds.time[-1].values)
now = tfmt(np.datetime64('now'), 's')
results_name = f'global_avg_sst_{start}_to_{end}.{now}'
if args.cluster or args.memory_opt_cluster:
  df.to_parquet(f'gs://xarray-sql-experiments/{results_name}/')
else:
  df.to_csv(results_name + '_*.csv')
