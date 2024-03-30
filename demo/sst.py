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

# Instead of letting users choose arbitrary time frames, we only allow
# the following choices. This design prevents users from accidentally
# processing way more data than they might have meant to. We don't
# want to bankrupt folks because they were off a few digits.
TIMEFRAMES = {
    'day': slice('1940-01-01', '1940-01-02'),
    'month': slice('1940-01-01', '1940-02-01'),
    'year': slice('1940-01-01', '1941-01-01'),
    'all': slice('1940-01-01', '2023-11-01'),
}

CLUSTERS = ['local', 'arm', 'mem-opt']


def rand_wx(times) -> xr.Dataset:
  """Produce a random ARCO-ERA5-like weather dataset."""
  np.random.seed(42)

  lat = np.linspace(-90, 90, num=720)
  lon = np.linspace(-180, 180, num=1440)
  time = xr.date_range(times.start, times.stop, freq='H')
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


parser = argparse.ArgumentParser()
parser.add_argument('--timeframe', choices=TIMEFRAMES.keys(), default='day')
parser.add_argument(
    '--cluster',
    choices=CLUSTERS,
    default='local',
    help='Choose the Dask cluster type. '
    'Either: a local cluster, ARM VMs or memory-optimized VMs in GCP via Coiled.',
)
parser.add_argument(
    '--fake',
    action='store_true',
    help='use local dummy data, default: ARCO-ERA5 data',
)

args = parser.parse_args()
timeframe = TIMEFRAMES[args.timeframe]

if args.cluster == 'arm':
  from coiled import Cluster

  cluster = Cluster(
      region='us-central1',
      spot_policy='spot_with_fallback',
      worker_vm_types='t2a-standard-16',  # 4 GiBs RAM per CPU, ARM.
  )

  client = cluster.get_client()
  cluster.adapt(minimum=1, maximum=100)
elif args.cluster == 'mem-opt':
  from coiled import Cluster

  cluster = Cluster(
      region='us-central1',
      spot_policy='spot_with_fallback',
      worker_vm_types='m3-ultramem-32',  # 30.5 GiBs RAM per CPU, x86.
  )

  client = cluster.get_client()
  cluster.adapt(minimum=1, maximum=25)
else:
  from dask.distributed import LocalCluster

  cluster = LocalCluster(processes=False)
  client = cluster.get_client()

if args.fake:
  era5_ds = rand_wx(timeframe).chunk({'time': 240, 'level': 1})
else:
  era5_ds = xr.open_zarr(
      'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/',
      chunks={'time': 240, 'level': 1},
  )

print('dataset opened.')

era5_sst_ds = era5_ds[['sea_surface_temperature']].sel(
    time=timeframe, level=1000
)

print(f'sst_size={era5_sst_ds.nbytes / 2**40:.5f}TiBs')

c = qr.Context()
# `time=48` produces 190 MiB chunks
# `time=96` produces 380 MiB chunks
# `time=192` produces 760 MiB chunks
# `time=240` produces 950 MiB chunks
# `time=720` produces 2851 MiB chunks --> utilizes 30 GiBs memory per CPU.
time_chunks = 96  # four day chunks.
if args.cluster == 'mem-opt':
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
now = np.datetime64('now', 's').astype(int)
results_name = f'global_avg_sst_{args.timeframe}_{now}'
if args.cluster == 'local':
  df.to_csv(results_name + '_*.csv')
else:
  df.to_parquet(f'gs://xarray-sql-experiments/{results_name}/')
