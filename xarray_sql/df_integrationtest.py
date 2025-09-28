"""Integration tests for xarray-sql."""

import pytest
import xarray as xr

from . import read_xarray


class TestEra5Integration:
  """Test integration with ERA5 dataset."""

  @pytest.mark.integration
  @pytest.mark.gcs
  def test_open_era5(self):
    """Test opening ERA5 dataset from Google Cloud Storage."""
    era5_ds = xr.open_zarr(
        'gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
        chunks={'time': 240, 'level': 1},
    )
    era5_wind_df = read_xarray(
        era5_ds[['u_component_of_wind', 'v_component_of_wind']]
    )

    expected_columns = [
        'time',
        'level',
        'latitude',
        'longitude',
        'u_component_of_wind',
        'v_component_of_wind',
    ]
    assert list(era5_wind_df.columns) == expected_columns
