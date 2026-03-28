"""DataFrame interchange protocol support for xarray SQL integration.

This package exposes helpers to register a Dataset accessor providing
`__dataframe__` so we can interoperate with other dataframe libraries.

The actual protocol implementation lives in `core.py`, while
`accessor.py` exposes user-facing registration utilities.

The DataFrame Interchange Protocol specification is defined at:
https://data-apis.org/dataframe-protocol/latest/API.html

Protocol version: 0
"""

from .accessor import register_dataset_dataframe_accessor
from .core import __dataframe_version__

__all__ = ["register_dataset_dataframe_accessor", "__dataframe_version__"]
