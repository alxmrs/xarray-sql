"""DataFrame interchange protocol support for xarray SQL integration.

This package exposes helpers to register a Dataset accessor providing
`__dataframe__` so we can interoperate with other dataframe libraries.

The actual protocol implementation lives in `core.py`, while
`accessor.py` exposes user-facing registration utilities.
"""

from .accessor import register_dataset_dataframe_accessor

__all__ = ["register_dataset_dataframe_accessor"]

