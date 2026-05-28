"""Parallel type enumeration for GW models.

``ParallelType`` is an IntEnum with members:

- ``SerialOnly`` — single-threaded execution (default).
- ``OpenMP``   — multi-threaded via OpenMP.
- ``CUDA``     — GPU-accelerated via CUDA.
"""

from ._parallel import _ParallelType

ParallelType = _ParallelType
