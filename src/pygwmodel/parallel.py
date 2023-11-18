from enum import IntEnum
from ._parallel import _ParallelType


class ParallelType(IntEnum):
    Serial = _ParallelType.SerialOnly
    OpenMP = _ParallelType.OpenMP
    CUDA = _ParallelType.CUDA