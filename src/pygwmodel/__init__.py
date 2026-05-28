"""pygwmodel — Python bindings for geographically weighted (GW) models.

Provides high-level GeoPandas-based interfaces to the C++ libgwmodel
library.  All algorithms support OpenMP multi-threading and some
support CUDA GPU acceleration.

Available models
----------------
- :class:`GWRBasic` — Basic Geographically Weighted Regression.
- :class:`GWRMultiscale` — Multiscale GWR (per-variable bandwidths).
- :class:`GTWR` — Geographically and Temporally Weighted Regression.
- :class:`GWAverage` — GW local summary statistics.
- :class:`GWCorrelation` — GW local correlation coefficients.

Spatial weighting
-----------------
- :class:`BandwidthWeight` — Bandwidth and kernel configuration.
- :class:`CRSDistance` — Euclidean / great-circle distance.
- :class:`CRSSTDistance` — Spatio-temporal distance (for GTWR).
"""

import os
import sys
from pathlib import Path

# delvewheel: patch

_dll_directory_handles = []


def _add_windows_dll_directories():
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    package_dir = Path(__file__).resolve().parent
    wheel_libs_dir = package_dir.parent / "pygwmodel.libs"
    if wheel_libs_dir.is_dir():
        _dll_directory_handles.append(os.add_dll_directory(str(wheel_libs_dir)))
        return

    candidates = [package_dir]

    for env_name in ("PYGWMODEL_DLL_DIRS", "PATH"):
        for item in os.environ.get(env_name, "").split(os.pathsep):
            if item:
                candidates.append(Path(item))

    for env_name in ("VCPKG_ROOT", "VCPKG_INSTALLATION_ROOT"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "installed" / "x64-windows" / "bin")

    candidates.append(Path("C:/vcpkg/installed/x64-windows/bin"))

    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        vs_root = Path(program_files_x86) / "Microsoft Visual Studio"
        candidates.extend(
            directory
            for directory in vs_root.glob(
                "*/**/VC/Redist/MSVC/*/*/x64/Microsoft.VC*.OpenMP.LLVM"
            )
        )

    for directory in candidates:
        if directory.is_dir():
            try:
                _dll_directory_handles.append(os.add_dll_directory(str(directory)))
            except OSError:
                pass


_add_windows_dll_directories()

from .gwr_basic import GWRBasic, ParallelType
from .gwr_multiscale import GWRMultiscale
from .gtwr import GTWR
from .spatial_weight import SpatialWeight, BandwidthWeight, CRSDistance, CRSSTDistance
from .gwss import GWAverage, GWCorrelation

if __name__ == "__main__":
    print("PyGWmodel Package.")
