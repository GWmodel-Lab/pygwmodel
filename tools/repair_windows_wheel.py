import os
import subprocess
import sys
from pathlib import Path


def _split_paths(value):
    return [Path(item) for item in value.split(os.pathsep) if item]


def _candidate_dll_dirs():
    candidates = []

    candidates.extend(_split_paths(os.environ.get("PYGWMODEL_DLL_DIRS", "")))
    candidates.extend(_split_paths(os.environ.get("PATH", "")))

    for env_name in ("VCPKG_ROOT", "VCPKG_INSTALLATION_ROOT"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "installed" / "x64-windows" / "bin")

    candidates.append(Path("C:/vcpkg/installed/x64-windows/bin"))

    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if program_files_x86:
        vs_root = Path(program_files_x86) / "Microsoft Visual Studio"
        candidates.extend(
            vs_root.glob("*/**/VC/Redist/MSVC/*/*/x64/Microsoft.VC*.OpenMP.LLVM")
        )

    seen = set()
    existing = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = str(resolved).casefold()
        if key not in seen and resolved.is_dir():
            seen.add(key)
            existing.append(resolved)
    return existing


def main():
    if len(sys.argv) != 3:
        print(
            "usage: repair_windows_wheel.py <wheel> <wheel-dir>",
            file=sys.stderr,
        )
        return 2

    wheel = Path(sys.argv[1]).resolve()
    wheel_dir = Path(sys.argv[2]).resolve()
    wheel_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "delvewheel",
        "repair",
        "--custom-patch",
        "--include",
        "gsl.dll;gslcblas.dll;libgcc_s_seh-1.dll;libgfortran-5.dll;libquadmath-0.dll;libwinpthread-1.dll",
        "--exclude",
        "msvcp140.dll;vcruntime140.dll;vcruntime140_1.dll",
        "--no-mangle-all",
        "--wheel-dir",
        str(wheel_dir),
    ]

    dll_dirs = _candidate_dll_dirs()
    if dll_dirs:
        command.extend(["--add-path", os.pathsep.join(str(path) for path in dll_dirs)])

    command.append(str(wheel))
    print("Repairing wheel with delvewheel")
    print("DLL search paths:")
    for path in dll_dirs:
        print(f"  {path}")
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
