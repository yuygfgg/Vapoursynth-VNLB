from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import shutil
import zipfile
from pathlib import Path


SYSTEM_DLL_EXCLUDE = {
    "api-ms-*.dll",
    "concrt*.dll",
    "ext-ms-*.dll",
    "mfc*.dll",
    "msvcp*.dll",
    "msvcr*.dll",
    "ucrtbase.dll",
    "ucrtbased.dll",
    "vccorlib*.dll",
    "vcomp*.dll",
    "vcruntime*.dll",
}


def _record_hash(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return f"sha256={digest.rstrip(b'=').decode()}", str(len(data))


def _rewrite_record(extract_dir: Path) -> None:
    dist_info_dirs = list(extract_dir.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise RuntimeError(
            f"expected exactly one dist-info directory, found {len(dist_info_dirs)}"
        )

    record = dist_info_dirs[0] / "RECORD"
    with record.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for path in sorted(extract_dir.rglob("*")):
            if path.is_dir():
                continue

            relative = path.relative_to(extract_dir).as_posix()
            if path == record:
                writer.writerow((relative, "", ""))
            else:
                writer.writerow((relative, *_record_hash(path)))


def _repack_wheel(extract_dir: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(extract_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(extract_dir).as_posix())


def repair_windows_wheel(
    wheel: Path,
    output_dir: Path,
    search_paths: list[str],
    plugin_dll_path: Path,
) -> Path:
    from delvewheel import _dll_utils

    extract_dir = wheel.parent.parent / "build" / "windows-wheel-repair" / wheel.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extract_dir)

    plugin_dll = extract_dir / plugin_dll_path
    if not plugin_dll.is_file():
        raise FileNotFoundError(f"missing plugin DLL: {plugin_dll}")

    os.environ["PATH"] = os.pathsep.join([*search_paths, os.environ.get("PATH", "")])
    dependency_paths, _, _, _ = _dll_utils.get_all_needed(
        str(plugin_dll),
        SYSTEM_DLL_EXCLUDE,
        None,
        "raise",
        False,
        False,
    )

    plugin_dir = plugin_dll.parent
    for dependency in sorted(dependency_paths, key=lambda item: Path(item).name.lower()):
        source = Path(dependency)
        target = plugin_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    dll_names = {path.name.lower() for path in plugin_dir.glob("*.dll")}
    if "libopenblas.dll" not in dll_names:
        raise RuntimeError("Windows wheel did not vendor libopenblas.dll")

    _rewrite_record(extract_dir)
    repaired_wheel = output_dir / wheel.name
    _repack_wheel(extract_dir, repaired_wheel)

    print("Vendored Windows DLLs:")
    for dll in sorted(path.name for path in plugin_dir.glob("*.dll")):
        print(f"  {dll}")

    return repaired_wheel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("-w", "--wheel-dir", type=Path, default=Path("wheelhouse"))
    parser.add_argument(
        "--plugin-dll",
        type=Path,
        default=Path("vapoursynth/plugins/vnlb/libvnlb.dll"),
    )
    parser.add_argument(
        "--add-path",
        action="append",
        default=[],
        help="additional DLL search path; may be passed multiple times",
    )
    args = parser.parse_args()

    for wheel in args.wheels:
        repair_windows_wheel(wheel, args.wheel_dir, args.add_path, args.plugin_dll)


if __name__ == "__main__":
    main()
