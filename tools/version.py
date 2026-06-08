from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\." r"(?P<minor>0|[1-9]\d*)\." r"(?P<patch>0|[1-9]\d*)$"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_project_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    version = pyproject["project"]["version"]
    if not isinstance(version, str):
        raise TypeError("[project].version must be a string")
    return version


def version_parts(version: str) -> tuple[int, int, int]:
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"unsupported project version: {version}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def plugin_version(version: str) -> str:
    major, minor, _ = version_parts(version)
    return f"{major}.{minor}"


def release_tag(version: str) -> str:
    major, minor, patch = version_parts(version)
    if patch == 0:
        return f"R{major}.{minor}"
    return f"R{major}.{minor}.{patch}"


def check_release_tag(version: str, actual_tag: str) -> int:
    expected_tag = release_tag(version)
    if actual_tag == expected_tag:
        return 0

    print(
        f"release tag mismatch: expected {expected_tag} for version {version}, "
        f"got {actual_tag}",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Report VNLB project versions.")
    parser.add_argument(
        "--release-tag",
        action="store_true",
        help="print the release tag derived from [project].version",
    )
    parser.add_argument(
        "--plugin-version",
        action="store_true",
        help="print the VapourSynth plugin major.minor version",
    )
    parser.add_argument(
        "--check-release-tag",
        metavar="TAG",
        help="verify that TAG matches the release tag for [project].version",
    )
    args = parser.parse_args()

    selected_modes = sum(
        [
            args.release_tag,
            args.plugin_version,
            args.check_release_tag is not None,
        ]
    )
    if selected_modes > 1:
        parser.error("select only one output or check mode")

    version = read_project_version(project_root())
    if args.check_release_tag is not None:
        return check_release_tag(version, args.check_release_tag)
    if args.release_tag:
        print(release_tag(version))
    elif args.plugin_version:
        print(plugin_version(version))
    else:
        print(version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
