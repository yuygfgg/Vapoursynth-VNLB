#!/usr/bin/env python3
"""Validate stored VNLB reference fixture files."""

from __future__ import annotations

import math
import sys
import tomllib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/reference_cases"


def load_npz(path: Path, key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain key {key!r}")
        return np.asarray(data[key])


def validate_case(case_dir: Path) -> None:
    with (case_dir / "case.toml").open("rb") as f:
        meta = tomllib.load(f)

    noisy = load_npz(case_dir / "input.npz", "noisy")
    clean = load_npz(case_dir / "input.npz", "clean")
    basic = load_npz(case_dir / "basic_ref.npz", "basic")
    final = load_npz(case_dir / "final_ref.npz", "final")

    expected_shape = (meta["frames"], meta["height"], meta["width"])
    arrays = {
        "noisy": noisy,
        "clean": clean,
        "basic": basic,
        "final": final,
    }
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise AssertionError(f"{case_dir.name}: {name} shape {array.shape} != {expected_shape}")
        if array.dtype != np.float32:
            raise AssertionError(f"{case_dir.name}: {name} dtype {array.dtype} != float32")
        if not np.isfinite(array).all():
            raise AssertionError(f"{case_dir.name}: {name} contains NaN or Inf")
        min_value = float(array.min())
        max_value = float(array.max())
        if name in {"noisy", "clean"}:
            if min_value < -1e-6 or max_value > 1.0 + 1e-6:
                raise AssertionError(
                    f"{case_dir.name}: {name} range [{min_value}, {max_value}] is outside [0, 1]"
                )
        elif min_value < -0.25 or max_value > 1.25:
            raise AssertionError(
                f"{case_dir.name}: {name} range [{min_value}, {max_value}] is implausible"
            )

    sigma = float(meta["noise"]["sigma_8bit"])
    if not math.isfinite(sigma) or sigma <= 0:
        raise AssertionError(f"{case_dir.name}: invalid sigma {sigma}")


def main() -> int:
    case_dirs = sorted(path for path in CASES.iterdir() if path.is_dir())
    if not case_dirs:
        raise AssertionError(f"no reference cases found under {CASES}")

    for case_dir in case_dirs:
        validate_case(case_dir)
        print(f"ok {case_dir.name}")

    print(f"validated {len(case_dirs)} reference fixture cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
