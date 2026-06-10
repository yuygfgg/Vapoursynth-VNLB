#!/usr/bin/env python3
"""Guard VapourSynth plugin output drift against pinned VNLB reference fixtures."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import numpy as np
import vapoursynth as vs


def load_npz(path: Path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data:
            raise KeyError(f"{path} does not contain {key!r}")
        return np.asarray(data[key], dtype=np.float32)


def ndarray_to_gray_clip(core: vs.Core, frames: np.ndarray) -> vs.VideoNode:
    if frames.ndim != 3:
        raise ValueError(f"expected T,H,W grayscale frames, got {frames.shape}")
    frame_count, height, width = frames.shape
    base = core.std.BlankClip(
        format=vs.GRAYS, width=width, height=height, length=frame_count, color=0
    )

    def selector(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        np.asarray(out[0])[:] = frames[n]
        return out

    return core.std.ModifyFrame(base, base, selector)


def clip_to_gray_array(node: vs.VideoNode) -> np.ndarray:
    return np.stack(
        [np.asarray(node.get_frame(n)[0]).copy() for n in range(node.num_frames)],
        axis=0,
    ).astype(np.float32, copy=False)


def run_vnlb_chain(core: vs.Core, noisy: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    params = meta["vnlb"]
    sigma = float(meta["noise"]["sigma_8bit"])
    common = {
        "sigma": sigma,
        "block_size": int(params["block_size"]),
        "patch_time": int(params["patch_time"]),
        "bm_range": int(params["bm_range"]),
        "radius": int(params["radius"]),
        "group_size": int(params["group_size"]),
        "rank": int(params["rank"]),
        "beta": float(params["beta"]),
        "block_step": max(1, int(params["block_size"]) // 2),
    }

    clip = ndarray_to_gray_clip(core, noisy)
    basic_params = {
        **common,
        "tau": 0.0,
        "variance_threshold": float(params["basic_variance_threshold"]),
    }
    basic_stack = core.vnlb.Basic(clip, **basic_params)
    basic = core.vnlb.Aggregate(
        basic_stack,
        src=clip,
        patch_time=common["patch_time"],
        radius=common["radius"],
    )

    final_params = {
        **common,
        "tau": float(params["final_tau_8bit"]),
        "variance_threshold": float(params["final_variance_threshold"]),
        "flat_areas": 1,
        "sigma_basic": 0.0,
    }
    final_stack = core.vnlb.Final(clip, ref=basic, **final_params)
    final = core.vnlb.Aggregate(
        final_stack,
        src=clip,
        patch_time=common["patch_time"],
        radius=common["radius"],
    )
    return clip_to_gray_array(basic), clip_to_gray_array(final)


def compare_array(case_name: str, stage: str, actual: np.ndarray, expected: np.ndarray, tolerances: dict) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{case_name} {stage}: shape {actual.shape} != {expected.shape}")
    if actual.dtype != np.float32:
        raise AssertionError(f"{case_name} {stage}: dtype {actual.dtype} != float32")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{case_name} {stage}: output contains NaN or Inf")

    difference = np.abs(actual - expected)
    max_abs = float(difference.max(initial=0.0))
    mean_abs = float(difference.mean())
    rmse = float(np.sqrt(np.mean(difference * difference)))
    print(
        f"{case_name} {stage}: max_abs={max_abs:.8g} "
        f"mean_abs={mean_abs:.8g} rmse={rmse:.8g}"
    )

    if max_abs > float(tolerances["max_abs"]):
        raise AssertionError(f"{case_name} {stage}: max_abs exceeds tolerance")
    if mean_abs > float(tolerances["mean_abs"]):
        raise AssertionError(f"{case_name} {stage}: mean_abs exceeds tolerance")
    if rmse > float(tolerances["rmse"]):
        raise AssertionError(f"{case_name} {stage}: rmse exceeds tolerance")


def run_case(core: vs.Core, case_dir: Path) -> None:
    with (case_dir / "case.toml").open("rb") as f:
        meta = tomllib.load(f)

    noisy = load_npz(case_dir / "input.npz", "noisy")
    basic_ref = load_npz(case_dir / "basic_ref.npz", "basic")
    final_ref = load_npz(case_dir / "final_ref.npz", "final")
    basic, final = run_vnlb_chain(core, noisy, meta)
    compare_array(case_dir.name, "basic", basic, basic_ref, meta["tolerance"])
    compare_array(case_dir.name, "final", final, final_ref, meta["tolerance"])


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: compare_reference.py <plugin-path> <source-root>", file=sys.stderr)
        return 2

    plugin_path = Path(sys.argv[1]).resolve()
    source_root = Path(sys.argv[2]).resolve()
    cases_root = source_root / "tests/reference_cases"

    core = vs.core
    core.std.LoadPlugin(path=str(plugin_path))
    case_dirs = sorted(path for path in cases_root.iterdir() if path.is_dir())
    if not case_dirs:
        raise AssertionError(f"no reference cases under {cases_root}")

    for case_dir in case_dirs:
        run_case(core, case_dir)
    print(f"validated {len(case_dirs)} reference comparison cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
