#!/usr/bin/env python3
"""Compare end-to-end VapourSynth outputs against fixed BestSource fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import vapoursynth as vs

CASE_TOLERANCE = 1.0e-6


def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    planes = [
        np.asarray(frame[plane]).copy() for plane in range(frame.format.num_planes)
    ]
    return np.stack(planes, axis=0).astype(np.float32, copy=False)


def clip_to_array(node: vs.VideoNode, frames: tuple[int, ...]) -> np.ndarray:
    return np.stack([frame_to_array(node.get_frame(frame)) for frame in frames], axis=0)


def run_chain(
    core: vs.Core,
    clip: vs.VideoNode,
    *,
    patch_time: int,
    radius: int,
    frames: tuple[int, ...],
    chroma: bool = False,
    mvfw: vs.VideoNode | None = None,
    mvbw: vs.VideoNode | None = None,
) -> np.ndarray:
    params = {
        "sigma": 5.1,
        "block_size": 2,
        "patch_time": patch_time,
        "bm_range": 1,
        "radius": radius,
        "group_size": 8,
        "rank": 4,
        "beta": 1.0,
        "tau": 0.0,
        "variance_threshold": 1.1,
        "block_step": 2,
    }
    if chroma:
        params["chroma"] = 1
    if mvfw is not None:
        params["mvfw"] = mvfw
    if mvbw is not None:
        params["mvbw"] = mvbw

    slot_count = (2 * radius) + patch_time
    expected_stack_height = clip.height * 2 * slot_count

    basic_stack = core.vnlb.Basic(clip, **params)
    if basic_stack.format.id != clip.format.id:
        raise AssertionError("Basic stack format does not match input")
    if basic_stack.width != clip.width or basic_stack.height != expected_stack_height:
        raise AssertionError("Basic stack dimensions are invalid")

    basic = core.vnlb.Aggregate(
        basic_stack, src=clip, patch_time=patch_time, radius=radius
    )
    final_stack = core.vnlb.Final(clip, ref=basic, **params)
    if final_stack.format.id != clip.format.id:
        raise AssertionError("Final stack format does not match input")
    if final_stack.width != clip.width or final_stack.height != expected_stack_height:
        raise AssertionError("Final stack dimensions are invalid")

    final = core.vnlb.Aggregate(
        final_stack, src=clip, patch_time=patch_time, radius=radius
    )
    return clip_to_array(final, frames)


def assert_basic_rejects_final_only_params(core: vs.Core) -> None:
    clip = core.std.BlankClip(format=vs.GRAYS, width=16, height=16, length=1, color=0)
    final_only_params = {
        "sigma_basic": 0.0,
        "gamma": 0.95,
        "flat_areas": 0,
    }
    for name, value in final_only_params.items():
        try:
            core.vnlb.Basic(clip, sigma=5.1, **{name: value})
        except vs.Error:
            continue
        raise AssertionError(f"Basic accepted final-only parameter {name!r}")


def assert_directional_temporal_args_override_radius(core: vs.Core) -> None:
    clip = core.std.BlankClip(format=vs.GRAYS, width=16, height=16, length=1, color=0)
    stack = core.vnlb.Basic(
        clip,
        sigma=5.1,
        block_size=2,
        patch_time=1,
        bm_range=1,
        radius=9,
        search_bwd=0,
        search_fwd=0,
        group_size=8,
        rank=4,
        block_step=2,
    )
    if stack.height != clip.height * 2:
        raise AssertionError("directional temporal args did not override radius")

    aggregated = core.vnlb.Aggregate(
        stack,
        src=clip,
        patch_time=1,
        radius=9,
        search_bwd=0,
        search_fwd=0,
    )
    aggregated.get_frame(0)


def assert_basic_rejects_invalid_params(core: vs.Core) -> None:
    clip = core.std.BlankClip(format=vs.GRAYS, width=16, height=16, length=1, color=0)
    invalid_params = {
        "sigma": float("inf"),
        "beta": float("inf"),
        "tau": float("inf"),
        "variance_threshold": float("inf"),
        "weight_alpha": float("inf"),
        "weight_beta": float("inf"),
        "weight_gamma": float("inf"),
        "weight_epsilon": float("inf"),
        "membership_noise_floor": float("inf"),
        "block_step": -1,
    }
    for name, value in invalid_params.items():
        params = {
            "sigma": 5.1,
            "block_size": 2,
            "patch_time": 1,
            "bm_range": 1,
            "radius": 0,
            "group_size": 8,
            "rank": 4,
            "block_step": 2,
            name: value,
        }
        try:
            stack = core.vnlb.Basic(clip, **params)
            stack.get_frame(0)
        except vs.Error:
            continue
        raise AssertionError(f"Basic accepted invalid parameter {name!r}")


def build_cases(
    core: vs.Core, source_root: Path, cache_dir: Path
) -> dict[str, np.ndarray]:
    image = core.bs.VideoSource(
        source=str(source_root / "tests/assets/downloads/earthrise.jpg"),
        cachepath=str(cache_dir / "earthrise.bsindex"),
    )
    image_gray = core.resize.Bicubic(image, format=vs.GRAYS, width=32, height=32)
    image_yuv = core.resize.Bicubic(image, format=vs.YUV444PS, width=32, height=32)

    video = core.bs.VideoSource(
        source=str(source_root / "tests/assets/downloads/wmap-990307_1280.mp4"),
        cachepath=str(cache_dir / "wmap.bsindex"),
    )
    video = core.std.Trim(video, first=0, length=4)
    video_gray = core.resize.Bicubic(video, format=vs.GRAYS, width=32, height=24)
    video_gray_mv_source = core.resize.Bicubic(
        video, format=vs.GRAY8, width=32, height=24
    )
    video_yuv = core.resize.Bicubic(video, format=vs.YUV444PS, width=32, height=24)
    super_clip = core.mv.Super(video_gray_mv_source, pel=1, hpad=16, vpad=16)
    mvfw = core.mv.Analyse(
        super_clip, isb=False, delta=1, blksize=8, overlap=4, search=3, searchparam=8
    )
    mvbw = core.mv.Analyse(
        super_clip, isb=True, delta=1, blksize=8, overlap=4, search=3, searchparam=8
    )

    return {
        "image_gray": run_chain(core, image_gray, patch_time=1, radius=0, frames=(0,)),
        "image_yuv444": run_chain(core, image_yuv, patch_time=1, radius=0, frames=(0,)),
        "image_yuv444_chroma": run_chain(
            core, image_yuv, patch_time=1, radius=0, frames=(0,), chroma=True
        ),
        "video_gray": run_chain(
            core, video_gray, patch_time=2, radius=1, frames=(0, 1, 2, 3)
        ),
        "video_gray_mvtools": run_chain(
            core,
            video_gray,
            patch_time=2,
            radius=1,
            frames=(0, 1, 2, 3),
            mvfw=mvfw,
            mvbw=mvbw,
        ),
        "video_yuv444": run_chain(
            core, video_yuv, patch_time=2, radius=1, frames=(0, 1, 2, 3)
        ),
        "video_yuv444_chroma": run_chain(
            core,
            video_yuv,
            patch_time=2,
            radius=1,
            frames=(0, 1, 2, 3),
            chroma=True,
        ),
    }


def compare_case(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    if actual.dtype != np.float32:
        raise AssertionError(f"{name}: actual dtype {actual.dtype} is not float32")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{name}: output contains NaN or Inf")

    difference = np.abs(actual - expected)
    max_abs = float(difference.max(initial=0.0))
    if max_abs > CASE_TOLERANCE:
        mean_abs = float(difference.mean())
        raise AssertionError(
            f"{name}: max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}, tolerance={CASE_TOLERANCE}"
        )


def main() -> int:
    if len(sys.argv) not in (4, 5):
        print(
            "usage: integration_bestsource.py <plugin-path> <source-root> <cache-dir> [--update]",
            file=sys.stderr,
        )
        return 2

    plugin_path = Path(sys.argv[1]).resolve()
    source_root = Path(sys.argv[2]).resolve()
    cache_dir = Path(sys.argv[3]).resolve()
    update = len(sys.argv) == 5 and sys.argv[4] == "--update"
    if len(sys.argv) == 5 and not update:
        print("unknown option: " + sys.argv[4], file=sys.stderr)
        return 2

    fixture_path = source_root / "tests/vs/fixtures/bestsource_outputs.npz"
    cache_dir.mkdir(parents=True, exist_ok=True)

    core = vs.core
    core.std.LoadPlugin(path=str(plugin_path))
    assert_basic_rejects_final_only_params(core)
    assert_directional_temporal_args_override_radius(core)
    assert_basic_rejects_invalid_params(core)
    actual = build_cases(core, source_root, cache_dir)

    if update:
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fixture_path, **actual)
        print(f"updated {fixture_path}")
        return 0

    if not fixture_path.exists():
        raise FileNotFoundError(fixture_path)

    with np.load(fixture_path) as expected:
        expected_keys = set(expected.files)
        actual_keys = set(actual.keys())
        if actual_keys != expected_keys:
            raise AssertionError(
                f"fixture keys {expected_keys} != actual keys {actual_keys}"
            )
        for name, values in actual.items():
            compare_case(name, values, expected[name])
            print(f"ok {name}")

    print("validated BestSource integration fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
