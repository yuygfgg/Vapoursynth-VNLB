#!/usr/bin/env python3
"""Compare local CPU and CUDA VNLB builds without touching plugin autoloading."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import vapoursynth as vs


def ndarray_to_gray_clip(core: vs.Core, frames: np.ndarray) -> vs.VideoNode:
    if frames.ndim != 3:
        raise ValueError(f"expected T,H,W grayscale frames, got {frames.shape}")
    frame_count, height, width = frames.shape
    base = core.std.BlankClip(
        format=vs.GRAYS,
        width=width,
        height=height,
        length=frame_count,
        color=0,
    )

    def selector(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        output = f.copy()
        np.asarray(output[0])[:] = frames[n]
        return output

    return core.std.ModifyFrame(base, base, selector)


def clip_to_array(node: vs.VideoNode, frames: list[int]) -> np.ndarray:
    return np.stack(
        [
            np.stack(
                [
                    np.asarray(node.get_frame(n)[plane]).copy()
                    for plane in range(node.format.num_planes)
                ],
                axis=0,
            )
            for n in frames
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def gradient_rms(values: np.ndarray) -> float:
    dx = values[..., :, 1:] - values[..., :, :-1]
    dy = values[..., 1:, :] - values[..., :-1, :]
    return float(np.sqrt((np.mean(dx * dx) + np.mean(dy * dy)) / 2.0))


def laplacian_rms(values: np.ndarray) -> float:
    center = values[..., 1:-1, 1:-1]
    laplacian = (
        values[..., :-2, 1:-1]
        + values[..., 2:, 1:-1]
        + values[..., 1:-1, :-2]
        + values[..., 1:-1, 2:]
        - (4.0 * center)
    )
    return rms(laplacian)


def report(stage: str, source: np.ndarray, cpu: np.ndarray, cuda: np.ndarray) -> None:
    difference = cuda - cpu
    print(
        f"{stage}: cpu<->cuda "
        f"mae={np.mean(np.abs(difference)):.8g} "
        f"rmse={rms(difference):.8g} "
        f"max={np.max(np.abs(difference)):.8g}"
    )
    print(
        f"{stage}: residual_rms cpu={rms(cpu - source):.8g} "
        f"cuda={rms(cuda - source):.8g}"
    )
    print(
        f"{stage}: gradient_rms cpu={gradient_rms(cpu):.8g} "
        f"cuda={gradient_rms(cuda):.8g}; "
        f"laplacian_rms cpu={laplacian_rms(cpu):.8g} "
        f"cuda={laplacian_rms(cuda):.8g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_plugin", type=Path)
    parser.add_argument("cuda_plugin", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        default=Path("tests/reference_cases/earthrise_gray/input.npz"),
    )
    source.add_argument("--video", type=Path)
    parser.add_argument("--frame", type=int, default=60000)
    parser.add_argument("--crop-width", type=int, default=192)
    parser.add_argument("--crop-height", type=int, default=128)
    parser.add_argument("--crop-left", type=int)
    parser.add_argument("--crop-top", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    core = vs.core
    core.num_threads = 1
    core.std.LoadPlugin(
        str(args.cpu_plugin.resolve()),
        forcens="vnlb_local",
        forceid="com.yuygfgg.vnlb.local",
    )
    core.std.LoadPlugin(
        str(args.cuda_plugin.resolve()),
        forcens="vnlbcu_local",
        forceid="com.yuygfgg.vnlbcu.local",
    )
    print(f"CPU plugin:  {core.vnlb_local.plugin_path}")
    print(f"CUDA plugin: {core.vnlbcu_local.plugin_path}")

    if args.video is None:
        with np.load(args.input) as archive:
            noisy = np.asarray(archive["noisy"], dtype=np.float32)
        source = ndarray_to_gray_clip(core, noisy)
        requested_frames = list(range(source.num_frames))
    else:
        source = core.bs.VideoSource(str(args.video.resolve()))
        source = core.resize.Bicubic(
            source,
            format=vs.YUV444PS,
            matrix_in_s="709",
            matrix_s="709",
            transfer_in_s="709",
            transfer_s="709",
            primaries_in_s="709",
            primaries_s="709",
        )
        if args.frame < 0 or args.frame >= source.num_frames:
            raise ValueError(
                f"frame {args.frame} is outside 0..{source.num_frames - 1}"
            )
        left = (
            (source.width - args.crop_width) // 2
            if args.crop_left is None
            else args.crop_left
        )
        top = (
            (source.height - args.crop_height) // 2
            if args.crop_top is None
            else args.crop_top
        )
        source = core.std.CropAbs(
            source,
            width=args.crop_width,
            height=args.crop_height,
            left=left,
            top=top,
        )
        requested_frames = [args.frame]
    common = {
        "sigma": 30.0,
        "block_size": 7,
        "block_step": 3,
        "group_size": 16,
        "cap_factor": 4.0,
        "model_cap_factor": 1.0,
        "bm_range": 13,
        "patch_time": 2,
        "radius": 1,
        "search_bwd": 1,
        "search_fwd": 1,
        "rank": 15,
        "beta": 1.0,
        "chroma": 1,
    }
    basic_parameters = {
        **common,
        "tau": 0.0,
        "variance_threshold": 2.7,
        "weight_alpha": 0.75,
        "weight_beta": 0.35,
        "weight_gamma": 1.0,
        "weight_epsilon": 1.0e-6,
        "membership_noise_floor": 0.25,
    }
    final_parameters = {
        **common,
        "tau": 400.0,
        "variance_threshold": 0.7,
        "sigma_basic": 0.0,
        "gamma": 0.95,
        "flat_areas": 1,
        "weight_alpha": 1.0,
        "weight_beta": 0.5,
        "weight_gamma": 1.0,
        "weight_epsilon": 1.0e-6,
        "membership_noise_floor": 0.25,
    }

    cpu_basic_stack = core.vnlb_local.Basic(source, **basic_parameters)
    cpu_basic = core.vnlb_local.Aggregate(
        cpu_basic_stack, source, patch_time=2, radius=1
    )
    cuda_basic = core.vnlbcu_local.Basic(
        source, **basic_parameters, num_streams=1, chunk_size=256
    )
    cpu_final_stack = core.vnlb_local.Final(
        source, ref=cpu_basic, **final_parameters
    )
    cpu_final = core.vnlb_local.Aggregate(
        cpu_final_stack, source, patch_time=2, radius=1
    )
    cuda_final = core.vnlbcu_local.Final(
        source,
        ref=cuda_basic,
        **final_parameters,
        num_streams=1,
        chunk_size=256,
    )

    source_values = clip_to_array(source, requested_frames)
    cpu_basic_values = clip_to_array(cpu_basic, requested_frames)
    cuda_basic_values = clip_to_array(cuda_basic, requested_frames)
    report("Basic", source_values, cpu_basic_values, cuda_basic_values)
    cpu_final_values = clip_to_array(cpu_final, requested_frames)
    cuda_final_values = clip_to_array(cuda_final, requested_frames)
    report("Final", source_values, cpu_final_values, cuda_final_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
