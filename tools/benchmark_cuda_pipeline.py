#!/usr/bin/env python3
"""Benchmark the CUDA VapourSynth pipeline and compare saved frame outputs."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs


def frame_to_array(frame: vs.VideoFrame) -> np.ndarray:
    return np.stack(
        [
            np.asarray(frame[plane], dtype=np.float32).copy()
            for plane in range(frame.format.num_planes)
        ],
        axis=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("plugin", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--stage", choices=("basic", "final"), default="final")
    parser.add_argument("--start", type=int, default=50)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--num-streams", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--reference", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    core = vs.core
    core.std.LoadPlugin(
        str(args.plugin.resolve()),
        forcens="vnlbcu_bench",
        forceid="com.yuygfgg.vnlbcu.bench",
    )
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
        "num_streams": args.num_streams,
        "chunk_size": args.chunk_size,
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

    basic = core.vnlbcu_bench.Basic(source, **basic_parameters)
    output = (
        basic
        if args.stage == "basic"
        else core.vnlbcu_bench.Final(
            source,
            ref=basic,
            **final_parameters,
        )
    )

    frame_numbers = list(range(args.start, args.start + args.count))
    if frame_numbers[-1] >= output.num_frames:
        raise ValueError(
            f"requested frame {frame_numbers[-1]} but clip has "
            f"{output.num_frames} frames"
        )

    timings: list[float] = []
    arrays: list[np.ndarray] = []
    for frame_number in frame_numbers:
        begin = time.perf_counter()
        frame = output.get_frame(frame_number)
        elapsed = time.perf_counter() - begin
        timings.append(elapsed)
        arrays.append(frame_to_array(frame))
        print(f"frame={frame_number} seconds={elapsed:.6f}")

    cold = timings[0]
    steady = timings[1:] if len(timings) > 1 else timings
    steady_total = sum(steady)
    print(f"cold_seconds={cold:.6f}")
    print(f"steady_mean_seconds={steady_total / len(steady):.6f}")
    print(f"steady_fps={len(steady) / steady_total:.6f}")

    values = np.stack(arrays, axis=0)
    if args.reference is not None:
        with np.load(args.reference) as archive:
            expected_frames = np.asarray(archive["frame_numbers"])
            expected_values = np.asarray(archive["values"], dtype=np.float32)
        if not np.array_equal(expected_frames, np.asarray(frame_numbers)):
            raise ValueError("reference frame numbers do not match")
        if expected_values.shape != values.shape:
            raise ValueError(
                f"reference shape {expected_values.shape} != {values.shape}"
            )
        difference = values - expected_values
        print(f"max_abs_diff={np.max(np.abs(difference)):.9g}")
        print(
            "rmse="
            f"{np.sqrt(np.mean(np.square(difference, dtype=np.float64))):.9g}"
        )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save,
            frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
            values=values,
        )
        print(f"saved={args.save.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
