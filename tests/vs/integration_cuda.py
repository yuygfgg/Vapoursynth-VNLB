#!/usr/bin/env python3

from __future__ import annotations

import math
import sys
from concurrent.futures import Future
from pathlib import Path

import vapoursynth as vs


def fail(message: str) -> None:
    raise RuntimeError(message)


def make_pattern_clip(core: vs.Core) -> vs.VideoNode:
    base = core.std.BlankClip(
        width=64,
        height=48,
        length=7,
        format=vs.GRAYS,
    )

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        result = f.copy()
        plane = memoryview(result[0])
        for y in range(result.height):
            for x in range(result.width):
                plane[y, x] = (
                    0.2
                    + (0.003 * x)
                    + (0.002 * y)
                    + (0.01 * n)
                    + (0.02 * math.sin((x + (2 * y) + n) / 7.0))
                )
        return result

    return core.std.ModifyFrame(base, clips=base, selector=fill)


def make_yuv_pattern_clip(core: vs.Core, gray: vs.VideoNode) -> vs.VideoNode:
    chroma_u = core.std.Expr(gray, "x 0.55 * 0.2 +")
    chroma_v = core.std.Expr(gray, "x 0.35 * 0.4 +")
    return core.std.ShufflePlanes(
        clips=[gray, chroma_u, chroma_v],
        planes=[0, 0, 0],
        colorfamily=vs.YUV,
    )


def request_all(clip: vs.VideoNode) -> list[vs.VideoFrame]:
    futures: list[Future[vs.VideoFrame]] = [
        clip.get_frame_async(frame) for frame in range(clip.num_frames)
    ]
    return [future.result(timeout=30.0) for future in futures]


def check_frames(
    frames: list[vs.VideoFrame], label: str, expected_count: int = 7
) -> None:
    if len(frames) != expected_count:
        fail(f"{label}: unexpected frame count")

    prior_mean = -math.inf
    for frame_number, frame in enumerate(frames):
        if frame.width != 64 or frame.height != 48:
            fail(f"{label}: output geometry changed")
        plane = memoryview(frame[0])
        total = 0.0
        minimum = math.inf
        maximum = -math.inf
        for y in range(frame.height):
            for x in range(frame.width):
                value = float(plane[y, x])
                if not math.isfinite(value):
                    fail(f"{label}: non-finite sample in frame {frame_number}")
                total += value
                minimum = min(minimum, value)
                maximum = max(maximum, value)
        mean = total / (frame.width * frame.height)
        if maximum - minimum < 0.05:
            fail(f"{label}: unexpectedly flat output frame {frame_number}")
        if mean <= prior_mean:
            fail(f"{label}: temporal pattern order was not preserved")
        prior_mean = mean


def compare_frames(
    actual: list[vs.VideoFrame],
    expected: list[vs.VideoFrame],
    label: str,
    tolerance: float = 2.0e-5,
) -> None:
    if len(actual) != len(expected):
        fail(f"{label}: frame counts differ")
    for frame_number, (actual_frame, expected_frame) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if (
            actual_frame.width != expected_frame.width
            or actual_frame.height != expected_frame.height
            or actual_frame.format.id != expected_frame.format.id
        ):
            fail(f"{label}: frame {frame_number} format or geometry differs")
        for plane_number in range(actual_frame.format.num_planes):
            actual_plane = memoryview(actual_frame[plane_number])
            expected_plane = memoryview(expected_frame[plane_number])
            for y in range(actual_frame.height):
                for x in range(actual_frame.width):
                    difference = abs(
                        float(actual_plane[y, x]) - float(expected_plane[y, x])
                    )
                    if not math.isfinite(difference):
                        fail(
                            f"{label}: frame {frame_number}, plane {plane_number}, "
                            f"sample ({x}, {y}) is non-finite"
                        )
                    if difference > tolerance:
                        fail(
                            f"{label}: frame {frame_number}, plane {plane_number}, "
                            f"sample ({x}, {y}) differs by {difference}"
                        )


def is_missing_cuda(error: BaseException) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "no cuda-capable device",
            "cuda driver version is insufficient",
            "cuda initialization error",
            "cudaerrornodevice",
        )
    )


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            "usage: integration_cuda.py /absolute/path/to/libvnlbcu.so "
            "[/absolute/path/to/libvnlb.so]"
        )
        return 2

    plugin = Path(sys.argv[1]).resolve()
    core = vs.core
    core.std.LoadPlugin(path=str(plugin))
    if not hasattr(core, "vnlbcu"):
        fail("plugin did not register the core.vnlbcu namespace")

    source = make_pattern_clip(core)
    try:
        if len(sys.argv) == 3:
            cpu_plugin = Path(sys.argv[2]).resolve()
            core.std.LoadPlugin(
                path=str(cpu_plugin),
                forcens="vnlb_cpu_test",
                forceid="com.yuygfgg.vnlb.cpu-test",
            )
            common_parameters = {
                "sigma": 8.0,
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
                **common_parameters,
                "tau": 0.0,
                "variance_threshold": 2.7,
                "weight_alpha": 0.75,
                "weight_beta": 0.35,
                "weight_gamma": 1.0,
                "weight_epsilon": 1.0e-6,
                "membership_noise_floor": 0.25,
            }
            final_parameters = {
                **common_parameters,
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

            cpu_basic_stack = core.vnlb_cpu_test.Basic(
                source, **basic_parameters
            )
            cpu_basic = core.vnlb_cpu_test.Aggregate(
                cpu_basic_stack, source, patch_time=2, radius=1
            )
            cuda_basic = core.vnlbcu.Basic(
                source,
                **basic_parameters,
                num_streams=1,
                chunk_size=64,
            )
            compare_frames(
                [cuda_basic.get_frame(3)],
                [cpu_basic.get_frame(3)],
                "CPU-compatible staggered Basic anchors",
                tolerance=5.0e-4,
            )

            cpu_final_stack = core.vnlb_cpu_test.Final(
                source, ref=cpu_basic, **final_parameters
            )
            cpu_final = core.vnlb_cpu_test.Aggregate(
                cpu_final_stack, source, patch_time=2, radius=1
            )
            cuda_final = core.vnlbcu.Final(
                source,
                ref=cuda_basic,
                **final_parameters,
                num_streams=1,
                chunk_size=64,
            )
            compare_frames(
                [cuda_final.get_frame(3)],
                [cpu_final.get_frame(3)],
                "CPU-compatible staggered Final anchors",
                tolerance=5.0e-4,
            )

        default_basic = core.vnlbcu.Basic(
            clip=source,
            sigma=8.0,
            chunk_size=64,
        )
        default_basic_frames = request_all(default_basic)
        check_frames(default_basic_frames, "Basic defaults")

        explicit_basic = core.vnlbcu.Basic(
            clip=source,
            sigma=8.0,
            block_size=10,
            block_step=5,
            group_size=16,
            bm_range=13,
            patch_time=2,
            search_bwd=1,
            search_fwd=1,
            rank=15,
            variance_threshold=3.7,
            num_streams=1,
            chunk_size=64,
        )
        explicit_basic_frames = request_all(explicit_basic)
        check_frames(explicit_basic_frames, "Basic explicit")
        compare_frames(
            default_basic_frames,
            explicit_basic_frames,
            "Basic omitted defaults versus explicit values",
        )

        zero_step_basic = core.vnlbcu.Basic(
            clip=source,
            sigma=8.0,
            group_size=16,
            block_step=0,
            rank=15,
            num_streams=3,
            chunk_size=64,
        )
        zero_step_basic_frames = request_all(zero_step_basic)
        check_frames(zero_step_basic_frames, "Basic block_step=0")
        compare_frames(
            zero_step_basic_frames,
            explicit_basic_frames,
            "Basic block_step=0 versus explicit half-patch step",
        )

        default_final = core.vnlbcu.Final(
            clip=source,
            ref=default_basic,
            sigma=8.0,
            chunk_size=64,
        )
        default_final_frames = request_all(default_final)
        check_frames(default_final_frames, "Final defaults")

        explicit_final = core.vnlbcu.Final(
            clip=source,
            ref=default_basic,
            sigma=8.0,
            block_size=10,
            block_step=5,
            group_size=16,
            bm_range=13,
            patch_time=2,
            search_bwd=1,
            search_fwd=1,
            rank=15,
            variance_threshold=max(0.0, 1.87 - (0.028 * 8.0)),
            num_streams=1,
            chunk_size=64,
        )
        explicit_final_frames = request_all(explicit_final)
        check_frames(explicit_final_frames, "Final explicit")
        compare_frames(
            default_final_frames,
            explicit_final_frames,
            "Final omitted defaults versus explicit values",
        )

        single_source = core.std.Trim(source, first=0, last=0)
        single_default = core.vnlbcu.Basic(
            clip=single_source,
            sigma=8.0,
            block_step=16,
            num_streams=1,
            chunk_size=16,
        )
        single_explicit = core.vnlbcu.Basic(
            clip=single_source,
            sigma=8.0,
            block_size=10,
            block_step=16,
            group_size=16,
            bm_range=13,
            patch_time=1,
            radius=0,
            rank=15,
            variance_threshold=3.7,
            num_streams=1,
            chunk_size=16,
        )
        single_default_frames = request_all(single_default)
        single_explicit_frames = request_all(single_explicit)
        check_frames(single_default_frames, "single-frame Basic", expected_count=1)
        compare_frames(
            single_default_frames,
            single_explicit_frames,
            "single-frame Gray defaults",
        )

        # Exercise the runtime-size dual-PCA path with the K/rank values from
        # the tuned reference RGB profile.  A sparse anchor grid keeps the
        # integration test small; the C++ group-filter test checks all samples
        # against a double-precision oracle.
        yuv_source = make_yuv_pattern_clip(core, source)
        default_large_basic = core.vnlbcu.Basic(
            clip=yuv_source,
            sigma=8.0,
            cap_factor=1.0,
            block_step=16,
            radius=0,
            chunk_size=32,
        )
        explicit_default_large_basic = core.vnlbcu.Basic(
            clip=yuv_source,
            sigma=8.0,
            block_size=7,
            group_size=16,
            rank=15,
            cap_factor=1.0,
            bm_range=13,
            patch_time=2,
            block_step=16,
            radius=0,
            variance_threshold=2.7,
            num_streams=1,
            chunk_size=32,
        )
        large_basic = core.vnlbcu.Basic(
            clip=yuv_source,
            sigma=8.0,
            block_size=7,
            group_size=100,
            rank=39,
            cap_factor=1.0,
            bm_range=13,
            patch_time=2,
            block_step=16,
            radius=0,
            variance_threshold=2.7,
            num_streams=1,
            chunk_size=32,
        )
        default_large_basic_frames = request_all(default_large_basic)
        explicit_default_large_basic_frames = request_all(
            explicit_default_large_basic
        )
        large_basic_frames = request_all(large_basic)
        check_frames(default_large_basic_frames, "YUV Basic defaults")
        check_frames(large_basic_frames, "Basic K100")
        compare_frames(
            default_large_basic_frames,
            explicit_default_large_basic_frames,
            "YUV Basic omitted defaults versus explicit values",
        )

        default_large_final = core.vnlbcu.Final(
            clip=yuv_source,
            ref=large_basic,
            sigma=8.0,
            cap_factor=1.0,
            block_step=16,
            radius=0,
            chunk_size=32,
        )
        explicit_default_large_final = core.vnlbcu.Final(
            clip=yuv_source,
            ref=large_basic,
            sigma=8.0,
            block_size=7,
            group_size=16,
            rank=15,
            cap_factor=1.0,
            bm_range=13,
            patch_time=2,
            block_step=16,
            radius=0,
            variance_threshold=max(
                0.2, 1.7 + ((1.2 - 1.7) * (8.0 - 10.0) / 10.0)
            ),
            num_streams=1,
            chunk_size=32,
        )
        large_final = core.vnlbcu.Final(
            clip=yuv_source,
            ref=large_basic,
            sigma=8.0,
            block_size=7,
            group_size=60,
            rank=39,
            cap_factor=1.0,
            bm_range=13,
            patch_time=2,
            block_step=16,
            radius=0,
            variance_threshold=max(
                0.2, 1.7 + ((1.2 - 1.7) * (8.0 - 10.0) / 10.0)
            ),
            num_streams=1,
            chunk_size=32,
        )
        default_large_final_frames = request_all(default_large_final)
        explicit_default_large_final_frames = request_all(
            explicit_default_large_final
        )
        large_final_frames = request_all(large_final)
        check_frames(default_large_final_frames, "YUV Final defaults")
        check_frames(large_final_frames, "Final K60")
        compare_frames(
            default_large_final_frames,
            explicit_default_large_final_frames,
            "YUV Final omitted defaults versus explicit values",
        )

        try:
            core.vnlbcu.Basic(
                clip=yuv_source,
                sigma=8.0,
                group_size=129,
                rank=39,
                radius=0,
            )
        except vs.Error as error:
            if "at most 128 model samples" not in str(error):
                raise
        else:
            fail("group_size=129 was not rejected by the CUDA plugin")
    except vs.Error as error:
        if is_missing_cuda(error):
            print(f"CUDA integration skipped: {error}")
            return 77
        raise

    print("vnlbcu VapourSynth integration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
