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


def check_frames(frames: list[vs.VideoFrame], label: str) -> None:
    if len(frames) != 7:
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
    if len(sys.argv) != 2:
        print("usage: integration_cuda.py /absolute/path/to/libvnlbcu.so")
        return 2

    plugin = Path(sys.argv[1]).resolve()
    core = vs.core
    core.std.LoadPlugin(path=str(plugin))
    if not hasattr(core, "vnlbcu"):
        fail("plugin did not register the core.vnlbcu namespace")

    source = make_pattern_clip(core)
    try:
        default_basic = core.vnlbcu.Basic(
            clip=source,
            sigma=8.0,
            rank=9,
            radius=1,
            num_streams=3,
            chunk_size=64,
        )
        default_basic_frames = request_all(default_basic)
        check_frames(default_basic_frames, "Basic defaults")

        explicit_basic = core.vnlbcu.Basic(
            clip=source,
            sigma=8.0,
            group_size=15,
            block_step=4,
            rank=9,
            radius=1,
            num_streams=3,
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
            group_size=15,
            block_step=0,
            rank=9,
            radius=1,
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
            rank=16,
            radius=1,
            num_streams=3,
            chunk_size=64,
        )
        default_final_frames = request_all(default_final)
        check_frames(default_final_frames, "Final defaults")

        explicit_final = core.vnlbcu.Final(
            clip=source,
            ref=default_basic,
            sigma=8.0,
            group_size=17,
            block_step=4,
            rank=16,
            radius=1,
            num_streams=3,
            chunk_size=64,
        )
        explicit_final_frames = request_all(explicit_final)
        check_frames(explicit_final_frames, "Final explicit")
        compare_frames(
            default_final_frames,
            explicit_final_frames,
            "Final omitted defaults versus explicit values",
        )

        # Exercise the runtime-size dual-PCA path with the K/rank values from
        # the tuned reference RGB profile.  A sparse anchor grid keeps the
        # integration test small; the C++ group-filter test checks all samples
        # against a double-precision oracle.
        yuv_source = make_yuv_pattern_clip(core, source)
        large_basic = core.vnlbcu.Basic(
            clip=yuv_source,
            sigma=8.0,
            group_size=100,
            rank=39,
            cap_factor=1.0,
            block_step=16,
            radius=0,
            num_streams=2,
            chunk_size=32,
        )
        large_basic_frames = request_all(large_basic)
        check_frames(large_basic_frames, "Basic K100")

        large_final = core.vnlbcu.Final(
            clip=yuv_source,
            ref=large_basic,
            sigma=8.0,
            group_size=60,
            rank=39,
            cap_factor=1.0,
            block_step=16,
            radius=0,
            num_streams=2,
            chunk_size=32,
        )
        large_final_frames = request_all(large_final)
        check_frames(large_final_frames, "Final K60")

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
