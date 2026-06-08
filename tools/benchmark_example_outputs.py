#!/usr/bin/env python3
"""Benchmark selected output nodes from example.py without opening vsview."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import vapoursynth as vs

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/Volumes/untitled/Cosmic.Princess.Kaguya.2026.1080p.NF.WEB-DL.MULTi.DDP5.1.H.264-VARYG.mkv"
)

BM_SHARED_PARAMS = {
    "sigma": 5.1,
    "block_step": 8,
    "bm_range": 9,
    "radius": 1,
}

COMMON_PARAMS = {
    **BM_SHARED_PARAMS,
    "block_size": 8,
    "patch_time": 1,
    "group_size": 8,
    "rank": 8,
    "beta": 1.0,
}

BASIC_PARAMS = {
    **COMMON_PARAMS,
    "tau": 0.0,
    "variance_threshold": 1.1,
    "flat_areas": 0,
}

FINAL_PARAMS = {
    **COMMON_PARAMS,
    "tau": 400.0,
    "variance_threshold": 1.7,
    "flat_areas": 1,
    "sigma_basic": 0.0,
}

BM3D_PARAMS = {
    **BM_SHARED_PARAMS,
    "ps_num": 2,
    "ps_range": 4,
}

NODE_NAMES = {
    0: "Source YUV444PS",
    1: "VNLB Basic",
    2: "VNLB Final",
    3: "VNLB Basic MVTools",
    4: "VNLB Final MVTools",
    5: "BM3D Ref",
    6: "BM3D Final",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "build/release/bench-example-ffms2"
    )
    parser.add_argument(
        "--source-filter",
        choices=("ffms2", "bestsource"),
        default="ffms2",
        help="Source filter used to open the input clip.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--middle",
        action="store_true",
        help="Benchmark a segment centered in the source clip.",
    )
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--max-cache-mb",
        type=int,
        default=0,
        help="Set VapourSynth core.max_cache_size in MB; 0 keeps the current value.",
    )
    parser.add_argument("--prefetch", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--sample-seconds", type=int, default=0)
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=ROOT / "build/release/benchmark-example.sample.txt",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[4, 6],
        choices=sorted(NODE_NAMES),
        help="Output node numbers from example.py to render.",
    )
    parser.add_argument(
        "--mvtools-opt",
        type=int,
        default=0,
        help="MVTools opt value. example.py currently uses opt=0.",
    )
    return parser.parse_args()


def default_plugin_path() -> Path:
    if platform.system() == "Darwin":
        return ROOT / "build/release/libvnlb.dylib"
    if platform.system() == "Windows":
        return ROOT / "build/release/vnlb.dll"
    return ROOT / "build/release/libvnlb.so"


def load_vnlb_plugin(core: vs.Core, plugin: Path | None) -> None:
    if hasattr(core, "vnlb"):
        return

    path = plugin or default_plugin_path()
    if not path.exists():
        raise FileNotFoundError(f"VNLB plugin not found: {path}")
    core.std.LoadPlugin(path=str(path.resolve()))


def load_source(
    core: vs.Core, source: Path, cache_dir: Path, source_filter: str
) -> tuple[vs.VideoNode, vs.VideoNode]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if source_filter == "ffms2":
        raw = core.ffms2.Source(
            source=str(source.resolve()),
            cachefile=str((cache_dir / "source.ffindex").resolve()),
        )
    else:
        raw = core.bs.VideoSource(
            source=str(source.resolve()),
            cachepath=str((cache_dir / "source.bsindex").resolve()),
        )
    src = core.resize.Bicubic(raw, format=vs.YUV444PS)
    mvsrc = core.resize.Bicubic(raw, format=vs.GRAY8)
    return src, mvsrc


def build_mvtools(
    core: vs.Core, mvsrc: vs.VideoNode, mvtools_opt: int
) -> tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode]:
    super_clip = core.mv.Super(mvsrc, pel=1, hpad=16, vpad=16, opt=mvtools_opt)
    mvfw = core.mv.Analyse(
        super_clip,
        isb=False,
        delta=1,
        blksize=8,
        overlap=4,
        search=3,
        searchparam=8,
        opt=mvtools_opt,
    )
    mvbw = core.mv.Analyse(
        super_clip,
        isb=True,
        delta=1,
        blksize=8,
        overlap=4,
        search=3,
        searchparam=8,
        opt=mvtools_opt,
    )

    mvsrc_sc = core.mv.SCDetection(mvsrc, mvfw)
    mvsrc_sc = core.mv.SCDetection(mvsrc_sc, mvbw)
    return mvfw, mvbw, mvsrc_sc


def run_vnlb(
    core: vs.Core,
    clip: vs.VideoNode,
    *,
    mvfw: vs.VideoNode | None = None,
    mvbw: vs.VideoNode | None = None,
) -> tuple[vs.VideoNode, vs.VideoNode]:
    basic_params = dict(BASIC_PARAMS)
    final_params = dict(FINAL_PARAMS)

    if mvfw is not None:
        basic_params["mvfw"] = mvfw
        final_params["mvfw"] = mvfw
    if mvbw is not None:
        basic_params["mvbw"] = mvbw
        final_params["mvbw"] = mvbw

    basic_stack = core.vnlb.Basic(clip, chroma=True, **basic_params)
    basic = core.vnlb.Aggregate(
        basic_stack,
        src=clip,
        patch_time=COMMON_PARAMS["patch_time"],
        radius=COMMON_PARAMS["radius"],
    )

    final_stack = core.vnlb.Final(clip, ref=basic, chroma=True, **final_params)
    final = core.vnlb.Aggregate(
        final_stack,
        src=clip,
        patch_time=COMMON_PARAMS["patch_time"],
        radius=COMMON_PARAMS["radius"],
    )
    return basic, final


def build_outputs(core: vs.Core, args: argparse.Namespace) -> dict[int, vs.VideoNode]:
    load_vnlb_plugin(core, args.plugin)

    src, mvsrc = load_source(core, args.source, args.cache_dir, args.source_filter)
    if args.middle:
        args.start = max(0, src.num_frames // 2 - args.frames // 2)
    src = core.std.Trim(src, first=args.start, length=args.frames)
    mvsrc = core.std.Trim(mvsrc, first=args.start, length=args.frames)

    outputs: dict[int, vs.VideoNode] = {0: src}

    needs_plain_vnlb = any(node in {1, 2} for node in args.nodes)
    needs_mv_vnlb = any(node in {3, 4} for node in args.nodes)
    needs_bm3d = any(node in {5, 6} for node in args.nodes)

    if needs_plain_vnlb:
        basic, final = run_vnlb(core, src)
        outputs[1] = basic
        outputs[2] = final

    if needs_mv_vnlb:
        mvfw, mvbw, mvsrc_sc = build_mvtools(core, mvsrc, args.mvtools_opt)
        src_mv = core.std.CopyFrameProps(
            src,
            prop_src=mvsrc_sc,
            props=["_SceneChangePrev", "_SceneChangeNext"],
        )
        basic_mv, final_mv = run_vnlb(core, src_mv, mvfw=mvfw, mvbw=mvbw)
        outputs[3] = basic_mv
        outputs[4] = final_mv

    if needs_bm3d:
        bm3d_ref = core.bm3d.VBasic(src, **BM3D_PARAMS).bm3d.VAggregate(
            radius=COMMON_PARAMS["radius"], sample=1
        )
        outputs[5] = bm3d_ref
        if 6 in args.nodes:
            outputs[6] = core.bm3d.VFinal(src, ref=bm3d_ref, **BM3D_PARAMS).bm3d.VAggregate(
                radius=COMMON_PARAMS["radius"], sample=1
            )

    return outputs


def checksum_frame(frame: vs.VideoFrame) -> float:
    checksum = 0.0
    for plane in range(frame.format.num_planes):
        values = np.asarray(frame[plane])
        checksum += float(values[0, 0])
        checksum += float(values[values.shape[0] // 2, values.shape[1] // 2])
    return checksum


def render_node(node: vs.VideoNode, prefetch: int) -> tuple[int, float]:
    iterator = node.frames(prefetch=prefetch) if prefetch > 0 else node.frames()
    count = 0
    checksum = 0.0
    for frame in iterator:
        checksum += checksum_frame(frame)
        count += 1
    return count, checksum


def warmup_node(node: vs.VideoNode, frames: int) -> None:
    for frame_number in range(min(frames, node.num_frames)):
        frame = node.get_frame(frame_number)
        checksum_frame(frame)


def sample_output_for_node(path: Path, node: int) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.node{node}{path.suffix}")
    return path / f"node{node}.sample.txt"


def start_sample(sample_seconds: int, output: Path) -> subprocess.Popen[bytes] | None:
    if sample_seconds <= 0:
        return None

    sample = shutil.which("sample")
    if sample is None:
        raise RuntimeError("sample command was not found")

    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sample, str(os.getpid()), str(sample_seconds), "-file", str(output)]
    )


def benchmark_node(node_index: int, node: vs.VideoNode, args: argparse.Namespace) -> None:
    if args.warmup > 0:
        warmup_node(node, args.warmup)

    sample_path = sample_output_for_node(args.sample_output, node_index)
    sample_process = start_sample(args.sample_seconds, sample_path)
    start = time.perf_counter()
    count, checksum = render_node(node, args.prefetch)
    elapsed = time.perf_counter() - start
    if sample_process is not None:
        sample_process.wait()

    print(
        f"node={node_index} name={NODE_NAMES[node_index]!r} "
        f"rendered={count} elapsed={elapsed:.3f}s fps={count / elapsed:.4f} "
        f"checksum={checksum:.9g}",
        flush=True,
    )
    if sample_process is not None:
        print(f"sample={sample_path}", flush=True)


def main() -> int:
    args = parse_args()
    core = vs.core
    if args.threads > 0:
        core.num_threads = args.threads
    if args.max_cache_mb > 0:
        core.max_cache_size = args.max_cache_mb

    outputs = build_outputs(core, args)
    print(
        "benchmark-example "
        f"pid={os.getpid()} threads={core.num_threads} "
        f"max_cache_mb={core.max_cache_size} prefetch={args.prefetch} "
        f"source={args.source} start={args.start} frames={args.frames} middle={args.middle} "
        f"source_filter={args.source_filter} nodes={args.nodes} "
        f"mvtools_opt={args.mvtools_opt}",
        flush=True,
    )
    for node_index in args.nodes:
        benchmark_node(node_index, outputs[node_index], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
