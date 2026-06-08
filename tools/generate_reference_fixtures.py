#!/usr/bin/env python3
"""Generate small VNLB reference fixtures from the pinned reference binary."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "3461d9d3b8d31baa0eb35e5acc60bedd283f1dd3"
REFERENCE_BINARY = ROOT / "third_party/reference/build/pariasm-vnlb/bin/vnlbayes"
REFERENCE_PATCH = ROOT / "third_party/reference/patches/pariasm-vnlb-macos-build.patch"
ASSETS = ROOT / "tests/assets/downloads"
CASES = ROOT / "tests/reference_cases"


@dataclass(frozen=True)
class VnlbParams:
    sigma: float = 12.0
    block_size: int = 4
    patch_time: int = 2
    bm_range: int = 3
    radius: int = 6
    group_size: int = 40
    rank: int = 16
    beta: float = 3.0
    basic_variance_threshold: float = 1.1
    final_variance_threshold: float = 1.8666666667
    final_tau_8bit: float = 400.0


@dataclass(frozen=True)
class Case:
    name: str
    frames: np.ndarray
    source: str
    params: VnlbParams = VnlbParams()


def ensure_gray_float(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError(f"expected T,H,W grayscale frames, got {frames.shape}")
    return np.clip(frames, 0.0, 1.0).astype(np.float32)


def resize_gray(image: Image.Image, size: int) -> np.ndarray:
    image = image.convert("L")
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def add_noise(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, sigma / 255.0, size=clean.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def synthetic_cases() -> list[Case]:
    h = w = 40
    t = 5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    constant = np.full((t, h, w), 0.42, dtype=np.float32)

    gradient = []
    for frame in range(t):
        gradient.append(
            0.15 + 0.70 * ((xx + yy + frame * 2.0) / (2.0 * (w - 1) + t * 2.0))
        )
    gradient = np.stack(gradient).astype(np.float32)

    moving_square = np.full((t, h, w), 0.18, dtype=np.float32)
    for frame in range(t):
        x0 = 6 + frame * 3
        y0 = 10 + frame
        moving_square[frame, y0 : y0 + 12, x0 : x0 + 12] = 0.78
        moving_square[frame, y0 + 3 : y0 + 8, x0 + 3 : x0 + 9] = 0.52

    low_texture = np.full((t, h, w), 0.50, dtype=np.float32)
    low_texture += 0.03 * np.sin((xx + yy)[None, :, :] / 7.0)
    low_texture += np.linspace(-0.015, 0.015, t, dtype=np.float32)[:, None, None]

    return [
        Case("constant_gray", ensure_gray_float(constant), "synthetic constant gray"),
        Case(
            "gradient_gray", ensure_gray_float(gradient), "synthetic diagonal gradient"
        ),
        Case(
            "moving_square_gray",
            ensure_gray_float(moving_square),
            "synthetic moving square",
        ),
        Case(
            "low_texture_gray",
            ensure_gray_float(low_texture),
            "synthetic low-texture sinusoid",
        ),
    ]


def downloaded_image_case() -> Case:
    image = Image.open(ASSETS / "earthrise.jpg")
    base = resize_gray(image, 48)
    frames = []
    for frame in range(5):
        shifted = np.roll(base, shift=frame - 2, axis=1)
        frames.append(shifted)
    return Case(
        "earthrise_gray",
        ensure_gray_float(np.stack(frames)),
        "NASA Apollo 8 Earthrise image via Wikimedia Commons public-domain file redirect",
    )


def downloaded_video_case() -> Case:
    video_path = ASSETS / "wmap-990307_1280.mp4"
    raw = iio.imread(video_path, index=None)
    if raw.ndim != 4:
        raise ValueError(f"expected video frames, got {raw.shape}")
    sample_indices = np.linspace(0, len(raw) - 1, 5, dtype=np.int64)
    frames = []
    for idx in sample_indices:
        image = Image.fromarray(raw[int(idx), :, :, :3])
        frames.append(resize_gray(image, 48))
    return Case(
        "wmap_video_gray",
        ensure_gray_float(np.stack(frames)),
        "NASA WMAP mission MP4 asset",
    )


def write_pfm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"expected grayscale PFM image, got {image.shape}")
    with path.open("wb") as f:
        f.write(f"Pf\n{image.shape[1]} {image.shape[0]}\n-1\n".encode("ascii"))
        np.ascontiguousarray(image, dtype="<f4").tofile(f)


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"Pf":
            raise ValueError(f"{path}: expected grayscale PFM, got {magic!r}")
        width_text, height_text = f.readline().split()
        width = int(width_text)
        height = int(height_text)
        scale = float(f.readline())
        dtype = "<f4" if scale < 0.0 else ">f4"
        data = np.fromfile(f, dtype=dtype, count=width * height)
    if data.size != width * height:
        raise ValueError(f"{path}: truncated PFM data")
    return data.reshape((height, width)).astype(np.float32, copy=False)


def save_reference_sequence(frames: np.ndarray, pattern_dir: Path) -> None:
    pattern_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        f32 = np.clip(frame * 255.0, 0.0, 255.0).astype(np.float32)
        write_pfm(pattern_dir / f"frame_{idx:03d}.pfm", f32)


def load_reference_sequence(pattern_dir: Path, count: int) -> np.ndarray:
    frames = []
    for idx in range(count):
        arr = read_pfm(pattern_dir / f"frame_{idx:03d}.pfm")
        frames.append(arr / 255.0)
    return np.stack(frames).astype(np.float32)


def run_reference(
    case: Case, noisy: np.ndarray, work: Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    params = case.params
    input_dir = work / "input"
    basic_dir = work / "basic"
    final_dir = work / "final"
    save_reference_sequence(noisy, input_dir)
    basic_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    command = [
        str(REFERENCE_BINARY),
        "-i",
        str(input_dir / "frame_%03d.pfm"),
        "-f",
        "0",
        "-l",
        str(noisy.shape[0] - 1),
        "-sigma",
        f"{params.sigma:g}",
        "-px1",
        str(params.block_size),
        "-pt1",
        str(params.patch_time),
        "-px2",
        str(params.block_size),
        "-pt2",
        str(params.patch_time),
        "-wx1",
        str(2 * params.bm_range + 1),
        "-wx2",
        str(2 * params.bm_range + 1),
        "-wt1",
        str(params.radius),
        "-wt2",
        str(params.radius),
        "-np1",
        str(params.group_size),
        "-np2",
        str(params.group_size),
        "-r1",
        str(params.rank),
        "-r2",
        str(params.rank),
        "-b1",
        f"{params.beta:g}",
        "-b2",
        f"{params.beta:g}",
        "-verbose",
        "false",
        "-bsic",
        str(basic_dir / "frame_%03d.pfm"),
        "-deno",
        str(final_dir / "frame_%03d.pfm"),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    basic = load_reference_sequence(basic_dir, noisy.shape[0])
    final = load_reference_sequence(final_dir, noisy.shape[0])
    return basic, final, command


def write_case_toml(path: Path, case: Case, command: list[str]) -> None:
    params = case.params
    command_text = ",\n".join(f"  {json.dumps(part)}" for part in command)
    path.write_text(textwrap.dedent(f"""\
            name = "{case.name}"
            source = "{case.source}"
            format = "Gray float32 PFM reference I/O, stored as float32 [0, 1]"
            width = {case.frames.shape[2]}
            height = {case.frames.shape[1]}
            frames = {case.frames.shape[0]}

            [noise]
            sigma_8bit = {params.sigma:g}
            seed = {stable_seed(case.name)}

            [vnlb]
            block_size = {params.block_size}
            patch_time = {params.patch_time}
            bm_range = {params.bm_range}
            radius = {params.radius}
            group_size = {params.group_size}
            rank = {params.rank}
            beta = {params.beta:g}
            basic_variance_threshold = {params.basic_variance_threshold:g}
            final_variance_threshold = {params.final_variance_threshold:g}
            final_tau_8bit = {params.final_tau_8bit:g}

            [reference]
            implementation = "pariasm/vnlb"
            commit = "{REFERENCE_COMMIT}"
            local_patch = "{REFERENCE_PATCH.relative_to(ROOT)}"
            binary = "{REFERENCE_BINARY.relative_to(ROOT)}"
            command = [
            {command_text}
            ]

            [tolerance]
            max_abs = 0.15
            mean_abs = 0.004
            rmse = 0.006
            """))


def stable_seed(name: str) -> int:
    seed = 2166136261
    for byte in name.encode("utf-8"):
        seed ^= byte
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def generate(keep_work: bool) -> None:
    if not REFERENCE_BINARY.exists():
        raise FileNotFoundError(f"reference binary not found: {REFERENCE_BINARY}")
    CASES.mkdir(parents=True, exist_ok=True)

    cases = synthetic_cases() + [downloaded_image_case(), downloaded_video_case()]
    for case in cases:
        case_dir = CASES / case.name
        work = case_dir / "_reference_work"
        if work.exists():
            shutil.rmtree(work)
        case_dir.mkdir(parents=True, exist_ok=True)
        clean = case.frames
        noisy = add_noise(clean, case.params.sigma, stable_seed(case.name))
        basic, final, command = run_reference(case, noisy, work)

        np.savez_compressed(case_dir / "input.npz", noisy=noisy, clean=clean)
        np.savez_compressed(case_dir / "basic_ref.npz", basic=basic)
        np.savez_compressed(case_dir / "final_ref.npz", final=final)
        write_case_toml(case_dir / "case.toml", case, command)

        if not keep_work:
            shutil.rmtree(work)
        print(
            f"generated {case.name}: {clean.shape[0]}x{clean.shape[1]}x{clean.shape[2]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-work", action="store_true", help="keep temporary PNG sequences"
    )
    args = parser.parse_args()
    generate(keep_work=args.keep_work)


if __name__ == "__main__":
    main()
