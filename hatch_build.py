from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import mac_platforms


class CustomBuildHook(BuildHookInterface):
    PLUGIN_DIR = Path("vapoursynth") / "plugins" / "vnlb"

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        if self.target_name != "wheel":
            return

        root = Path(self.root).resolve()
        stage_dir = root / self.PLUGIN_DIR

        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = f"py3-none-{self._platform_tag()}"

        self._clean_stage(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        build_dir_name = self._build_dir_name()
        cmake_args = [
            self._cmake(),
            "-S",
            str(root),
            "-G",
            env.get("CMAKE_GENERATOR", "Ninja"),
            "-D",
            "CMAKE_BUILD_TYPE=Release",
            "-D",
            "VNLB_BUILD_PLUGIN=ON",
            "-D",
            "VNLB_ENABLE_TESTS=OFF",
            "-D",
            f"Python3_EXECUTABLE={sys.executable}",
            "-D",
            f"VapourSynth_INCLUDE_DIR={self._vapoursynth_include_dir()}",
        ]

        if sys.platform == "darwin":
            deployment_target = env.setdefault("MACOSX_DEPLOYMENT_TARGET", "12.0")
            arch = (
                env.get("CMAKE_OSX_ARCHITECTURES")
                or env.get("CIBW_ARCHS_MACOS")
                or platform.machine()
                or "arm64"
            )
            arch = arch.replace(";", " ").split()[0]
            cmake_args.extend(
                [
                    "-D",
                    f"CMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}",
                    "-D",
                    f"CMAKE_OSX_ARCHITECTURES={arch}",
                ]
            )
        cmake_args.extend(shlex.split(env.get("CMAKE_ARGS", "")))
        build_dir = root / "build" / build_dir_name
        build_dir.mkdir(parents=True, exist_ok=True)
        cmake_args[3:3] = ["-B", str(build_dir)]

        subprocess.run(cmake_args, cwd=root, env=env, check=True)
        subprocess.run(
            [
                self._cmake(),
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "--target",
                "vnlb",
            ],
            cwd=root,
            env=env,
            check=True,
        )

        shutil.copy2(root / "packaging" / "manifest.vs", stage_dir / "manifest.vs")
        shutil.copy2(
            self._find_plugin_library(build_dir), stage_dir / self._library_name()
        )

    def clean(self, versions: list[str]) -> None:
        self._clean_stage(Path(self.root) / self.PLUGIN_DIR)

    @staticmethod
    def _clean_stage(stage_dir: Path) -> None:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)

    @staticmethod
    def _vapoursynth_include_dir() -> Path:
        command = [
            sys.executable,
            "-c",
            "import vapoursynth; print(vapoursynth.get_include(), end='')",
        ]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        include_dir = Path(completed.stdout)
        if not include_dir.is_dir():
            raise RuntimeError(
                f"VapourSynth include directory does not exist: {include_dir}"
            )
        return include_dir

    @staticmethod
    def _library_name() -> str:
        if sys.platform == "darwin":
            return "libvnlb.dylib"
        if sys.platform == "win32":
            return "libvnlb.dll"
        return "libvnlb.so"

    def _platform_tag(self) -> str:
        if sys.platform == "darwin":
            deployment_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET", "12.0")
            major, minor = (int(part) for part in deployment_target.split(".")[:2])
            arch = (
                os.environ.get("CMAKE_OSX_ARCHITECTURES")
                or os.environ.get("CIBW_ARCHS_MACOS")
                or platform.machine()
            )
            arch = arch.replace(";", " ").split()[0]
            return next(mac_platforms((major, minor), arch))

        if sys.platform.startswith("linux"):
            auditwheel_plat = os.environ.get("AUDITWHEEL_PLAT")
            if auditwheel_plat:
                return auditwheel_plat

        return sysconfig.get_platform().replace("-", "_").replace(".", "_")

    def _build_dir_name(self) -> str:
        return "wheel-" + self._platform_tag().replace("-", "_").replace(".", "_")

    def _cmake(self) -> str:
        return os.environ.get("CMAKE", "cmake")

    def _find_plugin_library(self, build_dir: Path) -> Path:
        library = build_dir / self._library_name()
        if library.exists():
            return library

        candidates = sorted(build_dir.rglob(self._library_name()))
        if candidates:
            return candidates[0]

        raise FileNotFoundError(f"could not find built VNLB plugin under {build_dir}")
