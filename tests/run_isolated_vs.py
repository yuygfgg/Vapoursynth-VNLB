#!/usr/bin/env python3
"""Run a VapourSynth test without auto-loading the installed VNLB plugin."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import vapoursynth as vs


class IsolatedPolicy(vs.EnvironmentPolicy):
    def __init__(self, discover_plugins: bool) -> None:
        self.discover_plugins = discover_plugins

    def on_policy_registered(self, api: vs.EnvironmentPolicyAPI) -> None:
        self.api = api
        flags = 0 if self.discover_plugins else int(vs.DISABLE_AUTO_LOADING)
        self.environment = api.create_environment(flags)

    def get_current_environment(self) -> vs.EnvironmentData | None:
        return self.environment

    def set_environment(
        self, environment: vs.EnvironmentData | None
    ) -> vs.EnvironmentData | None:
        previous = self.environment
        self.environment = environment
        return previous

    def close(self) -> None:
        if self.environment is not None:
            self.api.destroy_environment(self.environment)
            self.environment = None
        self.api.unregister_policy()

    def preserve_plugins(self, namespaces: list[str]) -> None:
        paths: dict[str, str] = {}
        for plugin in vs.core.plugins():
            if plugin.namespace in namespaces and plugin.plugin_path is not None:
                paths[plugin.namespace] = plugin.plugin_path

        missing = [namespace for namespace in namespaces if namespace not in paths]
        if missing:
            raise RuntimeError(
                "required auto-loaded VapourSynth plugins were not found: "
                + ", ".join(missing)
            )

        old_environment = self.environment
        self.environment = self.api.create_environment(int(vs.DISABLE_AUTO_LOADING))
        self.api.destroy_environment(old_environment)
        for namespace in namespaces:
            vs.core.std.LoadPlugin(path=paths[namespace])


def parse_arguments() -> tuple[list[str], Path, list[str]]:
    arguments = sys.argv[1:]
    preserved: list[str] = []
    while arguments[:1] == ["--preserve-plugin"]:
        if len(arguments) < 2:
            raise SystemExit("--preserve-plugin requires a namespace")
        preserved.append(arguments[1])
        arguments = arguments[2:]

    if not arguments:
        raise SystemExit(
            "usage: run_isolated_vs.py "
            "[--preserve-plugin namespace]... script.py [arguments ...]"
        )
    return preserved, Path(arguments[0]).resolve(), arguments[1:]


def main() -> None:
    preserved, script, script_arguments = parse_arguments()
    policy = IsolatedPolicy(bool(preserved))
    vs.register_policy(policy)
    try:
        if preserved:
            policy.preserve_plugins(preserved)
        sys.argv = [str(script), *script_arguments]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
