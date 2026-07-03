"""MCP server configuration loader and path patching."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .base import PROJECT_ROOT

ENV_PATTERN = re.compile(r".*/envs/([^/]+)/bin/python$")
PIXI_ENV_PATTERN = re.compile(r".*/\.pixi/envs/([^/]+)/bin/python$")
MCP_SOURCE = PROJECT_ROOT / "mcp_config.json"


def detect_pixi_project_root() -> str | None:
    """Detect Pixi project root by looking for pixi.toml."""
    candidate = PROJECT_ROOT / "pixi.toml"
    if candidate.is_file():
        return str(PROJECT_ROOT)
    return None


def detect_conda_base() -> str | None:
    """Detect conda/mamba base directory."""
    for cmd in ("conda", "mamba", "micromamba"):
        try:
            result = subprocess.run(
                [cmd, "info", "--base"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                base = result.stdout.strip()
                if base and Path(base).is_dir():
                    return base
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    for name in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        candidate = Path.home() / name
        if candidate.is_dir():
            return str(candidate)

    return None


def _rewrite_env_paths(
    env: dict,
    project_root: str,
    env_name: str | None,
    env_prefix: str | None,
) -> None:
    """Rewrite PYTHONPATH, CONDA_PREFIX, PATH etc. for an environment."""
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = project_root

    if env_name is None or env_prefix is None:
        return

    if "CONDA_PREFIX" in env:
        env["CONDA_PREFIX"] = env_prefix
    if "TRITON_PTXAS_BLACKWELL_PATH" in env:
        env["TRITON_PTXAS_BLACKWELL_PATH"] = f"{env_prefix}/bin/ptxas"
    if "PATH" in env:
        bin_dir = f"{env_prefix}/bin"
        env["PATH"] = re.sub(
            r"[^ ]*?/(?:envs|\.pixi/envs)/[^/]+/bin",
            bin_dir,
            env["PATH"],
            count=1,
        )


def load_mcp_servers(
    conda_base: str | None = None,
    pixi_root: str | None = None,
) -> dict[str, Any]:
    """Load mcp_config.json and rewrite env paths for this machine.

    Supports both conda and pixi environments. Pixi takes priority if both
    are available (pixi.toml present in project root).
    """
    with open(MCP_SOURCE) as fh:
        config = json.load(fh)

    project_root = str(PROJECT_ROOT)
    use_pixi = pixi_root is not None

    for server in config.get("mcpServers", {}).values():
        cmd = server.get("command", "")

        env_name: str | None = None
        env_prefix: str | None = None

        pixi_match = PIXI_ENV_PATTERN.search(cmd) or "PIXI_PROJECT" in cmd
        if use_pixi and pixi_match:
            m = re.search(r"\.pixi/envs/([^/]+)/bin/python", cmd)
            if not m:
                m = re.search(r"PIXI_PROJECT/\.pixi/envs/([^/]+)/bin/python", cmd)
            if m:
                env_name = m.group(1)
                env_prefix = f"{pixi_root}/.pixi/envs/{env_name}"
                server["command"] = f"{env_prefix}/bin/python"
        elif conda_base:
            match = ENV_PATTERN.match(cmd)
            if match:
                env_name = match.group(1)
                env_prefix = f"{conda_base}/envs/{env_name}"
                server["command"] = f"{env_prefix}/bin/python"

        _rewrite_env_paths(
            server.get("env", {}), project_root, env_name, env_prefix
        )

    return config.get("mcpServers", {})
