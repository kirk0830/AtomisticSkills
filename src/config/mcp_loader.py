"""MCP server configuration loader and path patching."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import PROJECT_ROOT

PIXI_ENV_PATTERN = re.compile(r".*/\.pixi/envs/([^/]+)/bin/python$")
MCP_SOURCE = PROJECT_ROOT / "mcp_config.json"


def detect_pixi_project_root() -> str | None:
    """Detect Pixi project root by looking for pixi.toml."""
    candidate = PROJECT_ROOT / "pixi.toml"
    if candidate.is_file():
        return str(PROJECT_ROOT)
    return None


def _rewrite_env_paths(
    env: dict,
    project_root: str,
    env_name: str | None,
    env_prefix: str | None,
) -> None:
    """Rewrite PYTHONPATH, PATH etc. for a Pixi environment."""
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = project_root

    if env_name is None or env_prefix is None:
        return

    if "TRITON_PTXAS_BLACKWELL_PATH" in env:
        env["TRITON_PTXAS_BLACKWELL_PATH"] = f"{env_prefix}/bin/ptxas"
    if "PATH" in env:
        bin_dir = f"{env_prefix}/bin"
        env["PATH"] = re.sub(
            r"[^ ]*?/\.pixi/envs/[^/]+/bin",
            bin_dir,
            env["PATH"],
            count=1,
        )


def load_mcp_servers(
    pixi_root: str | None = None,
) -> dict[str, Any]:
    """Load mcp_config.json and rewrite env paths for this machine.

    Uses Pixi environments (pixi.toml must be present in project root).
    """
    with open(MCP_SOURCE) as fh:
        config = json.load(fh)

    project_root = str(PROJECT_ROOT)

    for server in config.get("mcpServers", {}).values():
        cmd = server.get("command", "")

        env_name: str | None = None
        env_prefix: str | None = None

        if pixi_root is not None:
            m = re.search(r"\.pixi/envs/([^/]+)/bin/python", cmd)
            if not m:
                m = re.search(r"PIXI_PROJECT/\.pixi/envs/([^/]+)/bin/python", cmd)
            if m:
                env_name = m.group(1)
                env_prefix = f"{pixi_root}/.pixi/envs/{env_name}"
                server["command"] = f"{env_prefix}/bin/python"

        _rewrite_env_paths(
            server.get("env", {}), project_root, env_name, env_prefix
        )

    return config.get("mcpServers", {})
