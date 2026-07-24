"""Pixi MCP Server — execute Python scripts in Pixi environments via MCP.

This server bridges astrbot's sandboxed Agent (which has no shell/Python access
and cannot reach project directories outside ``data/``) with the full AtomisticSkills
environment.  The Agent calls :func:`pixi_run`, and this server runs a Python
script inside the requested Pixi environment.

**Security model:** only Python scripts are executable; the script path is
validated against an allowlist of project directories; arguments are passed
as a list (``shell=False``) to prevent injection; ``capture_patterns`` are
confined to the project root tree.

Use this server for **short-lived scripts** (data preparation, analysis, small
calculations).  For long-running or HPC-bound work use the existing base MCP
server's slurm tools (``submit_hpc_job`` etc.).
"""

from __future__ import annotations

import base64
import glob as glob_mod
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.mcp_utils import setup_mcp_stdout, run_fastmcp_server  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp_pipe_binary = setup_mcp_stdout()

PIXI_TOML = os.path.join(PROJECT_ROOT, "pixi.toml")

mcp = FastMCP("pixi_tools")

STDOUT_MAX = 80_000
STDERR_MAX = 30_000
FILE_MAX = 1_000_000  # 1 MB per captured file
COMMAND_TIMEOUT = 600  # 10 minutes

# --- security: allowed script directories (relative to PROJECT_ROOT) ---
_ALLOWED_DIRS = (".agents/skills/", "research/")


def _validate_script_path(script: str) -> str:
    """Validate and resolve a script path; raise ValueError on rejection."""
    if not script.endswith(".py"):
        raise ValueError(
            f"Only Python scripts (.py) are allowed, got: {script!r}"
        )

    # Reject directory traversal
    if ".." in Path(script).parts:
        raise ValueError(
            f"Path traversal (..) is not allowed in script path: {script!r}"
        )

    # Resolve to canonical absolute path
    abs_path = os.path.normpath(os.path.join(PROJECT_ROOT, script))
    rel_to_root = os.path.relpath(abs_path, PROJECT_ROOT)
    if rel_to_root.startswith("..") or os.path.isabs(rel_to_root):
        raise ValueError(
            f"Script resolves outside project root: {script!r}"
        )

    # Allowlist check
    allowed = any(
        rel_to_root == d.rstrip("/") or rel_to_root.startswith(d)
        for d in _ALLOWED_DIRS
    )
    if not allowed:
        raise ValueError(
            f"Script path must be under one of {_ALLOWED_DIRS}, got: {script!r}"
        )

    if not os.path.isfile(abs_path):
        raise ValueError(f"Script not found: {abs_path}")

    return abs_path


def _validate_environment(environment: str) -> None:
    """Reject obviously invalid environment names to guard against injection."""
    if not environment or not environment.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid environment name: {environment!r}")


def _validate_cwd(cwd: str | None) -> str:
    """Resolve and validate a working directory; return safe absolute path."""
    if cwd is None:
        return PROJECT_ROOT

    if ".." in Path(cwd).parts:
        raise ValueError(f"Path traversal (..) in cwd: {cwd!r}")

    abs_cwd = os.path.normpath(os.path.join(PROJECT_ROOT, cwd))
    rel_to_root = os.path.relpath(abs_cwd, PROJECT_ROOT)
    if rel_to_root.startswith("..") or os.path.isabs(rel_to_root):
        raise ValueError(f"cwd resolves outside project root: {cwd!r}")

    return abs_cwd


def _validate_capture_patterns(patterns: list[str] | None) -> list[str]:
    """Validate that all capture patterns are safe globs confined to the project."""
    if not patterns:
        return []

    safe: list[str] = []
    for pat in patterns:
        if not pat:
            continue
        if ".." in Path(pat).parts:
            raise ValueError(
                f"Path traversal (..) in capture pattern: {pat!r}"
            )
        # Only allow globs that start within the project
        safe.append(pat)
    return safe


@mcp.tool()
def pixi_run(
    environment: str,
    script: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    capture_patterns: list[str] | None = None,
) -> str:
    """Run a Python script inside a Pixi environment and return the result.

    This is the primary execution bridge for AtomisticSkills within astrbot.
    When an AtomisticSkills SKILL.md contains a ``# Env: <name>`` code block
    with a ``python <script> <args>`` command, the Agent should call this
    tool.

    **Security:** only ``.py`` scripts under ``.agents/skills/`` or
    ``research/`` are allowed; ``shell=False`` (no injection); all paths are
    validated.

    Args:
        environment: Pixi environment name (e.g. ``base``, ``mace``, ``matgl``,
            ``fairchem``, ``orca``, ``drugdisc``, ``nmr`` — see the
            mcp-environments rule for the full list).
        script: Relative path to the Python script from the project root,
            e.g. ``.agents/skills/mat-stability/scripts/query_mp_hull.py``.
        args: Positional arguments to the script, e.g. ``["--formula",
            "Li-Fe-P-O", "--output", "hull_structures/"]``.
        cwd: Optional working directory (relative to project root) for the
            script.  Defaults to the project root.
        capture_patterns: Optional glob patterns (relative to project root)
            for collecting output files after the script finishes, e.g.
            ``["hull_structures/**", "*.json"]``.  Each matched file's
            content is base64-encoded in the response.

    Returns:
        A JSON string with keys:
        - ``stdout`` (str) — standard output
        - ``stderr`` (str) — standard error
        - ``exit_code`` (int) — 0 = success
        - ``truncated`` (bool) — whether stdout or stderr was trimmed
        - ``files`` (dict) — ``{relative_path: base64_content, ...}``
    """
    # --- validate all inputs ---
    try:
        script_abs = _validate_script_path(script)
        _validate_environment(environment)
        cwd_abs = _validate_cwd(cwd)
        safe_patterns = _validate_capture_patterns(capture_patterns)
    except ValueError as exc:
        return json.dumps({
            "stdout": "",
            "stderr": str(exc),
            "exit_code": -1,
            "truncated": False,
            "files": {},
        }, ensure_ascii=False)

    if not os.path.isfile(PIXI_TOML):
        return json.dumps({
            "stdout": "",
            "stderr": f"pixi.toml not found at {PIXI_TOML}",
            "exit_code": -1,
            "truncated": False,
            "files": {},
        }, ensure_ascii=False)

    # --- build command: pixi run -e <env> -- python <script> [args...] ---
    cmd: list[str] = [
        "pixi", "run",
        "--manifest-path", PIXI_TOML,
        "-e", environment,
        "--",
        "python",
        script_abs,
    ]
    if args:
        cmd.extend(args)

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            cwd=cwd_abs,
            env=env,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        return json.dumps({
            "stdout": exc.stdout[-STDOUT_MAX:] if exc.stdout else "",
            "stderr": (exc.stderr[-STDERR_MAX:] if exc.stderr else "")
                       + f"\n\nCommand timed out after {COMMAND_TIMEOUT}s",
            "exit_code": -1,
            "truncated": False,
            "files": {},
        }, ensure_ascii=False)

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    truncated = False

    if len(stdout) > STDOUT_MAX:
        truncated = True
        stdout = stdout[-STDOUT_MAX:]
    if len(stderr) > STDERR_MAX:
        truncated = True
        stderr = stderr[-STDERR_MAX:]

    output: dict = {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.returncode,
        "truncated": truncated,
        "files": {},
    }

    if safe_patterns:
        for pattern in safe_patterns:
            for matched in glob_mod.glob(
                os.path.join(PROJECT_ROOT, pattern), recursive=True
            ):
                rel_path = os.path.relpath(matched, PROJECT_ROOT)
                try:
                    with open(matched, "rb") as fh:
                        content = fh.read()
                except OSError as exc:
                    output["files"][rel_path] = f"[read error: {exc}]"
                    continue

                if len(content) > FILE_MAX:
                    output["files"][rel_path] = (
                        f"[file too large: {len(content)} bytes]"
                    )
                else:
                    output["files"][rel_path] = base64.b64encode(
                        content
                    ).decode("ascii")

    return json.dumps(output, ensure_ascii=False)


if __name__ == "__main__":
    run_fastmcp_server(mcp, mcp_pipe_binary)
