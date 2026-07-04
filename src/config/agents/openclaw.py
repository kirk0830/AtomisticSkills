"""OpenClaw TUI/bot framework configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..base import PROJECT_ROOT, write_json, INSTRUCTION_STUB
from ..mcp_loader import load_mcp_servers


SKILLS_SRC = PROJECT_ROOT / ".agents" / "skills"
WORKFLOWS_SRC = PROJECT_ROOT / ".agents" / "workflows"
RULES_SRC = PROJECT_ROOT / ".agents" / "rules"


def detect_openclaw_workspace(cli_path: str | None = None) -> Path:
    """Return the OpenClaw workspace directory, or raise if not found."""
    if cli_path:
        return Path(cli_path).expanduser().resolve()

    env_path = None
    raw_env = os.environ.get("OPENCLAW_WORKSPACE")
    if raw_env:
        env_path = Path(raw_env).expanduser()

    candidates = [
        env_path,
        Path(".").resolve(),
        Path.home() / ".openclaw" / "workspace",
        Path.home() / "openclaw" / "workspace",
    ]

    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not detect OpenClaw workspace directory. "
        "Please pass it explicitly with --data-dir."
    )


def configure(servers: dict[str, Any], args: argparse.Namespace) -> int:
    """Configure OpenClaw TUI/bot framework."""
    if args.skills_only and args.mcp_only:
        print("Error: --skills-only and --mcp-only are mutually exclusive.", file=sys.stderr)
        return 1

    try:
        workspace_dir = detect_openclaw_workspace(args.data_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OpenClaw workspace] {workspace_dir}")

    if not args.mcp_only:
        _setup_openclaw_skills(workspace_dir)
        _setup_openclaw_instructions(workspace_dir)

    if not args.skills_only:
        _setup_openclaw_mcp(workspace_dir, servers, args)

    print()
    print("Done! OpenClaw is now configured with AtomisticSkills.")
    print("Launch OpenClaw with: openclaw --workspace", str(workspace_dir))
    return 0


def _setup_openclaw_skills(workspace_dir: Path) -> None:
    """Create symlinks for skills, workflows, and rules in OpenClaw workspace."""
    skills_dir = workspace_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    workflows_dir = workspace_dir / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    rules_dir = workspace_dir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)

    from ..base import symlink_dir_contents

    skills_linked = symlink_dir_contents(SKILLS_SRC, skills_dir)
    workflows_linked = symlink_dir_contents(WORKFLOWS_SRC, workflows_dir)
    rules_linked = symlink_dir_contents(RULES_SRC, rules_dir)

    print(f"[skills] Linked {skills_linked} skills to {skills_dir}")
    print(f"[workflows] Linked {workflows_linked} workflows to {workflows_dir}")
    print(f"[rules] Linked {rules_linked} rules to {rules_dir}")


def _setup_openclaw_instructions(workspace_dir: Path) -> None:
    """Create a quick-start instruction file for OpenClaw."""
    readme_path = workspace_dir / "ATOMISTICSKILLS_README.md"
    content = f"""\
# AtomisticSkills + OpenClaw Quick Start

## Launch OpenClaw
```bash
cd {workspace_dir}
openclaw tui
```

## Load Skills and Rules
Once inside the session, tell the agent:
```
absorb all of {PROJECT_ROOT} into context. most crucially is the files in {PROJECT_ROOT}/.agents — target the skills/, workflows/, and rules/ folders within .agent/
```

## Register MCP Servers
```
load all mcp tools from the mcp_config.json file in {PROJECT_ROOT}
```

## Test a Query
```
Search the Materials Project for the stable structure of LiFePO4 and report its bandgap.
```

## Key Paths
- Skills: {PROJECT_ROOT}/.agents/skills/
- Workflows: {PROJECT_ROOT}/.agents/workflows/
- Rules: {PROJECT_ROOT}/.agents/rules/
- MCP Config: {PROJECT_ROOT}/mcp_config.json
"""
    readme_path.write_text(content, encoding="utf-8")
    print(f"[instructions] Created quick-start guide -> {readme_path}")


def _setup_openclaw_mcp(workspace_dir: Path, servers: dict[str, Any], args: argparse.Namespace) -> int:
    """Configure MCP servers for OpenClaw."""
    config_dir = workspace_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    mcporter_config = config_dir / "mcporter.json"

    mcporter_data = {}
    if mcporter_config.exists():
        try:
            with open(mcporter_config) as fh:
                mcporter_data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    mcporter_data.setdefault("servers", {})
    mcporter_data["servers"].update(servers)

    write_json(mcporter_config, mcporter_data)
    print(f"[mcp] Wrote MCP config -> {mcporter_config}")

    if args.write_mcp_config:
        backup_config = config_dir / "atomisticskills_mcp.json"
        write_json(backup_config, {"mcpServers": servers})
        print(f"[mcp] Wrote backup config -> {backup_config}")

    if shutil.which("mcporter"):
        result = subprocess.run(
            ["mcporter", "config", "import", str(mcporter_config)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[mcp] Registered servers via mcporter")
        else:
            print(
                f"[WARN] mcporter config import failed (may need manual registration):\n"
                f"{result.stderr}",
                file=sys.stderr,
            )

    return 0