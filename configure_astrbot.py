#!/usr/bin/env python3
"""Configure AtomisticSkills for AstrBot.

AstrBot (https://docs.astrbot.app/) is a chatbot framework that sandboxes the
LLM agent to its own ``data/`` directory.  This script bridges that gap by:

1. Symlinking every project skill into ``<astrbot-data>/skills/`` so the agent
   can discover SKILL.md files.
2. Writing an index ``atomisticskills/SKILL.md`` that explains the framework.
3. Printing (and optionally saving) ready-to-paste MCP server JSON configs,
   with absolute paths rewritten for the local machine.

Usage:
    python configure_astrbot.py
    python configure_astrbot.py --data-dir /path/to/astrbot/data
    python configure_astrbot.py --data-dir /path/to/astrbot/data --use-uv
    python configure_astrbot.py --data-dir /path/to/astrbot/data --skills-only
    python configure_astrbot.py --data-dir /path/to/astrbot/data --mcp-only
    python configure_astrbot.py --data-dir /path/to/astrbot/data --write-mcp-config
    python configure_astrbot.py --list-servers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import configure_mcp

PROJECT_ROOT = Path(__file__).resolve().parent
SKILLS_SRC = PROJECT_ROOT / ".agents" / "skills"

INDEX_SKILL_NAME = "atomisticskills"


# ---------------------------------------------------------------------------
# Helpers: symlinks
# ---------------------------------------------------------------------------


def _symlink_target(path: Path) -> Path:
    """Return the absolute target path for a symlink."""
    target = path.readlink()
    if not target.is_absolute():
        target = path.parent / target
    return target


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is inside parent without requiring it to exist."""
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _remove_stale_project_skill_symlinks(
    skills_dir: Path,
    project_skills_dir: Path,
) -> int:
    """Remove symlinks that point to removed project skills."""
    removed = 0
    if not skills_dir.exists():
        return removed

    for entry in skills_dir.iterdir():
        if not entry.is_symlink():
            continue
        target = _symlink_target(entry)
        if _is_relative_to(target, project_skills_dir) and not target.exists():
            entry.unlink()
            removed += 1

    return removed


# ---------------------------------------------------------------------------
# AstrBot data directory detection
# ---------------------------------------------------------------------------


def detect_astrbot_data_dir(cli_path: str | None) -> Path:
    """Return the AstrBot data directory, or raise if not found."""
    if cli_path:
        return Path(cli_path).expanduser().resolve()

    env_path = None
    raw_env = os.environ.get("ASTRBOT_DATA_DIR")
    if raw_env:
        env_path = Path(raw_env).expanduser()

    candidates = [
        env_path,
        Path("data").resolve(),
        Path("..").resolve() / "astrbot" / "data",
        Path.home() / "astrbot" / "data",
    ]

    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not detect AstrBot data directory. "
        "Please pass it explicitly with --data-dir."
    )


# ---------------------------------------------------------------------------
# Skill linking
# ---------------------------------------------------------------------------


def link_skills_to_astrbot(data_dir: Path, project_root: Path) -> dict[str, int | list[str]]:
    """Symlink project skills into <data_dir>/skills and write the index skill."""
    project_skills_dir = project_root / ".agents" / "skills"
    skills_dir = data_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    removed = _remove_stale_project_skill_symlinks(skills_dir, project_skills_dir)

    linked = 0
    refreshed = 0
    skipped: list[str] = []
    conflicts: list[str] = []

    for project_skill in sorted(project_skills_dir.iterdir()):
        if not project_skill.is_dir():
            continue
        name = project_skill.name
        if name.startswith(".") or name.startswith("private-"):
            skipped.append(name)
            continue

        link_path = skills_dir / name

        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink():
                target = _symlink_target(link_path)
                if _is_relative_to(target, project_skills_dir):
                    link_path.unlink()
                    link_path.symlink_to(project_skill, target_is_directory=True)
                    refreshed += 1
                else:
                    skipped.append(name)
            else:
                conflicts.append(name)
        else:
            link_path.symlink_to(project_skill, target_is_directory=True)
            linked += 1

    write_index_skill(skills_dir, project_root)

    return {
        "linked": linked,
        "refreshed": refreshed,
        "skipped": len(skipped),
        "removed_stale": removed,
        "conflicts": conflicts,
    }


def write_index_skill(skills_dir: Path, project_root: Path) -> None:
    """Write the atomisticskills index SKILL.md into AstrBot's skills dir."""
    index_dir = skills_dir / INDEX_SKILL_NAME
    index_dir.mkdir(parents=True, exist_ok=True)
    skill_file = index_dir / "SKILL.md"

    skill_file.write_text(
        f"""\
---
name: atomisticskills
description: Use AtomisticSkills from {project_root} for atomistic research, materials simulation, molecular modeling, spectroscopy, MLIP, drug discovery, and scientific workflow tasks.
---

# AtomisticSkills

Use this skill when a task would benefit from the AtomisticSkills repository installed at:

`{project_root}`

Before acting, read the applicable project instructions directly from that repository:

1. Always read:
   - `{project_root}/.agents/rules/research-standards.md`
   - `{project_root}/.agents/rules/coding-standards.md`
   - `{project_root}/.agents/rules/mcp-environments.md`
2. For skill discovery, inspect:
   - `{project_root}/.agents/skills/*/SKILL.md`
3. For end-to-end protocols, inspect:
   - `{project_root}/.agents/workflows/`
4. Read the full selected `SKILL.md` or workflow file before following it.

If the current workspace is already `{project_root}` or one of its subdirectories, prefer the project-local AGENTS.md and project-local skills to avoid duplicate context.
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# MCP config generation
# ---------------------------------------------------------------------------


def _python_exists_for_server(server_cfg: dict[str, Any]) -> bool:
    """Return True if the server's configured Python interpreter exists."""
    cmd = server_cfg.get("command", "")
    return cmd and Path(cmd).exists()


def _to_uv_form(server_name: str, server_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a direct-path server config to AstrBot's ``env`` + ``uv`` form."""
    python_path = server_cfg["command"]
    args = list(server_cfg.get("args", []))

    env_args = []
    for key, value in server_cfg.get("env", {}).items():
        env_args.append(f"{key}={value}")

    # Build: env K=V ... uv run --python <path> python -m module
    uv_args = [
        *env_args,
        "uv",
        "run",
        "--python",
        str(python_path),
    ]

    # The original args are expected to be ["-m", "src.mcp_server.xxx_server"].
    # uv run still needs an explicit "python" before -m.
    uv_args.append("python")
    uv_args.extend(args)

    return {"command": "env", "args": uv_args}


def generate_astrbot_mcp_configs(
    project_root: Path,
    use_uv: bool,
) -> dict[str, dict[str, Any]]:
    """Load MCP configs and format them for AstrBot."""
    servers = configure_mcp.load_mcp_servers(pixi_root=str(project_root))
    result: dict[str, dict[str, Any]] = {}

    for name, cfg in servers.items():
        if not _python_exists_for_server(cfg):
            print(
                f"  [WARN] {name}: interpreter not found ({cfg.get('command')}). "
                f"Run 'pixi install -e {name}' first.",
                file=sys.stderr,
            )

        if use_uv:
            result[name] = _to_uv_form(name, cfg)
        else:
            result[name] = dict(cfg)

    return result


def print_mcp_configs(servers: dict[str, dict[str, Any]]) -> None:
    """Print per-server JSON blocks for copy-paste into AstrBot WebUI."""
    print("\n=== AstrBot MCP Server Configs ===")
    print("Paste each block into AstrBot WebUI -> MCP -> Add MCP Server.\n")

    for name, cfg in servers.items():
        print(f"--- {name} ---")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        print()


def write_mcp_config_file(data_dir: Path, servers: dict[str, dict[str, Any]]) -> Path:
    """Save all MCP configs to <data_dir>/config/atomisticskills_mcp.json."""
    config_dir = data_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "atomisticskills_mcp.json"
    config_file.write_text(
        json.dumps({"mcpServers": servers}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure AtomisticSkills for AstrBot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python configure_astrbot.py
  python configure_astrbot.py --data-dir /path/to/astrbot/data
  python configure_astrbot.py --data-dir /path/to/astrbot/data --use-uv
  python configure_astrbot.py --data-dir /path/to/astrbot/data --skills-only
  python configure_astrbot.py --data-dir /path/to/astrbot/data --mcp-only
  python configure_astrbot.py --data-dir /path/to/astrbot/data --write-mcp-config
  python configure_astrbot.py --list-servers
""",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="AstrBot data directory (default: auto-detect)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="AtomisticSkills project root (default: directory of this script)",
    )
    parser.add_argument(
        "--skills-only",
        action="store_true",
        help="Only create/refresh skill symlinks and index SKILL.md",
    )
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Only print MCP server JSON configs",
    )
    parser.add_argument(
        "--write-mcp-config",
        action="store_true",
        help="Also save MCP configs to <data-dir>/config/atomisticskills_mcp.json",
    )
    parser.add_argument(
        "--list-servers",
        action="store_true",
        help="List available MCP server names and exit",
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help=(
            "Output MCP configs in AstrBot's preferred 'env' + 'uv run' form "
            "instead of absolute Python paths"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()

    if args.list_servers:
        servers = configure_mcp.load_mcp_servers(pixi_root=str(project_root))
        for name in servers:
            print(name)
        return 0

    if args.mcp_only:
        servers = generate_astrbot_mcp_configs(project_root, use_uv=args.use_uv)
        print_mcp_configs(servers)
        return 0

    try:
        data_dir = detect_astrbot_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[AstrBot data dir] {data_dir}")

    stats = link_skills_to_astrbot(data_dir, project_root)
    print(
        "[skills] "
        f"Linked {stats['linked']}, refreshed {stats['refreshed']}, "
        f"skipped {stats['skipped']}, removed {stats['removed_stale']} stale"
    )
    if stats["conflicts"]:
        print(
            "[skills] Conflicts (non-symlink entries, left untouched): "
            + ", ".join(str(x) for x in stats["conflicts"]),
            file=sys.stderr,
        )
    print(f"  Index SKILL.md -> {data_dir / 'skills' / INDEX_SKILL_NAME / 'SKILL.md'}")

    if not args.skills_only:
        servers = generate_astrbot_mcp_configs(project_root, use_uv=args.use_uv)
        print_mcp_configs(servers)
        if args.write_mcp_config:
            config_file = write_mcp_config_file(data_dir, servers)
            print(f"[mcp] Wrote reference config -> {config_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
