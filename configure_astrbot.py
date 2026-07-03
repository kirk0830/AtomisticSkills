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
import sys
from pathlib import Path

from src.config.agents import astrbot as astrbot_config

PROJECT_ROOT = Path(__file__).resolve().parent


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
        from src.config.mcp_loader import load_mcp_servers

        servers = load_mcp_servers(pixi_root=str(project_root))
        for name in servers:
            print(name)
        return 0

    if args.mcp_only:
        servers = astrbot_config.generate_astrbot_mcp_configs(
            project_root, use_uv=args.use_uv
        )
        astrbot_config.print_mcp_configs(servers)
        return 0

    try:
        data_dir = astrbot_config.detect_astrbot_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[AstrBot data dir] {data_dir}")

    stats = astrbot_config.link_skills_to_astrbot(data_dir, project_root)
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
    print(
        f"  Index SKILL.md -> {data_dir / 'skills' / astrbot_config.INDEX_SKILL_NAME / 'SKILL.md'}"
    )

    if not args.skills_only:
        servers = astrbot_config.generate_astrbot_mcp_configs(
            project_root, use_uv=args.use_uv
        )
        astrbot_config.print_mcp_configs(servers)
        if args.write_mcp_config:
            config_file = astrbot_config.write_mcp_config_file(data_dir, servers)
            print(f"[mcp] Wrote reference config -> {config_file}")

    persona_path = astrbot_config.write_persona_file(data_dir, project_root)
    astrbot_config.print_persona_prompt(persona_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
