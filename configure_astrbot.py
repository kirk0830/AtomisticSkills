#!/usr/bin/env python3
"""Configure AtomisticSkills for AstrBot.

This is a legacy wrapper that delegates to the new `atomisticskills configure` CLI.
For the new unified interface, use:
    atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data

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

from src.cli import main as cli_main

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

    if args.list_servers:
        cli_main(["list-servers"])
        return 0

    cli_args = ["configure", "--agent", "astrbot"]

    if args.data_dir:
        cli_args.extend(["--data-dir", args.data_dir])
    if args.skills_only:
        cli_args.append("--skills-only")
    if args.mcp_only:
        cli_args.append("--mcp-only")
    if args.write_mcp_config:
        cli_args.append("--write-mcp-config")
    if args.use_uv:
        cli_args.append("--use-uv")

    return cli_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())