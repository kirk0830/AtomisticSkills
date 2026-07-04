"""AtomisticSkills CLI entry point.

Usage:
    atomisticskills configure --agent <agent> [options]
    atomisticskills list-agents
    atomisticskills list-servers
    atomisticskills --help
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .configure import add_configure_subcommand, configure_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atomisticskills",
        description="AtomisticSkills CLI: Configure and manage atomistic research workflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    add_configure_subcommand(subparsers)

    list_agents_parser = subparsers.add_parser(
        "list-agents",
        help="List supported AI agents for configuration",
        description="List all supported AI agents that can be configured with AtomisticSkills.",
    )

    list_servers_parser = subparsers.add_parser(
        "list-servers",
        help="List available MCP servers",
        description="List all available MCP servers that can be registered with agents.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "list-agents":
        from src.config.agents import astrbot, claude, codex, gemini, cursor, windsurf

        agents = [
            ("claude", "Claude Code (.mcp.json or ~/.claude/settings.json)"),
            ("codex", "OpenAI Codex CLI (.codex/config.toml)"),
            ("gemini", "Google Gemini CLI and IDE"),
            ("cursor", "Cursor IDE (.cursor/mcp.json)"),
            ("windsurf", "Windsurf IDE"),
            ("astrbot", "AstrBot chatbot framework"),
            ("openclaw", "OpenClaw TUI/bot framework"),
        ]
        print("Supported agents:")
        for name, desc in agents:
            print(f"  {name:10} - {desc}")
        return 0

    if args.command == "list-servers":
        from src.config.mcp_loader import load_mcp_servers, detect_pixi_project_root

        pixi_root = detect_pixi_project_root()
        servers = load_mcp_servers(pixi_root=pixi_root)
        print("Available MCP servers:")
        for name in servers:
            print(f"  {name}")
        return 0

    if args.command == "configure":
        return configure_cmd(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())