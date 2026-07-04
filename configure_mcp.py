#!/usr/bin/env python3
"""Configure AtomisticSkills MCP servers for any supported AI agent.

This is a legacy wrapper that delegates to the new `atomisticskills configure` CLI.
For the new unified interface, use:
    atomisticskills configure --agent <agent> [options]

Usage:
    python configure_mcp.py                        # auto-detect installed agents
    python configure_mcp.py --agent claude         # specific agent only
    python configure_mcp.py --agent claude codex   # multiple agents
    python configure_mcp.py --conda /path/to/miniforge3
    python configure_mcp.py --scope global         # write to global config only
    python configure_mcp.py --scope project        # write to project config only
    python configure_mcp.py --list-agents          # show detected agents
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from src.cli import main as cli_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure AtomisticSkills MCP servers for AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--agent",
        "-a",
        nargs="+",
        choices=["claude", "codex", "gemini", "cursor", "windsurf", "all"],
        default=None,
        metavar="AGENT",
        help="Agent(s) to configure: claude, codex, gemini, cursor, windsurf, all "
        "(default: auto-detect installed agents)",
    )
    parser.add_argument(
        "--conda",
        default=None,
        metavar="PATH",
        help="Path to conda/mamba base directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--pixi",
        action="store_true",
        default=None,
        help="Use Pixi environments (auto-detected if pixi.toml exists).",
    )
    parser.add_argument(
        "--no-pixi",
        action="store_true",
        help="Force conda mode even if pixi.toml is present.",
    )
    parser.add_argument(
        "--scope",
        choices=["project", "global", "both"],
        default="project",
        help="Where to write config: project dir, global user dir, or both "
        "(default: project).",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="Detect and print installed agents, then exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_agents:
        cli_main(["list-agents"])
        return

    if args.agent is None:
        from src.cli.configure import detect_agents

        agents = detect_agents()
        if not agents:
            print(
                "No supported agents auto-detected. "
                "Specify one with --agent (claude, codex, gemini, cursor, windsurf).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected agents: {', '.join(agents)}")
    elif "all" in args.agent:
        agents = ["claude", "codex", "gemini", "cursor", "windsurf"]
    else:
        agents = args.agent

    for agent in agents:
        cli_args = ["configure", "--agent", agent]
        if args.conda:
            cli_args.extend(["--conda", args.conda])
        if args.pixi:
            cli_args.append("--pixi")
        if args.no_pixi:
            cli_args.append("--no-pixi")
        if args.scope:
            cli_args.extend(["--scope", args.scope])
        cli_main(cli_args)


if __name__ == "__main__":
    main()