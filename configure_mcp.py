#!/usr/bin/env python3
"""Configure AtomisticSkills MCP servers for any supported AI agent.

Writes MCP server configs to the correct location for each agent, adapting
paths to the local Pixi installation.

Supported agents:
  claude   - Claude Code (.mcp.json or ~/.claude/settings.json)
  codex    - OpenAI Codex CLI (.codex/config.toml)
  gemini   - Google Gemini CLI and IDE (.gemini/settings.json, config/mcp_config.json, and custom plugin)
  cursor   - Cursor (.cursor/mcp.json)
  windsurf - Windsurf (~/.codeium/windsurf/mcp_config.json, global only)

Skills (.agents/skills/) and workflows (.agents/workflows/) are the universal
cross-platform paths — no changes needed for different agents.
Instruction files (CLAUDE.md / AGENTS.md / GEMINI.md) are auto-generated
for agents that don't already have one.

Usage:
    python configure_mcp.py                        # auto-detect installed agents
    python configure_mcp.py --agent claude         # specific agent only
    python configure_mcp.py --agent claude codex   # multiple agents
    python configure_mcp.py --scope global         # write to global config only
    python configure_mcp.py --scope project        # write to project config only
    python configure_mcp.py --list-agents          # show detected agents
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.config import (
    load_mcp_servers,
    detect_pixi_project_root,
    PROJECT_ROOT,
    MCP_SOURCE,
)
from src.config.agents import claude, codex, gemini, cursor, windsurf

KNOWN_AGENTS = ["claude", "codex", "gemini", "cursor", "windsurf"]

AGENT_WRITERS = {
    "claude": claude.configure,
    "codex": codex.configure,
    "gemini": gemini.configure,
    "cursor": cursor.configure,
    "windsurf": windsurf.configure,
}


def detect_agents() -> list[str]:
    """Return list of agent names that appear to be installed."""
    found = []

    if shutil.which("claude"):
        found.append("claude")

    if shutil.which("codex"):
        found.append("codex")

    if shutil.which("gemini"):
        found.append("gemini")

    cursor_global = Path.home() / ".cursor"
    if cursor_global.is_dir() or (PROJECT_ROOT / ".cursor").is_dir():
        found.append("cursor")

    windsurf_global = Path.home() / ".codeium" / "windsurf"
    if windsurf_global.is_dir():
        found.append("windsurf")

    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure AtomisticSkills MCP servers for AI agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--agent",
        "-a",
        nargs="+",
        choices=KNOWN_AGENTS + ["all"],
        default=None,
        metavar="AGENT",
        help=f"Agent(s) to configure: {', '.join(KNOWN_AGENTS)}, all "
        "(default: auto-detect installed agents)",
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
    args = parser.parse_args()

    if args.list_agents:
        detected = detect_agents()
        if detected:
            print("Detected agents:", ", ".join(detected))
        else:
            print("No supported agents detected.")
        return

    pixi_root = detect_pixi_project_root()
    if pixi_root is None:
        print(
            "Error: Could not find pixi.toml in project root.\n"
            "This project uses Pixi for environment management. "
            "Please ensure pixi.toml is present.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not MCP_SOURCE.exists():
        print(f"Error: {MCP_SOURCE} not found.", file=sys.stderr)
        sys.exit(1)

    servers = load_mcp_servers(pixi_root=pixi_root)

    if args.agent is None:
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
        agents = KNOWN_AGENTS
    else:
        agents = args.agent

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Env mode     : pixi (envs in .pixi/envs/)")
    print(f"Scope        : {args.scope}")
    print()

    for agent in agents:
        print(f"[{agent}]")
        AGENT_WRITERS[agent](servers, args.scope)
        print()

    print("Done. Skills (.agents/skills/) and workflows (.agents/workflows/)")
    print("are the cross-platform standard path — no changes needed there.")


if __name__ == "__main__":
    main()
