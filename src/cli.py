#!/usr/bin/env python3
"""AtomisticSkills unified command-line interface.

Usage:
    atomisticskills configure --agent <name> [options]
    atomisticskills list-agents
    atomisticskills list-servers

Examples:
    # Configure for IDE agents (auto-detect)
    atomisticskills configure

    # Configure specific agent with scope
    atomisticskills configure --agent claude --scope global
    atomisticskills configure --agent cursor codex

    # Configure AstrBot chatbot framework
    atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data
    atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data --skills-only
    atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data --write-mcp-config

    # List available agents
    atomisticskills list-agents

    # List available MCP servers
    atomisticskills list-servers
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
from src.config.agents import astrbot as astrbot_config

KNOWN_IDE_AGENTS = ["claude", "codex", "gemini", "cursor", "windsurf"]
KNOWN_AGENTS = KNOWN_IDE_AGENTS + ["astrbot"]

IDE_AGENT_WRITERS = {
    "claude": claude.configure,
    "codex": codex.configure,
    "gemini": gemini.configure,
    "cursor": cursor.configure,
    "windsurf": windsurf.configure,
}


# ---------------------------------------------------------------------------
# Subcommand: configure
# ---------------------------------------------------------------------------

def _add_configure_parser(subparsers: argparse._SubParsersAction) -> None:
    configure_parser = subparsers.add_parser(
        "configure",
        help="Configure AtomisticSkills for one or more AI agents",
        description="Write MCP server configs and/or skill symlinks for supported agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_configure_epilog(),
    )

    configure_parser.add_argument(
        "--agent",
        "-a",
        nargs="+",
        choices=KNOWN_AGENTS + ["all"],
        default=None,
        metavar="AGENT",
        help=f"Agent(s) to configure: {', '.join(KNOWN_AGENTS)}, all "
        "(default: auto-detect installed IDE agents)",
    )

    # --- IDE agent options ---
    configure_parser.add_argument(
        "--scope",
        choices=["project", "global", "both"],
        default="project",
        help="Where to write IDE agent config (default: project)",
    )

    # --- AstrBot-specific options ---
    configure_parser.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="AstrBot data directory (required for --agent astrbot)",
    )
    configure_parser.add_argument(
        "--skills-only",
        action="store_true",
        help="[astrbot] Only create/refresh skill symlinks and index SKILL.md",
    )
    configure_parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="[astrbot] Only print MCP server JSON configs",
    )
    configure_parser.add_argument(
        "--write-mcp-config",
        action="store_true",
        help="[astrbot] Also save MCP configs to <data-dir>/config/",
    )
    configure_parser.add_argument(
        "--use-uv",
        action="store_true",
        help="[astrbot] Output MCP configs in 'env' + 'uv run' form",
    )


def _configure_epilog() -> str:
    return """\
Examples:
  # Auto-detect installed IDE agents and configure (project scope)
  atomisticskills configure

  # Configure specific agents with global scope
  atomisticskills configure --agent claude --scope global
  atomisticskills configure --agent claude cursor --scope both

  # Configure AstrBot (skills only)
  atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data --skills-only

  # Configure AstrBot (MCP configs only)
  atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data --mcp-only --use-uv

  # Full AstrBot setup
  atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data --write-mcp-config
"""


def _detect_ide_agents() -> list[str]:
    """Return list of IDE agent names that appear to be installed."""
    found = []
    if shutil.which("claude"):
        found.append("claude")
    if shutil.which("codex"):
        found.append("codex")
    if shutil.which("gemini"):
        found.append("gemini")
    if (Path.home() / ".cursor").is_dir() or (PROJECT_ROOT / ".cursor").is_dir():
        found.append("cursor")
    if (Path.home() / ".codeium" / "windsurf").is_dir():
        found.append("windsurf")
    return found


def _resolve_agents(args: argparse.Namespace) -> list[str]:
    """Resolve the final list of agents from --agent / auto-detect."""
    if args.agent is None:
        agents = _detect_ide_agents()
        if not agents:
            print(
                "No supported IDE agents auto-detected. "
                f"Specify one with --agent ({', '.join(KNOWN_AGENTS)}).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Auto-detected agents: {', '.join(agents)}")
        return agents

    if "all" in args.agent:
        return KNOWN_AGENTS

    return args.agent


def _run_configure(args: argparse.Namespace) -> None:
    agents = _resolve_agents(args)

    # Split into IDE agents and astrbot
    ide_agents = [a for a in agents if a != "astrbot"]
    has_astrbot = "astrbot" in agents

    pixi_root = detect_pixi_project_root()
    if pixi_root is None:
        print(
            "Error: Could not find pixi.toml in project root.\n"
            "This project uses Pixi for environment management.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not MCP_SOURCE.exists():
        print(f"Error: {MCP_SOURCE} not found.", file=sys.stderr)
        sys.exit(1)

    # --- Configure IDE agents ---
    if ide_agents:
        servers = load_mcp_servers(pixi_root=pixi_root)
        print(f"Project root : {PROJECT_ROOT}")
        print(f"Env mode     : pixi (envs in .pixi/envs/)")
        print(f"Scope        : {args.scope}")
        print()

        for agent in ide_agents:
            print(f"[{agent}]")
            IDE_AGENT_WRITERS[agent](servers, args.scope)
            print()

    # --- Configure AstrBot ---
    if has_astrbot:
        _run_astrbot_configure(args)


def _run_astrbot_configure(args: argparse.Namespace) -> None:
    """Run AstrBot-specific configuration steps."""
    if args.mcp_only:
        servers = astrbot_config.generate_astrbot_mcp_configs(
            PROJECT_ROOT, use_uv=args.use_uv
        )
        astrbot_config.print_mcp_configs(servers)
        return

    try:
        data_dir = astrbot_config.detect_astrbot_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[AstrBot data dir] {data_dir}")

    stats = astrbot_config.link_skills_to_astrbot(data_dir, PROJECT_ROOT)
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
        f"  Index SKILL.md -> "
        f"{data_dir / 'skills' / astrbot_config.INDEX_SKILL_NAME / 'SKILL.md'}"
    )

    if not args.skills_only:
        servers = astrbot_config.generate_astrbot_mcp_configs(
            PROJECT_ROOT, use_uv=args.use_uv
        )
        astrbot_config.print_mcp_configs(servers)
        if args.write_mcp_config:
            config_file = astrbot_config.write_mcp_config_file(data_dir, servers)
            print(f"[mcp] Wrote reference config -> {config_file}")

    persona_path = astrbot_config.write_persona_file(data_dir, PROJECT_ROOT)
    astrbot_config.print_persona_prompt(persona_path)


# ---------------------------------------------------------------------------
# Subcommand: list-agents
# ---------------------------------------------------------------------------

def _run_list_agents(_args: argparse.Namespace) -> None:
    detected = _detect_ide_agents()
    if detected:
        print("Installed IDE agents:", ", ".join(detected))
    else:
        print("No IDE agents detected.")
    print(f"Also available (requires explicit --agent): astrbot")


# ---------------------------------------------------------------------------
# Subcommand: list-servers
# ---------------------------------------------------------------------------

def _run_list_servers(_args: argparse.Namespace) -> None:
    pixi_root = detect_pixi_project_root()
    if pixi_root is None:
        print("Error: pixi.toml not found.", file=sys.stderr)
        sys.exit(1)
    servers = load_mcp_servers(pixi_root=pixi_root)
    for name in sorted(servers):
        print(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atomisticskills",
        description="AtomisticSkills unified command-line interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", title="subcommands")

    _add_configure_parser(subparsers)

    subparsers.add_parser(
        "list-agents",
        help="List installed/supported AI agents",
    )
    subparsers.add_parser(
        "list-servers",
        help="List available MCP server names",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "configure":
        _run_configure(args)
    elif args.command == "list-agents":
        _run_list_agents(args)
    elif args.command == "list-servers":
        _run_list_servers(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
