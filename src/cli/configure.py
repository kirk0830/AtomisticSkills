"""Configure subcommand for AtomisticSkills CLI."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from src.config import (
    load_mcp_servers,
    detect_pixi_project_root,
    detect_conda_base,
    PROJECT_ROOT,
    MCP_SOURCE,
)
from src.config.agents import claude, codex, gemini, cursor, windsurf, astrbot


KNOWN_AGENTS = ["claude", "codex", "gemini", "cursor", "windsurf", "astrbot", "openclaw"]


def add_configure_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Add the configure subcommand to the parser."""
    parser = subparsers.add_parser(
        "configure",
        help="Configure AtomisticSkills for an AI agent",
        description="Configure AtomisticSkills MCP servers and skills for a specific AI agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--agent",
        "-a",
        required=True,
        choices=KNOWN_AGENTS,
        help=f"Agent to configure: {', '.join(KNOWN_AGENTS)}",
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
        help="Where to write config: project dir, global user dir, or both (default: project).",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Agent data directory (required for AstrBot, optional for others).",
    )

    parser.add_argument(
        "--skills-only",
        action="store_true",
        help="Only create/refresh skill symlinks and index SKILL.md (AstrBot/OpenClaw only).",
    )

    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Only print/generate MCP server configs.",
    )

    parser.add_argument(
        "--write-mcp-config",
        action="store_true",
        help="Save MCP configs to a file.",
    )

    parser.add_argument(
        "--use-uv",
        action="store_true",
        help=(
            "Output MCP configs in 'env' + 'uv run' form instead of absolute Python paths "
            "(AstrBot/OpenClaw only)."
        ),
    )


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

    astrbot_candidates = [
        Path("data").resolve(),
        Path("..").resolve() / "astrbot" / "data",
        Path.home() / "astrbot" / "data",
    ]
    for candidate in astrbot_candidates:
        if candidate.is_dir():
            found.append("astrbot")
            break

    openclaw_global = Path.home() / ".openclaw"
    if openclaw_global.is_dir():
        found.append("openclaw")

    return found


def configure_cmd(args: argparse.Namespace) -> int:
    """Execute the configure command."""
    pixi_root: str | None = None
    conda_base: str | None = None

    if args.no_pixi:
        use_pixi = False
    else:
        pixi_root = detect_pixi_project_root()
        use_pixi = pixi_root is not None

    if not use_pixi:
        conda_base = args.conda
        if conda_base is not None:
            if not Path(conda_base).is_dir():
                print(f"Error: {conda_base} is not a valid directory.", file=sys.stderr)
                return 1
        else:
            conda_base = detect_conda_base()
            if conda_base is None:
                print(
                    "Error: Could not auto-detect a conda/mamba installation.\n"
                    "Provide the base path explicitly: --conda /path/to/miniforge3",
                    file=sys.stderr,
                )
                return 1

    if not MCP_SOURCE.exists():
        print(f"Error: {MCP_SOURCE} not found.", file=sys.stderr)
        return 1

    servers = load_mcp_servers(conda_base=conda_base, pixi_root=pixi_root)

    print(f"Project root : {PROJECT_ROOT}")
    if use_pixi:
        print(f"Env mode     : pixi (envs in .pixi/envs/)")
    else:
        print(f"Env mode     : conda")
        print(f"Conda base   : {conda_base}")
    print(f"Scope        : {args.scope}")
    print()

    return _configure_agent(args.agent, servers, args)


def _configure_agent(agent: str, servers: dict[str, Any], args: argparse.Namespace) -> int:
    """Configure a specific agent."""
    print(f"[{agent}]")

    if agent == "astrbot":
        return _configure_astrbot(servers, args)
    elif agent == "openclaw":
        return _configure_openclaw(servers, args)
    elif agent in ["claude", "codex", "gemini", "cursor", "windsurf"]:
        return _configure_ide_agent(agent, servers, args)
    else:
        print(f"Unknown agent: {agent}", file=sys.stderr)
        return 1


def _configure_ide_agent(agent: str, servers: dict[str, Any], args: argparse.Namespace) -> int:
    """Configure IDE-based agents (claude, codex, gemini, cursor, windsurf)."""
    AGENT_WRITERS = {
        "claude": claude.configure,
        "codex": codex.configure,
        "gemini": gemini.configure,
        "cursor": cursor.configure,
        "windsurf": windsurf.configure,
    }

    AGENT_WRITERS[agent](servers, args.scope)
    print()
    print("Done. Skills (.agents/skills/) and workflows (.agents/workflows/)")
    print("are the cross-platform standard path — no changes needed there.")
    return 0


def _configure_astrbot(servers: dict[str, Any], args: argparse.Namespace) -> int:
    """Configure AstrBot chatbot framework."""
    if args.skills_only and args.mcp_only:
        print("Error: --skills-only and --mcp-only are mutually exclusive.", file=sys.stderr)
        return 1

    try:
        data_dir = astrbot.detect_astrbot_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[AstrBot data dir] {data_dir}")

    if not args.mcp_only:
        stats = astrbot.link_skills_to_astrbot(data_dir, PROJECT_ROOT)
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
            f"  Index SKILL.md -> {data_dir / 'skills' / astrbot.INDEX_SKILL_NAME / 'SKILL.md'}"
        )

    if not args.skills_only:
        astrbot_servers = astrbot.generate_astrbot_mcp_configs(
            PROJECT_ROOT, use_uv=args.use_uv
        )
        astrbot.print_mcp_configs(astrbot_servers)
        if args.write_mcp_config:
            config_file = astrbot.write_mcp_config_file(data_dir, astrbot_servers)
            print(f"[mcp] Wrote reference config -> {config_file}")

    persona_path = astrbot.write_persona_file(data_dir, PROJECT_ROOT)
    astrbot.print_persona_prompt(persona_path)

    return 0


def _configure_openclaw(servers: dict[str, Any], args: argparse.Namespace) -> int:
    """Configure OpenClaw TUI/bot framework."""
    from src.config.agents.openclaw import configure as openclaw_configure

    return openclaw_configure(servers, args)