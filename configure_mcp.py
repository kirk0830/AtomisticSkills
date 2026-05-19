#!/usr/bin/env python3
"""Configure AtomisticSkills MCP servers for any supported AI agent.

Writes MCP server configs to the correct location for each agent, adapting
paths to the local conda installation.

Supported agents:
  claude   - Claude Code (.mcp.json or ~/.claude/settings.json)
  codex    - OpenAI Codex CLI (.codex/config.toml)
  gemini   - Google Gemini CLI (.gemini/settings.json)
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
    python configure_mcp.py --conda /path/to/miniforge3
    python configure_mcp.py --scope global         # write to global config only
    python configure_mcp.py --scope project        # write to project config only
    python configure_mcp.py --list-agents          # show detected agents
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV_PATTERN = re.compile(r".*/envs/([^/]+)/bin/python$")
PROJECT_ROOT = Path(__file__).resolve().parent
MCP_SOURCE = PROJECT_ROOT / "mcp_config.json"

# Instruction file stub used when a target agent has no instruction file yet.
INSTRUCTION_STUB = """\
# AtomisticSkills Agent Instructions

This project uses AtomisticSkills — a framework for atomistic simulation
workflows combining literature, MLIP tools, and MCP servers.

See CLAUDE.md for the full instructions (Claude Code format, also applicable
to other agents). Skills are in .agents/skills/, workflows in .agents/workflows/.
"""


# ---------------------------------------------------------------------------
# Conda detection
# ---------------------------------------------------------------------------


def detect_conda_base() -> str | None:
    for cmd in ("conda", "mamba", "micromamba"):
        try:
            result = subprocess.run(
                [cmd, "info", "--base"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                base = result.stdout.strip()
                if base and Path(base).is_dir():
                    return base
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    for name in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        candidate = Path.home() / name
        if candidate.is_dir():
            return str(candidate)

    return None


# ---------------------------------------------------------------------------
# MCP config loading and path patching
# ---------------------------------------------------------------------------


def load_mcp_servers(conda_base: str) -> dict[str, Any]:
    """Load mcp_config.json and rewrite conda env paths for this machine."""
    with open(MCP_SOURCE) as fh:
        config = json.load(fh)

    project_root = str(PROJECT_ROOT)

    for server in config.get("mcpServers", {}).values():
        match = ENV_PATTERN.match(server.get("command", ""))
        if match:
            env_name = match.group(1)
            server["command"] = f"{conda_base}/envs/{env_name}/bin/python"
        env = server.get("env", {})
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = project_root

    return config.get("mcpServers", {})


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------

KNOWN_AGENTS = ["claude", "codex", "gemini", "cursor", "windsurf"]


def detect_agents() -> list[str]:
    """Return list of agent names that appear to be installed."""
    found = []

    # Claude Code
    if shutil.which("claude"):
        found.append("claude")

    # OpenAI Codex CLI
    if shutil.which("codex"):
        found.append("codex")

    # Gemini CLI
    if shutil.which("gemini"):
        found.append("gemini")

    # Cursor — check for global config dir or project dir
    cursor_global = Path.home() / ".cursor"
    if cursor_global.is_dir() or (PROJECT_ROOT / ".cursor").is_dir():
        found.append("cursor")

    # Windsurf
    windsurf_global = Path.home() / ".codeium" / "windsurf"
    if windsurf_global.is_dir():
        found.append("windsurf")

    return found


# ---------------------------------------------------------------------------
# Per-agent writers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict, *, merge_key: str | None = None) -> None:
    """Write JSON, optionally merging into an existing file under merge_key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    if merge_key:
        existing.setdefault(merge_key, {})
        existing[merge_key].update(data)
        data = existing
    else:
        existing.update(data)
        data = existing

    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def _write_instruction_file(path: Path, source: Path) -> None:
    """Write agent instruction file if it doesn't already exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # If the source (e.g. CLAUDE.md) exists, symlink; otherwise write stub.
    if source.exists():
        # Write a stub that points to CLAUDE.md rather than symlinking,
        # so the file is portable across checkouts.
        path.write_text(f"# See CLAUDE.md for full instructions\n\n{INSTRUCTION_STUB}")
    else:
        path.write_text(INSTRUCTION_STUB)
    print(f"  Created {path.relative_to(PROJECT_ROOT)}")


def configure_claude(servers: dict, scope: str) -> None:
    """Claude Code: .mcp.json (project) or ~/.claude/settings.json (global)."""
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".mcp.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".claude" / "settings.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Global MCP  → {path}")

    # CLAUDE.md already exists — nothing to do for instruction file.


def configure_codex(servers: dict, scope: str) -> None:
    """Codex CLI: .codex/config.toml (TOML format)."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # backport
        except ImportError:
            tomllib = None

    try:
        import tomli_w
    except ImportError:
        tomli_w = None

    def _write_toml(path: Path, servers: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        if path.exists() and tomllib:
            try:
                with open(path, "rb") as fh:
                    existing = tomllib.load(fh)
            except Exception:
                pass

        existing.setdefault("mcp_servers", {})
        for name, cfg in servers.items():
            existing["mcp_servers"][name] = {
                "command": cfg["command"],
                "args": cfg.get("args", []),
                **({"env": cfg["env"]} if cfg.get("env") else {}),
            }

        if tomli_w:
            with open(path, "wb") as fh:
                tomli_w.dump(existing, fh)
        else:
            # Fallback: write minimal TOML manually
            lines = []
            for name, cfg in existing.get("mcp_servers", {}).items():
                lines.append(f"\n[mcp_servers.{name}]")
                lines.append(f'command = {json.dumps(cfg["command"])}')
                args_str = ", ".join(json.dumps(a) for a in cfg.get("args", []))
                lines.append(f"args = [{args_str}]")
                for k, v in cfg.get("env", {}).items():
                    lines.append(f"{k} = {json.dumps(v)}")
            path.write_text("\n".join(lines) + "\n")

    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".codex" / "config.toml"
        _write_toml(path, servers)
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".codex" / "config.toml"
        _write_toml(path, servers)
        print(f"  Global MCP  → {path}")

    _write_instruction_file(
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "CLAUDE.md",
    )


def configure_gemini(servers: dict, scope: str) -> None:
    """Gemini CLI: .gemini/settings.json."""
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".gemini" / "settings.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".gemini" / "settings.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Global MCP  → {path}")

    _write_instruction_file(
        PROJECT_ROOT / "GEMINI.md",
        PROJECT_ROOT / "CLAUDE.md",
    )


def configure_cursor(servers: dict, scope: str) -> None:
    """Cursor: .cursor/mcp.json (project) or ~/.cursor/mcp.json (global)."""
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".cursor" / "mcp.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".cursor" / "mcp.json"
        _write_json(path, servers, merge_key="mcpServers")
        print(f"  Global MCP  → {path}")

    # Cursor uses .cursor/rules/*.mdc — write a minimal rule file if missing.
    rules_dir = PROJECT_ROOT / ".cursor" / "rules"
    rules_file = rules_dir / "atomisticskills.mdc"
    if not rules_file.exists():
        rules_dir.mkdir(parents=True, exist_ok=True)
        rules_file.write_text(
            "---\ndescription: AtomisticSkills research agent rules\nalwaysApply: true\n---\n\n"
            + INSTRUCTION_STUB
        )
        print(f"  Created {rules_file.relative_to(PROJECT_ROOT)}")


def configure_windsurf(servers: dict, scope: str) -> None:
    """Windsurf: global only (~/.codeium/windsurf/mcp_config.json)."""
    if scope == "project":
        print(
            "  Windsurf does not support project-level MCP config. "
            "Use --scope global or --scope both."
        )
        return

    path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    _write_json(path, servers, merge_key="mcpServers")
    print(f"  Global MCP  → {path}")

    # Windsurf reads .windsurfrules at project root.
    _write_instruction_file(
        PROJECT_ROOT / ".windsurfrules",
        PROJECT_ROOT / "CLAUDE.md",
    )


AGENT_WRITERS = {
    "claude": configure_claude,
    "codex": configure_codex,
    "gemini": configure_gemini,
    "cursor": configure_cursor,
    "windsurf": configure_windsurf,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        "--conda",
        default=None,
        metavar="PATH",
        help="Path to conda/mamba base directory (auto-detected if omitted).",
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

    # Resolve conda base
    conda_base: str | None = args.conda
    if conda_base is not None:
        if not Path(conda_base).is_dir():
            print(f"Error: {conda_base} is not a valid directory.", file=sys.stderr)
            sys.exit(1)
    else:
        conda_base = detect_conda_base()
        if conda_base is None:
            print(
                "Error: Could not auto-detect a conda/mamba installation.\n"
                "Provide the base path explicitly: --conda /path/to/miniforge3",
                file=sys.stderr,
            )
            sys.exit(1)

    if not MCP_SOURCE.exists():
        print(f"Error: {MCP_SOURCE} not found.", file=sys.stderr)
        sys.exit(1)

    servers = load_mcp_servers(conda_base)

    # Resolve agent list
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
    print(f"Conda base   : {conda_base}")
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
