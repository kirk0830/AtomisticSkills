"""Codex CLI MCP configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..base import (
    PROJECT_ROOT,
    write_instruction_file,
    symlink_target,
    is_relative_to,
    remove_stale_symlinks,
    build_global_reference_block,
    ATOMISTICSKILLS_GLOBAL_MARKER,
    upsert_marked_block,
)


def configure(servers: dict[str, Any], scope: str) -> None:
    """Configure MCP servers for OpenAI Codex CLI.

    - Project scope: .codex/config.toml
    - Global scope: ~/.codex/config.toml + global skills
    """
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".codex" / "config.toml"
        _write_toml(path, servers)
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".codex" / "config.toml"
        _write_toml(path, servers)
        print(f"  Global MCP  → {path}")

        global_agents = Path.home() / ".codex" / "AGENTS.md"
        upsert_marked_block(
            global_agents,
            ATOMISTICSKILLS_GLOBAL_MARKER,
            build_global_reference_block(),
        )
        print(f"  Global instructions → {global_agents}")
        _write_codex_global_skills()

    write_instruction_file(
        PROJECT_ROOT / "AGENTS.md",
        PROJECT_ROOT / "CLAUDE.md",
    )


def _toml_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(key)


def _toml_path(prefix: str, key: str) -> str:
    key_part = _toml_key(key)
    return f"{prefix}.{key_part}" if prefix else key_part


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


def _emit_toml_table(lines: list[str], prefix: str, table: dict) -> None:
    scalar_items = [
        (key, value) for key, value in table.items() if not isinstance(value, dict)
    ]
    child_items = [
        (key, value) for key, value in table.items() if isinstance(value, dict)
    ]

    if prefix:
        lines.append(f"[{prefix}]")
    for key, value in scalar_items:
        lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    if scalar_items and child_items:
        lines.append("")

    for index, (key, value) in enumerate(child_items):
        if lines and lines[-1] != "":
            lines.append("")
        child_prefix = _toml_path(prefix, key)
        _emit_toml_table(lines, child_prefix, value)
        if index != len(child_items) - 1 and lines[-1] != "":
            lines.append("")


def _write_toml(path: Path, servers: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}

    try:
        import tomllib  # type: ignore
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            tomllib = None  # type: ignore

    if path.exists() and tomllib:  # type: ignore
        try:
            with open(path, "rb") as fh:
                existing = tomllib.load(fh)  # type: ignore
        except Exception:
            pass

    existing.setdefault("mcp_servers", {})
    for name, cfg in servers.items():
        existing["mcp_servers"][name] = {
            "command": cfg["command"],
            "args": cfg.get("args", []),
            **({"env": cfg["env"]} if cfg.get("env") else {}),
        }

    try:
        import tomli_w  # type: ignore

        with open(path, "wb") as fh:
            tomli_w.dump(existing, fh)
    except ImportError:
        lines: list[str] = []
        _emit_toml_table(lines, "", existing)
        path.write_text("\n".join(lines).rstrip() + "\n")


def _write_codex_global_skills() -> None:
    """Expose project skills globally for Codex."""
    codex_skills_dir = Path.home() / ".codex" / "skills"
    codex_skills_dir.mkdir(parents=True, exist_ok=True)
    project_skills_dir = PROJECT_ROOT / ".agents" / "skills"
    removed = remove_stale_symlinks(codex_skills_dir, project_skills_dir)

    skill_dir = codex_skills_dir / "atomisticskills"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(_build_codex_index_skill())
    print(f"  Global skill -> {skill_file}")

    linked = 0
    skipped: list[str] = []
    for project_skill in sorted(project_skills_dir.iterdir()):
        if not project_skill.is_dir():
            continue

        global_skill = codex_skills_dir / project_skill.name
        if global_skill.exists() or global_skill.is_symlink():
            if global_skill.is_symlink():
                target = symlink_target(global_skill)
                if not is_relative_to(target, project_skills_dir):
                    skipped.append(project_skill.name)
                    continue
                global_skill.unlink()
            else:
                skipped.append(project_skill.name)
                continue

        global_skill.symlink_to(project_skill, target_is_directory=True)
        linked += 1

    print(
        f"  Global project skills -> {codex_skills_dir} "
        f"({linked} symlinks, {removed} stale removed)"
    )
    if skipped:
        separator = ", "
        print(f"  Skipped existing non-symlink skills: {separator.join(skipped)}")


def _build_codex_index_skill() -> str:
    """Build the Codex index skill content."""
    return f"""\
---
name: atomisticskills
description: Use AtomisticSkills from {PROJECT_ROOT} for atomistic research, materials simulation, molecular modeling, spectroscopy, MLIP, drug discovery, and scientific workflow tasks.
---

# AtomisticSkills

Use this skill when a task would benefit from the AtomisticSkills repository installed at:

`{PROJECT_ROOT}`

Before acting, read the applicable project instructions directly from that repository:

1. Always read:
   - `{PROJECT_ROOT}/.agents/rules/research-standards.md`
   - `{PROJECT_ROOT}/.agents/rules/coding-standards.md`
   - `{PROJECT_ROOT}/.agents/rules/mcp-environments.md`
2. For skill discovery, inspect:
   - `{PROJECT_ROOT}/.agents/skills/*/SKILL.md`
3. For end-to-end protocols, inspect:
   - `{PROJECT_ROOT}/.agents/workflows/`
4. Read the full selected `SKILL.md` or workflow file before following it.

If the current workspace is already `{PROJECT_ROOT}` or a subdirectory, prefer the project-local AGENTS.md and project-local skills to avoid duplicate context.
"""
