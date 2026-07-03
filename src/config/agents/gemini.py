"""Gemini CLI and IDE MCP configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base import (
    PROJECT_ROOT,
    write_json,
    write_instruction_file,
    reset_directory_symlink,
    build_global_reference_block,
    ATOMISTICSKILLS_GLOBAL_MARKER,
    upsert_marked_block,
)


def configure(servers: dict[str, Any], scope: str) -> None:
    """Configure MCP servers for Gemini CLI and IDE.

    - Project scope: .gemini/settings.json
    - Global scope: ~/.gemini/settings.json, ~/.gemini/config/mcp_config.json
    """
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".gemini" / "settings.json"
        write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        cli_path = Path.home() / ".gemini" / "settings.json"
        write_json(cli_path, servers, merge_key="mcpServers")
        print(f"  Global MCP (CLI) → {cli_path}")

        ide_path = Path.home() / ".gemini" / "config" / "mcp_config.json"
        write_json(ide_path, servers, merge_key="mcpServers")
        print(f"  Global MCP (IDE) → {ide_path}")

        _setup_gemini_plugin()
        _setup_gemini_global_instructions()

    write_instruction_file(
        PROJECT_ROOT / "GEMINI.md",
        PROJECT_ROOT / "CLAUDE.md",
    )


def _setup_gemini_plugin() -> None:
    """Set up Gemini IDE plugin with skills symlink."""
    plugin_dir = (
        Path.home()
        / ".gemini"
        / "config"
        / "plugins"
        / "Google.atomisticskills.atomisticskills"
    )
    plugin_dir.mkdir(parents=True, exist_ok=True)

    plugin_json = plugin_dir / "plugin.json"
    plugin_data = {
        "name": "atomisticskills",
        "description": "AtomisticSkills workspace plugin",
        "disabled": False,
    }
    with open(plugin_json, "w") as fh:
        json.dump(plugin_data, fh)
        fh.write("\n")
    print(f"  Created plugin config → {plugin_json}")

    skills_symlink = plugin_dir / "skills"
    target_skills = PROJECT_ROOT / ".agents" / "skills"
    symlink_action = reset_directory_symlink(skills_symlink, target_skills)
    print(
        f"  {symlink_action} skills symlink → {skills_symlink} to {target_skills}"
    )


def _setup_gemini_global_instructions() -> None:
    """Set up global GEMINI.md with AtomisticSkills reference."""
    global_md = Path.home() / ".gemini" / "GEMINI.md"
    block_content = _build_gemini_global_block()
    upsert_marked_block(global_md, ATOMISTICSKILLS_GLOBAL_MARKER, block_content)
    print(f"  Updated global instructions → {global_md}")


def _build_gemini_global_block() -> str:
    """Build the Gemini-specific global instruction block."""
    marker = ATOMISTICSKILLS_GLOBAL_MARKER
    project_root = str(PROJECT_ROOT)
    return f"""\
{marker}

If your current workspace is NOT {project_root} (or any of its subdirectories), and you need to perform atomistic research, materials discovery, molecular simulation, or related tasks:
- The AtomisticSkills repository is installed at {project_root}.
- You can access its Skills at {project_root}/.agents/skills/ and workflows at {project_root}/.agents/workflows/.
- Discover skills by running: grep -r "^description:" {project_root}/.agents/skills/*/SKILL.md
- Read and follow these rules from the AtomisticSkills repo:
  - [research-standards.md](file://{project_root}/.agents/rules/research-standards.md)
  - [coding-standards.md](file://{project_root}/.agents/rules/coding-standards.md)
  - [mcp-environments.md](file://{project_root}/.agents/rules/mcp-environments.md)
  - [skill-standards.md](file://{project_root}/.agents/rules/skill-standards.md)
  - [workflow-standards.md](file://{project_root}/.agents/rules/workflow-standards.md)
  - [plot-standards.md](file://{project_root}/.agents/rules/plot-standards.md)
- NOTE: If you are already inside the {project_root} directory, ignore this section to avoid loading duplicate rules or context.
"""
