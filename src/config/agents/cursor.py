"""Cursor IDE MCP configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import PROJECT_ROOT, write_json, INSTRUCTION_STUB


def configure(servers: dict[str, Any], scope: str) -> None:
    """Configure MCP servers for Cursor IDE.

    - Project scope: .cursor/mcp.json
    - Global scope: ~/.cursor/mcp.json
    """
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".cursor" / "mcp.json"
        write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".cursor" / "mcp.json"
        write_json(path, servers, merge_key="mcpServers")
        print(f"  Global MCP  → {path}")

    _setup_cursor_rules()


def _setup_cursor_rules() -> None:
    """Set up Cursor project rules if they don't exist."""
    rules_dir = PROJECT_ROOT / ".cursor" / "rules"
    rules_file = rules_dir / "atomisticskills.mdc"
    if not rules_file.exists():
        rules_dir.mkdir(parents=True, exist_ok=True)
        rules_file.write_text(
            "---\ndescription: AtomisticSkills research agent rules\nalwaysApply: true\n---\n\n"
            + INSTRUCTION_STUB
        )
        print(f"  Created {rules_file.relative_to(PROJECT_ROOT)}")
