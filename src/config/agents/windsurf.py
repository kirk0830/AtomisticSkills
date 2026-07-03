"""Windsurf IDE MCP configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import PROJECT_ROOT, write_json, write_instruction_file


def configure(servers: dict[str, Any], scope: str) -> None:
    """Configure MCP servers for Windsurf IDE.

    Windsurf only supports global MCP config.
    """
    if scope == "project":
        print(
            "  Windsurf does not support project-level MCP config. "
            "Use --scope global or --scope both."
        )
        return

    path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    write_json(path, servers, merge_key="mcpServers")
    print(f"  Global MCP  → {path}")

    write_instruction_file(
        PROJECT_ROOT / ".windsurfrules",
        PROJECT_ROOT / "CLAUDE.md",
    )
