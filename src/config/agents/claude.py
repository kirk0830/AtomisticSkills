"""Claude Code MCP configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import PROJECT_ROOT, write_json


def configure(servers: dict[str, Any], scope: str) -> None:
    """Configure MCP servers for Claude Code.

    - Project scope: .mcp.json
    - Global scope: ~/.claude.json
    """
    if scope in ("project", "both"):
        path = PROJECT_ROOT / ".mcp.json"
        write_json(path, servers, merge_key="mcpServers")
        print(f"  Project MCP → {path.relative_to(PROJECT_ROOT)}")

    if scope in ("global", "both"):
        path = Path.home() / ".claude.json"
        write_json(path, servers, merge_key="mcpServers")
        print(f"  Global MCP  → {path}")
