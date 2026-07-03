"""Configuration utilities for AtomisticSkills agent setup.

This package provides modular configuration generation for various AI agents
(Claude, Codex, Gemini, Cursor, Windsurf, AstrBot) using Jinja2 templates and
shared helper functions.
"""

from .base import (
    PROJECT_ROOT,
    TEMPLATES_DIR,
    get_jinja_env,
    symlink_target,
    is_relative_to,
    remove_stale_symlinks,
    symlink_dir_contents,
    reset_directory_symlink,
    upsert_marked_block,
    write_json,
    write_instruction_file,
    INSTRUCTION_STUB,
    ATOMISTICSKILLS_GLOBAL_MARKER,
    build_global_reference_block,
)
from .mcp_loader import (
    load_mcp_servers,
    detect_pixi_project_root,
    detect_conda_base,
    MCP_SOURCE,
)

__all__ = [
    "PROJECT_ROOT",
    "TEMPLATES_DIR",
    "get_jinja_env",
    "symlink_target",
    "is_relative_to",
    "remove_stale_symlinks",
    "symlink_dir_contents",
    "reset_directory_symlink",
    "upsert_marked_block",
    "write_json",
    "write_instruction_file",
    "INSTRUCTION_STUB",
    "ATOMISTICSKILLS_GLOBAL_MARKER",
    "build_global_reference_block",
    "load_mcp_servers",
    "detect_pixi_project_root",
    "detect_conda_base",
    "MCP_SOURCE",
]
