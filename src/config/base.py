"""Base configuration utilities shared across all agent config modules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = PROJECT_ROOT / ".agents" / "templates"

JINJA2_AVAILABLE = False
try:
    from jinja2 import Environment, FileSystemLoader

    JINJA2_AVAILABLE = True
except ImportError:
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore


INSTRUCTION_STUB = """\
# AtomisticSkills Agent Instructions

This project uses AtomisticSkills — a framework for atomistic simulation
workflows combining literature, MLIP tools, and MCP servers.

See CLAUDE.md for the full instructions (Claude Code format, also applicable
to other agents). Skills are in .agents/skills/, workflows in .agents/workflows/.
"""

ATOMISTICSKILLS_GLOBAL_MARKER = "# AtomisticSkills Global Reference"


def get_jinja_env():
    """Get Jinja2 environment, or None if Jinja2 is not available."""
    if not JINJA2_AVAILABLE or not TEMPLATES_DIR.exists():
        return None
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def symlink_target(path: Path) -> Path:
    """Return the absolute target path for a symlink."""
    target = path.readlink()
    if not target.is_absolute():
        target = path.parent / target
    return target


def is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is inside parent without requiring it to exist."""
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def remove_stale_symlinks(
    symlinks_dir: Path,
    target_dir: Path,
) -> int:
    """Remove symlinks that point to removed targets under target_dir."""
    removed = 0
    if not symlinks_dir.exists():
        return removed

    for entry in symlinks_dir.iterdir():
        if not entry.is_symlink():
            continue
        target = symlink_target(entry)
        if is_relative_to(target, target_dir) and not target.exists():
            entry.unlink()
            removed += 1

    return removed


def symlink_dir_contents(src_dir: Path, dst_dir: Path) -> int:
    """Symlink every entry inside src_dir into dst_dir.

    Stale symlinks whose target no longer exists are removed.
    Returns the count of active symlinks after the operation.
    """
    if not src_dir.exists():
        return 0

    dst_dir.mkdir(parents=True, exist_ok=True)

    for entry in dst_dir.iterdir():
        if not entry.is_symlink():
            continue
        target = symlink_target(entry)
        if is_relative_to(target, src_dir) and not target.exists():
            entry.unlink()

    linked = 0
    for src_entry in sorted(src_dir.iterdir()):
        if src_entry.name.startswith("."):
            continue
        link_path = dst_dir / src_entry.name
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink():
                target = symlink_target(link_path)
                if target.resolve() == src_entry.resolve():
                    linked += 1
                    continue
                link_path.unlink()
            else:
                continue
        link_path.symlink_to(src_entry, target_is_directory=src_entry.is_dir())
        linked += 1

    return linked


def reset_directory_symlink(link_path: Path, target_path: Path) -> str:
    """Replace a path with a directory symlink and report the action."""
    action = "Created"
    if link_path.exists() or link_path.is_symlink():
        action = "Refreshed" if link_path.is_symlink() else "Replaced"
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()

    link_path.symlink_to(target_path, target_is_directory=True)
    return action


def upsert_marked_block(path: Path, marker: str, block_content: str) -> None:
    """Append or replace a marked instruction block in a markdown file."""
    content = ""
    if path.exists():
        content = path.read_text()

    if marker in content:
        before = content.split(marker, 1)[0].rstrip()
        content = before + "\n\n" + block_content
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += "\n" + block_content

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.lstrip())


def write_json(path: Path, data: dict, *, merge_key: str | None = None) -> None:
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


def write_instruction_file(path: Path, source: Path) -> bool:
    """Write agent instruction file if it doesn't already exist.

    Returns True if a new file was created.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        path.write_text(f"# See CLAUDE.md for full instructions\n\n{INSTRUCTION_STUB}")
    else:
        path.write_text(INSTRUCTION_STUB)
    return True


def build_global_reference_block(project_root: Path | None = None) -> str:
    """Build the global reference block, using Jinja2 if available."""
    if project_root is None:
        project_root = PROJECT_ROOT

    env = get_jinja_env()
    if env is not None:
        template = env.get_template("common/_global_reference.md.j2")
        return template.render(project_root=str(project_root))

    return f"""\
{ATOMISTICSKILLS_GLOBAL_MARKER}

If your current workspace is NOT {project_root} (or any of its subdirectories), and the task involves atomistic research, materials simulation, drug discovery, spectroscopy, ML interatomic potentials, or related scientific workflows:
- The AtomisticSkills repository is installed at {project_root}.
- Rules live at {project_root}/.agents/rules/.
- Skills live at {project_root}/.agents/skills/.
- Workflows live at {project_root}/.agents/workflows/.
- First read these rules:
  - {project_root}/.agents/rules/research-standards.md
  - {project_root}/.agents/rules/coding-standards.md
  - {project_root}/.agents/rules/mcp-environments.md
- For skill discovery, scan descriptions with:
  grep -r "^description:" {project_root}/.agents/skills/*/SKILL.md
- For end-to-end protocols, inspect:
  find {project_root}/.agents/workflows -maxdepth 2 -type f
- Read the full SKILL.md or workflow file before following it.

If your current workspace IS {project_root} or one of its subdirectories, ignore this global reference because the project-local AGENTS.md and project skills are already available.
"""
