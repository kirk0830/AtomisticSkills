"""Shared ``# Env:`` code-block rewriter.

Converts SKILL.md code blocks from the ``# Env: <name>`` + ``python`` pattern
into pre-filled ``mcp_pixi_run()`` calls that non-shell agents (e.g. astrbot in
non-admin mode) can use directly.

This module is platform-agnostic: it only transforms text and does not depend
on any particular agent's filesystem layout.

Usage::

    from src.config.env_block_rewriter import transform_env_blocks

    rewritten = transform_env_blocks(skill_md_text)
"""

from __future__ import annotations

import re
import shlex


_ENV_BLOCK_RE = re.compile(
    r'```[^\n]*\n(.*?#\s*Env:\s[^\n]*\n.*?)```', re.DOTALL
)
_ENV_LINE_RE = re.compile(r'#\s*Env:\s*(\S+)')


def _extract_env_command(block: str) -> tuple[str, str, list[str]] | None:
    """Parse a ``# Env:`` code block and return ``(env, script, args)`` or ``None``.

    Handles line continuations (trailing ``\\``) and quoted arguments.
    """
    # --- extract environment name ---
    env_match = _ENV_LINE_RE.search(block)
    if not env_match:
        return None
    environment = env_match.group(1)

    # --- collect python command (handle line continuations with \) ---
    lines = block.split("\n")
    cmd_lines: list[str] = []
    in_command = False
    for line in lines:
        stripped = line.strip()
        if in_command:
            # Continuation of previous command
            if not stripped or stripped.startswith("#"):
                break
            cmd_lines.append(stripped.rstrip("\\").strip())
            if not line.rstrip().endswith("\\"):
                break
        elif stripped.startswith("python ") or stripped == "python":
            in_command = True
            cmd_lines.append(stripped[7:].rstrip("\\").strip())  # skip "python "
            if not line.rstrip().endswith("\\"):
                break

    if not cmd_lines:
        return None

    # --- join lines and split ---
    full_cmd = " ".join(cmd_lines).replace("\\'", "'").replace('\\"', '"')
    try:
        parts = shlex.split(full_cmd)
    except ValueError:
        parts = full_cmd.split()

    if not parts:
        return None

    script = parts[0]
    args = parts[1:]

    return environment, script, args


def _format_args_list(args: list[str]) -> str:
    """Format a list of args as a Python-like list string."""
    if not args:
        return "[]"
    return "[" + ", ".join(repr(a) for a in args) + "]"


def _rewrite_one_block(match: re.Match, *, restore_paths: bool = True) -> str:
    """Rewrite a single ``# Env:`` code block into an ``mcp_pixi_run()`` call.

    Parameters:
        match: The regex match object for one code block.
        restore_paths: If ``True``, restore script paths that were previously
            rewritten from ``.agents/skills/`` → ``../`` (e.g. by astrbot's
            sandbox rewriter) back to ``.agents/skills/`` form.  Set to
            ``False`` when the input text still has original paths.
    """
    block = match.group(0)
    parsed = _extract_env_command(block)
    if parsed is None:
        return block

    environment, script, args = parsed

    if restore_paths:
        script = re.sub(r'^\.\./([\w\-]+)/', r'.agents/skills/\1/', script)

    args_str = _format_args_list(args)
    lines = block.split("\n")
    fence = lines[0]  # ```bash or ```python etc.

    return "\n".join([
        fence,
        f"# Env: {environment}",
        "# Call via pixi MCP server:",
        f"mcp_pixi_run(environment={repr(environment)}, "
        f"script={repr(script)}, args={args_str})",
        "```",
    ])


def transform_env_blocks(text: str, *, restore_paths: bool = True) -> str:
    """Find ``# Env:`` code blocks in *text* and replace each with a
    pre-filled ``mcp_pixi_run()`` call.

    Parameters:
        text: The markdown content to transform.
        restore_paths: Whether to restore script paths from sandbox-relative
            (``../…``) to project-relative (``.agents/skills/…``).

    Returns:
        Transformed text with all ``# Env:`` blocks rewritten.
    """
    import functools

    return _ENV_BLOCK_RE.sub(
        functools.partial(_rewrite_one_block, restore_paths=restore_paths),
        text,
    )
