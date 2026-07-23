"""Utilities for proactively checking environment variables required by tools/skills.

All API keys, tokens, binary paths, and other external credentials used by the
project should be checked through these helpers so that the agent can surface
clear, actionable messages to the user before an external call fails.
"""

import os
from typing import Dict, List, Optional, Tuple


DOCS_LINK = "See docs/api_key_guide.md and docs/environment_variables.md for details."


def _format_missing_messages(
    missing: List[Tuple[str, str]],
    level: str = "required",
) -> str:
    """Format a human-readable message for missing environment variables.

    Args:
        missing: List of (variable_name, purpose) tuples.
        level: Either "required" or "recommended"; used in the header.

    Returns:
        A formatted multi-line message with a docs pointer.
    """
    header = (
        f"Missing {level} environment variable(s):"
        if level == "required"
        else f"Recommended environment variable(s) not set:"
    )
    lines = [header]
    for var_name, purpose in missing:
        lines.append(f"  - {var_name}: {purpose}")
    lines.append(DOCS_LINK)
    return "\n".join(lines)


def check_required_env_vars(required: Dict[str, str]) -> Optional[str]:
    """Check that all required environment variables are set.

    Args:
        required: Mapping from variable name to a short human-readable purpose.

    Returns:
        ``None`` if all variables are set, otherwise a formatted message
        explaining what is missing and where to find setup instructions.
    """
    missing: List[Tuple[str, str]] = []
    for var_name, purpose in required.items():
        if not os.getenv(var_name):
            missing.append((var_name, purpose))
    if not missing:
        return None
    return _format_missing_messages(missing, level="required")


def check_recommended_env_vars(recommended: Dict[str, str]) -> Optional[str]:
    """Check that recommended environment variables are set.

    Args:
        recommended: Mapping from variable name to a short human-readable purpose.

    Returns:
        ``None`` if all variables are set, otherwise a formatted warning
        message that can be prepended to a successful result.
    """
    missing: List[Tuple[str, str]] = []
    for var_name, purpose in recommended.items():
        if not os.getenv(var_name):
            missing.append((var_name, purpose))
    if not missing:
        return None
    return _format_missing_messages(missing, level="recommended")


def check_env_vars(
    required: Optional[Dict[str, str]] = None,
    recommended: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Check both required and recommended environment variables.

    Args:
        required: Required environment variables.
        recommended: Recommended (optional but encouraged) environment variables.

    Returns:
        A tuple ``(required_message, recommended_message)``.  Either entry is
        ``None`` when the corresponding group is fully set.
    """
    required_msg = None
    recommended_msg = None
    if required:
        required_msg = check_required_env_vars(required)
    if recommended:
        recommended_msg = check_recommended_env_vars(recommended)
    return required_msg, recommended_msg
