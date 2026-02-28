"""
Vendored Widom insertion implementation.

This package is vendored from COFclean to keep the `mat-sorption` skill
self-contained. The public API is re-exported as:

- `WidomInsertionResults`
- `run_widom_insertion()`

Requirements:
    - Conda environment: fairchem-agent
    - Required packages: ase, numpy, pydantic, pymatgen
"""

# Copyright (c) 2025 CuspAI
# Vendored from COFclean external/widom for mat-sorption skill.

from .run import WidomInsertionResults, run_widom_insertion

__all__ = ["WidomInsertionResults", "run_widom_insertion"]
