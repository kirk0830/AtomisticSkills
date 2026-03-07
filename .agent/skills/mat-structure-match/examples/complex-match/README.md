# Example: Complex Structure Matching

This example demonstrates how to match an unknown `Li10GeP2S12` (LGPS) structure, a complex solid-state electrolyte, against known polymorphs from the Materials Project.

## Run Instructions

1. Use the `search_materials_project_by_formula` MCP tool with the `return_all=True` flag to download all `Li10GeP2S12` candidate structures. It will produce the candidate files (such as `mp-696138.cif`, `mp-696128.cif`, and `mp-942733.cif`).

2. To see what a successful match to an experimental structure looks like, run the script against `known_experimental.cif`:

```bash
# Env: base-agent
python ../../scripts/match_structure.py known_experimental.cif candidates/ --output experimental_match.json
```
The output JSON will label `match_found: true` and identify the corresponding MP ID with `theoretical: False` (an ICSD-supported structure).

3. To see what a failure to match looks like, run it against `novel_structure.cif`, which is an artificially heavily distorted structural state:

```bash
# Env: base-agent
python ../../scripts/match_structure.py novel_structure.cif candidates/ --output novel_match.json
```
The output JSON will label `match_found: false`.
