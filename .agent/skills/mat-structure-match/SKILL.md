---
name: mat-structure-match
description: Determine if a given structure matches known experimental or theoretical structures, or compare two user-provided structures.
category: materials
---

# mat-structure-match

## Goal
To determine whether a user-provided structure (or list of structures) has been previously reported. This is done by matching the target structure against:
1. Known polymorphs in the Materials Project (MP). MP entries contain `theoretical` tags, and experimental structures will overlap with the Inorganic Crystal Structure Database (ICSD).
2. Any arbitrary candidate structure(s) provided by the user.

## Instructions

### Step 1: Search Materials Project for Candidate Structures
Before attempting to match a crystal structure, you need to download all known polymorphs for its chemical formula. Use the MP MCP tool `search_materials_project_by_formula` with the `return_all=True` flag to ensure you get *all* matching structures (both theoretical and experimental/ICSD), not just the ground state.

```bash
mcp_search_materials_project_by_formula(
    formula="Li10GeP2S12", # Formula of the target structure
    return_all=True, # Critical: Gets all polymorphs
    save_to_file="candidates" # Directory to save the candidates
)
```

**Note on ICSD:** MP ingests a large subset of experimental structures from ICSD, but **it does not contain the entire, up-to-date ICSD database**. MP typically includes older, well-ordered ICSD entries and may exclude recently added or highly disordered structures. Experimental structures will appear as `theoretical=False` in MP and will often contain ICSD IDs in their metadata. If you need to match against the *entire* ICSD, you must provide your own exported ICSD candidate files, as the full ICSD is proprietary and not available via public API.

### Step 2: Match Target Structure Against Candidates
Use the `match_structure.py` script to perform a symmetry-aware structural comparison between the user's structure and the candidates.

```bash
# Env: base-agent
python .agent/skills/mat-structure-match/scripts/match_structure.py target_structure.cif candidates/ --output match_results.json
```

**Custom Candidate Matching:** If a user provides two distinct structures and simply wants to know if they match each other, you can bypass Step 1 and provide the second structure as the candidate:

```bash
# Env: base-agent
python .agent/skills/mat-structure-match/scripts/match_structure.py target_structure_1.cif target_structure_2.xyz --output match_results.json
```

**Literature Fallback (Novel/Unmatched Structures):** 
If the script fails to find any *structural* match among the candidates, it will automatically extract the exact symmetry space group of your input structure (e.g., `P-31m`). It then queries the OpenAlex database for the target structure's chemical formula **AND** its crystal system/space group (e.g., `"Li2ZrCl6" AND ("P-31m" OR "trigonal")`). This ensures that even if a material (like the trigonal `Li2ZrCl6` from arXiv:2403.08237) is missing from Materials Project, the script will guarantee the literature reporting it actually matches your specific *polymorph*. It will output `literature_reported: true` and list the recent publications confirming its existence.

### Step 3: Literature Search (Optional)
If the user wants to know if the *composition* itself has been heavily studied or synthesized in particular conditions, perform a literature search using the `search_literature` MCP tool. 

```bash
mcp_search_literature(
    query="synthesis of Li10GeP2S12",
    limit=5,
    download=False
)
```

## Examples

Checking if a generated structure has been experimentally reported:
```bash
# Env: base-agent
mcp_search_materials_project_by_formula(formula="LiFePO4", return_all=True, save_to_file="LiFePO4_candidates")
python .agent/skills/mat-structure-match/scripts/match_structure.py generated_LFP.cif LiFePO4_candidates/ --output match_results.json
```

## Constraints
- **Environments**: The script uses the standard `StructureMatcher` from `pymatgen`. It MUST be run in the `base-agent` environment.
- **Match Tolerances**: The structural matching relies on fractional length tolerance (`--ltol`), site tolerance (`--stol`), and angle tolerance (`--angle_tol`). The defaults (0.2, 0.3, 5.0) are typically suitable for DFT-relaxed comparison against MP structures, but can be tweaked if the test structure is highly distorted or unrelaxed.

---

**Author:** Bowen Deng  
**Contact:** [GitHub @bowen-bd](https://github.com/bowen-bd)
