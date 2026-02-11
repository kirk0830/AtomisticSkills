---
name: protein-prep
description: Prepare macromolecular receptor structures (PDB/mmCIF or RCSB PDB ID) for docking or simulation by fixing common structure issues, adding hydrogens, and optionally generating AutoDock/Vina receptor PDBQT via Meeko.
category: drug-discovery
---

# protein-prep

## Goal
To prepare protein (and optionally nucleic acid) receptor structures for molecular docking (e.g., AutoDock Vina / AutoDock-GPU) or downstream simulation by:
1) retrieving coordinates from RCSB PDB (optional),
2) fixing common structural issues (missing atoms, nonstandard residues),
3) adding hydrogens at a target pH, and
4) optionally generating a receptor **PDBQT** using **Meeko's `mk_prepare_receptor.py`**, consistent with the AutoDock/Vina recommended workflow.

**Important context**: Docking accuracy is often limited by receptor preparation choices (protonation states, alternate locations, missing atoms/residues, and whether to retain key waters/cofactors). This skill automates the *baseline* workflow and produces a detailed JSON report so you can audit those choices.

## Instructions

### 1. Choose the right coordinate model (assembly vs asymmetric unit)
Many PDB entries have different "asymmetric unit" vs "biological assembly" coordinate sets. For docking against the biologically relevant oligomer/interface, prefer the biological assembly when appropriate.
See RCSB guidance on biological assemblies and file download services.

### 2. Prepare a docking-ready receptor from an RCSB PDB ID
This will:
- fetch the structure,
- (by default) **NOT** build missing residues (loops),
- replace common nonstandard residues when possible,
- remove waters + heterogens (configurable),
- add missing heavy atoms and hydrogens,
- write a prepared PDB + JSON summary,
- and generate receptor PDBQT using Meeko.

```bash
# Env: drugdisc-agent
python .agent/skills/protein-prep/scripts/prepare_protein.py \
  --pdb_id 1iep \
  --chains A \
  --ph 7.0 \
  --heterogens none \
  --missing_residues ignore \
  --output_dir protein_prep/
```

### 3. Keep cofactors/metal ions but remove crystallographic waters

If your binding site depends on a metal or cofactor (e.g., ZN, MG, HEM), keep it. A common choice is to remove water but keep non-water heterogens:

```bash
# Env: drugdisc-agent
python .agent/skills/protein-prep/scripts/prepare_protein.py \
  --pdb_id 1iep \
  --chains A \
  --heterogens non-water \
  --delete_resname SO4 GOL \
  --output_dir protein_prep_keep_cofactors/
```

### 4. Use a biological assembly (recommended when oligomerization matters)

```bash
# Env: drugdisc-agent
python .agent/skills/protein-prep/scripts/prepare_protein.py \
  --pdb_id 1iep \
  --assembly 1 \
  --chains A \
  --output_dir protein_prep_assembly1/
```

### 5. Prepare from a local structure file

```bash
# Env: drugdisc-agent
python .agent/skills/protein-prep/scripts/prepare_protein.py \
  --pdb_file receptor.pdb \
  --heterogens none \
  --output_dir protein_prep_local/
```

### 6. Skip PDBQT generation (simulation-focused prep)

```bash
# Env: drugdisc-agent
python .agent/skills/protein-prep/scripts/prepare_protein.py \
  --pdb_id 1iep \
  --no_pdbqt \
  --output_dir protein_prep_pdb_only/
```

### 7. Validate the output (strongly recommended)

After preparation:

* Inspect the JSON summary for **missing residues**, **nonstandard residue replacements**, and **atoms added**.
* Visually inspect the binding site and check for:
  * correct oligomeric state,
  * retained/removed cofactors and metal ions,
  * sensible protonation (especially histidines),
  * alternate locations resolved appropriately.

If protonation is critical, consider a hydrogen optimization / pKa-aware tool (e.g., Reduce/Reduce2, PROPKA/PDB2PQR/H++), then regenerate PDBQT from the protonated receptor.

## Examples

Preparing the c-Abl kinase domain receptor (Vina tutorial target) from PDB ID 1IEP:

```bash
# Env: drugdisc-agent
python .agent/skills/protein-prep/scripts/prepare_protein.py \
  --pdb_id 1iep \
  --chains A \
  --heterogens none \
  --missing_residues ignore \
  --ph 7.0 \
  --output_dir examples/1iep_receptor/
```

Outputs:

* `examples/1iep_receptor/1IEP_prepared.pdb` -- cleaned receptor with hydrogens
* `examples/1iep_receptor/1IEP_prepared.pdbqt` -- Vina/AutoDock-ready receptor
* `examples/1iep_receptor/1IEP_prepared.meeko.json` -- Meeko receptor config (useful for later export tooling)
* `examples/1iep_receptor/1IEP_summary.json` -- full preparation summary

## Constraints

* **Environment**: Requires `drugdisc-agent`.
* **Core dependencies**: `pdbfixer`, `openmm`, `meeko`. (Meeko must provide `mk_prepare_receptor.py` on PATH.)
* **Protonation is not "solved"**: Default pH-based hydrogen addition is a baseline. Binding-site protonation can dominate docking outcomes; treat this as a decision point, not a checkbox.
* **Missing residues**: By default, missing residues are not modeled; this avoids introducing uncertain loop models. If your binding site is incomplete, consider curated modeling.
* **Waters/cofactors**: The default removes them; retain only those known or strongly suspected to be functionally important.
* **Metals / covalent complexes / post-translational modifications**: Docking force fields and receptor typing can be fragile here; manual inspection and/or specialized protocols may be required.
* **PDBQT generation**: This skill uses Meeko's recommended receptor preparation interface rather than ad-hoc conversion.

Author: Matthew Cox
Contact: github username <mcox3406>
