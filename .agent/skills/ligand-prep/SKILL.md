---
name: ligand-prep
description: Prepare small-molecule ligands for docking/analysis: optional state enumeration, 3D conformer generation, MMFF/UFF minimization, and export to SDF + AutoDock PDBQT.
category: drug-discovery
---

# Ligand Preparation

## Goal
To prepare small-molecule ligands for molecular docking and downstream analysis by:
1) optionally enumerating relevant ligand ionization states (pH range) and tautomers,
2) generating 3D conformers with RDKit ETKDG,
3) minimizing with MMFF94 (fallback UFF),
4) exporting a docking-ready **PDBQT** (AutoDock-Vina) and an optimized **SDF**.

This skill emphasizes reproducibility and state correctness (protonation/tautomer/stereo), which are known major failure modes in docking workflows.

## Instructions

### 0. Decide ligand chemical state (critical)
Before generating 3D structures, ensure the ligand is **unambiguously specified**:
- Resolve stereochemistry (avoid undefined chiral centers).
- Consider multiple **protonation states** in a relevant pH window (often ~7.0–7.4, but depends on assay/compartment).
- Consider relevant **tautomers** when chemically ambiguous.

This script can *optionally* enumerate protonation states using Dimorphite-DL (open source, pH-window enumeration).

### 1. Prepare a ligand from SMILES (single molecule)
```bash
# Env: drugdisc-agent
python .agent/skills/ligand-prep/scripts/prepare_ligand.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --name aspirin \
  --output_dir ligand_prep/aspirin \
  --num_confs 50 \
  --prune_rms 0.5 \
  --max_iters 500
```

### 2. Prepare a ligand library from SDF (multi-molecule)

Processes **all molecules** in the SDF by default and uses each record's `_Name` when available.

```bash
# Env: drugdisc-agent
python .agent/skills/ligand-prep/scripts/prepare_ligand.py \
  --sdf ligands.sdf \
  --output_dir ligand_prep/library \
  --num_confs 25 \
  --write_all_confs
```

### 3. Batch preparation from a SMILES list

Input file format:

* `SMILES<TAB>NAME` (recommended)
* or `SMILES NAME`

```bash
# Env: drugdisc-agent
python .agent/skills/ligand-prep/scripts/prepare_ligand.py \
  --smiles_file ligands.smi \
  --output_dir ligand_prep/batch \
  --num_confs 30
```

### 4. Enumerate protonation states (optional, recommended when ambiguous)

Enumerate protonation states in a pH window and prepare **each generated state**.

```bash
# Env: drugdisc-agent
python .agent/skills/ligand-prep/scripts/prepare_ligand.py \
  --smiles "CC(=O)O" \
  --name acetic_acid \
  --output_dir ligand_prep/acetic_acid_states \
  --enumerate_protomers \
  --ph_min 6.8 \
  --ph_max 7.4 \
  --max_variants 16
```

### 5. Skip PDBQT generation

```bash
# Env: drugdisc-agent
python .agent/skills/ligand-prep/scripts/prepare_ligand.py \
  --smiles "c1ccccc1" \
  --name benzene \
  --no_pdbqt \
  --output_dir ligand_prep/benzene
```

### 6. Validate outputs (recommended)

* Confirm the final **formal charge** and that it matches your intended protomer.
* If stereochemistry is unassigned, fix input and re-run.
* If you enumerate states, dock *all major states* unless you have strong evidence to discard them.

## Examples

Prepare ibuprofen for docking:

```bash
# Env: drugdisc-agent
python .agent/skills/ligand-prep/scripts/prepare_ligand.py \
  --smiles "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O" \
  --name ibuprofen \
  --output_dir ligand_prep/ibuprofen \
  --num_confs 50
```

Expected outputs (per ligand/state):

* `<name>.sdf` - optimized 3D structure (best conformer)
* `<name>.pdbqt` - docking-ready PDBQT (if enabled)
* `<name>_allconfs.sdf` - optional ensemble dump
* `preparation_summary.json` - batch summary for the run

## Constraints

* **Environment**: Requires `drugdisc-agent`.
* **Core dependencies**: RDKit, Meeko. Meeko parameterizes ligands (atom types, partial charges, rotatable bonds) and writes PDBQT for AutoDock-Vina/AutoDock-GPU.
* **Docking format**: PDBQT includes polar H, partial charges, and AutoDock atom types (and flexibility metadata).
* **Force fields**: Uses MMFF94 for drug-like organics; falls back to UFF when MMFF typing is unavailable.
* **Ionization states**: Optional enumeration uses Dimorphite-DL (empirical rules + pH window). This is not a full pKa predictor; treat results as candidates to dock/inspect.
* **Tautomers**: Optional tautomer enumeration is heuristic; ambiguous systems may require specialized tools or manual curation.

Author: Matthew Cox
Contact: github username <mcox3406>
