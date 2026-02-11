---
name: admet-prediction
description: Compute RDKit physicochemical descriptors and rule-based drug-likeness heuristics (Ro5, Veber, QED) from SMILES.
category: drug-discovery
---

# admet-prediction

## Goal
Compute ADMET-relevant **physicochemical descriptors** and **rule-based drug-likeness heuristics** from SMILES strings using RDKit.

This skill reports:
- Core descriptors: molecular weight (average and exact), Wildman-Crippen cLogP, TPSA, HBD/HBA, rotatable bonds, ring counts, aromatic rings, heavy atoms, fractionCSP3, molar refractivity.
- Heuristics:
  - **Lipinski Rule of Five (Ro5)** compliance (≤ 1 violation) as a permeability/absorption triage heuristic.
  - **Veber** oral bioavailability heuristic (RB ≤ 10 and TPSA ≤ 140 Å²; plus reporting the alternative HBD+HBA ≤ 12 condition).
  - **QED** (Quantitative Estimate of Drug-likeness) score.

> Note: This does **not** predict experimental ADMET endpoints (e.g., clearance, CYP inhibition, hERG, Ames, etc.). It is an early-stage physchem/heuristics screen.

## Instructions

1. **Prepare environment**
   - Ensure RDKit is installed in the `drugdisc-agent` conda environment.
   - This skill runs entirely locally (no MCP tool calls).

2. **Single molecule analysis (SMILES → JSON)**
   ```bash
   # Env: drugdisc-agent
   python .agent/skills/admet-prediction/scripts/predict_admet.py \
       --smiles "CC(=O)Oc1ccccc1C(=O)O" \
       --output aspirin_admet.json
   ```

3. **Batch analysis from a SMILES file**
   - Input file format: one molecule per line, `SMILES<whitespace>Name`. Lines starting with `#` are ignored.
   ```bash
   # Env: drugdisc-agent
   python .agent/skills/admet-prediction/scripts/predict_admet.py \
       --smiles_file .agent/skills/admet-prediction/examples/compounds.smi \
       --output batch_admet.json
   ```

4. **Multiple SMILES on the command line**
   ```bash
   # Env: drugdisc-agent
   python .agent/skills/admet-prediction/scripts/predict_admet.py \
       --smiles \
         "CC(=O)Oc1ccccc1C(=O)O" \
         "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
         "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
       --output multi_admet.json
   ```

5. **TPSA mode for S/P-containing molecules (optional)**
   - RDKit's TPSA can optionally include S and P contributions.
   ```bash
   # Env: drugdisc-agent
   python .agent/skills/admet-prediction/scripts/predict_admet.py \
       --smiles "OC(=O)P(=O)(O)O" \
       --include_sandp_tpsa \
       --output foscarnet_admet.json
   ```

## Examples

Example `compounds.smi`:
```text
CN1C=NC2=C1C(=O)N(C(=O)N2C)C	caffeine
CC(=O)Oc1ccccc1C(=O)O	aspirin
CC(C)Cc1ccc(cc1)C(C)C(=O)O	ibuprofen
```

Run:
```bash
# Env: drugdisc-agent
python .agent/skills/admet-prediction/scripts/predict_admet.py \
    --smiles_file .agent/skills/admet-prediction/examples/compounds.smi \
    --output drug_admet.json
```

## Constraints
- **Environment**: Requires `drugdisc-agent` conda environment.
- **Dependencies**: RDKit (Chem, Descriptors, Lipinski, Crippen, QED).
- **Scope**: Outputs physchem descriptors + rule-based heuristics only; not ML/experimental ADMET prediction.
- **Ro5 interpretation**: A "pass" is defined here as **≤ 1 violation** (common industry convention).
- **Veber interpretation**: Primary check uses **TPSA ≤ 140 Å² and rotatable bonds ≤ 10**, and additionally reports the alternative **(HBD + HBA ≤ 12)** criterion.
- **Standardization**: If SMILES contains multiple fragments (e.g., salts, "."), results are reported but flagged with a warning; consider desalting/neutralization upstream for library triage.
- **TPSA option**: By default, TPSA uses RDKit's default behavior (no S/P); `--include_sandp_tpsa` includes S/P contributions.

Author: Matthew Cox
Contact: github username <mcox3406>
