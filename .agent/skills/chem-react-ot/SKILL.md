---
name: chem-react-ot
description: Generate transition state structures for chemical reactions using React-OT (Optimal Transport).
---

# `chem-react-ot` — React-OT Transition State Generation

Generates transition state (TS) structures given reactant and product structures using the React-OT model (Optimal Transport). React-OT is a generative model that predicts TS geometries directly without requiring an initial guesspath (like NEB).

**Category:** `chemistry`
**Environment:** `react-ot-agent`

## Key Features

- **Generative TS Prediction:** Predicts 3D transition state structures from 3D reactants and products.
- **Fast Inference:** Uses an ODE solver for generation, typically much faster than DFT-based NEB.
- **No Path Guess Required:** Directly generates the TS structure.

## Usage

### 1. Environment Setup

This skill requires the `react-ot-agent` conda environment. Ensure it is installed:

```bash
cd conda-envs/react-ot-agent
bash install.sh
```

### 2. Download Models

Before running the skill for the first time, download the pre-trained model weights:

```bash
# activate react-ot-agent first
conda activate react-ot-agent
python conda-envs/react-ot-agent/download_models.py
```

The checkpoint is saved to `~/.cache/react-ot/checkpoints/sb-pretrained.ckpt`.

### 3. Generate Transition State

Run the generation script with reactant and product files (xyz, cif, pdb, etc. - anything ASE reads).

```bash
# activate react-ot-agent
conda activate react-ot-agent

python .agent/skills/chem-react-ot/scripts/generate_ts.py \
    --reactants reactant.xyz \
    --products product.xyz \
    --output_dir results/ts_search
```

**Arguments:**

- `--reactants`: Path to reactant structure file(s). Can be a single file with multiple molecules or a list of files.
- `--products`: Path to product structure file(s).
- `--output_dir`: Directory to save the generated TS structure (`ts_generated.xyz`) and trajectory (`generation_traj.xyz`).
- `--nfe`: Number of function evaluations for the ODE solver (default: 10). Higher values might be more accurate but slower.
- `--checkpoint`: Path to custom model checkpoint (optional, defaults to downloaded one).

## Example

```bash
python .agent/skills/chem-react-ot/scripts/generate_ts.py \
    --reactants .agent/skills/chem-react-ot/examples/oxadiazole_isomerization/reactant.xyz \
    --products .agent/skills/chem-react-ot/examples/oxadiazole_isomerization/product.xyz \
    --output_dir .agent/skills/chem-react-ot/examples/oxadiazole_isomerization/output
```

## References

- [React-OT GitHub](https://github.com/deepprinciple/react-ot)
- Duan, C. et al. "React-OT: Optimal Transport for Generating Transition State in Chemical Reactions".
