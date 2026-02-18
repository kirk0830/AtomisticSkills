---
name: chem-conformer-search
description: Generate molecular conformers with RDKit ETKDG, relax with MLIPs, and rank by energy with Boltzmann weighting.
category: chemistry
---

# Molecular Conformer Search & Ranking

## Goal

Generate a diverse ensemble of low-energy conformers for a given molecule. The workflow combines:
1.  **Stochastic sampling** using RDKit's ETKDG algorithm (Experimental Torsion Distance Geometry).
2.  **High-accuracy relaxation** using Machine Learning Interatomic Potentials (MLIPs) to get near-DFT quality geometries and energies.
3.  **Deduplication** and **Boltzmann weighting** to identify the most relevant conformers at finite temperature.

> [!IMPORTANT]
> This skill is optimized for **organic molecules** and uses `MACE-OFF23` models by default. For inorganic clusters, switch to `MACE-OMAT` or `MatGL` models.

## 1. Prerequisites

- **Conda Environment**: `mace-agent` (recommended as it includes both `mace` and `rdkit`).
- **Input**: SMILES string or a structure file (`.xyz`, `.sdf`, `.mol2`, `.pdb`).

## 2. Methodology

1.  **Generation**: Generate `N` initial conformers using RDKit's `EmbedMultipleConfs` with ETKDGv3.
2.  **Relaxation**: Optimize the geometry of *each* conformer using the selected MLIP (fmax = 0.01 eV/Å).
3.  **Deduplication**: Compare relaxed conformers by RMSD (Root Mean Square Deviation). If RMSD < threshold (default 0.1 Å), keep only the lower-energy one.
4.  **Ranking**: Sort unique conformers by energy.
5.  **Boltzmann Weighting**: Calculate population probability $P_i$ at temperature $T$:
    $$P_i = \frac{e^{-(E_i - E_{min}) / k_B T}}{\sum_j e^{-(E_j - E_{min}) / k_B T}}$$

## 3. Usage

### Basic Usage (SMILES)

Generate 30 conformers for a molecule (e.g., aspirin) and relax with MACE-OFF23:

```bash
# Env: mace-agent
python .agent/skills/chem-conformer-search/scripts/conformer_search.py \
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \
    --num_conformers 30 \
    --output_dir research/aspirin_conformers
```

### Advanced Usage (Structure File + Options)

```bash
# Env: mace-agent
python .agent/skills/chem-conformer-search/scripts/conformer_search.py \
    --structure my_molecule.sdf \
    --num_conformers 100 \
    --rms_threshold 0.5 \
    --dedup_threshold 0.1 \
    --temperature 298.15 \
    --model_type mace \
    --model_name MACE-OFF23-small \
    --device cuda \
    --output_dir research/my_molecule_search
```

### Key Parameters

| Argument | Default | Description |
|:---|:---|:---|
| `--smiles` | - | SMILES string of the molecule |
| `--structure` | - | Path to input structure file (alternative to SMILES) |
| `--num_conformers` | 50 | Number of initial conformers to generate with RDKit |
| `--rms_threshold` | 0.2 | RDKit pruning threshold (Å) to discard similar initial conformers |
| `--dedup_threshold` | 0.1 | Post-relaxation RMSD threshold (Å) to merge identical conformers |
| `--fmax` | 0.01 | Force convergence criterion for relaxation (eV/Å) |
| `--temperature` | 298.15 | Temperature (K) for Boltzmann weighting |
| `--model_type` | `mace` | MLIP backend (`mace`, `matgl`, `fairchem`) |
| `--model_name` | `MACE-OFF23-small` | Specific model checkpoint to use |

## 4. Output Files

The output directory will contain:

1.  **`conformer_results.json`**: A summary file containing:
    -   List of unique conformers with their energies, relative energies, and Boltzmann weights.
    -   Paths to the corresponding XYZ files.
    -   Metadata (model used, parameters).
2.  **`conf_000.xyz`, `conf_001.xyz`, ...**: The relaxed structures of the unique conformers, sorted by energy (000 is the global minimum found).

## 5. Examples

See `examples/aspirin` for a complete example run on Acetylsalicylic acid.

## 6. References

- **ETKDG**: Riniker, S.; Landrum, G. A. "Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation", *J. Chem. Inf. Model.* **2015**, 55, 2562.
- **MACE-OFF23**: Kovács, D. P. et al. "MACE-OFF23: Transferable Machine Learning Force Fields for Organic Molecules", *arXiv* **2023**.
