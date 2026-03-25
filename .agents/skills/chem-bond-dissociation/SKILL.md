---
name: chem-bond-dissociation
description: Calculate homolytic and heterolytic bond dissociation energies (BDEs) for all single bonds in a molecule using MLIPs with RDKit fragmentation.
category: [chemistry]
---

# Bond Dissociation Energy Skill

## Goal

Calculate the **homolytic** and/or **heterolytic** bond dissociation energy (BDE) for each single bond in a molecule using Machine Learning Interatomic Potentials (MLIPs).

**Homolytic BDE** (radical fragments):
$$\text{BDE}_\text{homo}(A{-}B) = E(A\bullet) + E(B\bullet) - E(A{-}B)$$

**Heterolytic BDE** (ionic fragments, minimum over both polarity variants):
$$\text{BDE}_\text{hetero}(A{-}B) = \min\!\bigl(E(A^+)+E(B^-),\; E(A^-)+E(B^+)\bigr) - E(A{-}B)$$

> [!IMPORTANT]
> This skill computes BDEs by relaxing both the intact molecule and fragments with an MLIP. For purpose-trained GNN models that predict BDE directly from SMILES (MAE ~0.6 kcal/mol), consider [ALFABET](https://bde.ml.nrel.gov) or BonDNet instead.

## Background

BDE is a fundamental thermodynamic quantity that determines:
- **Drug metabolism**: CYP450 enzymes abstract H from the weakest C–H bond
- **Electrolyte stability**: Which bonds break first under electrochemical voltage
- **Combustion chemistry**: Rate-determining bond-breaking steps in fuel oxidation
- **Polymer degradation**: Weakest links in polymer backbone chains

A 2024 study (Zubatyuk et al., *JCTC*) demonstrated that MACE potentials achieve BDE RMSE of 1.37 kcal/mol for aliphatic C–H bonds in drug-like molecules, outperforming semi-empirical methods and ALFABET for BDE **ranking**.

## 1. Prerequisites

- **Conda Environment**: `mace-agent` (includes RDKit, ASE, and MACE)
- **Input**: SMILES string or structure file (`.sdf`, `.mol2`)
- **RDKit**: Required for bond identification and molecular fragmentation

## 2. Choosing a Foundation Potential

Refer to the [foundation-potentials skill](../ml-foundation-potentials/SKILL.md) for model selection.

> [!IMPORTANT]
> **Model requirements by cleavage mode:**
>
> | Mode | Recommended model | `supports_charge_spin` |
> |:---|:---|:---|
> | `homolytic` | `MACE-OFF23-small/medium/large` | Not required |
> | `heterolytic` or `both` | `MACE-OMOL-extra-large` | ✅ Required |
> | `heterolytic` or `both` | FairChem UMA/ESEN with `--task_name omol` | ✅ Required |
>
> If you request `--cleavage both` with a model that does **not** support charge/spin,
> the skill will log a warning and silently fall back to homolytic-only.
> Using `--cleavage heterolytic` with an unsupported model raises an error.

## 3. Calculation Workflow

### Step 1: Provide a molecule

```bash
# SMILES input (most common)
--smiles "CCO"

# Or from a structure file
--structure molecule.sdf
```

### Step 2: Run BDE calculation

**Homolytic only** (default, no charge/spin needed):
```bash
# Env: mace-agent
python .agents/skills/chem-bond-dissociation/scripts/calculate_bde.py \
    --smiles CCO \
    --all_bonds \
    --cleavage homolytic \
    --model_type mace \
    --model_name MACE-OFF23-small \
    --output_dir research/my_folder/bde_results
```

**Both homolytic and heterolytic** (requires charge/spin-aware model):
```bash
# Env: mace-agent
python .agents/skills/chem-bond-dissociation/scripts/calculate_bde.py \
    --smiles CCO \
    --all_bonds \
    --cleavage both \
    --model_type mace \
    --model_name MACE-OMOL-extra-large \
    --output_dir research/my_folder/bde_results_both
```

**Heterolytic only** with FairChem:
```bash
# Env: fairchem-agent
python .agents/skills/chem-bond-dissociation/scripts/calculate_bde.py \
    --smiles CCO \
    --all_bonds \
    --cleavage heterolytic \
    --model_type fairchem \
    --model_name uma-s-1p1 \
    --task_name omol \
    --output_dir research/my_folder/bde_hetero
```

### Key Parameters

| Argument | Default | Description |
|:---|:---|:---|
| `--smiles` | — | SMILES string of the molecule |
| `--structure` | — | Path to structure file (`.sdf`, `.mol2`) |
| `--bond` | — | Specific bond as atom indices `"i-j"` (0-indexed) |
| `--all_bonds` | `True` | Compute BDE for all single bonds |
| `--include_h_bonds` | `False` | Include X–H bonds |
| `--cleavage` | `homolytic` | `homolytic`, `heterolytic`, or `both` |
| `--model_type` | `mace` | MLIP backend (`mace`, `matgl`, `fairchem`) |
| `--model_name` | auto | Model checkpoint (default: `MACE-OFF23-small` for homolytic, `MACE-OMOL-extra-large` for hetero/both) |
| `--task_name` | — | Task head for multi-task models (e.g. `omol` for FairChem UMA) |
| `--fmax` | `0.01` | Force convergence for relaxation (eV/Å) |
| `--output_dir` | required | Output directory |

## 4. Output Files

- **`bde_results.json`** — Full results including:
  - `metadata`: model name, cleavage mode, `supports_charge_spin`, SMILES, etc.
  - `intact_energy_eV`: Energy of the relaxed intact molecule
  - `bonds`: List of per-bond results:
    - `bde_eV`, `bde_kJ_mol`, `bde_kcal_mol`: Homolytic BDE (if computed)
    - `heterolytic_bde_eV`, `heterolytic_bde_kJ_mol`, `heterolytic_bde_kcal_mol`: Best heterolytic BDE (if computed)
    - `heterolytic_best_variant`: Which polarity won (`"frag1+ / frag2-"` or `"frag1- / frag2+"`)
    - `heterolytic_variants`: Raw results for both polarity variants
  - `weakest_bond_homolytic`, `weakest_bond_heterolytic`: Summary of weakest bonds
  - `bonds_ranked_by_homolytic_bde`, `bonds_ranked_by_heterolytic_bde`: Sorted tables

- **`intact_relaxed.xyz`**: Relaxed intact molecule
- **`frag_bond{N}_homo_{1,2}.xyz`**: Homolytic radical fragments
- **`frag_bond{N}_hetero_pos_neg_{1,2}.xyz`**: Heterolytic cation/anion fragments (variant A)
- **`frag_bond{N}_hetero_neg_pos_{1,2}.xyz`**: Heterolytic anion/cation fragments (variant B)

## 5. Examples

### Ethanol — Homolytic BDE

```bash
# Env: mace-agent
python .agents/skills/chem-bond-dissociation/scripts/calculate_bde.py \
    --smiles CCO \
    --all_bonds \
    --include_h_bonds \
    --cleavage homolytic \
    --model_type mace \
    --model_name MACE-OFF23-small \
    --output_dir .agents/skills/chem-bond-dissociation/examples/ethanol
```

Experimental BDEs for ethanol (Blanksby & Ellison, 2003):
| Bond | Experimental BDE (kcal/mol) |
|:---|:---|
| O–H | ~104 |
| C–H (methyl) | ~101 |
| C–H (methylene) | ~95 |
| C–C | ~85 |
| C–O | ~92 |

## 6. Constraints

- **Radical spin states**: For homolytic BDE, MLIPs are generally "electron-agnostic" and treat fragments as neutral regardless of spin state. BDE **ranking** is typically more reliable than absolute values.
- **Ionic states**: Heterolytic BDE requires MACE-OMOL or FairChem (omol task), which are trained on charged molecules. Other models silently ignore charge/spin (wrapper.supports_charge_spin == False), so they cannot be used for heterolytic BDE.
- **Ring bonds**: Breaking bonds in rings produces a single open-chain diradical. The script will warn and skip ring bonds.
- **Accuracy**: Expect ~2–5 kcal/mol error for homolytic BDEs with MACE-OFF23. Heterolytic accuracy is less benchmarked with current MLIPs.
- **Environments**:
  - `mace-agent` for MACE models
  - `matgl-agent` for MatGL/CHGNet models (homolytic only)
  - `fairchem-agent` for FairChem/UMA models

## References

- Blanksby & Ellison, "Bond Dissociation Energies of Organic Molecules", *Acc. Chem. Res.* **2003**, 36, 255.
- St. John et al., "Prediction of organic homolytic bond dissociation enthalpies at near chemical accuracy with sub-second computational cost", *Nat. Commun.* **2020**, 11, 2328. (ALFABET)
- Zubatyuk et al., "A Transferable MACE Potential for Open- and Closed-Shell Drug-Like Molecules", *J. Chem. Theory Comput.* **2024**.

---

**Author:** Bowen Deng  
**Contact:** [GitHub @bowen-bd](https://github.com/bowen-bd)
