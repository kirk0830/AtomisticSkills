---
trigger: model_decision
description: Stress units in MLIPs, ASE calculators, and DFT labels (eV/A^3 vs. GPa)
---

# Stress Unit Standards

Standardizing stress units is critical for consistency between DFT results, MLIP training, and atomistic simulations.

## Project Standard: eV/Å³
Following the **ASE (Atomic Simulation Environment)** standard, all internal representations of stress in this project use **eV/Å³**.

| Component | Standard Unit | Notes |
| :--- | :--- | :--- |
| **ASE Calculator** | `eV/Å³` | Standard ASE behavior. |
| **Static Prediction** | `eV/Å³` | Returned by `base.py`'s `static_calculation`. |
| **Training Labels** | `eV/Å³` | Saved to `training_data.json` by Atomate2/Parsing tools. |
| **Simulations** | `eV/Å³` | MatCalc (MD, Relaxation) expects this magnitude. |
| **VASP (Raw)** | `kB` | Converted to eV/Å³ during parsing (`kB * 0.00062415`). |
| **VASP (Report)** | `GPa` | VASP OUTCAR often reports GPa; we standardize to eV/Å³. |

## MLIP Trainer Requirements
Some MLIP trainers expect different units for their loss functions. Conversions are handled automatically inside the respective `fine_tune` methods:

- **MACE**: Targets `eV/Å³` (No conversion needed).
- **FAIRCHEM**: Targets `eV/Å³` (No conversion needed).
- **MatGL**: Targets **GPa**. Labels are converted from `eV/Å³` $\to$ `GPa` inside `MATGLWrapper._prepare_training_data`.

## Common Conversion Factors
- **1 GPa** $\approx$ **0.0062415 eV/Å³** (`ase.units.GPa`)
- **1 eV/Å³** $\approx$ **160.2176 GPa** (`1.0 / ase.units.GPa`)
- **1 kB (VASP)** = **0.1 GPa** $\approx$ **0.00062415 eV/Å³**
