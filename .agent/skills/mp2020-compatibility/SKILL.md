---
name: mp2020-compatibility
description: Energy corrections needed when using certain MLIPs for phase diagram construction / formation energy calculations.
---

# MP2020 Compatibility

## Goal
To apply [Materials Project 2020 Compatibility](https://pymatgen.org/pymatgen.entries.compatibility.html) schemes to MLIP-predicted energies. This is required for models trained on GGA/GGA+U mixed data (e.g., MPtrj data) when constructing convex hulls or phase diagrams to ensure compatibility with the Materials Project database.

> [!IMPORTANT]
> **Do NOT apply this to r2SCAN models.** 
> Only use this for models trained on GGA/GGA+U mixed data.

## Compatible Models
This correction is **REQUIRED** for:
- **MACE-MH-1** `omat_pbe` head (default)
- **MACE-MH-0** `omat_pbe` head
- **MACE-OMAT-0-small**, **MACE-OMAT-0-medium**
- **MACE-MP-0-small**, **MACE-MP-0-medium**, **MACE-MP-0-large** (Legacy)
- **M3GNet-MP-2021.2.8-PES**
- **M3GNet-MP-2021.2.8-DIRECT-PES**
- **uma-s-1**, **uma-s-1p1**, **uma-m-1p1** (with `omat` head)

This correction is **NOT** for:
- **CHGNet** (e.g. `CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES`)
- **MACE-MATPES-r2SCAN-0**
- **TensorNet-MatPES-r2SCAN-v2025.1-PES**
- **SevenNet** (if trained on r2SCAN)

> [!NOTE]
> The full list of compatible models can be found in `resources/gga-ggau-mixed-mlips.yaml`.

## Instructions

### 1. Check Compatibility
Use the `check_compatibility.py` script to programmatically determine if a model/head requires correction.

```bash
python .agent/skills/mp2020-compatibility/scripts/check_compatibility.py --name "MACE-MH-1" --head "omat_pbe"
# Exit code 0 if required, 1 if not.
```

### 2. Apply Correction to a Structure
Use the `apply_correction.py` script to calculate the corrected energy for a single structure.

To run on a directory of structure files (batch mode):
```bash
# Energy defaults to 0.0 if not specified (useful for just checking corrections)
python .agent/skills/mp2020-compatibility/scripts/apply_correction.py /path/to/structure_dir/
```

To run on a specific file with known energy:
```bash
# Env: base-agent
python .agent/skills/mp2020-compatibility/scripts/apply_correction.py structure.cif --energy -123.45
```

### 2. Batch Processing
For referencing or phase diagram generation, apply this correction to every entry before computing E_hull.

## Constraints
- **Environment**: `base-agent` (requires `pymatgen`).
- **Input Energy**: Must be the *total energy* in eV (not per atom).
