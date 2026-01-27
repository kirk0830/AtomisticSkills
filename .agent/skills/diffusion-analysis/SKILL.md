---
name: diffusion-analysis
description: Calculate ionic diffusion coefficients and activation energy from MD trajectories using pymatgen.
---

# Diffusion Analysis

## Goal
To accurately calculate the ionic diffusivity ($D$) and activation energy ($E_a$) of specific atomic species in a material using Molecular Dynamics (MD) trajectories.

## Instructions

1.  **MD Preparation**: Run NVT MD simulations at multiple temperatures (e.g., 800K, 1000K, 1200K) using the `run_md` tool. Ensure supercells are > 10 Å and duration is > 50 ps.
2.  **Analysis**: For each temperature, execute the [analyze_diffusion.py](scripts/analyze_diffusion.py) script. This script uses `pymatgen.analysis.diffusion.analyzer` for robust MSD calculation.
    ```bash
    python scripts/analyze_diffusion.py \
        <trajectory_path> --species <element> --temperature <T> --start_skip 50
    ```
3.  **Arrhenius Fitting**: Manually collect the $D$ values and fit to the Arrhenius equation: $\ln(D) = \ln(D_0) - E_a / (k_B T)$.

## Examples

Calculating Li diffusion in a solid electrolyte at 1000K:
```bash
python scripts/analyze_diffusion.py \
    results/md_1000K/trajectory.traj --species Li --temperature 1000
```

## Constraints
- **Trajectory Format**: Must be in ASE `.traj` format.
- **Diffusive Regime**: Ensure the MSD plot (saved to `output_dir`) is linear before accepting results.
- **Atom Count**: Recommended for systems with >100 atoms to reduce noise.
