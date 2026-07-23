---
name: mat-defect-energy-dft
description: Calculate charged defect formation energies and transition level diagrams using pymatgen-analysis-defects and ASE + Quantum ESPRESSO or CP2K.
category: [materials]
---

# Point-Defect Formation Energy (DFT)

## Goal
To calculate the formation energy of point defects (vacancies, substitutions, interstitials) including **charged defect states** and **finite-size corrections** using DFT via ASE + Quantum ESPRESSO or ASE + CP2K. This produces formation energy diagrams showing defect charge transition levels as a function of Fermi energy.

$$E_f[D^q] = E[D^q] - E[\text{bulk}] + \sum_i \Delta n_i \mu_i + q(E_\text{VBM} + \Delta E_F) + E_\text{corr}$$

where $q$ is the charge state, $E_\text{VBM}$ is the valence band maximum, $\Delta E_F$ is the Fermi energy relative to VBM, and $E_\text{corr}$ is the finite-size correction (simplified Madelung correction).

## Prerequisites / Environment Check

This skill uses ASE + Quantum ESPRESSO or ASE + CP2K. Confirm the variables for your chosen engine are set before running calculations.

- **For Quantum ESPRESSO**:
  - `ESPRESSO_PSEUDO` (required) — Path to a directory containing Quantum ESPRESSO pseudopotentials.
  - `ASE_ESPRESSO_COMMAND` or `ESPRESSO_COMMAND` (optional) — Command to run `pw.x`.
- **For CP2K**:
  - `CP2K_DATA_DIR` (required) — Path to CP2K data directory containing `BASIS_SET` and `POTENTIAL` files.
  - `ASE_CP2K_COMMAND` or `CP2K_COMMAND` (optional) — Command to run CP2K.
- `DFT_ENGINE` (optional) — Global default engine (`qe` or `cp2k`), can be overridden by `--engine`.

See `docs/api_key_guide.md`, `docs/environment_variables.md`, and `docs/hpc_job_submission.md` for setup details.

Before running this skill, verify these variables are set for the chosen engine. If any required variable is missing, ask the user to set it before proceeding.

## Instructions

### 1. Obtain Bulk Structure
Start with a relaxed primitive cell:
```bash
mcp_base_search_materials_project_by_formula(formula="MgO", save_to_file="MgO.cif")
```

### 2. Generate Defect Structures
Use `pymatgen-analysis-defects` to generate all symmetry-unique defect supercells with charge states:
```bash
# Env: base
python .agents/skills/mat-defect-energy-dft/scripts/generate_defect_structures.py \
    --bulk MgO.cif \
    --supercell_size 3 3 3 \
    --defect_type vacancy \
    --charge_range -2 2 \
    --output dft_defects/
```

This generates:
- POSCAR files for each defect × charge state
- A `defect_index.json` mapping defect names to charge states and structures
- Pristine supercell for the bulk reference

### 3. Run DFT Calculations (ASE + QE/CP2K)
Submit calculations via the local ASE runner:

```bash
# Quantum ESPRESSO
# Env: qe
python .agents/skills/mat-defect-energy-dft/scripts/run_defect_calculations.py \
    --engine qe \
    --bulk dft_defects/pristine_supercell.cif \
    --defect-dir dft_defects \
    --defect-index dft_defects/defect_index.json \
    --output-dir dft_defect_calcs \
    --calc-type relax \
    --kpts 2 2 2 \
    --ecutwfc 50 \
    --pseudo-dir $ESPRESSO_PSEUDO

# CP2K
# Env: cp2k
python .agents/skills/mat-defect-energy-dft/scripts/run_defect_calculations.py \
    --engine cp2k \
    --bulk dft_defects/pristine_supercell.cif \
    --defect-dir dft_defects \
    --defect-index dft_defects/defect_index.json \
    --output-dir dft_defect_calcs \
    --calc-type relax \
    --kpts 2 2 2 \
    --cutoff 400 \
    --xc PBE
```

The runner:
- Runs a bulk reference calculation.
- Runs each defect + charge state in its own subdirectory.
- Writes `results.json` (DFTResult format) for each calculation.
- Applies the appropriate total charge for charged defects through engine-specific input.

### 4. Parse Results and Compute Formation Energies
After DFT calculations complete:
```bash
# Env: base
python .agents/skills/mat-defect-energy-dft/scripts/parse_defect_results.py \
    --bulk_dir dft_defect_calcs/pristine_supercell/ \
    --defect_dir dft_defect_calcs/ \
    --defect_index dft_defects/defect_index.json \
    --dielectric 9.8 \
    --output formation_energies.json \
    --plot formation_energy_diagram.png
```

The script:
- Parses `results.json` from ASE/QE/CP2K runners (preferred) or legacy VASP outputs.
- Applies a simplified Madelung finite-size correction for charged defects.
- Determines band gap from the bulk calculation.
- Constructs the formation energy diagram.

### 5. Interpret Results
The formation energy diagram shows:
- **Slopes**: Each line segment has slope = charge state $q$
- **Transition levels**: Intersections where the stable charge state changes ($\epsilon(q/q')$)
- **Low formation energy → high concentration**: Defects with low $E_f$ at the Fermi level are most abundant

## Examples

### Oxygen Vacancy in MgO

```bash
# 1. Get MgO
mcp_base_search_materials_project_by_formula(formula="MgO")

# 2. Generate defects with charges -2 to +2
python .agents/skills/mat-defect-energy-dft/scripts/generate_defect_structures.py \
    --bulk MgO.cif --supercell_size 3 3 3 --defect_type vacancy --charge_range -2 2 --output mgo_defects/

# 3. Run DFT with Quantum ESPRESSO
# Env: qe
python .agents/skills/mat-defect-energy-dft/scripts/run_defect_calculations.py \
    --engine qe \
    --bulk mgo_defects/pristine_supercell.cif \
    --defect-dir mgo_defects \
    --defect-index mgo_defects/defect_index.json \
    --output-dir mgo_dft \
    --calc-type relax \
    --kpts 2 2 2 \
    --pseudo-dir $ESPRESSO_PSEUDO

# 4. Parse and plot (after DFT completes)
# Env: base
python .agents/skills/mat-defect-energy-dft/scripts/parse_defect_results.py \
    --bulk_dir mgo_dft/pristine_supercell/ --defect_dir mgo_dft/ \
    --defect_index mgo_defects/defect_index.json --dielectric 9.8 --output mgo_fe.json
```

## Constraints

- **DFT engine**: Requires Quantum ESPRESSO (`ESPRESSO_PSEUDO`) or CP2K (`CP2K_DATA_DIR`). VASP outputs are still parsed for legacy data.
- **Supercell size**: Use at least 3×3×3 for cubic systems. Charged defect corrections are less reliable for small cells.
- **Dielectric constant**: The simplified Madelung correction requires the static dielectric constant of the host material. Use experimental or computed values.
- **Functional**: PBE underestimates band gaps → transition levels may be shifted. For accurate results, use HSE06 hybrid functional (requires custom input).
- **Environments**:
  - Structure generation: `base` (pymatgen-analysis-defects)
  - DFT submission: `qe` or `cp2k` (ASE local runner)
  - Post-processing: `base`
- **Charged defects**: The ASE runner sets the total cell charge via engine-specific input. For CP2K, verify that the charge is correctly propagated to the generated input.
- **For neutral defects only (no DFT)**: See [mat-defect-energy](../mat-defect-energy/SKILL.md) for MLIP-based approach.

---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
