# Model-Specific LAMMPS Scripts

## Goal
Provide one runnable script per MLIP backend for `mat-lammps-md`.

## Current Coverage
- MatGL/CHGNet: `run_matgl_cu_phase_transition.sh` (implemented)
- FairChem: `run_fairchem_co_cu111_adsorption.sh` (implemented)
- MACE: `run_mace_na2si3o7_quench.sh` (implemented)

## MatGL Example: Cu Phase-Transition Scan

This example runs a simple heat-and-hold workflow for fcc Copper (Cu) with
ASE + CHGNet. The script:

1. creates an fcc Cu supercell data file;
2. loads a MatGL potential in ASE;
3. runs heat-and-hold MD;
4. stores logs/trajectory for post-analysis of a transition signature
   (e.g., abrupt energy/volume slope change).

```bash
# Env: matgl-agent
bash .agent/skills/mat-lammps-md/examples/model-scripts/run_matgl_cu_phase_transition.sh
```

## Output
- `./out-matgl-cu-phase-transition/cu_fcc.data`
- `./out-matgl-cu-phase-transition/log.matgl`
- `./out-matgl-cu-phase-transition/cu_heat.lammpstrj`
- `./out-matgl-cu-phase-transition/run_summary.json`

Optional model override:
```bash
CHGNET_MODEL=0.3.0 bash .agent/skills/mat-lammps-md/examples/model-scripts/run_matgl_cu_phase_transition.sh
```

## FairChem Example: CO Adsorption on Cu(111)

This example runs adsorption energy with FAIR-Chem `lmp_fc`:

1. builds a clean Cu(111) slab, isolated CO molecule, and adsorbed CO/Cu(111);
2. evaluates each with the same FairChem setup;
3. computes adsorption energy as:
   `E_ads = E(CO/Cu111) - E(Cu111) - E(CO)`.

```bash
# Env: fairchem-agent
bash .agent/skills/mat-lammps-md/examples/model-scripts/run_fairchem_co_cu111_adsorption.sh
```

### FairChem Output
- `./out-fairchem-co-cu111/cu111_clean.data`
- `./out-fairchem-co-cu111/co_gas.data`
- `./out-fairchem-co-cu111/co_on_cu111.data`
- `./out-fairchem-co-cu111/energies.json`
- `./out-fairchem-co-cu111/adsorption_summary.txt`

## MACE Example: Thermal Quench of Na2Si3O7 Glass

This example runs a melt-quench protocol for sodium silicate (`Na2Si3O7`) with
a MACE-bound LAMMPS binary:

1. auto-generates a Na2Si3O7 seed structure (or uses `INPUT_STRUCTURE`);
2. auto-prepares a LAMMPS MACE model (`mace_auto-lammps.pt`) unless `MODEL_FILE` is given;
3. runs melt/quench/hold.

```bash
# Env: mace-agent
bash .agent/skills/mat-lammps-md/examples/model-scripts/run_mace_na2si3o7_quench.sh
```

Optional overrides:
```bash
# Env: mace-agent
INPUT_STRUCTURE=./na2si3o7_initial.cif \
MACE_MODEL_NAME=MACE-MP-medium \
MACE_HEAD=omat_pbe \
bash .agent/skills/mat-lammps-md/examples/model-scripts/run_mace_na2si3o7_quench.sh
```

### MACE Output
- `./out-mace-na2si3o7-quench/na2si3o7_initial.data`
- `./out-mace-na2si3o7-quench/in.na2si3o7_quench_mace`
- `./out-mace-na2si3o7-quench/log.lammps`
- `./out-mace-na2si3o7-quench/na2si3o7_quench.lammpstrj`
- `./out-mace-na2si3o7-quench/na2si3o7_glass_final.data`

## Notes
- The MACE quench input follows the same pattern; if needed, edit only the
  "MACE model hookup" block in `in.na2si3o7_quench_mace`.
- For model choice and recommended checkpoints, see
  [ml-foundation-potentials](../../ml-foundation-potentials/SKILL.md).
