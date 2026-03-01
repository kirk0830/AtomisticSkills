---
name: chem-sorption
description: Relax porous frameworks and compute gas adsorption (Henry coefficient, heat of adsorption, isotherms) using UMA (FairChem) from the CLI.
category: materials
---

# chem-sorption

## Goal

To run a **relax → Widom / GCMC** workflow for porous frameworks (e.g. COFs, MOFs) using the **UMA** (Universal Materials Accelerator) model via FairChem: (1) relax the structure; (2) optionally compute Henry coefficients and heats of adsorption via Widom insertion; (3) optionally compute adsorption isotherms via Grand Canonical Monte Carlo (GCMC). All steps are invoked from the command line with Python scripts; no database manager or high-throughput pipeline.

## Prerequisites

- **Conda environment**: `fairchem-agent` (UMA/FairChem, ASE, PyTorch).
- **UMA checkpoint**: Path to a `.pt` checkpoint (e.g. `uma-s-1p1`, `uma-m-1p1`, or a fine-tuned model).
- **Input**: A framework structure in CIF (or XYZ) format. For Widom and GCMC, use the **relaxed** structure from step 1.

## Instructions

1. **Relax the framework**: Relax the framework with UMA (positions and optionally cell). Outputs go to `output_dir/`: all CIF/XYZ files produced plus `relax_results.json`.

```bash
# Env: fairchem-agent
python .agent/skills/chem-sorption/scripts/run_relax_uma.py \
    --structure path/to/framework.cif \
    --name MYCOF \
    --weights path/to/uma.pt \
    --task-name omol \
    --output-dir ./out \
    --fmax 0.05 \
    --steps 1000
```

**Key parameters:**
- `--structure`: Input CIF or XYZ.
- `--name`: Identifier for the framework (used in output filenames, e.g. `name.relaxed.cif`).
- `--weights`: Path to UMA checkpoint (`.pt`).
- `--task-name`: UMA task (`omol`, `omat`, `odac`, `omc`, `oc20`). Use `odac` for MOFs/COFs when available.
- `--output-dir`: Directory for outputs (default: current directory). Writes CIF/XYZ + relax_results.json.
- `--fmax`, `--steps`: Force convergence (eV/Å) and max optimization steps.
- `--fixed-cell`: Optimize atoms only (no cell relaxation).
- `--min-plane-dist`: Minimum interplanar distance (Å) before building a supercell (default 12).
- `--supercell`: Optional pre-relax supercell, e.g. `2,2,2`.

2. **(Optional) Widom insertion** — Henry coefficient and heat of adsorption. Use the **relaxed** structure from step 1. Writes a single JSON (default `output_dir/widom_results.json`, or `--output`).

```bash
# Env: fairchem-agent
python .agent/skills/chem-sorption/scripts/run_widom_uma.py \
    --structure ./out/MYCOF.relaxed.cif \
    --name MYCOF \
    --weights path/to/uma.pt \
    --task-name omol \
    --gas CO2 \
    --temperature 298 \
    --num-insertions 5000 \
    --output-dir ./out
```

**Key parameters:**
- `--structure`: Relaxed framework CIF (from step 1).
- `--gas`: Gas molecule (e.g. `CO2`, `N2`).
- `--temperature`: Temperature in Kelvin.
- `--num-insertions`: Number of Widom insertion attempts (default 5000).
- `--min-interplanar-distance`: Min interplanar distance before supercell (default 12 Å).

3. **(Optional) GCMC** — adsorption isotherm. Use the **relaxed** structure. Supports **single-component** (CO2 or N2) and **multicomponent** mixtures (e.g. CO2+N2). Both modes write a JSON plus PNGs (single-component default `output_dir/gcmc_results.json`, multicomponent default `output_dir/multi_gcmc.json`, `output_dir/nmols*.png`, `output_dir/energy.png`, or `--output` for JSON path).

**3a. Single-component GCMC (CO2 or N2):**

```bash
# Env: fairchem-agent
python .agent/skills/chem-sorption/scripts/run_gcmc_uma.py \
    --cif ./out/MYCOF.relaxed.cif \
    --output-dir ./out \
    --weights path/to/uma.pt \
    --task-name odac \
    --steps 100000 \
    --temperature-K 298 \
    --pressure-bar 1 \
    --adsorbate CO2 \
    --scheme gcmc
```

**Key parameters (single-component):**
- `--cif`: Relaxed framework CIF.
- `--output-dir`: Directory for results (default: current directory). Writes `gcmc_results.json`, `nmols.png`, `energy.png`.
- `--output`: Optional path for the GCMC results JSON (overrides `output_dir/gcmc_results.json`).
- `--weights`: Path to UMA checkpoint (`.pt`).
- `--task-name`: UMA task (e.g. `odac` for adsorption).
- `--steps`: Number of MC steps.
- `--temperature-K`, `--pressure-bar`: Temperature (K) and pressure (bar).
- `--adsorbate`: `CO2` or `N2`.
- `--scheme`: `gcmc` or `hmc`.
- `--restart-traj`, `--restart-frame`: Optional restart from a previous trajectory.

**3b. Multicomponent GCMC (mixtures, e.g. CO2+N2):**

```bash
# Env: fairchem-agent
python .agent/skills/chem-sorption/scripts/run_gcmc_uma_multi.py \
    --cif ./out/MYCOF.relaxed.cif \
    --output-dir ./out \
    --weights path/to/uma.pt \
    --task-name odac \
    --steps 100000 \
    --temperature-K 298 \
    --gases CO2 N2 \
    --y 0.15 0.85 \
    --p-total-bar 1.0 \
    --scheme gcmc
```

**Key parameters (multicomponent):**
- `--cif`: Relaxed framework CIF.
- `--output-dir`: Directory for results (default: current directory). Writes `multi_gcmc.json`, `nmols_*.png` (per species), combined `nmols.png`, `energy.png`.
- `--output`: Optional path for the GCMC results JSON (overrides `output_dir/multi_gcmc.json`).
- `--weights`, `--task-name`, `--steps`, `--temperature-K`, `--scheme`, `--restart-traj`, `--restart-frame`: Same meaning as single-component GCMC.
- `--gases`: List of gas species (e.g. `CO2 N2`, order-free).
- `--y`: Mole fractions aligned with `--gases` (will be normalized).
- `--p-total-bar`: Total pressure in bar; partial pressures are `p_i = y_i * p_total`.
- `--kij`: Optional Peng–Robinson binary interaction entries, repeatable (e.g. `--kij CO2,N2,0.0`).

## Examples

**Full workflow (relax → Widom → GCMC):**

```bash
# Env: fairchem-agent
# 1. Relax (writes to ./results/)
python .agent/skills/chem-sorption/scripts/run_relax_uma.py \
    --structure my_cof.cif --name COF-1 --weights /path/to/uma.pt --output-dir ./results

# 2. Widom (use a relaxed CIF from step 1; see relax_results.json for paths)
python .agent/skills/chem-sorption/scripts/run_widom_uma.py \
    --structure ./results/COF-1.relaxed.cif \
    --name COF-1 --weights /path/to/uma.pt --gas CO2 --temperature 298 --output-dir ./results

# 3. GCMC
python .agent/skills/chem-sorption/scripts/run_gcmc_uma.py \
    --cif ./results/COF-1.relaxed.cif \
    --output-dir ./results \
    --weights /path/to/uma.pt --steps 50000 --temperature-K 298 --pressure-bar 1 --adsorbate CO2
```

## Output layout

All outputs are **JSON** (plus CIF/XYZ for relax, which are needed for downstream steps).

- **Relax**: `output_dir/` → all CIF and XYZ files produced (e.g. `name.relaxed.cif`, `name.relaxed.xyz`, `name_supercell.relaxed.cif` if supercell built), plus `relax_results.json` (paths to those files and optional comparison metrics).
- **Widom**: One standalone JSON file (default `output_dir/widom_results.json`, or `--output`). Contains COFclean-style Henry fields: `adsorbates`, `temperature_K`, `henry_mol_kg_Pa`, `henry_stderr_mol_kg_Pa`, `heat_of_adsorption_kJ_mol`, `heat_of_adsorption_std_kJ_mol`, `source`, `method`, etc.
- **GCMC (single + multicomponent)**: One standalone JSON file (single-component default `output_dir/gcmc_results.json`, multicomponent default `output_dir/multi_gcmc.json`, or `--output`) plus PNGs. Single-component: `nmols.png`, `energy.png`. Multicomponent: per-species `nmols_<species>.png`, optional combined `nmols.png`, and `energy.png`. JSON contains COFclean-style isotherm: `adsorbates`, `temperature_K`, `source`, `units`, `points` (`p_total`, `p_partial` per species, `q_total`, `q_by_adsorbate`), plus optional `wall_time_s`, `steps_per_s`.

## Constraints

- **Environment**: All three scripts require the **fairchem-agent** conda environment (UMA, ASE, PyTorch, pydantic, pymatgen; GCMC uses a vendored copy of ASE-MC shipped under `scripts/ase_mc`).
- **UMA only**: This skill uses UMA (FairChem) only.
- **Input structure**: For Widom and GCMC, use a **relaxed** framework CIF from the relax script for consistent results.
- **Gases**: Widom supports gases buildable with `ase.build.molecule` (e.g. CO2, N2). GCMC scripts support CO2 and N2 (Peng–Robinson parameters) for both single-component and multicomponent mixtures (`--gases`/`--y`).
- **Paths**: Run commands from the **project root** so that `python .agent/skills/chem-sorption/scripts/run_*.py` resolves and script imports (relax_common, widom_common, etc.) work.

## References

- FairChem / UMA: [FairChem](https://github.com/Open-Catalyst-Project/fairchem)
- Widom, B., “Some Topics in the Theory of Fluids”, *J. Chem. Phys.* **39** (11), 2808–2812 (1963). [DOI](https://doi.org/10.1063/1.1734110)
- Metropolis, N. et al., “Equation of State Calculations by Fast Computing Machines”, *J. Chem. Phys.* **21** (6), 1087–1092 (1953). [DOI](https://doi.org/10.1063/1.1699114)
- Peng, D.-Y. and Robinson, D. B., “A New Two-Constant Equation of State”, *Ind. Eng. Chem. Fundam.* **15** (1), 59–64 (1976). [DOI](https://doi.org/10.1021/i160057a011)
- Widom code adapted from: Anuroop Sriram, Logan M. Brabson, Xiaohan Yu, Sihoon Choi, Kareem Abdelmaqsoud, Elias Moubarak, Pim de Haan, Sindy Löwe, Johann Brehmer, John R. Kitchin, Max Welling, C. Lawrence Zitnick, Zachary Ulissi, Andrew J. Medford, David S. Sholl, “The Open DAC 2025 Dataset for Sorbent Discovery in Direct Air Capture” (arXiv:2508.03162). [DOI](https://doi.org/10.48550/arXiv.2508.03162)
- GCMC code adapted from: Woodrow N. Wilson, Vivek S. Bharadwaj, Neeraj Rai, “Python Library for Monte Carlo Simulations with Ab Initio and Machine-Learned Interatomic Potentials”, *J. Chem. Theory Comput.* **21** (20), 2025.

---

**Author:** Artur Lyssenko  
**Contact:** [GitHub @arturlyssenko12](https://github.com/arturlyssenko12)
