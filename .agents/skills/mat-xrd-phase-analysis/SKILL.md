---
name: mat-xrd-phase-analysis
description: Phase identification from experimental XRD using DARA's tree search (Ray-based).
category: [materials]
---

# XRD Phase Analysis (DARA Tree Search)

## Goal

Identify crystalline phases present in an XRD pattern by using DARA's phase search (parallelized tree search with BGMN), following the DARA tutorial  
([Tutorial 2: Phase analysis with tree search](https://cedergrouphub.github.io/dara/notebooks/phase_search.html)).

This skill:
- Downloads candidate CIFs from COD for a given chemical system (e.g. `Ge-O-Zn`).
- Runs DARA's `search_phases` to search mixtures of phases.
- Saves plots and a human-readable report alongside the pattern.

## Requirements

- Conda environment: `xrd-agent` (see `conda-envs/xrd-agent`).
- Dependencies: `dara-xrd` (and its Ray/BGMN stack).
- BGMN: DARA will prompt to download or use a local BGMN installation.
- For clusters where only some nodes have internet:
  - You can optionally use a **two-step workflow** (download CIFs on a node with internet, run Ray search elsewhere).

## Script

`scripts/phase_search.py`

- Uses:
  - `CODDatabase.get_cifs_by_chemsys(chemical_system, dest_dir=...)`
  - `search_phases(pattern_path, phases, wavelength, instrument_profile)`
- Output directory (default):
  - `phase_analysis_results/` under the **same directory as the XRD pattern**.

## Usage

Use this script when the node where you run it can reach the COD servers (for automatic CIF download), or when you have a local directory of CIFs you want to search.

```bash
# Env: xrd-agent
python .agents/skills/mat-xrd-phase-analysis/scripts/phase_search.py \
  --xrd_data .agents/skills/mat-xrd-phase-analysis/examples/GeO2-ZnO/GeO2-ZnO_700C_60min.xrdml \
  --chemical_system "Ge-O-Zn"
```

This will:
1. Create `phase_analysis_results/` next to the XRD file.
2. Download and filter CIFs for `Ge-O-Zn` into `phase_analysis_results/cifs/`.
3. Start a local Ray cluster.
4. Run `search_phases(...)`.
5. Write plots and reports into `phase_analysis_results/`.

You can also provide your own CIFs:

```bash
python .../phase_search.py \
  --xrd_data pattern.xrdml \
  --cif_dir /path/to/my_cifs
```

In this case no internet / COD download is used; only `--cif_dir` is searched.

Specific example:

```bash
python .../phase_search.py \
  --xrd_data .agents/skills/mat-xrd-phase-analysis/examples/GeO2-ZnO/GeO2-ZnO_700C_60min.xrdml \
  --cif_dir .agents/skills/mat-xrd-phase-analysis/examples/GeO2-ZnO/phase_analysis_results/cifs
```

## Arguments (`phase_search.py`)

- `--xrd_data` (required):
  - Path to XRD pattern (`.xy`, `.xrdml`, or `.raw`).
- `--chemical_system`:
  - String like `"Ge-O-Zn"`. Used with COD to download CIFs when `--cif_dir` is not given.
- `--cif_dir`:
  - Directory containing `.cif` files to search. When set, skips COD download.
- `--output_dir`:
  - Custom output directory. Default: `<xrd_dir>/phase_analysis_results/`.
- `--wavelength` (optional):
  - X-ray source: `Cu`, `Co`, `Cr`, `Fe`, `Mo`, or a numeric wavelength in nm. Default: `Cu`.
- `--instrument_profile` (optional):
  - BGMN instrument profile. Default: `Aeris-fds-Pixcel1d-Medipix3`.
- `--quiet`:
  - Suppress verbose logging/prints.

## Outputs

All outputs go under `<output_dir>` (default: `phase_analysis_results/` under the XRD directory):

- **CIFs**
  - `cifs/`: Candidate phases for the chemical system (when downloaded).

- **Search results**
  - `results_summary.json`:
    - `pattern`, `chemical_system`, `cif_dir`
    - `num_solutions`, `best_rwp`
    - `solutions`: list of `{rank, rwp, phase_files}`.
  - `phase_search_report.txt`:
    - Text report similar to the DARA tutorial:
      - Total number of solutions and best `Rwp`.
      - `Rwp of solution i = ... %` for each solution.
      - `Phases found in solution 0:` with grouped phase file names.
      - Paths to the results directory and summary JSON.

- **Plots**
  - `solution_0_refinement.html`, `solution_0_refinement.png`
  - `solution_1_refinement.html`, `solution_1_refinement.png`, etc. (if multiple solutions)
  - Plots come from `SearchResult.visualize()` (observed / calculated / difference pattern, like the tutorial).

## Notes and Constraints

- **Ray / cluster config**:
  - This script assumes you can start a local Ray cluster with `ray.init(address="local", ...)`.
  - On HPC systems with a preconfigured Ray/GCS, you may need a clean environment or a job script that does not inherit cluster-level Ray environment variables.
- **Internet vs no-internet**:
  - On clusters with **internet on compute nodes**, use Mode A (single-step).
  - On clusters with **internet only on login nodes**, use Mode B (two-step).
- **Performance**:
  - Phase search can be CPU-intensive; adjust `num_cpus` in the script’s `ray.init(...)` if needed.

## Related Skills

- **`mat-xrd-refinement`**:
  - For known-phase Rietveld refinement (given specific CIFs).
- **`mat-xrd-calculator`**:
  - For calculating theoretical XRD patterns from crystal structures.

