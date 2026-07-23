---
name: chem-dft-orca-singlepoint
description: Run a DFT or Coupled Cluster single-point energy calculation (with optional gradients/Hessian) on a molecular structure with ORCA, either locally or via HPC cluster submission.
category: [chemistry]
---

# DFT Single-Point Calculation with ORCA

## Goal

Compute the DFT electronic energy and optionally forces (gradients) and/or the Hessian for a given molecular structure with the ORCA quantum chemistry program.

This skill supports two execution modes:
1. **Local mode** — Uses the SCINE wrapper for automated input generation, output parsing, and error handling (existing behavior)
2. **HPC mode** — Submits the calculation to an HPC cluster via Slurm (login node or SSH), using the unified HPC module

> [!IMPORTANT]
> Before running, ask the user which execution mode they prefer:
> - **Local**: ORCA runs on the current machine (requires `ORCA_BINARY_PATH`)
> - **HPC**: Submit to a Slurm cluster (requires HPC configuration — see [HPC Configuration](#hpc-configuration))
>
> If HPC mode is chosen, ask the user for: partition/queue name, number of CPU cores, wall time limit, and any required modules (e.g. `orca/5.0.4`, `openmpi/4.1.5`).

> [!NOTE]
> This skill is for **standard DFT single-point calculations** on molecular (non-periodic) systems. For advanced methods, multi-reference calculations, or properties not exposed here, use the [advanced ORCA skill](../chem-dft-orca-advanced-calculation/SKILL.md). For geometry optimization, use the [ORCA optimization skill](../chem-dft-orca-optimization/SKILL.md).

## Prerequisites / Environment Check

Before choosing an execution mode, confirm the required environment variables and configuration are set.

- **Local mode**: `ORCA_BINARY_PATH` must point to the ORCA executable. Without it, local ORCA calculations cannot run. Download ORCA from https://www.faccts.de/orca (or your HPC support team), extract it, and set `export ORCA_BINARY_PATH=/path/to/orca`.
- **HPC mode**: HPC connection must be configured via `~/.atomistic_skills.yaml` or environment variables such as `HPC_MODE`, `HPC_SSH_HOST`, `HPC_SSH_USER`, `HPC_SSH_KEY`, and `HPC_MODULES_ORCA`. See `docs/hpc_job_submission.md` and `docs/environment_variables.md`.

Before running this skill, verify that the variables for the chosen mode are set. If anything required is missing, ask the user to set it before proceeding.

## 1. Prerequisites

### Local Mode
- **Pixi environment:** `orca` with `scine_utilities` and `ase` installed
- **ORCA binary:** The environment variable `ORCA_BINARY_PATH` must point to the ORCA executable
  ```bash
  export ORCA_BINARY_PATH=/path/to/orca
  ```
  > [!WARNING]
  > ORCA **cannot** be installed from `conda-forge`. The `orca` package on conda-forge is an unrelated Python workflow library. You must download the ORCA quantum chemistry binary separately (e.g. from the ORCA forum or your HPC support team), extract it, and set `ORCA_BINARY_PATH` to the absolute path of the `orca` executable. Verify it runs with `$ORCA_BINARY_PATH --version` before starting calculations.
- **Input structure:** A molecular structure file readable by ASE (`.xyz`, `.cif`, `.mol`, etc.)

### HPC Mode
- **Pixi environment:** Any environment with `atomistic-skills` installed (the HPC module is in `src.utils.hpc`)
- **HPC cluster:** A Slurm-based cluster accessible either:
  - **Login node mode**: Agent running on the login node (sbatch available in PATH)
  - **SSH mode**: Agent running locally, submitting via SSH (requires SSH key)
- **HPC configuration:** Set up via `~/.atomistic_skills.yaml` or environment variables (see [HPC Configuration](#hpc-configuration))
- **ORCA on cluster:** ORCA must be installed on the HPC cluster and loadable via `module load`
- **Input structure:** Same as local mode

## 2. Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--structure` | (required) | Path to input structure file |
| `--charge` | `0` | Molecular charge |
| `--spin_multiplicity` | `1` | Spin multiplicity (2S+1) |
| `--functional` | `PBE` | DFT functional (e.g. `PBE`, `B3LYP`, `wB97X-V`, `PBE0`) |
| `--basis_set` | `def2-SVP` | Basis set (e.g. `def2-SVP`, `def2-TZVP`, `def2-TZVPP`) |
| `--dispersion` | None | Dispersion correction (e.g. `D3BJ`, `D4`) |
| `--solvation` | None | Implicit solvation model: `CPCM` or `SMD` |
| `--solvent` | None | Solvent name (e.g. `water`, `ethanol`, `dmso`); required if `--solvation` is set |
| `--special_option` | `NOSOSCF` | ORCA special option passed to SCINE calculator. Set to empty string to disable. |
| `--nprocs` | `1` | Number of CPU cores for ORCA |
| `--compute_gradients` | off | Flag to also compute forces |
| `--compute_hessian` | off | Flag to also compute the Hessian matrix |
| `--calculator_settings` | None | Extra SCINE calculator settings as a JSON string (see below) |
| `--output_dir` | auto | Output directory |

## 3. Running a Calculation

### Option A: Local Execution (SCINE wrapper)

#### Basic energy calculation

```bash
# Env: orca
python .agents/skills/chem-dft-orca-singlepoint/scripts/run_singlepoint.py \
    --structure molecule.xyz \
    --output_dir research/my_project/singlepoint
```

#### Energy + forces with a hybrid functional and dispersion

```bash
# Env: orca
python .agents/skills/chem-dft-orca-singlepoint/scripts/run_singlepoint.py \
    --structure molecule.xyz \
    --functional B3LYP \
    --basis_set def2-TZVP \
    --dispersion D3BJ \
    --compute_gradients \
    --nprocs 4 \
    --output_dir research/my_project/singlepoint
```

#### With implicit solvation

```bash
# Env: orca
python .agents/skills/chem-dft-orca-singlepoint/scripts/run_singlepoint.py \
    --structure molecule.xyz \
    --functional PBE0 \
    --basis_set def2-TZVP \
    --solvation CPCM \
    --solvent water \
    --compute_gradients \
    --output_dir research/my_project/singlepoint_solvated
```

#### With extra SCINE calculator settings

For settings not exposed as dedicated flags, pass a JSON string via `--calculator_settings`. SCINE is strict about types, so JSON ensures values are passed with the correct type (int, float, string).

```bash
# Env: orca
python .agents/skills/chem-dft-orca-singlepoint/scripts/run_singlepoint.py \
    --structure molecule.xyz \
    --functional B3LYP \
    --basis_set def2-TZVP \
    --calculator_settings '{"max_scf_iterations": 128}' \
    --output_dir research/my_project/singlepoint_custom
```

> [!IMPORTANT]
> A popular functional choice is 'wB97M-V' which can only be used with the hack "--functional '' --dispersion '' --special_option wB97M-V"
> This hack will work for any functional choice that includes hyphens

#### Hessian calculation

```bash
# Env: orca
python .agents/skills/chem-dft-orca-singlepoint/scripts/run_singlepoint.py \
    --structure molecule.xyz \
    --functional B3LYP \
    --basis_set def2-TZVP \
    --dispersion D3BJ \
    --compute_gradients \
    --compute_hessian \
    --charge 0 \
    --spin_multiplicity 1 \
    --nprocs 8 \
    --output_dir research/my_project/singlepoint_full
```

#### Beyond DFT calculation

ORCA also supports post-HF methods useful for reference calculations, such as local coupled cluster DLPNO-CCSD(T).
This is also available through this skill.

```bash
# Env: orca
python .agents/skills/chem-dft-orca-singlepoint/scripts/run_singlepoint.py \
    --structure molecule.xyz \
    --functional DLPNO-CCSD(T) \
    --basis_set def2-TZVP \
    --charge 0 \
    --spin_multiplicity 1 \
    --nprocs 8 \
    --output_dir research/my_project/singlepoint_full
```

### Option B: HPC Cluster Submission

Use the `OrcaHPCRunner` to submit calculations to an HPC cluster. This generates the ORCA input file, submits a Slurm job, waits for completion, and parses the results.

#### Basic HPC submission

```python
# Env: orca (or any env with atomistic-skills installed)
from src.utils.dft.orca_hpc import OrcaHPCRunner

runner = OrcaHPCRunner(mode="hpc")

result = runner.run_singlepoint(
    structure_path="molecule.xyz",
    functional="B3LYP",
    basis_set="def2-TZVP",
    dispersion="D3BJ",
    charge=0,
    spin_multiplicity=1,
    nprocs=16,              # CPU cores per job
    compute_gradients=True,
    output_dir="research/my_project/singlepoint_hpc",
    poll_interval=30,       # Check job status every 30s
)

print(f"Job ID: {result.job_id}")
print(f"Energy: {result.energy_eV:.6f} eV")
print(f"SCF converged: {result.scf_converged}")
print(f"Success: {result.success}")
```

#### With custom HPC settings

```python
# Env: orca
from src.utils.dft.orca_hpc import OrcaHPCRunner

# Pass HPC config directly, or use ~/.atomistic_skills.yaml
runner = OrcaHPCRunner(
    mode="hpc",
    hpc_config={
        "mode": "ssh",
        "host": "cluster.university.edu",
        "user": "your_username",
        "key_path": "~/.ssh/id_ed25519",
        "remote_work_dir": "/work/your_username/orca_calcs",
    },
)

result = runner.run_singlepoint(
    structure_path="molecule.xyz",
    functional="PBE0",
    basis_set="def2-TZVP",
    nprocs=32,
    output_dir="/work/your_username/orca_calcs/benzene",
    job_name="benzene_pbe0",
    timeout=3600,  # 1 hour timeout
)
```

#### Hessian calculation on HPC

```python
# Env: orca
from src.utils.dft.orca_hpc import OrcaHPCRunner

runner = OrcaHPCRunner(mode="hpc")
result = runner.run_singlepoint(
    structure_path="molecule.xyz",
    functional="B3LYP",
    basis_set="def2-TZVP",
    dispersion="D3BJ",
    nprocs=16,
    compute_gradients=True,
    compute_hessian=True,
    output_dir="research/my_project/hessian_hpc",
)
```

## 4. Useful standards to adhere to

- Transition state structures should be calculated with an unrestricted method (e.g. '--calculator_settings {"spin_mode": "unrestricted"}')
- For open-shell systems, different spin multiplicities should be calculated with separate single-point calculations

## 5. Output Files

- `singlepoint_results.json`: Structured results containing:
  - `energy_hartree`, `energy_eV`: Electronic energy in Hartree and eV
  - `forces_eV_per_Ang`: Forces array (if `--compute_gradients` was set)
  - `max_force_eV_per_Ang`, `rms_force_eV_per_Ang`: Force summary statistics
  - `hessian_eV_per_Ang2`: Hessian matrix (if `--compute_hessian` was set)
  - Input parameters (functional, basis set, charge, etc.) for reproducibility
- `input_structure.xyz`: Copy of the input structure

## 6. Constraints

- **Non-periodic systems only:** This skill is designed for molecules, clusters, and finite systems. ORCA does not handle periodic boundary conditions.
- **Standard methods:** For multi-reference methods (CASSCF, NEVPT2), excited-state calculations (TD-DFT, EOM-CCSD), or other advanced features, use the [advanced ORCA skill](../chem-dft-orca-advanced-calculation/SKILL.md).
- **ORCA binary (local mode):** `ORCA_BINARY_PATH` must be set and point to a working ORCA installation.
- **HPC cluster (HPC mode):** ORCA must be available on the cluster via `module load`. Configure via `~/.atomistic_skills.yaml`.
- **Environment:** Local mode requires the `orca` pixi environment. HPC mode works from any environment with `atomistic-skills` installed.
- **Solvation:** When using `--solvation`, you must also provide `--solvent`. Available solvents depend on the chosen model (CPCM/SMD); common names like `water`, `ethanol`, `dmso`, `acetonitrile`, `thf` are supported.
- **Spin multiplicity:** Provide the spin multiplicity $2S+1$ (e.g. 1 for singlet, 2 for doublet, 3 for triplet), not the number of unpaired electrons.

> [!CAUTION]
> ORCA is a commercially/academically licensed quantum-chemistry binary and is **not** available on conda-forge. The `orca` package on conda-forge is an unrelated Python workflow library, not the quantum-chemistry ORCA program. You must download and install ORCA separately, then set `ORCA_BINARY_PATH` to the absolute path of the `orca` executable. Before running calculations, verify that the path exists and is executable (e.g., `test -x "$ORCA_BINARY_PATH"` or run `$ORCA_BINARY_PATH --version`).

## 7. HPC Configuration

### Config File: ~/.atomistic_skills.yaml

```yaml
hpc:
  # Profile: generic, nersc_perlmutter, mit_supercloud, etc.
  profile: "nersc_perlmutter"

  # Mode: auto, local, ssh
  mode: "auto"

  # SSH config (for ssh mode)
  ssh_host: "cluster.university.edu"
  ssh_user: "your_username"
  ssh_key: "~/.ssh/id_ed25519"
  ssh_remote_work_dir: "/work/your_username/orca_calcs"

  # ORCA modules (overrides profile defaults)
  modules:
    orca: ["orca/5.0.4", "openmpi/4.1.5"]
```

### Environment Variables

```bash
# Quick override without editing config file
export HPC_MODE=ssh
export HPC_SSH_HOST=cluster.university.edu
export HPC_SSH_USER=your_username
export HPC_SSH_KEY=~/.ssh/id_ed25519
export HPC_MODULES_ORCA="orca/5.0.4,openmpi/4.1.5"
```

### Built-in Profiles

| Profile | Default Partition | ORCA Modules |
|---------|------------------|--------------|
| `generic` | None | None |
| `nersc_perlmutter` | `cpu` | `orca/5.0.4`, `openmpi/4.1.5` |
| `mit_supercloud` | `batch` | `orca/5.0` |
| `umich_arc` | `standard` | `orca/5.0.4` |

See [HPC Job Submission docs](../../../docs/hpc_job_submission.md) for full configuration reference.

## References

- Neese, F., "Software update: The ORCA program system—Version 5.0", *WIREs Comput. Mol. Sci.*, 2022. [DOI](https://doi.org/10.1002/wcms.1606)
- Weymuth, T. et al., "SCINE—Software for Chemical Interaction Networks", *J. Chem. Phys.*, 2024. [DOI](https://doi.org/10.1063/5.0206974)

---

**Author:** Miguel Steiner
**Contact:** [GitHub @steinmig](https://github.com/steinmig)
