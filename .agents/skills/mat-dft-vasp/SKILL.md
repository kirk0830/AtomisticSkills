---
name: mat-dft-vasp
description: Prepare VASP input files, run DFT calculations (locally, via HPC cluster, or remotely via atomate2), and parse VASP output results.
category: materials
---

# mat-dft-vasp

> [!CAUTION]
> **Legacy skill.** VASP is a commercial periodic DFT engine. AtomisticSkills is migrating to the open-source conda-forge alternatives **CP2K** and **Quantum ESPRESSO**, which are available via the `cp2k` and `qe` Pixi environments. For new periodic DFT work, prefer the forthcoming `mat-dft-cp2k` / `mat-dft-qe` skills (see [DFT migration report](../../../.trae/documents/dft_migration_report.md)). VASP support is retained here only for backwards compatibility.

## Goal

To prepare VASP input files (INCAR, POTCAR, KPOINTS, POSCAR) locally for a structure or list of structures, and to parse the resulting VASP output files (`vasprun.xml`, `OUTCAR`) to extract the final energies, forces, stress, and geometries.

This skill supports three execution modes:
1. **Manual mode** — Generate inputs locally, run VASP manually on any cluster
2. **HPC mode** — Generate inputs and submit directly to a Slurm cluster via the unified HPC module
3. **Atomate2 mode** — Full workflow management via atomate2 + jobflow-remote (recommended for complex workflows)

> [!IMPORTANT]
> Before running, ask the user which execution mode they prefer:
> - **Manual**: Generate inputs, user submits jobs themselves
> - **HPC**: Auto-submit to a Slurm cluster (login node or SSH), simple single-job calculations
> - **Atomate2**: Full workflow system with MongoDB, dynamic error handling, automated result parsing
>
> If HPC mode is chosen, ask for: partition/queue, number of nodes, tasks per node, wall time, and required modules (e.g. `vasp/6.4.2-cpu`).

> [!TIP]
> **When to use which mode?**
> - **HPC mode** (`VaspHPCRunner`): Quick single calculations, simple relaxations, no MongoDB setup needed
> - **Atomate2 mode**: Complex workflows (NEB, defect calculations, phase diagrams), database integration, dynamic error correction
> - **Manual mode**: When you need full control over job submission

## Prerequisites / Environment Check

VASP is a licensed periodic DFT engine. Before running, confirm the variables for your chosen mode are set. For new periodic DFT work, consider **CP2K** or **Quantum ESPRESSO** (installed via the `cp2k` / `qe` Pixi environments) instead; they are the preferred open-source alternatives and do not require these credentials.

- `VASP_CMD` or `ATOMATE2_VASP_CMD` (required for local VASP / Atomate2) — Command to run VASP, e.g. `mpirun -np 16 vasp_std`. `ATOMATE2_VASP_CMD` takes precedence. Without it, local VASP/Atomate2 execution will fail.
- `PMG_VASP_PSP_DIR` (required for local POTCAR generation) — Path to a valid VASP POTCAR directory. Without it, POTCAR files cannot be generated. Can also be set in `~/.pmgrc.yaml`.
- `ATOMATE2_REMOTE_PROJECT` (recommended for remote Atomate2) — Default project name for remote job submission. Only needed when `jobflow-remote` cannot auto-detect the project.

See `docs/api_key_guide.md`, `docs/environment_variables.md`, and `docs/hpc_job_submission.md` for setup details.

Before running this skill, verify these variables are set for the chosen mode. If any required variable is missing, ask the user to set it before proceeding.

## Instructions

### Step 1. Prepare VASP Inputs

Use the `prepare_vasp_inputs.py` script to generate local input files from a structure (CIF, XYZ, POSCAR) or a directory of structures.

```bash
# Env: base
python .agents/skills/mat-dft-vasp/scripts/prepare_vasp_inputs.py \
    <structure-path> \
    <output-dir> \
    --preset_type matpes-r2scan \
    --calculation_type relaxation
```

Parameters:
- `structure_path`: Path to a single structure or a directory of structures.
- `output_dir`: Location to write the inputs. If `structure_path` is a directory, subdirectories will be created.
- `--preset_type`: Standard VASP presets. Options include `omat`, `mp`, `matpes-pbe`, and `matpes-r2scan`.
- `--calculation_type`: Defaults to `relaxation`. Use `static` for SCF static single-point.

### Step 2. Run VASP Calculation

#### Option A: Manual Submission

Once inputs are generated, submit the VASP jobs to an HPC or local cluster manually:

```bash
# Copy generated inputs to cluster
scp -r output_dir/ cluster:/work/your_username/vasp_calc/

# SSH to cluster and submit
ssh cluster
cd /work/your_username/vasp_calc
sbatch submit_script.sh
```

#### Option B: HPC Auto-Submission (VaspHPCRunner)

Use the `VaspHPCRunner` to automatically generate inputs, submit to HPC, and parse results:

```python
# Env: base (or any env with atomistic-skills + pymatgen installed)
from src.utils.dft.vasp_hpc import VaspHPCRunner
from pymatgen.core import Structure

# Load structure
structure = Structure.from_file("structure.cif")

# Create runner (auto-detects local/hpc mode)
runner = VaspHPCRunner(mode="hpc")

# Run static calculation
result = runner.run_static(
    structure=structure,
    xc="PBE",
    encut=520,
    kpoints=[4, 4, 4],
    nodes=2,
    ntasks_per_node=32,
    output_dir="research/my_project/vasp_static",
    poll_interval=60,
)

print(f"Job ID: {result.job_id}")
print(f"Energy: {result.energy_eV:.6f} eV")
print(f"Converged: {result.converged}")
```

#### Run geometry optimization on HPC

```python
# Env: base
from src.utils.dft.vasp_hpc import VaspHPCRunner
from pymatgen.core import Structure

structure = Structure.from_file("structure.cif")
runner = VaspHPCRunner(mode="hpc")

result = runner.run_relax(
    structure=structure,
    xc="PBE",
    encut=520,
    kpoints_density=40,  # K-points per reciprocal atom
    nodes=4,
    ntasks_per_node=32,
    output_dir="research/my_project/vasp_relax",
    poll_interval=60,
    timeout=86400,  # 24 hour timeout
)

print(f"Final energy: {result.energy_eV:.6f} eV")
if result.final_structure:
    result.final_structure.to_file("research/my_project/vasp_relax/CONTCAR_final.cif")
```

#### Option C: Atomate2 Workflow (Full System)

For complex workflows with MongoDB integration:

```bash
# Env: atomate2
# Use the MCP atomate2 server tools:
# - mcp_atomate2_run_atomate2_vasp_calculation
# - mcp_atomate2_get_atomate2_job_status
# - mcp_atomate2_get_atomate2_results_by_id
```

See the [Atomate2 remote setup guide](../../../pixi.toml (feature: atomate2) / atomate2_remote_worker_setup.md) for configuration details.

### Step 3. Parse VASP Results

After the VASP calculation has concluded, extract the output data (energy, forces, stress, structure) using `parse_vasp_results.py`. This handles both single directories (containing a `vasprun.xml`) and root directories with multiple subdirectories.

```bash
# Env: base
python .agents/skills/mat-dft-vasp/scripts/parse_vasp_results.py \
    <vasp-output-dir> \
    --save_to_file parsed_results.json
```

For HPC mode, the `VaspHPCRunner` already parses results automatically and returns a `VaspResult` object.

## HPC Configuration

### Config File: ~/.atomistic_skills.yaml

```yaml
hpc:
  profile: "nersc_perlmutter"
  mode: "auto"

  # SSH config (for ssh mode)
  ssh_host: "cluster.university.edu"
  ssh_user: "your_username"
  ssh_key: "~/.ssh/id_ed25519"
  ssh_remote_work_dir: "/work/your_username/vasp_calcs"

  # VASP modules (overrides profile defaults)
  modules:
    vasp: ["vasp/6.4.2-cpu", "intel/2023"]
    vasp_gpu: ["vasp/6.4.2-gpu", "cuda/12.2"]
```

### Environment Variables

```bash
export HPC_MODE=ssh
export HPC_SSH_HOST=cluster.university.edu
export HPC_SSH_USER=your_username
export HPC_SSH_KEY=~/.ssh/id_ed25519
export HPC_MODULES_VASP="vasp/6.4.2-cpu,intel/2023"
export PMG_VASP_PSP_DIR=/path/to/potcars  # Required for POTCAR generation
```

See [HPC Job Submission docs](../../../docs/hpc_job_submission.md) for full configuration reference.

## Constraints

- **Environments**: Manual/HPC mode scripts require the `base` pixi environment. Atomate2 mode requires the `atomate2` environment.
- **POTCARs**: Both `prepare_vasp_inputs.py` and `VaspHPCRunner` require `PMG_VASP_PSP_DIR` to point to a valid POTCAR directory.
- **Parsing Robustness**: The parser requires at a minimum `vasprun.xml` to succeed. `OUTCAR` is read supplementary.
- **HPC mode limitations**: `VaspHPCRunner` supports static and relax calculations only. For NEB, band structure, or other complex workflows, use Atomate2.
- **VASP license**: Users must have a valid VASP license to use VASP and POTCAR files.

## References

- Kresse, G. & Furthmüller, J., "Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set". *Physical Review B*, 54, 11169. [DOI](https://doi.org/10.1103/PhysRevB.54.11169)
- See also: [HPC Job Submission Documentation](../../../docs/hpc_job_submission.md)

---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)