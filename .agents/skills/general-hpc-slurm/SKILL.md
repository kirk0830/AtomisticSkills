---
name: general-hpc-slurm
description: Submit, monitor, and retrieve results from HPC Slurm jobs. Use this for ANY heavy computation — never run DFT, MD, or MLIP relaxation locally on the login node.
category: [general, hpc]
---

# HPC Slurm Job Management

## Goal

Submit computational jobs (DFT, MD, MLIP relaxation, phonon, etc.) to an HPC cluster
via Slurm, monitor their progress, and retrieve results.

> [!CAUTION]
> **Never run heavy computation locally on the login node.** All CPU/GPU-intensive
> work must be submitted as Slurm jobs through this skill. Violating this rule may
> impact other users on the login node and may trigger cluster policy enforcement.

## Prerequisites

- AstrBot must be running on an HPC login node with `sbatch`/`squeue`/`sacct` in PATH.
- MCP server `base` must be configured and running.
- Optional: `~/.atomistic_skills.yaml` for default Slurm parameters.

## Workflow

### 1. Submit a Job

Use the `submit_hpc_job` MCP tool:

```
mcp_base_submit_hpc_job(
    name="my_calculation",
    command="python run_calc.py --input input.cif --output results/",
    partition="md",
    ntasks_per_node=32,
    cpus_per_task=1,
    gres="gpu:1",
    time_limit="02:00:00",
    work_dir="/path/to/research/dir",
    modules="orca/5.0.4,openmpi/4.1.5"
)
```

Key parameters:
- `name`: Short descriptive name (e.g. "vasp_LiFePO4", "orca_SP_water")
- `command`: The shell command to execute on the compute node
- `partition`: Slurm queue name (default: "md")
- `gres`: GPU resources. Use `"gpu:1"` for GPU jobs, empty string `""` for CPU-only
- `time_limit`: Wall time in HH:MM:SS
- `modules`: Comma-separated module names to load before execution

### 2. Monitor Job

Check status periodically:

```
mcp_base_check_hpc_job_status(job_id="12345")
```

Possible states: `PENDING` → `RUNNING` → `COMPLETED` / `FAILED` / `TIMEOUT`

### 3. Wait for Completion

For jobs with known duration, wait synchronously:

```
mcp_base_wait_for_hpc_job(
    job_id="12345",
    poll_interval=30,
    timeout=7200   # 2 hours
)
```

### 4. Read Results

After the job completes:

```
mcp_base_read_hpc_job_output(
    job_id="12345",
    file_path="/path/to/research/dir/output.log"
)
```

## Job Specification Examples

### CPU-only DFT (ORCA):

```
name: "orca_singlepoint_H2O"
command: "$(which orca || echo orca) orca.inp > orca.out"
partition: "md"
ntasks_per_node: 32
cpus_per_task: 1
gres: ""                          # No GPU
time_limit: "04:00:00"
modules: "orca/5.0.4,openmpi/4.1.5"
```

### GPU-accelerated MLIP relaxation:

```
name: "mace_relax_LiFePO4"
command: "python relax.py --model MACE-OMAT-0 --input POSCAR"
partition: "md"
ntasks_per_node: 32
cpus_per_task: 1
gres: "gpu:1"
time_limit: "12:00:00"
```

## Configuration

Default Slurm parameters can be set in `~/.atomistic_skills.yaml`.
See `.atomistic_skills.yaml.example` in the project root for a complete template.

## Related Skills

- [chem-dft-orca-singlepoint](../chem-dft-orca-singlepoint/SKILL.md) — ORCA DFT calculations
- [mat-dft-vasp](../mat-dft-vasp/SKILL.md) — VASP DFT calculations
- [ml-mlip-benchmark](../ml-mlip-benchmark/SKILL.md) — MLIP benchmarking (submit via Slurm)

## Related Rules

- [hpc-standards.md](atomisticskills/rules/hpc-standards.md) — HPC job submission rules (MUST READ)
- [mcp-environments.md](atomisticskills/rules/mcp-environments.md) — Pixi environment to MCP server mapping
