---
trigger: always_on
---

# HPC / Slurm Job Submission Rules

When submitting computational work through AtomisticSkills, follow these rules.

## 1. Never Run Heavy Computation Locally

- **Login nodes are for job submission only.** Do NOT run DFT, MD, MLIP relaxation,
  phonon calculations, or any other CPU/GPU-intensive work directly on the login node.
- Before executing any computation, confirm the resource requirements:
  - CPU-only light task (< 1 minute)? → OK to run locally.
  - CPU/GPU heavy task? → Submit via `submit_hpc_job` MCP tool.
  - DFT calculation (ORCA, VASP, CP2K, QE)? → **Always submit via Slurm**.
  - MD simulation or MLIP relaxation? → **Always submit via Slurm**.

## 2. Job Submission Workflow

The standard HPC workflow through MCP tools:

```
1. submit_hpc_job(name, command, ...)  →  returns job_id
2. check_hpc_job_status(job_id)        →  monitor progress (optional)
3. wait_for_hpc_job(job_id, ...)       →  block until COMPLETED/FAILED
4. read_hpc_job_output(job_id, path)   →  read results
```

## 3. Job Specification Guidelines

- **Partition/Queue**: Use the appropriate queue for your workload type.
  - GPU jobs: use the GPU partition (e.g. `md` with `gres="gpu:1"`)
  - CPU-only jobs: use the CPU partition with empty `gres`
- **Time limit**: Be realistic but not excessive.
  - Short tests: `00:30:00`
  - Single-point DFT: `02:00:00`
  - Geometry optimization: `12:00:00`
  - MD production run: `24:00:00` or more
- **Resources**: Match the job to the node.
  - Default: `ntasks_per_node=32, cpus_per_task=1`
  - GPU: add `gres="gpu:1"`
- **Modules**: Load via the `modules` parameter (comma-separated) or configure
  default modules in `~/.atomistic_skills.yaml`.

## 4. Configuration

HPC configuration is loaded from (in priority order):
1. Explicit parameters in the `submit_hpc_job` call
2. Environment variables (`HPC_MODE`, `HPC_PROFILE`, etc.)
3. `~/.atomistic_skills.yaml` config file
4. Built-in profiles (`generic`, `nersc_perlmutter`, etc.)

See `.atomistic_skills.yaml.example` in the project root for a template.
