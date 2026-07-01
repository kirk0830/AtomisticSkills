# HPC Job Submission Module

A unified interface for submitting and managing computational jobs on HPC clusters.
Supports two execution modes:

- **Local mode**: Run `sbatch` directly from the current machine (e.g., HPC login node)
- **SSH mode**: Submit jobs via SSH to a remote cluster (e.g., from your laptop)

## Quick Start

### Mode 1: Local (Login Node)

```python
from src.utils.hpc import JobManager, JobSpec

manager = JobManager.from_env()  # auto-detects local mode

spec = JobSpec(
    name="vasp_calc",
    command="vasp_std",
    nodes=2,
    ntasks_per_node=32,
    partition="cpu",
    qos="regular",
    time_limit="24:00:00",
    work_dir="/global/cfs/cdirs/m5068/my_project/calc_01",
    modules=["vasp/6.4.2-cpu"],
    pre_run="export OMP_NUM_THREADS=1",
)

job_id = manager.submit(spec)
print(f"Submitted job: {job_id}")

status = manager.status(job_id)
print(f"State: {status.state}")

# Wait for completion (polls every 30s)
final_status = manager.wait_for_completion(job_id, poll_interval=30)
print(f"Final state: {final_status.state}")
```

### Mode 2: SSH (Remote)

```python
from src.utils.hpc import JobManager

# Via environment variables:
# export HPC_MODE=ssh
# export HPC_SSH_HOST=perlmutter-p1.nersc.gov
# export HPC_SSH_USER=your_username
# export HPC_SSH_KEY=~/.ssh/nersc
# export HPC_REMOTE_WORK_DIR=/global/cfs/cdirs/m5068/my_project

manager = JobManager.from_env()

# Or directly:
from src.utils.hpc.slurm_ssh import SlurmSSHBackend
backend = SlurmSSHBackend(
    host="perlmutter-p1.nersc.gov",
    user="your_username",
    key_path="~/.ssh/nersc",
    remote_work_dir="/global/cfs/cdirs/m5068/my_project",
)
manager = JobManager(backend)
```

### Environment Variables

| Variable | Mode | Description |
|----------|------|-------------|
| `HPC_MODE` | both | `auto`, `local`, or `ssh` (default: `auto`) |
| `HPC_SSH_HOST` | ssh | SSH hostname |
| `HPC_SSH_USER` | ssh | SSH username |
| `HPC_SSH_KEY` | ssh | Path to SSH private key |
| `HPC_SSH_PORT` | ssh | SSH port (default: 22) |
| `HPC_REMOTE_WORK_DIR` | ssh | Remote working directory |
| `HPC_WORK_DIR` | local | Local working directory |

## API Reference

### JobSpec

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | (required) | Job name |
| `command` | str | (required) | Command to execute |
| `nodes` | int | 1 | Number of nodes |
| `ntasks_per_node` | int | 1 | Tasks per node |
| `cpus_per_task` | int | 1 | CPUs per task |
| `partition` | str | None | Slurm partition |
| `qos` | str | None | Slurm QoS |
| `time_limit` | str | None | Wall time (HH:MM:SS) |
| `memory_per_node` | str | None | Memory per node |
| `gres` | str | None | Generic resources (e.g., `gpu:4`) |
| `work_dir` | str | None | Working directory |
| `output_file` | str | `%x-%j.out` | stdout file pattern |
| `error_file` | str | `%x-%j.err` | stderr file pattern |
| `account` | str | None | Account name |
| `email` | str | None | Email for notifications |
| `email_type` | str | `ALL` | Email trigger type |
| `modules` | list | [] | Modules to load |
| `pre_run` | str | "" | Pre-run shell commands |
| `post_run` | str | "" | Post-run shell commands |
| `environment` | dict | {} | Environment variables |
| `extra_directives` | list | [] | Extra SBATCH directives |
| `metadata` | dict | {} | User metadata |

### JobStatus

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | str | Job ID |
| `state` | JobState | Current state |
| `name` | str | Job name |
| `queue` | str | Partition/queue |
| `nodes` | int | Number of nodes |
| `submit_time` | float | Submission timestamp |
| `start_time` | float | Start timestamp |
| `end_time` | float | End timestamp |
| `elapsed_time` | str | Elapsed time string |
| `exit_code` | int | Exit code |
| `error_message` | str | Error message |
| `work_dir` | str | Working directory |

### JobState Enum

- `PENDING`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`
- `TIMEOUT`
- `UNKNOWN`

### JobManager Methods

| Method | Description |
|--------|-------------|
| `submit(job_spec)` | Submit a job, return job_id |
| `status(job_id)` | Get job status |
| `cancel(job_id)` | Cancel a job |
| `list_jobs(user, state)` | List jobs |
| `wait_for_completion(job_id, ...)` | Block until job completes |
| `read_output(job_id, path)` | Read output file |
| `upload(local, remote)` | Upload file |
| `download(remote, local)` | Download file |
| `check_available()` | Check if backend is accessible |

## Examples

### VASP Calculation (Local Mode)

```python
from src.utils.hpc import JobManager, JobSpec

manager = JobManager.from_env()

spec = JobSpec(
    name="si_static",
    command="vasp_std",
    nodes=1,
    ntasks_per_node=16,
    cpus_per_task=2,
    partition="cpu",
    qos="regular",
    time_limit="04:00:00",
    work_dir="/global/cfs/cdirs/m5068/si_calc",
    modules=["vasp/6.4.2-cpu"],
    pre_run="export OMP_NUM_THREADS=2",
    environment={"VASP_COMMAND": "vasp_std"},
)

job_id = manager.submit(spec)
print(f"VASP job submitted: {job_id}")
```

### ORCA Calculation (SSH Mode)

```python
from src.utils.hpc import JobManager, JobSpec

manager = JobManager.from_config({
    "mode": "ssh",
    "host": "hpc.university.edu",
    "user": "john_doe",
    "key_path": "~/.ssh/id_ed25519",
    "remote_work_dir": "/work/john_doe/orca_calcs",
})

spec = JobSpec(
    name="benzene_opt",
    command="orca input.inp > output.out",
    nodes=1,
    ntasks_per_node=8,
    partition="compute",
    time_limit="12:00:00",
    work_dir="/work/john_doe/orca_calcs/benzene",
    modules=["orca/5.0.4", "openmpi/4.1.5"],
    pre_run="export OMPI_NUM_THREADS=1",
)

# Upload input files first
manager.upload("input.inp", "/work/john_doe/orca_calcs/benzene/input.inp")

job_id = manager.submit(spec)
print(f"ORCA job submitted: {job_id}")
```

### LAMMPS MD Simulation

```python
from src.utils.hpc import JobManager, JobSpec

manager = JobManager.from_env()

spec = JobSpec(
    name="lammps_npt",
    command="lmp -in in.npt",
    nodes=4,
    ntasks_per_node=48,
    partition="compute",
    time_limit="48:00:00",
    gres="gpu:4",
    work_dir="/scratch/li_diffusion",
    modules=["lammps/2024-gpu", "cuda/12.2"],
    pre_run="export OMP_NUM_THREADS=1",
)

job_id = manager.submit(spec)
final_status = manager.wait_for_completion(job_id, poll_interval=60)
print(f"LAMMPS finished with state: {final_status.state}")
```

## Architecture

```
src/utils/hpc/
├── __init__.py          # Public API
├── base.py              # Abstract base classes
│   ├── JobSpec          # Job specification
│   ├── JobStatus        # Job status information
│   ├── JobState         # Enum of job states
│   └── HPCBackend       # Abstract backend interface
├── job_template.py      # Slurm script generation
├── slurm_local.py       # Local Slurm backend
├── slurm_ssh.py         # SSH-based Slurm backend
└── job_manager.py       # High-level job manager + factory
```

## Security Notes

- **SSH mode**: SSH keys are read from filesystem, never hard-coded. No password support.
- **Local mode**: Uses system Slurm commands directly; inherits user permissions.
- **Environment variables**: All sensitive configuration via env vars or external config files.
- **No secrets in code**: Never put passwords or keys in Python source code.
