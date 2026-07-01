# HPC Job Submission Module

A unified interface for submitting and managing computational jobs on HPC clusters.
Supports two execution modes:

- **Local mode**: Run `sbatch` directly from the current machine (e.g., HPC login node)
- **SSH mode**: Submit jobs via SSH to a remote cluster (e.g., from your laptop)

## Configuration System

### Configuration Priority (Highest to Lowest)

1. **JobSpec explicit values** — Hardcoded in the job spec
2. **Environment variables** — `HPC_MODULES_<APP>`, `HPC_MODE`, `HPC_PROFILE`
3. **Config file** — `~/.atomistic_skills.yaml`
4. **Built-in profile defaults** — Predefined for common HPC centers

### Config File: ~/.atomistic_skills.yaml

```yaml
MP_API_KEY: "your_mp_api_key_here"

# HPC Configuration
hpc:
  # Profile: generic, nersc_perlmutter, mit_supercloud, etc.
  profile: "nersc_perlmutter"
  
  # Execution mode: auto, local, ssh
  mode: "auto"
  
  # Default time limit for jobs
  default_time_limit: "01:00:00"
  
  # SSH configuration (for ssh mode)
  ssh_host: null
  ssh_user: null
  ssh_key: null
  ssh_port: 22
  ssh_remote_work_dir: "~/hpc_jobs"
  
  # Local work directory (for local mode)
  local_work_dir: null
  
  # Application-specific modules (override profile defaults)
  modules:
    vasp: ["vasp/6.4.2-cpu"]
    vasp_gpu: ["vasp/6.4.2-gpu", "cuda/12.2"]
    orca: ["orca/5.0.4", "openmpi/4.1.5"]
    lammps: ["lammps/2024"]
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `HPC_MODE` | Execution mode: `auto`, `local`, `ssh` |
| `HPC_PROFILE` | Profile name: `nersc_perlmutter`, `generic`, etc. |
| `HPC_MODULES_<APP>` | Modules for specific app: `HPC_MODULES_VASP="vasp/6.4.2"` |
| `HPC_SSH_HOST` | SSH hostname |
| `HPC_SSH_USER` | SSH username |
| `HPC_SSH_KEY` | SSH key path |
| `HPC_SSH_PORT` | SSH port (default: 22) |

## Built-in Profiles

| Profile | Description | Default Partition | Modules Example |
|---------|-------------|------------------|-----------------|
| `generic` | Minimal defaults | None | None |
| `nersc_perlmutter` | NERSC Perlmutter CPU | `cpu` | `vasp: ["vasp/6.4.2-cpu"]` |
| `nersc_perlmutter_gpu` | NERSC Perlmutter GPU | `gpu_gres` | `vasp_gpu: ["vasp/6.4.2-gpu"]` |
| `mit_supercloud` | MIT SuperCloud | `batch` | `vasp: ["vasp/6.4"]` |
| `umich_arc` | University of Michigan ARC | `standard` | `vasp: ["vasp/6.4"]` |

## Quick Start

### Local Mode (Login Node)

```python
from src.utils.hpc import JobManager, JobSpec, HPCConfigLoader

# Load configuration
loader = HPCConfigLoader()

# Create manager (auto-detects local mode if sbatch available)
manager = JobManager.from_config(loader.get_backend_config())

# Resolve job spec with profile defaults
resolved = loader.resolve_job_spec(
    {"name": "vasp_calc", "command": "vasp_std"},
    app="vasp",
    profile="nersc_perlmutter",
)
spec = JobSpec.from_dict(resolved)

# Submit
job_id = manager.submit(spec)
print(f"Submitted: {job_id}")
```

### SSH Mode (Remote)

```python
from src.utils.hpc import JobManager, JobSpec, HPCConfigLoader

# Configure via env vars or config file:
# export HPC_MODE=ssh
# export HPC_SSH_HOST=perlmutter-p1.nersc.gov
# export HPC_SSH_USER=your_username
# export HPC_SSH_KEY=~/.ssh/nersc

loader = HPCConfigLoader()
manager = JobManager.from_config(loader.get_backend_config())

# Resolve with app-specific modules
resolved = loader.resolve_job_spec(
    {"name": "orca_opt", "command": "orca input.inp"},
    app="orca",
)
spec = JobSpec.from_dict(resolved)

# Upload input files first
manager.upload("input.inp", "/global/cfs/cdirs/m5068/my_project/input.inp")

# Submit
job_id = manager.submit(spec)
```

### With Environment Variable Override

```bash
# Override modules without editing config file
export HPC_MODULES_VASP="vasp/6.5.0-cpu intel/2024"
export HPC_PROFILE="nersc_perlmutter"
```

```python
# Python will automatically pick up the env vars
loader = HPCConfigLoader()
modules = loader.get_modules("vasp")  # Returns ["vasp/6.5.0-cpu", "intel/2024"]
```

## Jinja2 Templates

Script generation uses Jinja2 templates for flexibility:

- Built-in template: [templates/slurm_base.j2](file:///workspace/src/utils/hpc/templates/slurm_base.j2)
- Custom templates: User can provide their own via `template_path`
- Conditional sections: Only include directives when values are set
- Fallback: String-based generator if Jinja2 not available

```python
from src.utils.hpc import generate_slurm_script, JobSpec

spec = JobSpec(name="test", command="echo hello", modules=["python"])
script = generate_slurm_script(spec)

# With custom template
script = generate_slurm_script(spec, template_path="my_custom_template.j2")
```

## API Reference

### HPCConfigLoader

| Method | Description |
|--------|-------------|
| `get_hpc_config()` | Load and merge all config sources |
| `get_modules(app, profile, explicit)` | Resolve module list for an app |
| `get_backend_config()` | Get config dict for JobManager |
| `resolve_job_spec(spec, app, profile)` | Fill in defaults from profile |
| `create_sample_config(path)` | Generate sample config file |

### JobSpec

See [base.py](file:///workspace/src/utils/hpc/base.py) for full field list.

### JobManager

| Method | Description |
|--------|-------------|
| `submit(spec)` | Submit a job |
| `status(job_id)` | Check status |
| `cancel(job_id)` | Cancel job |
| `wait_for_completion(job_id, ...)` | Block until done |
| `upload(local, remote)` | Upload file |
| `download(remote, local)` | Download file |

## Module Structure

```
src/utils/hpc/
├── __init__.py          # Public API
├── base.py              # JobSpec, JobStatus, JobState, HPCBackend
├── profiles.py          # Built-in HPC profiles
├── config_loader.py     # Config loading (file + env vars + profiles)
├── job_template.py      # Jinja2 script generation
├── slurm_local.py       # Local Slurm backend
├── slurm_ssh.py         # SSH Slurm backend
├── job_manager.py       # JobManager + factory
└── templates/
    └── slurm_base.j2    # Default Slurm template
```

## Security Notes

- SSH keys read from filesystem only (no hard-coded secrets)
- No password support (keys are more secure)
- Config via env vars or YAML files (no secrets in code)
- Optional dependencies (jinja2, yaml) gracefully handled