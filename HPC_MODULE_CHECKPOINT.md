# HPC Job Submission Module Checkpoint

## Date: 2026-07-01

This checkpoint records the implementation of a unified HPC job submission module
with configuration system and Jinja2 templates.

---

## 1. Completed Implementation

### Module Structure
```
src/utils/hpc/
├── __init__.py          # Public API exports
├── base.py              # JobSpec, JobStatus, JobState, HPCBackend
├── profiles.py          # Built-in HPC profiles (5 profiles)
├── config_loader.py     # Config loading (file + env vars + profiles)
├── job_template.py      # Jinja2 script generation + fallback
├── slurm_local.py       # Local Slurm backend
├── slurm_ssh.py         # SSH Slurm backend
├── job_manager.py       # JobManager + factory
└── templates/
    └── slurm_base.j2    # Default Jinja2 template
```

### Built-in Profiles
| Profile | Description |
|---------|-------------|
| `generic` | Minimal defaults for any Slurm cluster |
| `nersc_perlmutter` | NERSC Perlmutter CPU nodes |
| `nersc_perlmutter_gpu` | NERSC Perlmutter GPU nodes |
| `mit_supercloud` | MIT SuperCloud cluster |
| `umich_arc` | University of Michigan ARC |

### Configuration Priority
1. JobSpec explicit values (highest)
2. Environment variables (`HPC_MODULES_<APP>`, etc.)
3. Config file (`~/.atomistic_skills.yaml`)
4. Profile defaults (lowest)

---

## 2. Key Features

### Jinja2 Templates
- ✅ Conditional sections (only include when needed)
- ✅ Custom template support (`template_path`)
- ✅ `quote` filter for shell escaping
- ✅ Fallback generator if Jinja2 unavailable

### Configuration System
- ✅ Config file: `~/.atomistic_skills.yaml`
- ✅ Environment variables: `HPC_MODE`, `HPC_PROFILE`, `HPC_MODULES_<APP>`
- ✅ Profile defaults for common HPC centers
- ✅ App-specific module resolution
- ✅ JobSpec resolution with profile defaults

### Backends
- ✅ Local: Direct `sbatch` on login node
- ✅ SSH: Submit via SSH to remote cluster
- ✅ Auto-detection: Checks if `sbatch` available

### Graceful Dependency Handling
- ✅ `yaml` optional — warning if unavailable
- ✅ `jinja2` optional — uses fallback generator

---

## 3. Usage Example

```python
from src.utils.hpc import JobManager, JobSpec, HPCConfigLoader

loader = HPCConfigLoader()

# Resolve job spec with profile defaults
resolved = loader.resolve_job_spec(
    {"name": "vasp_calc", "command": "vasp_std"},
    app="vasp",
    profile="nersc_perlmutter",
)
# Result includes: modules=["vasp/6.4.2-cpu"], partition="cpu", qos="regular"

spec = JobSpec.from_dict(resolved)

# Create manager (auto-detects local/ssh mode)
manager = JobManager.from_config(loader.get_backend_config())

# Submit
job_id = manager.submit(spec)
```

---

## 4. Files Created

| File | Lines | Purpose |
|------|-------|---------|
| [base.py](file:///workspace/src/utils/hpc/base.py) | 142 | Core data classes |
| [profiles.py](file:///workspace/src/utils/hpc/profiles.py) | 105 | HPC profile definitions |
| [config_loader.py](file:///workspace/src/utils/hpc/config_loader.py) | 175 | Config loading system |
| [job_template.py](file:///workspace/src/utils/hpc/job_template.py) | 143 | Jinja2 + fallback |
| [slurm_local.py](file:///workspace/src/utils/hpc/slurm_local.py) | 180 | Local backend |
| [slurm_ssh.py](file:///workspace/src/utils/hpc/slurm_ssh.py) | 170 | SSH backend |
| [job_manager.py](file:///workspace/src/utils/hpc/job_manager.py) | 80 | Manager + factory |
| [templates/slurm_base.j2](file:///workspace/src/utils/hpc/templates/slurm_base.j2) | 65 | Jinja2 template |
| [docs/hpc_job_submission.md](file:///workspace/docs/hpc_job_submission.md) | 209 | Usage documentation |

---

## 5. Next Steps

- Integration with ORCA skills
- Integration with VASP (alternative to jobflow-remote)
- Add more HPC profiles (users can contribute)
- Unit tests with mocked subprocess

---

## 6. Related Documents

- [Usage Documentation](file:///workspace/docs/hpc_job_submission.md)
- [Security Fix Checkpoint](file:///workspace/SECURITY_FIX_CHECKPOINT.md)

---

*Checkpoint updated: 2026-07-01*