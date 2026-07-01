# HPC Job Submission Module Checkpoint

## Date: 2026-07-01

This checkpoint records the implementation of a unified HPC job submission module.

---

## 1. Module Overview

**Location**: [src/utils/hpc/](file:///workspace/src/utils/hpc/)

**Purpose**: Provide a unified interface for submitting and managing computational jobs on HPC clusters, supporting two execution modes:
- **Local mode**: Direct `sbatch` on login node (no SSH)
- **SSH mode**: Submit via SSH to remote cluster

**Files Created**:
| File | Description |
|------|-------------|
| [base.py](file:///workspace/src/utils/hpc/base.py) | Abstract base classes (JobSpec, JobStatus, JobState, HPCBackend) |
| [job_template.py](file:///workspace/src/utils/hpc/job_template.py) | Slurm script generator (string-based, will migrate to Jinja2) |
| [slurm_local.py](file:///workspace/src/utils/hpc/slurm_local.py) | Local Slurm backend |
| [slurm_ssh.py](file:///workspace/src/utils/hpc/slurm_ssh.py) | SSH-based Slurm backend |
| [job_manager.py](file:///workspace/src/utils/hpc/job_manager.py) | High-level JobManager + factory |
| [__init__.py](file:///workspace/src/utils/hpc/__init__.py) | Public API exports |
| [docs/hpc_job_submission.md](file:///workspace/docs/hpc_job_submission.md) | Usage documentation |

---

## 2. Key Design Decisions

### Backend Architecture
- **Abstract HPCBackend**: Defines common interface (submit, status, cancel, list_jobs, etc.)
- **SlurmLocalBackend**: Uses subprocess to call `sbatch`, `squeue`, `scancel` directly
- **SlurmSSHBackend**: Uses SSH + SCP for remote submission and file transfer
- **JobManager**: Factory + convenience methods (wait_for_completion, upload/download)

### Security Approach
- SSH keys read from filesystem, never hard-coded
- No password support (keys are more secure)
- Environment variables for configuration (no secrets in code)
- SSH options: `StrictHostKeyChecking=accept-new`, `ConnectTimeout=10`

### JobSpec Design
- Comprehensive Slurm options (nodes, ntasks, cpus, partition, qos, time, memory, gres)
- Environment setup (modules, pre_run, post_run, environment variables)
- Output handling (output_file, error_file, email notifications)
- Extensible (extra_directives, metadata)

---

## 3. Current Implementation Status

| Feature | Status |
|---------|--------|
| JobSpec dataclass | ✅ Done |
| JobStatus dataclass | ✅ Done |
| JobState enum | ✅ Done |
| HPCBackend abstract class | ✅ Done |
| Slurm script generator (string-based) | ✅ Done |
| SlurmLocalBackend | ✅ Done |
| SlurmSSHBackend | ✅ Done |
| JobManager factory | ✅ Done |
| Environment variable config | ✅ Done |
| Usage documentation | ✅ Done |
| Jinja2 templates | ⏸️ Pending |
| Module environment config system | ⏸️ Pending |
| Integration with ORCA skills | ⏸️ Pending |
| Integration with VASP (Atomate2) | ⏸️ Pending |

---

## 4. Pending Improvements

### 4.1 Jinja2 Templates (User Request)

**Current**: String-based script generation in `job_template.py`
**Proposed**: Migrate to Jinja2 templates for:
- Better readability and maintainability
- Easier customization (user-provided templates)
- Conditional sections (e.g., only include `gres` if GPU requested)
- Template inheritance (base template + custom overrides)

**Implementation Plan**:
1. Create `src/utils/hpc/templates/` directory
2. Add `slurm_base.j2` template
3. Update `job_template.py` to use Jinja2
4. Support custom template paths in JobSpec

### 4.2 Module Environment Configuration (User Request)

**Problem**: Users need to configure `module load` commands before running jobs. Current approach is ad-hoc (hardcoded in JobSpec.modules or pre_run).

**Proposed Solutions** (need discussion):

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Config file** | `~/.hpc_modules.yaml` or `hpc_config.yaml` | Persistent, versionable | Needs file management |
| **B. Environment variables** | `HPC_MODULES_VASP="vasp/6.4.2-cpu"` | Simple, no files | Many env vars for many apps |
| **C. Per-skill config** | Each skill has `config.yaml` | Skill-specific | Config fragmentation |
| **D. Profile templates** | Pre-defined profiles (NERSC, generic, etc.) | Easy for common cases | Not flexible enough |
| **E. Hybrid** | Config file + env vars + profiles | Maximum flexibility | Complex implementation |

**Recommended**: Option E (Hybrid)
- Base config: `~/.atomistic_skills.yaml` (already exists for MP_API_KEY, etc.)
- App-specific: `HPC_MODULES_<APP>` env vars override
- Profile templates: Built-in profiles for common HPC centers

---

## 5. Integration Roadmap

### ORCA Skills
- [chem-dft-orca-singlepoint](file:///workspace/.agents/skills/chem-dft-orca-singlepoint/SKILL.md)
- [chem-dft-orca-optimization](file:///workspace/.agents/skills/chem-dft-orca-optimization/SKILL.md)
- [chem-dft-orca-advanced-calculation](file:///workspace/.agents/skills/chem-dft-orca-advanced-calculation/SKILL.md)

**Current**: Local execution via `ORCA_BINARY_PATH`
**Proposed**: HPC module integration
- JobSpec with ORCA-specific defaults
- Module config: `HPC_MODULES_ORCA="orca/5.0.4 openmpi/4.1.5"`
- SSH mode for remote submission

### VASP (Atomate2)
- [atomate2_utils.py](file:///workspace/src/utils/dft/atomate2_utils.py)

**Current**: jobflow-remote + MongoDB
**Proposed**: Optional HPC module integration
- For users without jobflow-remote setup
- Simpler single-job submission path

---

## 6. Security Considerations

### SSH Mode
- ✅ SSH keys from filesystem only
- ✅ No password support
- ✅ StrictHostKeyChecking enabled
- ⚠️ Key expiration not handled (e.g., NERSC 24h keys)
- 💡 Proposed: Key refresh detection + warning

### Local Mode
- ✅ Uses system Slurm commands
- ✅ Inherits user permissions
- ⚠️ No authentication needed (assumes login node)

### Environment Variables
- ✅ Config via env vars (no secrets in code)
- ⚠️ Shell history may expose keys in env var values
- 💡 Proposed: Use config file for sensitive paths

---

## 7. Next Steps

1. **Jinja2 Templates**: Migrate script generation to Jinja2
2. **Module Config System**: Design and implement hybrid config approach
3. **ORCA Integration**: Update ORCA skills to use HPC module
4. **VASP Integration**: Add HPC module as alternative to jobflow-remote
5. **Testing**: Add unit tests for backends (mock subprocess)
6. **Documentation**: Update skills to reference HPC module

---

## 8. Related Documents

- [HPC Job Submission Usage](file:///workspace/docs/hpc_job_submission.md)
- [Atomate2 Remote Setup](file:///workspace/conda-envs/atomate2-agent/atomate2_remote_worker_setup.md)
- [Environment Variables Guide](file:///workspace/docs/environment_variables.md)
- [Security Fix Checkpoint](file:///workspace/SECURITY_FIX_CHECKPOINT.md)

---

*Checkpoint created: 2026-07-01*