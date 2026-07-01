# HPC Job Submission Module Checkpoint

## Date: 2026-07-01

This checkpoint records the implementation of a unified HPC job submission module
with configuration system, Jinja2 templates, and ORCA/VASP integrations.

---

## 1. Module Overview

**Core HPC Module**: [src/utils/hpc/](file:///workspace/src/utils/hpc/)

**DFT Integrations**:
- [src/utils/dft/orca_hpc.py](file:///workspace/src/utils/dft/orca_hpc.py) — ORCA HPC runner
- [src/utils/dft/vasp_hpc.py](file:///workspace/src/utils/dft/vasp_hpc.py) — VASP HPC runner

---

## 2. Core HPC Module

### File Structure
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

### Configuration Priority
1. JobSpec explicit values (highest)
2. Environment variables (`HPC_MODULES_<APP>`, etc.)
3. Config file (`~/.atomistic_skills.yaml`)
4. Profile defaults (lowest)

### Built-in Profiles
| Profile | Description |
|---------|-------------|
| `generic` | Minimal defaults for any Slurm cluster |
| `nersc_perlmutter` | NERSC Perlmutter CPU nodes |
| `nersc_perlmutter_gpu` | NERSC Perlmutter GPU nodes |
| `mit_supercloud` | MIT SuperCloud cluster |
| `umich_arc` | University of Michigan ARC |

---

## 3. ORCA Integration

**File**: [src/utils/dft/orca_hpc.py](file:///workspace/src/utils/dft/orca_hpc.py)

### Features
- ✅ Two modes: `local` (direct) and `hpc` (cluster submission)
- ✅ Auto-detect mode based on environment
- ✅ Input file generation from parameters (no SCINE dependency)
- ✅ Output parsing (energy, SCF convergence, timing)
- ✅ HPC job submission, status polling, result retrieval
- ✅ Single-point energy calculations
- ✅ Gradient (force) calculations
- ✅ Hessian (frequency) calculations

### Key Classes/Functions
- `OrcaHPCRunner` — Main runner class
- `generate_orca_input()` — Generate ORCA input from parameters
- `parse_orca_output()` — Parse ORCA output file
- `OrcaSinglepointResult` — Result dataclass

### Usage Example
```python
from src.utils.dft.orca_hpc import OrcaHPCRunner

runner = OrcaHPCRunner(mode="hpc")  # or "local" or "auto"

result = runner.run_singlepoint(
    structure_path="molecule.xyz",
    functional="B3LYP",
    basis_set="def2-TZVP",
    charge=0,
    spin_multiplicity=1,
    nprocs=16,
    compute_gradients=True,
)

print(f"Energy: {result.energy_eV:.6f} eV")
print(f"SCF converged: {result.scf_converged}")
```

### Relationship to SCINE
- SCINE wrapper still available for local mode (existing behavior)
- OrcaHPCRunner provides HPC capability without SCINE dependency
- Both approaches are valid, user can choose

---

## 4. VASP Integration

**File**: [src/utils/dft/vasp_hpc.py](file:///workspace/src/utils/dft/vasp_hpc.py)

### Features
- ✅ Two modes: `local` (direct) and `hpc` (cluster submission)
- ✅ Auto-detect mode based on environment
- ✅ Input generation via pymatgen (INCAR, POSCAR, KPOINTS, POTCAR)
- ✅ Output parsing (energy, structure, forces, convergence)
- ✅ Static (single-point) calculations
- ✅ Geometry optimization (relax) calculations
- ✅ HPC job submission, status polling, result retrieval

### Key Classes/Functions
- `VaspHPCRunner` — Main runner class
- `generate_vasp_input()` — Generate VASP input files
- `parse_vasp_output()` — Parse VASP output files
- `VaspResult` — Result dataclass

### Usage Example
```python
from src.utils.dft.vasp_hpc import VaspHPCRunner
from pymatgen.core import Structure

# Load structure
structure = Structure.from_file("POSCAR")

runner = VaspHPCRunner(mode="hpc")  # or "local" or "auto"

result = runner.run_static(
    structure=structure,
    xc="PBE",
    encut=520,
    kpoints=[4, 4, 4],
    nodes=2,
    ntasks_per_node=32,
)

print(f"Energy: {result.energy_eV:.6f} eV")
print(f"Converged: {result.converged}")
```

### Relationship to Atomate2
- **VaspHPCRunner**: Lightweight, single-job, simple API
- **Atomate2**: Full workflow system, MongoDB, jobflow-remote
- Use VaspHPCRunner for: quick single calculations, simple relaxations
- Use Atomate2 for: complex workflows, NEB, defect calculations, database integration

---

## 5. Configuration System

### Config File: ~/.atomistic_skills.yaml
```yaml
MP_API_KEY: "your_mp_api_key_here"

hpc:
  profile: "nersc_perlmutter"
  mode: "auto"
  default_time_limit: "01:00:00"
  
  # SSH mode config
  ssh_host: null
  ssh_user: null
  ssh_key: null
  ssh_port: 22
  ssh_remote_work_dir: "~/hpc_jobs"
  
  # App-specific modules (override profile defaults)
  modules:
    vasp: ["vasp/6.4.2-cpu"]
    orca: ["orca/5.0.4", "openmpi/4.1.5"]
    lammps: ["lammps/2024"]
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| `HPC_MODE` | `auto`, `local`, `ssh` |
| `HPC_PROFILE` | Profile name |
| `HPC_MODULES_<APP>` | Modules for app (comma-separated) |
| `HPC_SSH_HOST` | SSH hostname |
| `HPC_SSH_USER` | SSH username |
| `HPC_SSH_KEY` | SSH key path |
| `ORCA_BINARY_PATH` | ORCA binary path (local mode) |
| `VASP_CMD` | VASP command (local mode) |

---

## 6. Implementation Status

| Component | Status |
|-----------|--------|
| Core HPC module (base + backends) | ✅ Done |
| Jinja2 templates | ✅ Done |
| Configuration system | ✅ Done |
| Built-in profiles | ✅ Done |
| ORCA integration | ✅ Done |
| VASP integration | ✅ Done |
| Documentation (hpc_job_submission.md) | ✅ Done |
| ORCA skill updates | ⏸️ Pending (backward compatible) |
| VASP skill updates | ⏸️ Pending (Atomate2 still primary) |
| Unit tests | ⏸️ Pending |
| More HPC profiles | ⏸️ Pending (community contributions) |

---

## 7. Design Decisions

### Why two VASP paths (HPC + Atomate2)?
- **Different use cases**: Quick single job vs. complex workflows
- **Backward compatibility**: Atomate2 users don't need to change
- **Gradual migration**: Users can adopt HPC module when ready
- **Different dependencies**: HPC module lighter (no MongoDB, no jobflow)

### Why direct input generation (not SCINE for ORCA)?
- HPC mode doesn't need SCINE wrapper overhead
- Input generation is simple and well-understood
- Keeps HPC path independent from SCINE
- SCINE still available for local mode users

### Graceful degradation
- All optional dependencies (jinja2, yaml, numpy, pymatgen) handled gracefully
- Fallback generators if advanced features unavailable
- Clear error messages when dependencies missing

---

## 8. Security Notes

- SSH keys read from filesystem only (never hard-coded)
- No password support (SSH keys are more secure)
- Configuration via env vars or YAML files (no secrets in code)
- Optional dependencies handled safely (graceful degradation)

---

## 9. Related Documents

- [HPC Job Submission Usage](file:///workspace/docs/hpc_job_submission.md)
- [Security Fix Checkpoint](file:///workspace/SECURITY_FIX_CHECKPOINT.md)
- [Atomate2 Remote Setup](file:///workspace/conda-envs/atomate2-agent/atomate2_remote_worker_setup.md)

---

*Checkpoint updated: 2026-07-01*