"""
HPC Job Submission Utilities

A unified interface for submitting and managing jobs on HPC clusters
via multiple backends (local Slurm, SSH Slurm, etc.).

Configuration System:
    - Config file: ~/.atomistic_skills.yaml
    - Environment variables: HPC_MODE, HPC_PROFILE, HPC_MODULES_<APP>, etc.
    - Built-in profiles: nersc_perlmutter, mit_supercloud, etc.

Quick Start:
    from src.utils.hpc import JobManager, JobSpec, HPCConfigLoader
    
    # Load configuration
    loader = HPCConfigLoader()
    
    # Create manager with resolved config
    manager = JobManager.from_config(loader.get_backend_config())
    
    # Resolve job spec with defaults
    resolved = loader.resolve_job_spec({"name": "vasp_calc"}, app="vasp")
    spec = JobSpec.from_dict(resolved)
    
    # Submit
    job_id = manager.submit(spec)

Profiles:
    - generic: Minimal defaults for any Slurm cluster
    - nersc_perlmutter: NERSC Perlmutter CPU nodes
    - nersc_perlmutter_gpu: NERSC Perlmutter GPU nodes
    - mit_supercloud: MIT SuperCloud
    - umich_arc: University of Michigan ARC

Modules:
    - base: Core classes (JobSpec, JobStatus, JobState, HPCBackend)
    - job_template: Jinja2-based Slurm script generation
    - profiles: Built-in HPC profiles
    - config_loader: Configuration loading (file + env vars + profiles)
    - slurm_local: Local Slurm backend
    - slurm_ssh: SSH-based Slurm backend
    - job_manager: High-level job management interface
"""

from src.utils.hpc.base import (
    HPCBackend,
    JobSpec,
    JobState,
    JobStatus,
)
from src.utils.hpc.config_loader import (
    HPCConfig,
    HPCConfigLoader,
)
from src.utils.hpc.job_manager import (
    JobManager,
    create_backend,
)
from src.utils.hpc.job_template import generate_slurm_script
from src.utils.hpc.profiles import (
    HPCProfile,
    get_profile,
    get_modules_for_app,
    list_profiles,
)

__all__ = [
    "HPCBackend",
    "HPCConfig",
    "HPCConfigLoader",
    "HPCProfile",
    "JobManager",
    "JobSpec",
    "JobState",
    "JobStatus",
    "create_backend",
    "generate_slurm_script",
    "get_modules_for_app",
    "get_profile",
    "list_profiles",
]