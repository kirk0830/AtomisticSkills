"""
HPC Job Submission Utilities

A unified interface for submitting and managing jobs on HPC clusters
via multiple backends (local Slurm, SSH Slurm, etc.).

Quick Start:
    from src.utils.hpc import JobManager, JobSpec

    # Create manager from environment variables
    manager = JobManager.from_env()

    # Submit a job
    spec = JobSpec(
        name="vasp_calc",
        command="vasp_std",
        nodes=2,
        ntasks_per_node=32,
        partition="cpu",
        time_limit="24:00:00",
        modules=["vasp/6.4.2-cpu"],
    )
    job_id = manager.submit(spec)

    # Check status
    status = manager.status(job_id)
    print(f"Job state: {status.state}")

Backends:
    - slurm_local: Direct sbatch on the same machine (login node)
    - slurm_ssh: Submit via SSH to a remote cluster

Modules:
    - base: Core classes (JobSpec, JobStatus, JobState, HPCBackend)
    - job_template: Slurm script generation
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
from src.utils.hpc.job_manager import (
    JobManager,
    create_backend,
)
from src.utils.hpc.job_template import generate_slurm_script

__all__ = [
    "HPCBackend",
    "JobManager",
    "JobSpec",
    "JobState",
    "JobStatus",
    "create_backend",
    "generate_slurm_script",
]
