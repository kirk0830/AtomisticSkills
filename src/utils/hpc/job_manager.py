"""
Unified HPC Job Manager - single interface for all backends.

Provides a factory function to create the appropriate backend based on
configuration, and a high-level JobManager class with convenience methods.

Usage:
    from src.utils.hpc import JobManager, create_backend

    # Auto-detect backend
    backend = create_backend(
        mode="local",  # or "ssh"
        ...
    )

    # Or use JobManager
    manager = JobManager.from_config(config_dict)
    job_id = manager.submit(job_spec)
    status = manager.wait_for_completion(job_id)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.hpc.base import HPCBackend, JobSpec, JobState, JobStatus

logger = logging.getLogger(__name__)


def create_backend(
    mode: str = "auto",
    **kwargs,
) -> HPCBackend:
    if mode == "auto":
        mode = _auto_detect_mode()

    if mode == "local":
        from src.utils.hpc.slurm_local import SlurmLocalBackend
        return SlurmLocalBackend(**kwargs)

    elif mode == "ssh":
        from src.utils.hpc.slurm_ssh import SlurmSSHBackend
        return SlurmSSHBackend(**kwargs)

    else:
        raise ValueError(f"Unknown backend mode: {mode}. Use 'local', 'ssh', or 'auto'.")


def _auto_detect_mode() -> str:
    import shutil
    if shutil.which("sbatch") and shutil.which("squeue"):
        return "local"
    return "ssh"


class JobManager:
    def __init__(self, backend: HPCBackend):
        self.backend = backend

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "JobManager":
        mode = config.get("mode", "auto")
        backend_kwargs = {k: v for k, v in config.items() if k != "mode"}
        backend = create_backend(mode=mode, **backend_kwargs)
        return cls(backend)

    @classmethod
    def from_env(cls) -> "JobManager":
        config: Dict[str, Any] = {}

        mode = os.environ.get("HPC_MODE", "auto")
        config["mode"] = mode

        if mode == "ssh":
            host = os.environ.get("HPC_SSH_HOST")
            if not host:
                raise ValueError("HPC_SSH_HOST must be set for SSH mode")
            config["host"] = host
            config["user"] = os.environ.get("HPC_SSH_USER")
            config["key_path"] = os.environ.get("HPC_SSH_KEY")
            config["port"] = int(os.environ.get("HPC_SSH_PORT", "22"))
            config["remote_work_dir"] = os.environ.get(
                "HPC_REMOTE_WORK_DIR", "~/hpc_jobs"
            )

        if mode == "local":
            work_dir = os.environ.get("HPC_WORK_DIR")
            if work_dir:
                config["work_dir"] = work_dir

        return cls.from_config(config)

    def submit(self, job_spec: Union[JobSpec, Dict[str, Any]]) -> str:
        if isinstance(job_spec, dict):
            job_spec = JobSpec.from_dict(job_spec)
        return self.backend.submit(job_spec)

    def status(self, job_id: str) -> JobStatus:
        return self.backend.status(job_id)

    def cancel(self, job_id: str) -> bool:
        return self.backend.cancel(job_id)

    def list_jobs(
        self,
        user: Optional[str] = None,
        state: Optional[JobState] = None,
    ) -> List[JobStatus]:
        return self.backend.list_jobs(user=user, state=state)

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: Optional[int] = None,
    ) -> JobStatus:
        status = self.backend.status(job_id)
        return status.wait(self.backend, poll_interval=poll_interval, timeout=timeout)

    def read_output(self, job_id: str, output_path: str) -> str:
        return self.backend.read_file(job_id, output_path)

    def upload(self, local_path: str, remote_path: str) -> bool:
        return self.backend.upload_file(local_path, remote_path)

    def download(self, remote_path: str, local_path: str) -> bool:
        return self.backend.download_file(remote_path, local_path)

    def check_available(self) -> bool:
        return self.backend.check_available()
