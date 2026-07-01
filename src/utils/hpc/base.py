"""
Base classes for HPC job submission backends.

Provides abstract interfaces for job specification, status tracking,
and backend implementations (Slurm local, Slurm SSH, etc.).

Usage:
    from src.utils.hpc.base import JobSpec, JobStatus, HPCBackend
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


class JobState(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class JobSpec:
    name: str
    command: str
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 1
    partition: Optional[str] = None
    qos: Optional[str] = None
    time_limit: Optional[str] = None
    memory_per_node: Optional[str] = None
    gres: Optional[str] = None
    work_dir: Optional[str] = None
    output_file: str = "%x-%j.out"
    error_file: str = "%x-%j.err"
    account: Optional[str] = None
    email: Optional[str] = None
    email_type: str = "ALL"
    modules: List[str] = field(default_factory=list)
    pre_run: str = ""
    post_run: str = ""
    environment: Dict[str, str] = field(default_factory=dict)
    extra_directives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobSpec":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class JobStatus:
    job_id: str
    state: JobState
    name: Optional[str] = None
    queue: Optional[str] = None
    nodes: Optional[int] = None
    submit_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_time: Optional[str] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    work_dir: Optional[str] = None
    raw_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_complete(self) -> bool:
        return self.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.TIMEOUT)

    @property
    def is_running(self) -> bool:
        return self.state == JobState.RUNNING

    @property
    def is_pending(self) -> bool:
        return self.state == JobState.PENDING

    def wait(
        self,
        backend: "HPCBackend",
        poll_interval: int = 30,
        timeout: Optional[int] = None,
    ) -> "JobStatus":
        start = time.time()
        while True:
            status = backend.status(self.job_id)
            self.state = status.state
            if status.is_complete:
                return status
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Job {self.job_id} did not complete within {timeout}s")
            time.sleep(poll_interval)


class HPCBackend:
    name: str = "base"

    def submit(self, job_spec: JobSpec) -> str:
        raise NotImplementedError

    def status(self, job_id: str) -> JobStatus:
        raise NotImplementedError

    def cancel(self, job_id: str) -> bool:
        raise NotImplementedError

    def list_jobs(self, user: Optional[str] = None, state: Optional[JobState] = None) -> List[JobStatus]:
        raise NotImplementedError

    def read_file(self, job_id: str, remote_path: str) -> str:
        raise NotImplementedError

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        raise NotImplementedError

    def download_file(self, remote_path: str, local_path: str) -> bool:
        raise NotImplementedError

    def check_available(self) -> bool:
        raise NotImplementedError
