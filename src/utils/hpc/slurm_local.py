"""
Slurm local backend for HPC job submission.

Submits jobs directly via sbatch from the current machine (e.g., a login node).
No SSH required - assumes the agent is running on the same machine where Slurm
commands are available.

Typical use case:
  - Agent running on an HPC login node
  - sbatch/squeue/scancel available in PATH
  - Direct file system access to the working directory

Usage:
    from src.utils.hpc.slurm_local import SlurmLocalBackend
    backend = SlurmLocalBackend()
    job_id = backend.submit(job_spec)
    status = backend.status(job_id)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from src.utils.hpc.base import HPCBackend, JobSpec, JobState, JobStatus
from src.utils.hpc.job_template import generate_slurm_script

logger = logging.getLogger(__name__)


class SlurmLocalBackend(HPCBackend):
    name = "slurm_local"

    def __init__(self, work_dir: Optional[str] = None):
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._scripts_dir = self.work_dir / "_job_scripts"
        self._scripts_dir.mkdir(exist_ok=True)

    def check_available(self) -> bool:
        return shutil.which("sbatch") is not None and shutil.which("squeue") is not None

    def submit(self, job_spec: JobSpec) -> str:
        if not self.check_available():
            raise RuntimeError("Slurm commands (sbatch/squeue) not found in PATH")

        script_content = generate_slurm_script(job_spec)
        script_name = f"{job_spec.name}_{os.getpid()}.sh"
        script_path = self._scripts_dir / script_name
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        logger.info(f"Submitting job: {job_spec.name}")
        logger.debug(f"Script: {script_path}")

        submit_dir = Path(job_spec.work_dir) if job_spec.work_dir else self.work_dir
        submit_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(submit_dir),
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

            job_id = result.stdout.strip()
            logger.info(f"Job submitted: {job_id}")
            return job_id

        except subprocess.TimeoutExpired:
            raise RuntimeError("sbatch timed out")

    def status(self, job_id: str) -> JobStatus:
        try:
            result = subprocess.run(
                [
                    "sacct",
                    "-j", job_id,
                    "--format=JobID,JobName,State,Partition,NNodes,Submit,Start,End,Elapsed,ExitCode,WorkDir",
                    "--parsable2",
                    "--noheader",
                    "-n",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0 or not result.stdout.strip():
                result = subprocess.run(
                    [
                        "squeue",
                        "-j", job_id,
                        "--format=%i|%j|%T|%P|%D|%V|%S|%M|%Z",
                        "--noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    return JobStatus(job_id=job_id, state=JobState.UNKNOWN)

                line = result.stdout.strip().split("\n")[0]
                parts = line.split("|")
                state = self._parse_state(parts[2] if len(parts) > 2 else "UNKNOWN")
                return JobStatus(
                    job_id=job_id,
                    state=state,
                    name=parts[1] if len(parts) > 1 else None,
                    queue=parts[3] if len(parts) > 3 else None,
                    nodes=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None,
                    submit_time=self._parse_time(parts[5]) if len(parts) > 5 else None,
                    start_time=self._parse_time(parts[6]) if len(parts) > 6 else None,
                    elapsed_time=parts[7] if len(parts) > 7 else None,
                    work_dir=parts[8] if len(parts) > 8 else None,
                    raw_output=result.stdout,
                )

            lines = result.stdout.strip().split("\n")
            main_line = next(
                (l for l in lines if l.strip() and not l.strip().startswith(job_id + ".")),
                lines[0],
            )
            parts = main_line.split("|")

            state = self._parse_state(parts[2] if len(parts) > 2 else "UNKNOWN")
            exit_code = None
            if len(parts) > 9 and ":" in parts[9]:
                try:
                    exit_code = int(parts[9].split(":")[0])
                except (ValueError, IndexError):
                    pass

            return JobStatus(
                job_id=job_id,
                state=state,
                name=parts[1] if len(parts) > 1 else None,
                queue=parts[3] if len(parts) > 3 else None,
                nodes=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None,
                submit_time=self._parse_time(parts[5]) if len(parts) > 5 else None,
                start_time=self._parse_time(parts[6]) if len(parts) > 6 else None,
                end_time=self._parse_time(parts[7]) if len(parts) > 7 else None,
                elapsed_time=parts[8] if len(parts) > 8 else None,
                exit_code=exit_code,
                work_dir=parts[10] if len(parts) > 10 else None,
                raw_output=result.stdout,
            )

        except subprocess.TimeoutExpired:
            return JobStatus(job_id=job_id, state=JobState.UNKNOWN,
                             error_message="sacct/squeue timed out")
        except Exception as e:
            logger.warning(f"Failed to get status for job {job_id}: {e}")
            return JobStatus(job_id=job_id, state=JobState.UNKNOWN,
                             error_message=str(e))

    def cancel(self, job_id: str) -> bool:
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info(f"Job cancelled: {job_id}")
                return True
            logger.warning(f"scancel failed: {result.stderr.strip()}")
            return False
        except subprocess.TimeoutExpired:
            return False

    def list_jobs(
        self,
        user: Optional[str] = None,
        state: Optional[JobState] = None,
    ) -> List[JobStatus]:
        cmd = ["squeue", "--format=%i|%j|%T|%P|%D|%V|%S|%M|%Z", "--noheader"]
        if user:
            cmd.extend(["-u", user])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return []

            jobs = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("|")
                job_state = self._parse_state(parts[2] if len(parts) > 2 else "UNKNOWN")
                if state and job_state != state:
                    continue
                jobs.append(JobStatus(
                    job_id=parts[0] if len(parts) > 0 else "",
                    state=job_state,
                    name=parts[1] if len(parts) > 1 else None,
                    queue=parts[3] if len(parts) > 3 else None,
                    nodes=int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None,
                    submit_time=self._parse_time(parts[5]) if len(parts) > 5 else None,
                    start_time=self._parse_time(parts[6]) if len(parts) > 6 else None,
                    elapsed_time=parts[7] if len(parts) > 7 else None,
                    work_dir=parts[8] if len(parts) > 8 else None,
                ))
            return jobs

        except subprocess.TimeoutExpired:
            return []

    def read_file(self, job_id: str, remote_path: str) -> str:
        path = Path(remote_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {remote_path}")
        return path.read_text()

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            src = Path(local_path)
            dst = Path(remote_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.warning(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            src = Path(remote_path)
            dst = Path(local_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.warning(f"Download failed: {e}")
            return False

    @staticmethod
    def _parse_state(slurm_state: str) -> JobState:
        s = slurm_state.strip().upper()
        state_map = {
            "PENDING": JobState.PENDING,
            "PD": JobState.PENDING,
            "RUNNING": JobState.RUNNING,
            "R": JobState.RUNNING,
            "COMPLETED": JobState.COMPLETED,
            "CD": JobState.COMPLETED,
            "FAILED": JobState.FAILED,
            "F": JobState.FAILED,
            "CANCELLED": JobState.CANCELLED,
            "CA": JobState.CANCELLED,
            "TIMEOUT": JobState.TIMEOUT,
            "TO": JobState.TIMEOUT,
            "CONFIGURING": JobState.PENDING,
            "CF": JobState.PENDING,
            "COMPLETING": JobState.RUNNING,
            "CG": JobState.RUNNING,
            "SUSPENDED": JobState.RUNNING,
            "S": JobState.RUNNING,
        }
        for key, val in state_map.items():
            if s.startswith(key):
                return val
        return JobState.UNKNOWN

    @staticmethod
    def _parse_time(time_str: str) -> Optional[float]:
        from datetime import datetime
        if not time_str or time_str.strip() in ("", "N/A", "Unknown"):
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y-%H:%M:%S"):
            try:
                return datetime.strptime(time_str.strip(), fmt).timestamp()
            except ValueError:
                continue
        return None
