"""
Slurm SSH backend for HPC job submission.

Submits jobs to a remote Slurm cluster via SSH.
Uses SSH key-based authentication (no passwords in code).

Typical use case:
  - Agent running on local machine
  - Remote HPC cluster accessible via SSH
  - SSH key configured for password-less login

Security notes:
  - SSH keys are read from the filesystem, never hard-coded
  - Supports ssh-agent and key files
  - No password support (keys are more secure)

Usage:
    from src.utils.hpc.slurm_ssh import SlurmSSHBackend
    backend = SlurmSSHBackend(
        host="perlmutter-p1.nersc.gov",
        user="your_username",
        key_path="~/.ssh/nersc",
        remote_work_dir="/global/cfs/cdirs/m5068/your_project",
    )
    job_id = backend.submit(job_spec)
    status = backend.status(job_id)
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from src.utils.hpc.base import HPCBackend, JobSpec, JobState, JobStatus
from src.utils.hpc.job_template import generate_slurm_script

logger = logging.getLogger(__name__)


class SlurmSSHBackend(HPCBackend):
    name = "slurm_ssh"

    def __init__(
        self,
        host: str,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        port: int = 22,
        remote_work_dir: str = "~/hpc_jobs",
        ssh_options: Optional[List[str]] = None,
    ):
        self.host = host
        self.user = user
        self.port = port
        self.key_path = os.path.expanduser(key_path) if key_path else None
        self.remote_work_dir = remote_work_dir
        self.ssh_options = ssh_options or [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
        ]

    def _ssh_cmd(self, command: str) -> List[str]:
        cmd = ["ssh"]
        cmd.extend(self.ssh_options)
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        target = f"{self.user}@{self.host}" if self.user else self.host
        cmd.extend([target, command])
        return cmd

    def _run_ssh(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess:
        cmd = self._ssh_cmd(command)
        logger.debug(f"SSH: {' '.join(cmd[:5])}...")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _scp_cmd(self, source: str, dest: str, upload: bool = True) -> List[str]:
        cmd = ["scp"]
        cmd.extend(self.ssh_options)
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        if self.port != 22:
            cmd.extend(["-P", str(self.port)])
        target = f"{self.user}@{self.host}" if self.user else self.host
        if upload:
            cmd.extend([source, f"{target}:{dest}"])
        else:
            cmd.extend([f"{target}:{source}", dest])
        return cmd

    def _run_scp(
        self, source: str, dest: str, upload: bool = True, timeout: int = 60
    ) -> subprocess.CompletedProcess:
        cmd = self._scp_cmd(source, dest, upload)
        logger.debug(f"SCP: {' '.join(cmd[:5])}...")
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def check_available(self) -> bool:
        try:
            result = self._run_ssh("which sbatch squeue")
            return result.returncode == 0 and "sbatch" in result.stdout
        except Exception:
            return False

    def submit(self, job_spec: JobSpec) -> str:
        script_content = generate_slurm_script(job_spec)
        remote_script = f"{self.remote_work_dir}/_job_scripts/{job_spec.name}_{os.getpid()}.sh"

        self._run_ssh(f"mkdir -p {shlex.quote(f'{self.remote_work_dir}/_job_scripts')}")

        local_temp = Path(f"/tmp/{job_spec.name}_{os.getpid()}.sh")
        local_temp.write_text(script_content)
        local_temp.chmod(0o755)

        try:
            scp_result = self._run_scp(str(local_temp), remote_script, upload=True)
            if scp_result.returncode != 0:
                raise RuntimeError(f"SCP upload failed: {scp_result.stderr.strip()}")

            submit_dir = job_spec.work_dir or self.remote_work_dir
            result = self._run_ssh(
                f"cd {shlex.quote(submit_dir)} && sbatch --parsable {shlex.quote(remote_script)}",
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

            job_id = result.stdout.strip()
            logger.info(f"Job submitted via SSH: {job_id}")
            return job_id
        finally:
            local_temp.unlink(missing_ok=True)

    def status(self, job_id: str) -> JobStatus:
        try:
            sacct_cmd = (
                f"sacct -j {shlex.quote(job_id)} "
                f"--format=JobID,JobName,State,Partition,NNodes,Submit,Start,End,Elapsed,ExitCode,WorkDir "
                f"--parsable2 --noheader -n"
            )
            result = self._run_ssh(sacct_cmd, timeout=30)

            if result.returncode != 0 or not result.stdout.strip():
                squeue_cmd = (
                    f"squeue -j {shlex.quote(job_id)} "
                    f"--format='%i|%j|%T|%P|%D|%V|%S|%M|%Z' --noheader"
                )
                result = self._run_ssh(squeue_cmd, timeout=30)
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
                             error_message="SSH command timed out")
        except Exception as e:
            logger.warning(f"Failed to get status for job {job_id}: {e}")
            return JobStatus(job_id=job_id, state=JobState.UNKNOWN,
                             error_message=str(e))

    def cancel(self, job_id: str) -> bool:
        try:
            result = self._run_ssh(f"scancel {shlex.quote(job_id)}", timeout=30)
            if result.returncode == 0:
                logger.info(f"Job cancelled via SSH: {job_id}")
                return True
            logger.warning(f"scancel failed: {result.stderr.strip()}")
            return False
        except Exception:
            return False

    def list_jobs(
        self,
        user: Optional[str] = None,
        state: Optional[JobState] = None,
    ) -> List[JobStatus]:
        cmd = "squeue --format='%i|%j|%T|%P|%D|%V|%S|%M|%Z' --noheader"
        if user:
            cmd += f" -u {shlex.quote(user)}"

        try:
            result = self._run_ssh(cmd, timeout=30)
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
        except Exception:
            return []

    def read_file(self, job_id: str, remote_path: str) -> str:
        result = self._run_ssh(f"cat {shlex.quote(remote_path)}", timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to read file: {result.stderr.strip()}")
        return result.stdout

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            result = self._run_scp(local_path, remote_path, upload=True, timeout=120)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            result = self._run_scp(remote_path, local_path, upload=False, timeout=120)
            return result.returncode == 0
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
