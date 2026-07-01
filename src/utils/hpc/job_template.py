"""
Job script template generator for HPC schedulers.

Generates submission scripts for Slurm with proper headers,
module loading, environment setup, and command execution.

Usage:
    from src.utils.hpc.job_template import generate_slurm_script
    script = generate_slurm_script(job_spec)
"""

from __future__ import annotations

import shlex
from typing import Optional

from src.utils.hpc.base import JobSpec


def _slurm_directive(key: str, value: Optional[str]) -> str:
    if value is None:
        return ""
    return f"#SBATCH --{key}={value}\n"


def _slurm_flag(key: str, enabled: bool) -> str:
    if not enabled:
        return ""
    return f"#SBATCH --{key}\n"


def generate_slurm_script(job_spec: JobSpec, shebang: str = "#!/bin/bash") -> str:
    lines = [shebang, ""]

    lines.append("# ============================================================")
    lines.append("#  SLURM Directives")
    lines.append("# ============================================================")
    lines.append("")

    lines.append(_slurm_directive("job-name", job_spec.name))
    lines.append(_slurm_directive("nodes", str(job_spec.nodes)))
    lines.append(_slurm_directive("ntasks-per-node", str(job_spec.ntasks_per_node)))
    lines.append(_slurm_directive("cpus-per-task", str(job_spec.cpus_per_task)))
    lines.append(_slurm_directive("partition", job_spec.partition))
    lines.append(_slurm_directive("qos", job_spec.qos))
    lines.append(_slurm_directive("time", job_spec.time_limit))
    lines.append(_slurm_directive("mem", job_spec.memory_per_node))
    lines.append(_slurm_directive("gres", job_spec.gres))
    lines.append(_slurm_directive("output", job_spec.output_file))
    lines.append(_slurm_directive("error", job_spec.error_file))
    lines.append(_slurm_directive("account", job_spec.account))
    lines.append(_slurm_directive("mail-user", job_spec.email))
    if job_spec.email:
        lines.append(_slurm_directive("mail-type", job_spec.email_type))

    for directive in job_spec.extra_directives:
        if directive.strip().startswith("#SBATCH"):
            lines.append(directive.rstrip() + "\n")
        else:
            lines.append(f"#SBATCH {directive}\n")

    lines.append("")
    lines.append("# ============================================================")
    lines.append("#  Environment Setup")
    lines.append("# ============================================================")
    lines.append("")

    if job_spec.work_dir:
        lines.append(f"cd {shlex.quote(job_spec.work_dir)}")
        lines.append("")

    if job_spec.modules:
        lines.append("# Load modules")
        for mod in job_spec.modules:
            lines.append(f"module load {mod}")
        lines.append("")

    if job_spec.environment:
        lines.append("# Environment variables")
        for key, value in job_spec.environment.items():
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.append("")

    if job_spec.pre_run:
        lines.append("# Pre-run setup")
        lines.append(job_spec.pre_run.rstrip())
        lines.append("")

    lines.append("# ============================================================")
    lines.append("#  Job Execution")
    lines.append("# ============================================================")
    lines.append("")

    lines.append("echo \"=== Job started at $(date) ===\"")
    lines.append("echo \"Job ID: $SLURM_JOB_ID\"")
    lines.append("echo \"Job Name: $SLURM_JOB_NAME\"")
    lines.append("echo \"Node list: $SLURM_JOB_NODELIST\"")
    lines.append("echo \"Working directory: $(pwd)\"")
    lines.append("")

    lines.append("# Run the main command")
    lines.append(job_spec.command)
    lines.append("EXIT_CODE=$?")
    lines.append("")

    if job_spec.post_run:
        lines.append("# Post-run cleanup")
        lines.append(job_spec.post_run.rstrip())
        lines.append("")

    lines.append("echo \"=== Job finished at $(date) with exit code $EXIT_CODE ===\"")
    lines.append("exit $EXIT_CODE")

    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)
