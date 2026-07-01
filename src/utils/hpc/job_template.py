"""
Job script template generator for HPC schedulers.

Uses Jinja2 templates for flexible script generation with:
- Conditional sections (only include directives when needed)
- Custom template support (user can provide their own)
- Template inheritance (extend base templates)

Usage:
    from src.utils.hpc.job_template import generate_slurm_script
    script = generate_slurm_script(job_spec)
    
    # With custom template:
    script = generate_slurm_script(job_spec, template_path="my_template.j2")
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Optional

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from src.utils.hpc.base import JobSpec

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _get_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["quote"] = lambda s: shlex.quote(str(s))
    return env


def generate_slurm_script(
    job_spec: JobSpec,
    template_name: str = "slurm_base.j2",
    template_path: Optional[str] = None,
) -> str:
    """
    Generate a Slurm submission script from a JobSpec.

    Args:
        job_spec: Job specification with all configuration.
        template_name: Name of the built-in template file (default: slurm_base.j2).
        template_path: Path to a custom template file (overrides template_name).

    Returns:
        Generated script content as a string.
    """
    if not JINJA2_AVAILABLE:
        logger.warning("jinja2 module not available, using fallback generator")
        return _fallback_generate(job_spec)

    if template_path:
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        template = Template(template_file.read_text())
    else:
        env = _get_jinja_env()
        try:
            template = env.get_template(template_name)
        except Exception as e:
            logger.warning(f"Failed to load template {template_name}: {e}")
            return _fallback_generate(job_spec)

    return template.render(job=job_spec)


def _fallback_generate(job_spec: JobSpec, shebang: str = "#!/bin/bash") -> str:
    """Fallback string-based script generation (no Jinja2 dependency)."""
    lines = [shebang, ""]
    lines.append("# SLURM Directives")
    lines.append("")

    def directive(key: str, value: Optional[str]) -> str:
        if value is None:
            return ""
        return f"#SBATCH --{key}={value}"

    lines.append(directive("job-name", job_spec.name))
    lines.append(directive("nodes", str(job_spec.nodes)))
    lines.append(directive("ntasks-per-node", str(job_spec.ntasks_per_node)))
    lines.append(directive("cpus-per-task", str(job_spec.cpus_per_task)))
    lines.append(directive("partition", job_spec.partition))
    lines.append(directive("qos", job_spec.qos))
    lines.append(directive("time", job_spec.time_limit))
    lines.append(directive("mem", job_spec.memory_per_node))
    lines.append(directive("gres", job_spec.gres))
    lines.append(directive("output", job_spec.output_file))
    lines.append(directive("error", job_spec.error_file))
    lines.append(directive("account", job_spec.account))
    lines.append(directive("mail-user", job_spec.email))
    if job_spec.email:
        lines.append(directive("mail-type", job_spec.email_type))

    for extra in job_spec.extra_directives:
        if extra.strip().startswith("#SBATCH"):
            lines.append(extra)
        else:
            lines.append(f"#SBATCH {extra}")

    lines.append("")
    lines.append("# Environment Setup")
    lines.append("")

    if job_spec.work_dir:
        lines.append(f"cd {shlex.quote(job_spec.work_dir)}")

    if job_spec.modules:
        lines.append("# Load modules")
        for mod in job_spec.modules:
            lines.append(f"module load {mod}")

    if job_spec.environment:
        lines.append("# Environment variables")
        for key, value in job_spec.environment.items():
            lines.append(f"export {key}={shlex.quote(value)}")

    if job_spec.pre_run:
        lines.append("# Pre-run setup")
        lines.append(job_spec.pre_run)

    lines.append("")
    lines.append("# Job Execution")
    lines.append("echo 'Job started at $(date)'")
    lines.append(job_spec.command)
    lines.append("EXIT_CODE=$?")

    if job_spec.post_run:
        lines.append("# Post-run cleanup")
        lines.append(job_spec.post_run)

    lines.append("echo 'Job finished at $(date) with exit code $EXIT_CODE'")
    lines.append("exit $EXIT_CODE")

    return "\n".join(line for line in lines if line) + "\n"