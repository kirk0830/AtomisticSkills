"""
Built-in HPC profiles for common computing centers.

Profiles provide default configuration for specific HPC systems,
including partition names, QoS, modules, and typical job specs.

Usage:
    from src.utils.hpc.profiles import get_profile, list_profiles
    profile = get_profile("nersc_perlmutter")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HPCProfile:
    name: str
    description: str
    partition: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None
    time_limit: Optional[str] = None
    modules: Dict[str, List[str]] = field(default_factory=dict)
    partition_aliases: Dict[str, str] = field(default_factory=dict)
    default_nodes: int = 1
    default_ntasks_per_node: int = 1
    default_cpus_per_task: int = 1
    default_gpu_partition: Optional[str] = None
    default_cpu_partition: Optional[str] = None
    constraint: Optional[str] = None
    gres: Optional[str] = None
    pre_run_commands: str = ""
    extra_directives: List[str] = field(default_factory=list)


BUILTIN_PROFILES: Dict[str, HPCProfile] = {
    "generic": HPCProfile(
        name="generic",
        description="Generic Linux cluster with Slurm (minimal defaults)",
        default_nodes=1,
        default_ntasks_per_node=4,
        default_cpus_per_task=1,
        time_limit="01:00:00",
    ),
    "nersc_perlmutter": HPCProfile(
        name="nersc_perlmutter",
        description="NERSC Perlmutter supercomputer",
        partition="cpu",
        qos="regular",
        constraint="cpu",
        time_limit="24:00:00",
        default_cpu_partition="cpu",
        default_gpu_partition="gpu_gres",
        default_nodes=1,
        default_ntasks_per_node=32,
        default_cpus_per_task=4,
        modules={
            "vasp": ["vasp/6.4.2-cpu"],
            "vasp_gpu": ["vasp/6.4.2-gpu", "cudatoolkit/12.2"],
            "lammps": ["lammps/2024-cpu"],
            "lammps_gpu": ["lammps/2024-gpu", "cudatoolkit/12.2"],
            "orca": ["orca/5.0.4", "openmpi/4.1.5"],
        },
        partition_aliases={"cpu": "cpu", "gpu": "gpu_gres"},
        pre_run_commands="export OMP_NUM_THREADS=4",
        extra_directives=["--constraint=cpu"],
    ),
    "nersc_perlmutter_gpu": HPCProfile(
        name="nersc_perlmutter_gpu",
        description="NERSC Perlmutter GPU nodes",
        partition="gpu_gres",
        qos="regular",
        constraint="gpu",
        time_limit="24:00:00",
        default_gpu_partition="gpu_gres",
        default_nodes=1,
        default_ntasks_per_node=4,
        default_cpus_per_task=32,
        gres="gpu:4",
        modules={
            "vasp_gpu": ["vasp/6.4.2-gpu", "cudatoolkit/12.2"],
            "lammps_gpu": ["lammps/2024-gpu", "cudatoolkit/12.2"],
        },
        pre_run_commands="export OMP_NUM_THREADS=32",
        extra_directives=["--constraint=gpu", "--gres=gpu:4"],
    ),
    "mit_supercloud": HPCProfile(
        name="mit_supercloud",
        description="MIT SuperCloud cluster",
        partition="batch",
        qos="normal",
        time_limit="24:00:00",
        default_nodes=1,
        default_ntasks_per_node=36,
        default_cpus_per_task=1,
        modules={
            "vasp": ["vasp/6.4"],
            "lammps": ["lammps/stable"],
            "orca": ["orca/5.0"],
        },
    ),
    "umich_arc": HPCProfile(
        name="umich_arc",
        description="University of Michigan ARC cluster",
        partition="standard",
        qos="normal",
        time_limit="48:00:00",
        default_nodes=1,
        default_ntasks_per_node=36,
        modules={
            "vasp": ["vasp/6.4"],
            "lammps": ["lammps/2024"],
            "orca": ["orca/5.0.4"],
        },
    ),
}


def get_profile(name: str) -> HPCProfile:
    """Get a profile by name. Returns 'generic' if not found."""
    return BUILTIN_PROFILES.get(name, BUILTIN_PROFILES["generic"])


def list_profiles() -> List[str]:
    """List all available profile names."""
    return list(BUILTIN_PROFILES.keys())


def get_profile_description(name: str) -> str:
    """Get description for a profile."""
    profile = get_profile(name)
    return profile.description


def get_modules_for_app(profile_name: str, app_name: str) -> List[str]:
    """Get module list for a specific application in a profile."""
    profile = get_profile(profile_name)
    return profile.modules.get(app_name, [])