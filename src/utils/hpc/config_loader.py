"""
HPC configuration loader with priority-based resolution.

Loads configuration from multiple sources with defined priority:
1. JobSpec explicit values (highest priority)
2. Environment variables (HPC_MODULES_<APP>, HPC_MODE, etc.)
3. Config file (~/.atomistic_skills.yaml)
4. Built-in profile defaults (lowest priority)

Usage:
    from src.utils.hpc.config_loader import HPCConfigLoader
    loader = HPCConfigLoader()
    
    # Get modules for VASP on Perlmutter
    modules = loader.get_modules("vasp", profile="nersc_perlmutter")
    
    # Resolve full JobSpec with defaults
    resolved_spec = loader.resolve_job_spec(raw_spec, app="vasp")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from src.utils.hpc.profiles import get_profile, get_modules_for_app, HPCProfile

logger = logging.getLogger(__name__)

CONFIG_FILE_DEFAULTS = [
    Path.home() / ".atomistic_skills.yaml",
    Path.home() / ".atomistic_skills.yml",
    Path.home() / ".config" / "atomistic_skills" / "config.yaml",
]


@dataclass
class HPCConfig:
    profile: str = "generic"
    mode: str = "auto"
    modules: Dict[str, List[str]] = field(default_factory=dict)
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_key: Optional[str] = None
    ssh_port: int = 22
    ssh_remote_work_dir: str = "~/hpc_jobs"
    local_work_dir: Optional[str] = None
    default_time_limit: str = "01:00:00"


class HPCConfigLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config_cache: Optional[Dict[str, Any]] = None

    def _find_config_file(self) -> Optional[Path]:
        if self.config_path and self.config_path.exists():
            return self.config_path
        for path in CONFIG_FILE_DEFAULTS:
            if path.exists():
                return path
        return None


    def _load_config_file(self) -> Dict[str, Any]:
        if self._config_cache is not None:
            return self._config_cache

        config_path = self._find_config_file()
        if not config_path:
            logger.debug("No config file found, using defaults")
            self._config_cache = {}
            return self._config_cache

        try:
            if YAML_AVAILABLE:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
                logger.debug(f"Loaded config from {config_path}")
            else:
                logger.warning(f"yaml module not available, cannot load {config_path}")
                config = {}
            self._config_cache = config
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            self._config_cache = {}
            return self._config_cache

    def get_hpc_config(self) -> HPCConfig:
        file_config = self._load_config_file()
        hpc_section = file_config.get("hpc", {})
        env_config = self._load_from_env()

        return HPCConfig(
            profile=env_config.get("profile") or hpc_section.get("profile", "generic"),
            mode=env_config.get("mode") or hpc_section.get("mode", "auto"),
            modules=hpc_section.get("modules", {}),
            ssh_host=env_config.get("ssh_host") or hpc_section.get("ssh_host"),
            ssh_user=env_config.get("ssh_user") or hpc_section.get("ssh_user"),
            ssh_key=env_config.get("ssh_key") or hpc_section.get("ssh_key"),
            ssh_port=int(env_config.get("ssh_port") or hpc_section.get("ssh_port", 22)),
            ssh_remote_work_dir=hpc_section.get("ssh_remote_work_dir", "~/hpc_jobs"),
            local_work_dir=hpc_section.get("local_work_dir"),
            default_time_limit=hpc_section.get("default_time_limit", "01:00:00"),
        )

    def _load_from_env(self) -> Dict[str, str]:
        env_config = {}
        if os.environ.get("HPC_MODE"):
            env_config["mode"] = os.environ["HPC_MODE"]
        if os.environ.get("HPC_PROFILE"):
            env_config["profile"] = os.environ["HPC_PROFILE"]
        if os.environ.get("HPC_SSH_HOST"):
            env_config["ssh_host"] = os.environ["HPC_SSH_HOST"]
        if os.environ.get("HPC_SSH_USER"):
            env_config["ssh_user"] = os.environ["HPC_SSH_USER"]
        if os.environ.get("HPC_SSH_KEY"):
            env_config["ssh_key"] = os.environ["HPC_SSH_KEY"]
        if os.environ.get("HPC_SSH_PORT"):
            env_config["ssh_port"] = os.environ["HPC_SSH_PORT"]
        return env_config

    def get_modules(
        self,
        app: str,
        profile: Optional[str] = None,
        explicit_modules: Optional[List[str]] = None,
    ) -> List[str]:
        if explicit_modules:
            return explicit_modules

        env_key = f"HPC_MODULES_{app.upper()}"
        if os.environ.get(env_key):
            env_val = os.environ[env_key]
            return [m.strip() for m in env_val.split(",") if m.strip()]

        config = self.get_hpc_config()
        config_modules = config.modules.get(app, [])
        if config_modules:
            return config_modules

        profile_name = profile or config.profile
        return get_modules_for_app(profile_name, app)

    def get_backend_config(self) -> Dict[str, Any]:
        config = self.get_hpc_config()
        backend_config: Dict[str, Any] = {"mode": config.mode}
        if config.mode == "ssh":
            if config.ssh_host:
                backend_config["host"] = config.ssh_host
            if config.ssh_user:
                backend_config["user"] = config.ssh_user
            if config.ssh_key:
                backend_config["key_path"] = config.ssh_key
            if config.ssh_port != 22:
                backend_config["port"] = config.ssh_port
            backend_config["remote_work_dir"] = config.ssh_remote_work_dir
        if config.mode == "local":
            if config.local_work_dir:
                backend_config["work_dir"] = config.local_work_dir
        return backend_config

    def resolve_job_spec(
        self,
        spec: Dict[str, Any],
        app: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        config = self.get_hpc_config()
        profile_obj = get_profile(profile or config.profile)

        resolved: Dict[str, Any] = {
            "name": spec.get("name", "unnamed"),
            "command": spec.get("command", ""),
            "nodes": spec.get("nodes", profile_obj.default_nodes),
            "ntasks_per_node": spec.get("ntasks_per_node", profile_obj.default_ntasks_per_node),
            "cpus_per_task": spec.get("cpus_per_task", profile_obj.default_cpus_per_task),
            "time_limit": spec.get("time_limit") or profile_obj.time_limit or config.default_time_limit,
            "partition": spec.get("partition") or profile_obj.partition,
            "qos": spec.get("qos") or profile_obj.qos,
            "account": spec.get("account") or profile_obj.account,
        }

        if app:
            resolved["modules"] = self.get_modules(
                app, profile=profile, explicit_modules=spec.get("modules")
            )

        for key in ["gres", "work_dir", "output_file", "error_file", "email",
                    "email_type", "pre_run", "post_run", "environment",
                    "extra_directives", "metadata"]:
            if spec.get(key):
                resolved[key] = spec[key]

        if profile_obj.pre_run_commands and not resolved.get("pre_run"):
            resolved["pre_run"] = profile_obj.pre_run_commands

        if profile_obj.extra_directives and not resolved.get("extra_directives"):
            resolved["extra_directives"] = profile_obj.extra_directives

        return resolved

    def create_sample_config(self, output_path: Optional[str] = None) -> str:
        sample = """# AtomisticSkills Configuration
# Copy to ~/.atomistic_skills.yaml and customize

# Materials Project API Key (required for base tools)
MP_API_KEY: "your_mp_api_key_here"

# HPC Configuration
hpc:
  # Profile: generic, nersc_perlmutter, mit_supercloud, etc.
  profile: "generic"
  
  # Execution mode: auto, local, ssh
  mode: "auto"
  
  # Default time limit for jobs
  default_time_limit: "01:00:00"
  
  # SSH configuration (for ssh mode)
  ssh_host: null
  ssh_user: null
  ssh_key: null
  ssh_port: 22
  ssh_remote_work_dir: "~/hpc_jobs"
  
  # Local work directory (for local mode)
  local_work_dir: null
  
  # Application-specific modules
  # These override profile defaults
  modules:
    vasp: ["vasp/6.4.2-cpu"]
    vasp_gpu: ["vasp/6.4.2-gpu", "cuda/12.2"]
    orca: ["orca/5.0.4", "openmpi/4.1.5"]
    lammps: ["lammps/2024"]
    lammps_gpu: ["lammps/2024-gpu", "cuda/12.2"]
"""
        if output_path:
            Path(output_path).write_text(sample)
            logger.info(f"Sample config written to {output_path}")
        return sample