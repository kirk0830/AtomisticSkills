"""
VASP HPC integration - submit VASP calculations to HPC clusters.

A lightweight alternative to Atomate2 for simple VASP calculations.
For complex workflows (multi-step relaxations, NEB, etc.), use Atomate2.

Supports two execution modes:
- local: Direct execution (VASP binary on current machine)
- hpc: Submit to HPC cluster via Slurm (local or SSH)

Features:
- Input generation via pymatgen (INCAR, POSCAR, KPOINTS, POTCAR)
- Job submission via HPC module
- Output parsing (energy, forces, structure)
- Backward compatible with VASP_CMD environment variable

Usage:
    from src.utils.dft.vasp_hpc import VaspHPCRunner
    
    runner = VaspHPCRunner(mode="hpc")
    result = runner.run_static(
        structure=structure_object,
        xc="PBE",
        encut=520,
        kpoints=[4, 4, 4],
    )
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

EV_PER_HARTREE = 27.211386246


@dataclass
class VaspResult:
    energy_eV: Optional[float] = None
    energy_per_atom_eV: Optional[float] = None
    final_structure: Optional[Any] = None  # pymatgen Structure
    forces: Optional[List[List[float]]] = None
    max_force: Optional[float] = None
    converged: bool = False
    n_ionic_steps: int = 0
    n_electronic_steps: int = 0
    job_id: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)


def _check_pymatgen() -> bool:
    """Check if pymatgen is available."""
    try:
        import pymatgen
        return True
    except ImportError:
        return False


def generate_vasp_input(
    structure: Any,
    output_dir: str,
    xc: str = "PBE",
    encut: float = 520.0,
    kpoints: Union[List[int], str] = "Gamma",
    kpoints_density: Optional[float] = None,
    incar_settings: Optional[Dict[str, Any]] = None,
    potcar_functional: str = "PBE",
    calc_type: str = "static",
) -> Dict[str, Any]:
    """
    Generate VASP input files (INCAR, POSCAR, KPOINTS, POTCAR).

    Args:
        structure: pymatgen Structure object
        output_dir: Output directory
        xc: Exchange-correlation functional
        encut: Plane-wave cutoff energy in eV
        kpoints: K-point mesh (list of 3 ints) or 'Gamma' or 'Monkhorst'
        kpoints_density: K-points per reciprocal atom (overrides kpoints)
        incar_settings: Extra INCAR settings
        potcar_functional: POTCAR functional (PBE, LDA, etc.)
        calc_type: Calculation type: 'static', 'relax', 'scf'

    Returns:
        Dict with information about generated inputs
    """
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints, Potcar
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    os.makedirs(output_dir, exist_ok=True)

    if not isinstance(structure, Structure):
        from pymatgen.core import Structure as _Structure
        if hasattr(structure, "get_chemical_formula"):
            structure = _Structure.from_dict(structure.as_dict())
        else:
            raise ValueError("structure must be a pymatgen Structure object")

    poscar = Poscar(structure)
    poscar.write_file(os.path.join(output_dir, "POSCAR"))

    default_incar = {
        "PREC": "Accurate",
        "ENCUT": encut,
        "EDIFF": 1e-6,
        "IBRION": -1 if calc_type == "static" else 2,
        "NSW": 0 if calc_type == "static" else 99,
        "ISIF": 2 if calc_type == "static" else 3,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "ALGO": "Normal",
        "LREAL": False,
        "LWAVE": False,
        "LCHARG": False,
        "NELM": 200,
        "EDIFFG": -0.01 if calc_type == "relax" else None,
    }

    if xc.upper() in ("PBE", "GGA"):
        default_incar["GGA"] = "PE"
    elif xc.upper() in ("LDA",):
        pass
    else:
        default_incar["GGA"] = xc

    if calc_type == "relax":
        default_incar["IBRION"] = 2
        default_incar["NSW"] = 99
        default_incar["ISIF"] = 3
        default_incar["EDIFFG"] = -0.01

    if incar_settings:
        for key, value in incar_settings.items():
            default_incar[key] = value

    default_incar = {k: v for k, v in default_incar.items() if v is not None}

    incar = Incar(default_incar)
    incar.write_file(os.path.join(output_dir, "INCAR"))

    if kpoints_density:
        from pymatgen.io.vasp.inputs import Kpoints as _Kpoints
        kpts = _Kpoints.automatic_density(structure, kpoints_density)
    elif isinstance(kpoints, (list, tuple)) and len(kpoints) == 3:
        kpts = Kpoints.gamma_automatic(kpoints)
    elif kpoints == "Monkhorst":
        kpts = Kpoints.monkhorst_automatic([1, 1, 1])
    else:
        kpts = Kpoints.gamma_automatic([1, 1, 1])

    kpts.write_file(os.path.join(output_dir, "KPOINTS"))

    try:
        potcar_symbols = structure.symbol_set
        potcar = Potcar(symbols=[str(s) for s in potcar_symbols], functional=potcar_functional)
        potcar.write_file(os.path.join(output_dir, "POTCAR"))
        potcar_available = True
    except Exception as e:
        logger.warning(f"POTCAR generation failed: {e}")
        potcar_available = False

    return {
        "output_dir": output_dir,
        "n_atoms": len(structure),
        "formula": structure.composition.reduced_formula,
        "potcar_available": potcar_available,
    }


def parse_vasp_output(output_dir: str) -> VaspResult:
    """
    Parse VASP output files (OUTCAR, CONTCAR, etc.) and extract results.

    Args:
        output_dir: Directory containing VASP output files

    Returns:
        VaspResult with parsed results
    """
    result = VaspResult(output_dir=output_dir)

    outcar_path = os.path.join(output_dir, "OUTCAR")
    contcar_path = os.path.join(output_dir, "CONTCAR")

    if not os.path.exists(outcar_path):
        result.error_message = "OUTCAR not found"
        return result

    try:
        from pymatgen.io.vasp.outputs import Outcar, Vasprun

        vasprun_path = os.path.join(output_dir, "vasprun.xml")
        if os.path.exists(vasprun_path):
            try:
                vr = Vasprun(vasprun_path)
                result.energy_eV = vr.final_energy
                result.converged = vr.converged
                result.final_structure = vr.final_structure
                if vr.final_structure:
                    result.energy_per_atom_eV = result.energy_eV / len(vr.final_structure)
                if vr.ionic_steps:
                    result.n_ionic_steps = len(vr.ionic_steps)
                    last_step = vr.ionic_steps[-1]
                    if hasattr(last_step, "forces") and last_step.forces is not None:
                        import numpy as np
                        result.forces = last_step.forces.tolist()
                        result.max_force = float(np.max(np.abs(last_step.forces)))
            except Exception as e:
                logger.warning(f"vasprun.xml parsing failed: {e}")

        if result.energy_eV is None and os.path.exists(outcar_path):
            try:
                outcar = Outcar(outcar_path)
                if hasattr(outcar, "final_energy"):
                    result.energy_eV = outcar.final_energy
                if hasattr(outcar, "final_structure") and result.final_structure is None:
                    result.final_structure = outcar.final_structure
                if hasattr(outcar, "n_ionic_steps"):
                    result.n_ionic_steps = outcar.n_ionic_steps
            except Exception as e:
                logger.warning(f"OUTCAR parsing failed: {e}")

        if result.final_structure is None and os.path.exists(contcar_path):
            try:
                from pymatgen.io.vasp.inputs import Poscar
                poscar = Poscar.from_file(contcar_path)
                result.final_structure = poscar.structure
            except Exception:
                pass

        if result.energy_eV is not None and result.converged:
            result.success = True

    except ImportError:
        logger.warning("pymatgen not available, basic parsing only")
        result = _parse_vasp_basic(outcar_path, result)

    return result


def _parse_vasp_basic(outcar_path: str, result: VaspResult) -> VaspResult:
    """Basic OUTCAR parsing without pymatgen."""
    with open(outcar_path, "r") as f:
        text = f.read()

    import re

    energy_match = re.search(r"free  energy   TOTEN  =\s+([-\d.]+) eV", text)
    if energy_match:
        result.energy_eV = float(energy_match.group(1))

    if "reached required accuracy" in text:
        result.converged = True
        result.success = True

    return result


class VaspHPCRunner:
    """
    VASP calculation runner with HPC support.

    Lightweight alternative to Atomate2 for simple VASP calculations.
    For complex workflows, use Atomate2 (Atomate2Handler).

    Supports two modes:
    - local: Direct execution (VASP binary on current machine)
    - hpc: Submit to HPC cluster via Slurm
    """

    def __init__(
        self,
        mode: str = "auto",
        vasp_cmd: Optional[str] = None,
        hpc_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize VASP runner.

        Args:
            mode: Execution mode: 'local', 'hpc', or 'auto'
            vasp_cmd: VASP command (e.g., 'mpirun -np 16 vasp_std')
            hpc_config: HPC configuration dict
        """
        self.mode = mode
        self.vasp_cmd = vasp_cmd or os.environ.get("VASP_CMD") or os.environ.get("ATOMATE2_VASP_CMD")
        self._hpc_manager = None
        self._hpc_config_loader = None
        self._hpc_config = hpc_config or {}

        if mode == "auto":
            self.mode = self._auto_detect_mode()

    def _auto_detect_mode(self) -> str:
        from src.utils.hpc import HPCConfigLoader

        loader = HPCConfigLoader()
        config = loader.get_hpc_config()
        if config.mode != "auto":
            return config.mode

        if self.vasp_cmd:
            return "local"

        if os.environ.get("HPC_MODE") == "hpc":
            return "hpc"

        return "local"

    def _get_hpc_manager(self):
        if self._hpc_manager is not None:
            return self._hpc_manager

        from src.utils.hpc import JobManager, HPCConfigLoader

        loader = HPCConfigLoader()
        self._hpc_config_loader = loader

        if self._hpc_config:
            backend_config = self._hpc_config
        else:
            backend_config = loader.get_backend_config()

        self._hpc_manager = JobManager.from_config(backend_config)
        return self._hpc_manager

    def run_static(
        self,
        structure: Any,
        xc: str = "PBE",
        encut: float = 520.0,
        kpoints: Union[List[int], str] = "Gamma",
        kpoints_density: Optional[float] = None,
        incar_settings: Optional[Dict[str, Any]] = None,
        output_dir: str = ".",
        poll_interval: int = 60,
        timeout: Optional[int] = None,
        job_name: Optional[str] = None,
        nodes: int = 1,
        ntasks_per_node: int = 16,
    ) -> VaspResult:
        """
        Run a static (single-point) VASP calculation.

        Args:
            structure: pymatgen Structure object
            xc: Exchange-correlation functional
            encut: Plane-wave cutoff energy (eV)
            kpoints: K-point mesh or 'Gamma'
            kpoints_density: K-points per reciprocal atom
            incar_settings: Extra INCAR settings
            output_dir: Output directory
            poll_interval: HPC poll interval (seconds)
            timeout: HPC timeout (seconds)
            job_name: Custom job name
            nodes: Number of nodes (HPC mode)
            ntasks_per_node: Tasks per node (HPC mode)

        Returns:
            VaspResult with calculation results
        """
        input_params = {
            "xc": xc,
            "encut": encut,
            "kpoints": kpoints,
            "calc_type": "static",
        }

        os.makedirs(output_dir, exist_ok=True)

        if not _check_pymatgen():
            return VaspResult(
                error_message="pymatgen is required for VASP input generation",
                input_params=input_params,
            )

        try:
            gen_info = generate_vasp_input(
                structure=structure,
                output_dir=output_dir,
                xc=xc,
                encut=encut,
                kpoints=kpoints,
                kpoints_density=kpoints_density,
                incar_settings=incar_settings,
                calc_type="static",
            )
        except Exception as e:
            return VaspResult(
                error_message=f"Input generation failed: {e}",
                input_params=input_params,
            )

        if not gen_info.get("potcar_available", False):
            return VaspResult(
                error_message="POTCAR not available. Set PMG_VASP_PSP_DIR.",
                input_params=input_params,
            )

        if self.mode == "local":
            return self._run_vasp_local(
                output_dir=output_dir,
                input_params=input_params,
            )
        else:
            return self._run_vasp_hpc(
                output_dir=output_dir,
                input_params=input_params,
                poll_interval=poll_interval,
                timeout=timeout,
                job_name=job_name,
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
            )

    def run_relax(
        self,
        structure: Any,
        xc: str = "PBE",
        encut: float = 520.0,
        kpoints: Union[List[int], str] = "Gamma",
        kpoints_density: Optional[float] = None,
        incar_settings: Optional[Dict[str, Any]] = None,
        output_dir: str = ".",
        poll_interval: int = 60,
        timeout: Optional[int] = None,
        job_name: Optional[str] = None,
        nodes: int = 1,
        ntasks_per_node: int = 16,
    ) -> VaspResult:
        """
        Run a geometry optimization (relaxation) VASP calculation.

        Args:
            structure: pymatgen Structure object
            xc: Exchange-correlation functional
            encut: Plane-wave cutoff energy (eV)
            kpoints: K-point mesh or 'Gamma'
            kpoints_density: K-points per reciprocal atom
            incar_settings: Extra INCAR settings
            output_dir: Output directory
            poll_interval: HPC poll interval (seconds)
            timeout: HPC timeout (seconds)
            job_name: Custom job name
            nodes: Number of nodes (HPC mode)
            ntasks_per_node: Tasks per node (HPC mode)

        Returns:
            VaspResult with calculation results
        """
        input_params = {
            "xc": xc,
            "encut": encut,
            "kpoints": kpoints,
            "calc_type": "relax",
        }

        os.makedirs(output_dir, exist_ok=True)

        if not _check_pymatgen():
            return VaspResult(
                error_message="pymatgen is required for VASP input generation",
                input_params=input_params,
            )

        try:
            gen_info = generate_vasp_input(
                structure=structure,
                output_dir=output_dir,
                xc=xc,
                encut=encut,
                kpoints=kpoints,
                kpoints_density=kpoints_density,
                incar_settings=incar_settings,
                calc_type="relax",
            )
        except Exception as e:
            return VaspResult(
                error_message=f"Input generation failed: {e}",
                input_params=input_params,
            )

        if not gen_info.get("potcar_available", False):
            return VaspResult(
                error_message="POTCAR not available. Set PMG_VASP_PSP_DIR.",
                input_params=input_params,
            )

        if self.mode == "local":
            return self._run_vasp_local(
                output_dir=output_dir,
                input_params=input_params,
            )
        else:
            return self._run_vasp_hpc(
                output_dir=output_dir,
                input_params=input_params,
                poll_interval=poll_interval,
                timeout=timeout,
                job_name=job_name,
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
            )

    def _run_vasp_local(
        self,
        output_dir: str,
        input_params: Dict[str, Any],
    ) -> VaspResult:
        if not self.vasp_cmd:
            result = VaspResult(input_params=input_params, output_dir=output_dir)
            result.error_message = "VASP_CMD not set"
            result.success = False
            return result

        logger.info(f"Running VASP locally: {self.vasp_cmd}")
        try:
            subprocess.run(
                self.vasp_cmd,
                shell=True,
                cwd=output_dir,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            result = VaspResult(input_params=input_params, output_dir=output_dir)
            result.error_message = str(e)
            return result

        return parse_vasp_output(output_dir)

    def _run_vasp_hpc(
        self,
        output_dir: str,
        input_params: Dict[str, Any],
        poll_interval: int,
        timeout: Optional[int],
        job_name: Optional[str],
        nodes: int,
        ntasks_per_node: int,
    ) -> VaspResult:
        from src.utils.hpc import JobSpec, HPCConfigLoader

        manager = self._get_hpc_manager()
        loader = self._hpc_config_loader or HPCConfigLoader()

        job_name = job_name or f"vasp_{input_params.get('calc_type', 'calc')}"

        vasp_exec = self.vasp_cmd or "vasp_std"
        command = f"mpirun -np {nodes * ntasks_per_node} {vasp_exec}"

        try:
            if not manager.check_available():
                raise RuntimeError("HPC backend not available")

            resolved = loader.resolve_job_spec(
                {
                    "name": job_name,
                    "command": command,
                    "nodes": nodes,
                    "ntasks_per_node": ntasks_per_node,
                    "cpus_per_task": 1,
                    "work_dir": output_dir,
                    "output_file": f"{job_name}-%j.out",
                    "error_file": f"{job_name}-%j.err",
                },
                app="vasp",
            )
            job_spec = JobSpec.from_dict(resolved)

            logger.info(f"Submitting VASP job to HPC: {job_name}")
            job_id = manager.submit(job_spec)
            logger.info(f"Job submitted: {job_id}")

            status = manager.wait_for_completion(
                job_id, poll_interval=poll_interval, timeout=timeout
            )
            logger.info(f"Job {job_id} finished with state: {status.state}")

            result = parse_vasp_output(output_dir)
            result.job_id = job_id
            result.input_params = input_params
            return result

        except Exception as e:
            logger.error(f"HPC VASP calculation failed: {e}")
            result = VaspResult(
                error_message=str(e),
                input_params=input_params,
                output_dir=output_dir,
            )
            return result