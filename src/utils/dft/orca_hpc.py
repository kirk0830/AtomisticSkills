"""
ORCA HPC integration - submit ORCA calculations to HPC clusters.

Supports two execution modes:
- local: Direct execution via SCINE wrapper (existing behavior)
- hpc: Submit to HPC cluster via Slurm (local or SSH)

For HPC mode:
1. Generate ORCA input file from parameters
2. Submit job to HPC cluster
3. Wait for completion
4. Parse output and return results

Usage:
    from src.utils.dft.orca_hpc import OrcaHPCRunner
    
    runner = OrcaHPCRunner(mode="hpc")
    result = runner.run_singlepoint(
        structure_path="molecule.xyz",
        functional="B3LYP",
        basis_set="def2-TZVP",
        nprocs=16,
    )
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386246
BOHR_PER_ANGSTROM = 0.529177210903

ENERGY_PATTERN = re.compile(r"FINAL SINGLE POINT ENERGY\s+([-\d.]+)")
SCF_CONVERGED_PATTERN = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
SCF_NOT_CONVERGED_PATTERN = re.compile(r"SCF NOT CONVERGED")
TOTAL_RUN_TIME_PATTERN = re.compile(r"TOTAL RUN TIME:\s+(.+)")


@dataclass
class OrcaSinglepointResult:
    energy_hartree: Optional[float] = None
    energy_eV: Optional[float] = None
    scf_converged: bool = False
    scf_cycles: Optional[int] = None
    run_time: Optional[str] = None
    output_text: str = ""
    job_id: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None
    forces_eV_per_Ang: Optional[List[List[float]]] = None
    hessian_eV_per_Ang2: Optional[List[List[float]]] = None
    input_params: Dict[str, Any] = field(default_factory=dict)


def generate_orca_input(
    structure_path: str,
    functional: str = "PBE",
    basis_set: str = "def2-SVP",
    charge: int = 0,
    spin_multiplicity: int = 1,
    dispersion: Optional[str] = None,
    solvation: Optional[str] = None,
    solvent: Optional[str] = None,
    nprocs: int = 1,
    memory_per_proc: int = 4096,
    compute_gradients: bool = False,
    compute_hessian: bool = False,
    special_options: Optional[List[str]] = None,
    extra_lines: Optional[List[str]] = None,
) -> str:
    """
    Generate an ORCA input file from parameters.

    Args:
        structure_path: Path to structure file (.xyz, .cif, etc.)
        functional: DFT functional name
        basis_set: Basis set name
        charge: Molecular charge
        spin_multiplicity: Spin multiplicity (2S+1)
        dispersion: Dispersion correction (D3BJ, D4, etc.)
        solvation: Solvation model (CPCM, SMD)
        solvent: Solvent name
        nprocs: Number of processors
        memory_per_proc: Memory per processor in MB
        compute_gradients: Whether to compute gradients
        compute_hessian: Whether to compute Hessian
        special_options: List of special options
        extra_lines: Extra lines to add to input file

    Returns:
        ORCA input file content as string
    """
    import ase.io

    atoms = ase.io.read(structure_path)

    lines = []

    method_line = f"! {functional} {basis_set}"
    if dispersion:
        method_line += f" {dispersion}"
    if special_options:
        for opt in special_options:
            method_line += f" {opt}"
    lines.append(method_line)

    if compute_gradients and not compute_hessian:
        lines.append("! ENGRAD")
    if compute_hessian:
        lines.append("! FREQ")

    lines.append("")
    lines.append(f"%pal nprocs {nprocs} end")
    lines.append(f"%maxcore {memory_per_proc}")
    lines.append("")

    if solvation and solvent:
        lines.append(f"%cpcm")
        lines.append(f"  solvent {solvent}")
        lines.append(f"end")
        lines.append("")

    if extra_lines:
        for line in extra_lines:
            lines.append(line)
        lines.append("")

    lines.append("* xyz")
    lines.append(f"{charge} {spin_multiplicity}")

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    for sym, pos in zip(symbols, positions):
        lines.append(f"  {sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}")

    lines.append("*")
    lines.append("")

    return "\n".join(lines)


def parse_orca_output(output_path: str) -> OrcaSinglepointResult:
    """
    Parse ORCA output file and extract key results.

    Args:
        output_path: Path to ORCA output file

    Returns:
        OrcaSinglepointResult with parsed results
    """
    result = OrcaSinglepointResult()

    if not os.path.exists(output_path):
        result.error_message = f"Output file not found: {output_path}"
        return result

    with open(output_path, "r") as f:
        text = f.read()

    result.output_text = text

    energy_match = ENERGY_PATTERN.search(text)
    if energy_match:
        result.energy_hartree = float(energy_match.group(1))
        result.energy_eV = result.energy_hartree * HARTREE_TO_EV

    scf_match = SCF_CONVERGED_PATTERN.search(text)
    if scf_match:
        result.scf_converged = True
        result.scf_cycles = int(scf_match.group(1))

    if SCF_NOT_CONVERGED_PATTERN.search(text):
        result.scf_converged = False

    time_match = TOTAL_RUN_TIME_PATTERN.search(text)
    if time_match:
        result.run_time = time_match.group(1).strip()

    if "ERROR" in text.upper() and result.energy_hartree is None:
        result.success = False
        error_lines = []
        for line in text.split("\n"):
            if "ERROR" in line.upper() or "error" in line.lower():
                error_lines.append(line.strip())
        if error_lines:
            result.error_message = "; ".join(error_lines[:5])
    elif result.energy_hartree is not None and result.scf_converged:
        result.success = True

    return result


class OrcaHPCRunner:
    """
    ORCA calculation runner with HPC support.

    Supports two modes:
    - local: Direct execution (SCINE wrapper or binary)
    - hpc: Submit to HPC cluster via Slurm
    """

    def __init__(
        self,
        mode: str = "auto",
        orca_binary: Optional[str] = None,
        hpc_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ORCA runner.

        Args:
            mode: Execution mode: 'local', 'hpc', or 'auto'
            orca_binary: Path to ORCA binary (for local mode)
            hpc_config: HPC configuration dict for HPC mode
        """
        self.mode = mode
        self.orca_binary = orca_binary or os.environ.get("ORCA_BINARY_PATH")
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

        if self.orca_binary and os.path.isfile(self.orca_binary):
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

    def run_singlepoint(
        self,
        structure_path: str,
        functional: str = "PBE",
        basis_set: str = "def2-SVP",
        charge: int = 0,
        spin_multiplicity: int = 1,
        dispersion: Optional[str] = None,
        solvation: Optional[str] = None,
        solvent: Optional[str] = None,
        nprocs: int = 1,
        memory_per_proc: int = 4096,
        compute_gradients: bool = False,
        compute_hessian: bool = False,
        special_options: Optional[List[str]] = None,
        output_dir: str = ".",
        poll_interval: int = 30,
        timeout: Optional[int] = None,
        job_name: Optional[str] = None,
    ) -> OrcaSinglepointResult:
        """
        Run a single-point energy calculation.

        Args:
            structure_path: Path to input structure
            functional: DFT functional
            basis_set: Basis set
            charge: Molecular charge
            spin_multiplicity: Spin multiplicity
            dispersion: Dispersion correction
            solvation: Solvation model
            solvent: Solvent name
            nprocs: Number of processors
            memory_per_proc: Memory per processor (MB)
            compute_gradients: Compute forces
            compute_hessian: Compute Hessian
            special_options: Extra ORCA options
            output_dir: Output directory
            poll_interval: HPC poll interval (seconds)
            timeout: HPC timeout (seconds)
            job_name: Custom job name for HPC

        Returns:
            OrcaSinglepointResult with calculation results
        """
        os.makedirs(output_dir, exist_ok=True)

        input_params = {
            "functional": functional,
            "basis_set": basis_set,
            "charge": charge,
            "spin_multiplicity": spin_multiplicity,
            "dispersion": dispersion,
            "solvation": solvation,
            "solvent": solvent,
            "nprocs": nprocs,
            "compute_gradients": compute_gradients,
            "compute_hessian": compute_hessian,
        }

        if self.mode == "local":
            return self._run_singlepoint_local(
                structure_path=structure_path,
                functional=functional,
                basis_set=basis_set,
                charge=charge,
                spin_multiplicity=spin_multiplicity,
                dispersion=dispersion,
                solvation=solvation,
                solvent=solvent,
                nprocs=nprocs,
                memory_per_proc=memory_per_proc,
                compute_gradients=compute_gradients,
                compute_hessian=compute_hessian,
                special_options=special_options,
                output_dir=output_dir,
                input_params=input_params,
            )
        else:
            return self._run_singlepoint_hpc(
                structure_path=structure_path,
                functional=functional,
                basis_set=basis_set,
                charge=charge,
                spin_multiplicity=spin_multiplicity,
                dispersion=dispersion,
                solvation=solvation,
                solvent=solvent,
                nprocs=nprocs,
                memory_per_proc=memory_per_proc,
                compute_gradients=compute_gradients,
                compute_hessian=compute_hessian,
                special_options=special_options,
                output_dir=output_dir,
                poll_interval=poll_interval,
                timeout=timeout,
                job_name=job_name,
                input_params=input_params,
            )

    def _run_singlepoint_local(
        self,
        structure_path: str,
        functional: str,
        basis_set: str,
        charge: int,
        spin_multiplicity: int,
        dispersion: Optional[str],
        solvation: Optional[str],
        solvent: Optional[str],
        nprocs: int,
        memory_per_proc: int,
        compute_gradients: bool,
        compute_hessian: bool,
        special_options: Optional[List[str]],
        output_dir: str,
        input_params: Dict[str, Any],
    ) -> OrcaSinglepointResult:
        input_content = generate_orca_input(
            structure_path=structure_path,
            functional=functional,
            basis_set=basis_set,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            dispersion=dispersion,
            solvation=solvation,
            solvent=solvent,
            nprocs=nprocs,
            memory_per_proc=memory_per_proc,
            compute_gradients=compute_gradients,
            compute_hessian=compute_hessian,
            special_options=special_options,
        )

        input_path = os.path.join(output_dir, "orca.inp")
        output_path = os.path.join(output_dir, "orca.out")

        with open(input_path, "w") as f:
            f.write(input_content)

        if self.orca_binary:
            logger.info(f"Running ORCA locally: {self.orca_binary}")
            try:
                cmd = [self.orca_binary, input_path]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=output_dir,
                    timeout=timeout if 'timeout' in dir() else None,
                )
                with open(output_path, "w") as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n--- STDERR ---\n")
                        f.write(result.stderr)
            except Exception as e:
                res = OrcaSinglepointResult(input_params=input_params)
                res.success = False
                res.error_message = str(e)
                return res
        else:
            logger.warning("ORCA binary not set, generating input only")
            res = OrcaSinglepointResult(input_params=input_params)
            res.success = False
            res.error_message = "ORCA binary not available (input generated only)"
            return res

        parsed = parse_orca_output(output_path)
        parsed.input_params = input_params
        return parsed

    def _run_singlepoint_hpc(
        self,
        structure_path: str,
        functional: str,
        basis_set: str,
        charge: int,
        spin_multiplicity: int,
        dispersion: Optional[str],
        solvation: Optional[str],
        solvent: Optional[str],
        nprocs: int,
        memory_per_proc: int,
        compute_gradients: bool,
        compute_hessian: bool,
        special_options: Optional[List[str]],
        output_dir: str,
        poll_interval: int,
        timeout: Optional[int],
        job_name: Optional[str],
        input_params: Dict[str, Any],
    ) -> OrcaSinglepointResult:
        from src.utils.hpc import JobSpec, HPCConfigLoader

        manager = self._get_hpc_manager()
        loader = self._hpc_config_loader or HPCConfigLoader()

        input_content = generate_orca_input(
            structure_path=structure_path,
            functional=functional,
            basis_set=basis_set,
            charge=charge,
            spin_multiplicity=spin_multiplicity,
            dispersion=dispersion,
            solvation=solvation,
            solvent=solvent,
            nprocs=nprocs,
            memory_per_proc=memory_per_proc,
            compute_gradients=compute_gradients,
            compute_hessian=compute_hessian,
            special_options=special_options,
        )

        job_name = job_name or f"orca_{functional}_{basis_set}".replace("/", "_")
        remote_dir = f"{output_dir}"

        input_filename = "orca.inp"
        output_filename = "orca.out"

        temp_input = os.path.join(output_dir, input_filename)
        with open(temp_input, "w") as f:
            f.write(input_content)

        try:
            if not manager.check_available():
                raise RuntimeError("HPC backend not available")

            resolved = loader.resolve_job_spec(
                {
                    "name": job_name,
                    "command": f"$(which orca || echo orca) {input_filename} > {output_filename}",
                    "ntasks_per_node": nprocs,
                    "cpus_per_task": 1,
                    "work_dir": output_dir,
                    "output_file": f"{job_name}-%j.out",
                    "error_file": f"{job_name}-%j.err",
                },
                app="orca",
            )
            job_spec = JobSpec.from_dict(resolved)

            logger.info(f"Submitting ORCA job to HPC: {job_name}")
            job_id = manager.submit(job_spec)
            logger.info(f"Job submitted: {job_id}")

            status = manager.wait_for_completion(
                job_id, poll_interval=poll_interval, timeout=timeout
            )
            logger.info(f"Job {job_id} finished with state: {status.state}")

            try:
                output_text = manager.read_output(job_id, f"{output_dir}/{output_filename}")
                local_output = os.path.join(output_dir, output_filename)
                with open(local_output, "w") as f:
                    f.write(output_text)

                result = parse_orca_output(local_output)
            except Exception as e:
                logger.warning(f"Failed to read output: {e}")
                result = OrcaSinglepointResult(
                    error_message=f"Failed to read output: {e}",
                    input_params=input_params,
                )

            result.job_id = job_id
            result.input_params = input_params
            return result

        except Exception as e:
            logger.error(f"HPC ORCA calculation failed: {e}")
            result = OrcaSinglepointResult(
                error_message=str(e),
                input_params=input_params,
            )
            return result