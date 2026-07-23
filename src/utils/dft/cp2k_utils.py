"""
CP2K utilities and runners for MLIP Agent.

All functions use ASE calculators and expect ASE 3.22+.
Units: energy eV, forces eV/Å, stress eV/Å³.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dft_common import DFTResult

logger = logging.getLogger(__name__)


def _get_cp2k_command(command: Optional[str] = None) -> Optional[str]:
    """Resolve the CP2K command from explicit value or environment variables."""
    if command:
        return command
    for env_var in ("ASE_CP2K_COMMAND", "CP2K_COMMAND"):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def _get_cp2k_data_dir(data_dir: Optional[str] = None) -> Optional[str]:
    """Resolve the CP2K data directory."""
    if data_dir:
        return data_dir
    return os.environ.get("CP2K_DATA_DIR")


def _infer_basis_potential_files(
    data_dir: Optional[str] = None,
    basis_set_file: Optional[str] = None,
    potential_file: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer BASIS_SET and POTENTIAL file paths from CP2K_DATA_DIR.

    Args:
        data_dir: CP2K data directory.
        basis_set_file: Explicit basis set file path.
        potential_file: Explicit potential file path.

    Returns:
        Tuple (basis_set_file, potential_file).
    """
    resolved_dir = _get_cp2k_data_dir(data_dir)
    if not resolved_dir:
        return basis_set_file, potential_file

    data_path = Path(resolved_dir)
    if basis_set_file is None:
        for candidate in ("BASIS_SET", "BASIS_MOLOPT"):
            candidate_path = data_path / candidate
            if candidate_path.exists():
                basis_set_file = str(candidate_path)
                break
    if potential_file is None:
        candidate_path = data_path / "POTENTIAL"
        if candidate_path.exists():
            potential_file = str(candidate_path)

    return basis_set_file, potential_file


def build_cp2k_calculator(
    atoms: Any,
    calc_type: str = "static",
    command: Optional[str] = None,
    basis_set_file: Optional[str] = None,
    potential_file: Optional[str] = None,
    cutoff: float = 400.0,
    rel_cutoff: float = 50.0,
    kpts: Tuple[int, int, int] = (4, 4, 4),
    spin_polarized: bool = False,
    initial_magmoms: Optional[List[float]] = None,
    xc: str = "PBE",
    input_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """
    Build an ASE CP2K calculator.

    Args:
        atoms: ASE Atoms object.
        calc_type: "static" or "relax".
        command: CP2K command. Falls back to ASE_CP2K_COMMAND or CP2K_COMMAND.
        basis_set_file: Path to CP2K basis set file. Inferred from
            CP2K_DATA_DIR if not provided.
        potential_file: Path to CP2K potential file. Inferred from
            CP2K_DATA_DIR if not provided.
        cutoff: Plane-wave cutoff in Ry.
        rel_cutoff: Relative cutoff in Ry.
        kpts: Monkhorst-Pack k-point mesh.
        spin_polarized: Whether to run spin-polarized calculation.
        initial_magmoms: Initial magnetic moments in Bohr magneton.
        xc: Exchange-correlation functional.
        input_data: Additional CP2K input sections (dict of section -> keywords).
        **kwargs: Extra arguments passed to the CP2K calculator.

    Returns:
        ASE CP2K calculator instance.
    """
    try:
        from ase.calculators.cp2k import CP2K
    except ImportError as exc:
        raise ImportError("ASE with CP2K support is required.") from exc

    resolved_command = _get_cp2k_command(command)
    if not resolved_command:
        logger.warning(
            "No CP2K command provided. Set ASE_CP2K_COMMAND or CP2K_COMMAND."
        )

    basis_set_file, potential_file = _infer_basis_potential_files(
        data_dir=kwargs.pop("data_dir", None),
        basis_set_file=basis_set_file,
        potential_file=potential_file,
    )

    if basis_set_file is None or potential_file is None:
        logger.warning(
            "CP2K basis/potential files not found. Set basis_set_file, "
            "potential_file, or CP2K_DATA_DIR."
        )

    # Default basis set and pseudo-potential names per element.
    symbols = atoms.get_chemical_symbols()
    unique_symbols = sorted(set(symbols))
    basis_set = kwargs.pop("basis_set", "DZVP-MOLOPT-SR-GTH")
    potential = kwargs.pop("potential", "GTH-" + xc.upper())

    if spin_polarized or initial_magmoms is not None:
        spin_polarized = True

    calc_kwargs: Dict[str, Any] = {
        "command": resolved_command,
        "xc": xc,
        "cutoff": cutoff,
        "rel_cutoff": rel_cutoff,
        "basis_set": basis_set,
        "potential": potential,
        "poisson_solver": kwargs.pop("poisson_solver", "auto"),
        "kpts": kpts,
        "inp": input_data or {},
    }
    if basis_set_file:
        calc_kwargs["basis_set_file"] = basis_set_file
    if potential_file:
        calc_kwargs["potential_file"] = potential_file
    if spin_polarized:
        calc_kwargs["spin_polarized"] = True
        if initial_magmoms is not None:
            calc_kwargs["magmoms"] = initial_magmoms

    calc_kwargs.update(kwargs)
    return CP2K(**calc_kwargs)


def _run_cp2k(
    atoms: Any,
    output_dir: str,
    calc_type: str = "static",
    fmax: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run a CP2K calculation and return a raw results dict."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calc = build_cp2k_calculator(atoms, calc_type=calc_type, **kwargs)
    atoms.calc = calc

    if calc_type == "relax" and fmax is not None:
        try:
            from ase.optimize import BFGS

            optimizer = BFGS(
                atoms,
                trajectory=str(output_path / "relax.traj"),
                logfile=str(output_path / "relax.log"),
            )
            optimizer.run(fmax=fmax)
        except Exception as exc:
            logger.error(f"CP2K relax failed: {exc}")
            raise
    else:
        atoms.get_potential_energy()

    return parse_cp2k_results(output_dir)


def run_cp2k_static(atoms: Any, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Run a CP2K single-point calculation.

    Args:
        atoms: ASE Atoms object.
        output_dir: Directory for input/output files.
        **kwargs: Arguments passed to build_cp2k_calculator.

    Returns:
        Dict with keys: energy, forces, stress, final_atoms, converged.
        Units: energy eV, forces eV/Å, stress eV/Å³.
    """
    return _run_cp2k(atoms, output_dir, calc_type="static", **kwargs)


def run_cp2k_relax(
    atoms: Any,
    output_dir: str,
    fmax: float = 0.05,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a CP2K geometry relaxation.

    Args:
        atoms: ASE Atoms object.
        output_dir: Directory for input/output files.
        fmax: Force convergence criterion in eV/Å.
        **kwargs: Arguments passed to build_cp2k_calculator.

    Returns:
        Dict with keys: energy, forces, stress, final_atoms, converged.
        Units: energy eV, forces eV/Å, stress eV/Å³.
    """
    return _run_cp2k(atoms, output_dir, calc_type="relax", fmax=fmax, **kwargs)


def parse_cp2k_mulliken_spins(output_file: str) -> Optional[List[float]]:
    """
    Parse Mulliken spin populations from a CP2K output file.

    Args:
        output_file: Path to the CP2K output file.

    Returns:
        List of atomic spin moments in Bohr magneton, or None if not found.
    """
    path = Path(output_file)
    if not path.exists():
        logger.warning(f"CP2K output file not found: {output_file}")
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")

    # Locate the last Mulliken Population Analysis block.
    matches = list(re.finditer(r"Mulliken Population Analysis", text))
    if not matches:
        logger.warning("No Mulliken Population Analysis block found")
        return None

    block_start = matches[-1].end()
    block = text[block_start: block_start + 5000]
    lines = block.splitlines()

    spins: List[float] = []
    parsing = False
    for line in lines:
        # Header line usually contains "Atom", "Charge", "Spin", etc.
        if "Atom" in line and "Spin" in line:
            parsing = True
            continue
        if not parsing:
            continue
        if not line.strip() or line.strip().startswith("-"):
            if spins:
                break
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                spin = float(parts[-1])
                spins.append(spin)
            except ValueError:
                continue
        elif spins:
            break

    if not spins:
        logger.warning("Could not extract Mulliken spins from output")
        return None

    logger.info(f"Extracted {len(spins)} Mulliken spins from {output_file}")
    return spins


def parse_cp2k_results(
    output_dir: str,
    parse_magmoms: bool = False,
) -> Dict[str, Any]:
    """
    Parse CP2K results from an output directory.

    Args:
        output_dir: Directory containing CP2K output files.
        parse_magmoms: If True, extract Mulliken spin populations.

    Returns:
        Dict with keys: energy, forces, stress, final_atoms, converged,
        magnetic_moments.
        Units: energy eV, forces eV/Å, stress eV/Å³.
    """
    output_path = Path(output_dir)
    result: Dict[str, Any] = {
        "engine": "cp2k",
        "energy": None,
        "forces": None,
        "stress": None,
        "final_atoms": None,
        "converged": False,
        "magnetic_moments": None,
        "raw_output_dir": str(output_path),
    }

    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_path}")
        return result

    # CP2K calculator writes output to a file named cp2k-*.out by default.
    out_files = sorted(output_path.glob("cp2k-*.out"))
    if not out_files:
        out_files = sorted(output_path.glob("*.out"))

    if not out_files:
        logger.warning(f"No CP2K output files found in {output_path}")
        return result

    out_file = out_files[0]

    try:
        from ase.io import read

        final_atoms = read(str(out_file), format="cp2k-output")
        if final_atoms is not None:
            result["final_atoms"] = final_atoms
            result["energy"] = float(final_atoms.get_potential_energy())
            result["forces"] = final_atoms.get_forces()
            result["stress"] = final_atoms.get_stress(voigt=False)
    except Exception as exc:
        logger.warning(f"ASE parsing of {out_file} failed: {exc}")

    # Text-based convergence check.
    try:
        text = out_file.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"SCF run converged", text, re.IGNORECASE):
            result["converged"] = True
        elif re.search(r"SCF run NOT converged", text, re.IGNORECASE):
            result["converged"] = False
        elif result["energy"] is not None:
            # ASE read succeeded, assume converged.
            result["converged"] = True
    except Exception as exc:
        logger.warning(f"Text parsing of {out_file} failed: {exc}")

    if parse_magmoms:
        try:
            result["magnetic_moments"] = parse_cp2k_mulliken_spins(str(out_file))
        except Exception as exc:
            logger.warning(f"Parsing magnetic moments failed: {exc}")

    return result
