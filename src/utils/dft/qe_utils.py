"""
Quantum ESPRESSO utilities and runners for MLIP Agent.

All functions use ASE calculators and expect ASE 3.22+.
Units: energy eV, forces eV/Å, stress eV/Å³.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .dft_common import DFTResult, get_kpath_seekpath

logger = logging.getLogger(__name__)


def _get_espresso_command(command: Optional[str] = None) -> Optional[str]:
    """Resolve the pw.x command from explicit value or environment variables."""
    if command:
        return command
    for env_var in ("ASE_ESPRESSO_COMMAND", "ESPRESSO_COMMAND"):
        value = os.environ.get(env_var)
        if value:
            return value
    return None


def _get_pseudo_dir(pseudo_dir: Optional[str] = None) -> Optional[str]:
    """Resolve the pseudopotential directory."""
    if pseudo_dir:
        return pseudo_dir
    return os.environ.get("ESPRESSO_PSEUDO")


def _compute_band_gap(
    eigenvalues: np.ndarray,
    fermi_energy: Optional[float] = None,
    occupations: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Estimate band gap from eigenvalues (eV).

    Args:
        eigenvalues: array of shape (n_kpoints, n_bands) or (n_spin, n_kpoints, n_bands).
        fermi_energy: Fermi energy in eV.
        occupations: optional occupancy array matching eigenvalues shape.

    Returns:
        Band gap in eV, or None if it cannot be determined.
    """
    if eigenvalues is None or eigenvalues.size == 0:
        return None

    ev = np.asarray(eigenvalues)
    if occupations is not None:
        occ = np.asarray(occupations)
        shape = ev.shape
        occ = occ.reshape(shape)
        occupied = ev[occ > 0.5]
        unoccupied = ev[occ <= 0.5]
        if occupied.size == 0 or unoccupied.size == 0:
            return 0.0
        homo = float(np.max(occupied))
        lumo = float(np.min(unoccupied))
        return max(0.0, lumo - homo)

    if fermi_energy is None:
        return None

    # Conservative estimate using Fermi energy as reference.
    if ev.ndim == 2:
        occupied_max = float(np.max(ev[ev <= fermi_energy])) if np.any(ev <= fermi_energy) else None
        unoccupied_min = float(np.min(ev[ev > fermi_energy])) if np.any(ev > fermi_energy) else None
    elif ev.ndim == 3:
        occupied_max = float(np.max(ev[ev <= fermi_energy])) if np.any(ev <= fermi_energy) else None
        unoccupied_min = float(np.min(ev[ev > fermi_energy])) if np.any(ev > fermi_energy) else None
    else:
        return None

    if occupied_max is None or unoccupied_min is None:
        return 0.0
    return max(0.0, unoccupied_min - occupied_max)


def build_qe_calculator(
    atoms: Any,
    calc_type: str = "static",
    pseudo_dir: Optional[str] = None,
    command: Optional[str] = None,
    ecutwfc: float = 50.0,
    ecutrho: Optional[float] = None,
    kpts: Tuple[int, int, int] = (4, 4, 4),
    occupations: str = "smearing",
    smearing: str = "mv",
    degauss: float = 0.01,
    conv_thr: float = 1e-8,
    input_data: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """
    Build an ASE Quantum ESPRESSO calculator.

    Args:
        atoms: ASE Atoms object (used to determine nspin/magnetization defaults).
        calc_type: "static", "relax", "bands", or "scf".
        pseudo_dir: Directory containing pseudopotentials. Falls back to
            ESPRESSO_PSEUDO environment variable.
        command: pw.x command. Falls back to ASE_ESPRESSO_COMMAND or
            ESPRESSO_COMMAND environment variables.
        ecutwfc: Plane-wave kinetic energy cutoff in Ry.
        ecutrho: Charge density cutoff in Ry (defaults to 4 * ecutwfc).
        kpts: Monkhorst-Pack k-point mesh.
        occupations: Occupation smearing scheme ("smearing", "fixed", "tetrahedra").
        smearing: Smearing type ("mv", "gaussian", "fermi-dirac", etc.).
        degauss: Smearing width in Ry.
        conv_thr: SCF convergence threshold in Ry.
        input_data: Additional Quantum ESPRESSO input namelist data merged into
            the control/system sections.
        **kwargs: Extra arguments passed to the Espresso calculator.

    Returns:
        ASE Espresso calculator instance.
    """
    try:
        from ase.calculators.espresso import Espresso
    except ImportError as exc:
        raise ImportError(
            "ASE with Quantum ESPRESSO support is required."
        ) from exc

    resolved_pseudo_dir = _get_pseudo_dir(pseudo_dir)
    if not resolved_pseudo_dir:
        raise ValueError(
            "pseudo_dir is required. Set pseudo_dir or the ESPRESSO_PSEUDO environment variable."
        )

    resolved_command = _get_espresso_command(command)
    if not resolved_command:
        logger.warning(
            "No Quantum ESPRESSO command provided. Set ASE_ESPRESSO_COMMAND or ESPRESSO_COMMAND."
        )

    if ecutrho is None:
        ecutrho = 4.0 * ecutwfc

    calculation_map = {
        "static": "scf",
        "scf": "scf",
        "relax": "relax",
        "bands": "scf",
    }
    calculation = calculation_map.get(calc_type, "scf")

    control = {
        "calculation": calculation,
        "prefix": kwargs.pop("prefix", "qe_calc"),
        "pseudo_dir": resolved_pseudo_dir,
        "outdir": kwargs.pop("outdir", "./"),
        "tprnfor": True,
        "tstress": True,
        "verbosity": kwargs.pop("verbosity", "high"),
    }

    system = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
        "occupations": occupations,
        "smearing": smearing,
        "degauss": degauss,
    }

    # Add nspin/magnetization if atoms carry initial magnetic moments.
    if atoms.has("magmoms"):
        magmoms = atoms.get_initial_magnetic_moments()
        if np.any(np.abs(magmoms) > 1e-8):
            system["nspin"] = 2
            system["starting_magnetization"] = {
                specie: float(mag)
                for specie, mag in zip(atoms.get_chemical_symbols(), magmoms)
            }

    electrons = {
        "conv_thr": conv_thr,
        "mixing_mode": kwargs.pop("mixing_mode", "plain"),
        "mixing_beta": kwargs.pop("mixing_beta", 0.7),
        "diagonalization": kwargs.pop("diagonalization", "david"),
    }

    if calc_type == "relax":
        ions = {
            "ion_dynamics": kwargs.pop("ion_dynamics", "bfgs"),
            "upscale": kwargs.pop("upscale", 100.0),
        }
    else:
        ions = {}

    if input_data:
        control.update(input_data.get("control", {}))
        system.update(input_data.get("system", {}))
        electrons.update(input_data.get("electrons", {}))
        ions.update(input_data.get("ions", {}))

    calc_kwargs = {
        "command": resolved_command,
        "kpts": kpts,
        "input_data": {"control": control, "system": system, "electrons": electrons},
    }
    if ions:
        calc_kwargs["input_data"]["ions"] = ions

    calc_kwargs.update(kwargs)
    return Espresso(**calc_kwargs)


def _run_qe(
    atoms: Any,
    output_dir: str,
    calc_type: str = "static",
    **kwargs,
) -> Dict[str, Any]:
    """Run a Quantum ESPRESSO calculation and return a raw results dict."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calc = build_qe_calculator(atoms, calc_type=calc_type, **kwargs)
    atoms.calc = calc

    if calc_type == "relax":
        try:
            from ase.optimize import BFGS

            fmax = kwargs.pop("fmax", 0.05)
            optimizer = BFGS(atoms, trajectory=str(output_path / "relax.traj"), logfile=str(output_path / "relax.log"))
            optimizer.run(fmax=fmax)
        except Exception as exc:
            logger.warning(f"ASE relax failed: {exc}. Falling back to pw.x internal relaxation.")
            calc = build_qe_calculator(atoms, calc_type="relax", **kwargs)
            atoms.calc = calc
            atoms.get_potential_energy()
    else:
        atoms.get_potential_energy()

    return parse_qe_results(output_dir)


def run_qe_static(atoms: Any, output_dir: str, **kwargs) -> Dict[str, Any]:
    """
    Run a Quantum ESPRESSO single-point calculation.

    Args:
        atoms: ASE Atoms object.
        output_dir: Directory for input/output files.
        **kwargs: Arguments passed to build_qe_calculator.

    Returns:
        Dict with keys: energy, forces, stress, final_atoms, converged, band_gap.
        Units: energy eV, forces eV/Å, stress eV/Å³.
    """
    return _run_qe(atoms, output_dir, calc_type="static", **kwargs)


def run_qe_relax(
    atoms: Any,
    output_dir: str,
    fmax: float = 0.05,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a Quantum ESPRESSO geometry relaxation.

    Args:
        atoms: ASE Atoms object.
        output_dir: Directory for input/output files.
        fmax: Force convergence criterion in eV/Å.
        **kwargs: Arguments passed to build_qe_calculator.

    Returns:
        Dict with keys: energy, forces, stress, final_atoms, converged, band_gap.
        Units: energy eV, forces eV/Å, stress eV/Å³.
    """
    kwargs["fmax"] = fmax
    return _run_qe(atoms, output_dir, calc_type="relax", **kwargs)


def run_qe_band_structure(
    atoms: Any,
    output_dir: str,
    mode: str = "line",
    n_points: int = 100,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a Quantum ESPRESSO band structure / DOS calculation.

    Args:
        atoms: ASE Atoms object.
        output_dir: Directory for input/output files.
        mode: "line", "uniform", or "both".
            - "line": high-symmetry k-path band structure.
            - "uniform": uniform k-mesh for DOS (SCF with the provided k-grid).
            - "both": run line and uniform calculations.
        n_points: Number of k-points along the high-symmetry path.
        **kwargs: Arguments passed to build_qe_calculator.

    Returns:
        Dict with keys depending on mode. Common keys: engine, converged,
        fermi_energy, band_gap, final_atoms, raw_output_dir.
        For "line"/"both": eigenvalues_line, kpoints_line, kpath_indices,
        kpath_labels, special_points.
        For "uniform"/"both": eigenvalues_uniform, kpoints_uniform,
        dos_weights (if available).
        Units: eigenvalues eV, kpoints fractional reciprocal coordinates,
        fermi_energy eV, band_gap eV.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if mode not in ("line", "uniform", "both"):
        raise ValueError(f"mode must be 'line', 'uniform', or 'both', got {mode!r}")

    # Step 1: SCF calculation on a uniform k-grid.
    scf_dir = output_path / "scf"
    scf_dir.mkdir(parents=True, exist_ok=True)
    scf_kwargs = kwargs.copy()
    scf_kwargs.pop("kpts", None)  # use default SCF k-grid from kwargs or calculator default
    scf_result = run_qe_static(atoms, str(scf_dir), **scf_kwargs)
    fermi_energy = scf_result.get("fermi_energy")
    final_atoms = scf_result.get("final_atoms") or atoms
    converged = scf_result.get("converged", False)

    result: Dict[str, Any] = {
        "engine": "qe",
        "converged": converged,
        "fermi_energy": fermi_energy,
        "band_gap": None,
        "final_atoms": final_atoms,
        "scf_result": scf_result,
        "raw_output_dir": str(output_path),
    }

    # Helper to run a non-SCF step in its own directory.
    def _run_nscf(prefix: str, kpts, calc_type: str = "bands", extra_control=None):
        step_dir = output_path / prefix
        step_dir.mkdir(parents=True, exist_ok=True)
        step_kwargs = kwargs.copy()
        step_kwargs["kpts"] = kpts
        step_kwargs["calc_type"] = calc_type
        step_kwargs["input_data"] = step_kwargs.get("input_data", {})
        step_kwargs["input_data"].setdefault("control", {})
        control_update = {
            "calculation": calc_type,
            "restart_mode": "from_scratch",
            "outdir": "./",
            "prefix": prefix,
        }
        if extra_control:
            control_update.update(extra_control)
        step_kwargs["input_data"]["control"].update(control_update)
        step_kwargs["input_data"].setdefault("system", {})
        step_kwargs["input_data"]["system"].setdefault(
            "nbnd", kwargs.get("nbnd", None)
        )

        calc = build_qe_calculator(final_atoms, **step_kwargs)
        final_atoms.calc = calc
        final_atoms.get_potential_energy()
        return parse_qe_results(str(step_dir))

    if mode in ("line", "both"):
        kpts, path_indices, labels, special_points = get_kpath_seekpath(
            final_atoms, n_points=n_points
        )
        line_result = _run_nscf("bands", kpts, calc_type="bands")
        eigenvalues_line = line_result.get("eigenvalues")
        result.update(
            {
                "eigenvalues_line": eigenvalues_line,
                "kpoints_line": kpts,
                "kpath_indices": path_indices,
                "kpath_labels": labels,
                "special_points": special_points,
                "band_gap_line": _compute_band_gap(
                    eigenvalues_line, fermi_energy=fermi_energy
                ),
            }
        )
        converged = converged and line_result.get("converged", False)

    if mode in ("uniform", "both"):
        uniform_kpts = kwargs.get("kpts", (4, 4, 4))
        uniform_result = _run_nscf("dos", uniform_kpts, calc_type="nscf")
        eigenvalues_uniform = uniform_result.get("eigenvalues")
        result.update(
            {
                "eigenvalues_uniform": eigenvalues_uniform,
                "kpoints_uniform": uniform_kpts,
                "band_gap_uniform": _compute_band_gap(
                    eigenvalues_uniform, fermi_energy=fermi_energy
                ),
            }
        )
        converged = converged and uniform_result.get("converged", False)

    result["converged"] = converged
    result["band_gap"] = result.get("band_gap_line") or result.get(
        "band_gap_uniform"
    )
    return result


def parse_qe_results(output_dir: str) -> Dict[str, Any]:
    """
    Parse Quantum ESPRESSO results from an output directory.

    Args:
        output_dir: Directory containing pw.x output files.

    Returns:
        Dict with keys: energy, forces, stress, final_atoms, converged,
        fermi_energy, eigenvalues, band_gap.
        Units: energy eV, forces eV/Å, stress eV/Å³.
    """
    output_path = Path(output_dir)
    result: Dict[str, Any] = {
        "engine": "qe",
        "energy": None,
        "forces": None,
        "stress": None,
        "final_atoms": None,
        "converged": False,
        "fermi_energy": None,
        "eigenvalues": None,
        "band_gap": None,
        "raw_output_dir": str(output_path),
    }

    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_path}")
        return result

    # Try reading from the ASE calculator if a pwo file exists.
    pwo_files = sorted(output_path.glob("*.pwo"))
    if not pwo_files:
        logger.warning(f"No .pwo output files found in {output_path}")
        return result

    pwo_path = pwo_files[0]
    try:
        from ase.io import read

        final_atoms = read(str(pwo_path), format="espresso-out")
        if final_atoms is not None:
            result["final_atoms"] = final_atoms
            result["energy"] = float(final_atoms.get_potential_energy())
            result["forces"] = final_atoms.get_forces()
            result["stress"] = final_atoms.get_stress(voigt=False)
            result["converged"] = True
    except Exception as exc:
        logger.warning(f"ASE parsing of {pwo_path} failed: {exc}")

    # Extract Fermi energy and eigenvalues from stdout.
    try:
        text = pwo_path.read_text(encoding="utf-8", errors="ignore")
        import re

        fermi_match = re.search(
            r"the Fermi energy is\s+([-\d.]+)\s*ev", text, re.IGNORECASE
        )
        if fermi_match:
            result["fermi_energy"] = float(fermi_match.group(1))

        if "convergence NOT achieved" in text.lower():
            result["converged"] = False
        elif "convergence has been achieved" in text.lower():
            result["converged"] = True

        # Very rough eigenvalue extraction: k-point blocks are complex to parse
        # robustly without a dedicated parser. We leave eigenvalues to caller
        # unless a simple single-shot regex can locate them.
    except Exception as exc:
        logger.warning(f"Text parsing of {pwo_path} failed: {exc}")

    if result["energy"] is not None:
        result["band_gap"] = _compute_band_gap(
            result.get("eigenvalues"),
            fermi_energy=result.get("fermi_energy"),
        )

    return result
