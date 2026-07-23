"""
Run DFT calculations for defect structures using ASE + QE or ASE + CP2K.

This script replaces the atomate2 VASP path for mat-defect-energy-dft with a
local ASE runner that supports both Quantum ESPRESSO and CP2K.

Usage:
    python run_defect_calculations.py \
        --engine qe \
        --bulk dft_defects/pristine_supercell.cif \
        --defect-dir dft_defects \
        --defect-index dft_defects/defect_index.json \
        --output-dir dft_defect_calcs \
        --calc-type relax \
        --pseudo-dir $ESPRESSO_PSEUDO

Requirements:
    - Pixi environment: qe or cp2k (ase)
    - Pseudopotentials / basis sets configured for the chosen engine
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.utils.dft import run_cp2k_relax, run_cp2k_static, run_qe_relax, run_qe_static
from src.utils.dft.dft_common import write_dft_results, DFTResult

logger = logging.getLogger(__name__)


def _load_atoms(structure_path: str):
    """Load an ASE Atoms object from a structure file."""
    from ase.io import read

    atoms = read(structure_path)
    if atoms is None:
        raise ValueError(f"Could not read structure from {structure_path}")
    return atoms


def _add_charge_to_kwargs(engine: str, charge: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of kwargs with total charge set for the chosen engine."""
    kwargs = kwargs.copy()
    if charge == 0:
        return kwargs

    kwargs.setdefault("input_data", {})
    if engine == "qe":
        kwargs["input_data"].setdefault("system", {})
        kwargs["input_data"]["system"]["tot_charge"] = float(charge)
    elif engine == "cp2k":
        # Best-effort charge setting via ASE CP2K input_data.
        # ASE CP2K may also accept a top-level ``charge`` keyword in newer versions.
        kwargs["input_data"].setdefault("FORCE_EVAL", {})
        kwargs["input_data"]["FORCE_EVAL"].setdefault("SUBSYS", {})
        kwargs["input_data"]["FORCE_EVAL"]["SUBSYS"]["CHARGE"] = charge
    else:
        raise ValueError(f"Unknown engine: {engine}")
    return kwargs


def run_bulk(engine: str, bulk_path: str, output_dir: str, calc_type: str, kwargs: Dict[str, Any]):
    """Run a bulk reference calculation."""
    atoms = _load_atoms(bulk_path)
    output_path = Path(output_dir) / "pristine_supercell"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning bulk reference ({engine}, {calc_type}) ...")
    if engine == "qe":
        result = run_qe_relax(atoms, str(output_path), **kwargs) if calc_type == "relax" else run_qe_static(atoms, str(output_path), **kwargs)
    else:
        result = run_cp2k_relax(atoms, str(output_path), **kwargs) if calc_type == "relax" else run_cp2k_static(atoms, str(output_path), **kwargs)

    _write_result(engine, result, str(output_path / "results.json"))
    print(f"  Energy: {result.get('energy')} eV, Converged: {result.get('converged')}")
    return result


def run_defect(engine: str, defect_info: Dict[str, Any], output_dir: str, calc_type: str, kwargs: Dict[str, Any]):
    """Run a single defect + charge calculation."""
    name = defect_info["name"]
    structure_path = defect_info["file"]
    charge = defect_info["charge"]

    atoms = _load_atoms(structure_path)
    output_path = Path(output_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)

    charged_kwargs = _add_charge_to_kwargs(engine, charge, kwargs)

    print(f"\nRunning defect {name} (q={charge:+d}) ...")
    if engine == "qe":
        result = run_qe_relax(atoms, str(output_path), **charged_kwargs) if calc_type == "relax" else run_qe_static(atoms, str(output_path), **charged_kwargs)
    else:
        result = run_cp2k_relax(atoms, str(output_path), **charged_kwargs) if calc_type == "relax" else run_cp2k_static(atoms, str(output_path), **charged_kwargs)

    _write_result(engine, result, str(output_path / "results.json"))
    print(f"  Energy: {result.get('energy')} eV, Converged: {result.get('converged')}")
    return result


def _write_result(engine: str, result: Dict[str, Any], output_path: str):
    """Write a result dict as a DFTResult JSON file."""
    dft_result = DFTResult(
        engine=engine,
        energy=result.get("energy"),
        forces=_to_array(result.get("forces")),
        stress=_to_array(result.get("stress")),
        final_atoms=result.get("final_atoms"),
        converged=bool(result.get("converged", False)),
        band_gap=result.get("band_gap"),
        magnetic_moments=_to_list(result.get("magnetic_moments")),
        raw_output_dir=result.get("raw_output_dir"),
        metadata={"source": engine},
    )
    write_dft_results(dft_result, output_path)


def _to_array(value):
    if value is None:
        return None
    return np.asarray(value)


def _to_list(value):
    if value is None:
        return None
    return list(value)


def main():
    parser = argparse.ArgumentParser(
        description="Run DFT calculations for defect structures with QE or CP2K."
    )
    parser.add_argument(
        "--engine",
        choices=["qe", "cp2k"],
        required=True,
        help="DFT engine to use",
    )
    parser.add_argument(
        "--bulk",
        required=True,
        help="Path to pristine supercell structure",
    )
    parser.add_argument(
        "--defect-dir",
        required=True,
        help="Directory containing defect CIF files (used for output organization)",
    )
    parser.add_argument(
        "--defect-index",
        required=True,
        help="Path to defect_index.json from generate_defect_structures.py",
    )
    parser.add_argument(
        "--output-dir",
        default="./dft_defect_calcs",
        help="Output directory (default: ./dft_defect_calcs)",
    )
    parser.add_argument(
        "--calc-type",
        choices=["static", "relax"],
        default="relax",
        help="Calculation type (default: relax)",
    )
    # Common DFT parameters
    parser.add_argument(
        "--ecutwfc",
        type=float,
        default=50.0,
        help="QE: plane-wave cutoff in Ry (default: 50)",
    )
    parser.add_argument(
        "--ecutrho",
        type=float,
        default=None,
        help="QE: charge-density cutoff in Ry (default: 4*ecutwfc)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=400.0,
        help="CP2K: plane-wave cutoff in Ry (default: 400)",
    )
    parser.add_argument(
        "--rel-cutoff",
        type=float,
        default=50.0,
        help="CP2K: relative cutoff in Ry (default: 50)",
    )
    parser.add_argument(
        "--kpts",
        type=int,
        nargs=3,
        default=[2, 2, 2],
        metavar=("KX", "KY", "KZ"),
        help="Monkhorst-Pack k-grid (default: 2 2 2)",
    )
    parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="QE: pseudopotential directory (default: $ESPRESSO_PSEUDO)",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="Engine command (default: engine-specific env var)",
    )
    parser.add_argument(
        "--smearing",
        default="mv",
        help="QE: smearing type (default: mv)",
    )
    parser.add_argument(
        "--degauss",
        type=float,
        default=0.01,
        help="QE: smearing width in Ry (default: 0.01)",
    )
    parser.add_argument(
        "--conv-thr",
        type=float,
        default=1e-8,
        help="QE: SCF convergence threshold in Ry (default: 1e-8)",
    )
    parser.add_argument(
        "--xc",
        default="PBE",
        help="CP2K: exchange-correlation functional (default: PBE)",
    )
    parser.add_argument(
        "--basis-set-file",
        default=None,
        help="CP2K: basis set file path (default: inferred from $CP2K_DATA_DIR)",
    )
    parser.add_argument(
        "--potential-file",
        default=None,
        help="CP2K: potential file path (default: inferred from $CP2K_DATA_DIR)",
    )
    parser.add_argument(
        "--spin-polarized",
        action="store_true",
        help="Run spin-polarized calculation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    with open(args.defect_index, "r", encoding="utf-8") as f:
        defect_index = json.load(f)

    kpts = tuple(args.kpts)

    if args.engine == "qe":
        kwargs: Dict[str, Any] = {
            "pseudo_dir": args.pseudo_dir,
            "command": args.command,
            "ecutwfc": args.ecutwfc,
            "ecutrho": args.ecutrho,
            "kpts": kpts,
            "smearing": args.smearing,
            "degauss": args.degauss,
            "conv_thr": args.conv_thr,
        }
    else:
        kwargs = {
            "command": args.command,
            "cutoff": args.cutoff,
            "rel_cutoff": args.rel_cutoff,
            "kpts": kpts,
            "xc": args.xc,
            "basis_set_file": args.basis_set_file,
            "potential_file": args.potential_file,
            "spin_polarized": args.spin_polarized,
        }

    # Bulk reference
    run_bulk(args.engine, args.bulk, args.output_dir, args.calc_type, kwargs)

    # Defect calculations
    for defect_info in defect_index["defects"]:
        try:
            run_defect(args.engine, defect_info, args.output_dir, args.calc_type, kwargs)
        except Exception as exc:
            logger.error(f"Failed to run defect {defect_info['name']}: {exc}")
            continue

    print(f"\n✓ Defect calculations finished. Results in {args.output_dir}")

    # Save input configs for reproducibility
    from src.utils.config_utils import save_skill_inputs

    save_skill_inputs(args, args.output_dir)


if __name__ == "__main__":
    main()
