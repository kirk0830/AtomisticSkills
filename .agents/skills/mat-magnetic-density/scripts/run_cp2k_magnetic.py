"""
Run spin-polarized CP2K calculations and extract magnetic moments.

This script replaces the atomate2 + VASP path for mat-magnetic-density with a
local ASE + CP2K runner.

Usage:
    python run_cp2k_magnetic.py Fe.cif \
        --output-dir ./Fe_magnetic \
        --spin-polarized \
        --initial-magmoms 2.2 \
        --cutoff 400 \
        --kpts 4 4 4

Requirements:
    - Pixi environment: cp2k (ase)
    - CP2K data directory (BASIS_SET, POTENTIAL) configured via CP2K_DATA_DIR
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.utils.dft import run_cp2k_relax, run_cp2k_static
from src.utils.dft.dft_common import DFTResult, write_dft_results

logger = logging.getLogger(__name__)


def _load_atoms(structure_path: str):
    """Load an ASE Atoms object from a structure file."""
    from ase.io import read

    atoms = read(structure_path)
    if atoms is None:
        raise ValueError(f"Could not read structure from {structure_path}")
    return atoms


def _parse_magmoms(spec: str, n_atoms: int) -> List[float]:
    """Parse a compact magnetic-moment specification into a list."""
    magmoms: List[float] = []
    for token in spec.split(","):
        token = token.strip()
        if "*" in token:
            count_str, value_str = token.split("*", 1)
            count = int(count_str.strip())
            value = float(value_str.strip())
            magmoms.extend([value] * count)
        else:
            magmoms.append(float(token))
    if len(magmoms) == 1:
        magmoms = magmoms * n_atoms
    if len(magmoms) != n_atoms:
        raise ValueError(
            f"Number of magnetic moments ({len(magmoms)}) does not match "
            f"number of atoms ({n_atoms})"
        )
    return magmoms


def main():
    parser = argparse.ArgumentParser(
        description="Run spin-polarized CP2K calculations and extract magnetic moments."
    )
    parser.add_argument(
        "structure",
        help="Input structure file (CIF, POSCAR, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        default="./cp2k_magnetic",
        help="Output directory (default: ./cp2k_magnetic)",
    )
    parser.add_argument(
        "--calc-type",
        choices=["static", "relax"],
        default="static",
        help="Calculation type (default: static)",
    )
    parser.add_argument(
        "--spin-polarized",
        action="store_true",
        help="Run a spin-polarized calculation",
    )
    parser.add_argument(
        "--initial-magmoms",
        type=str,
        default=None,
        help="Initial magnetic moments, e.g. '2.2' or '5*2.2,0.0'",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=400.0,
        help="Plane-wave cutoff in Ry (default: 400)",
    )
    parser.add_argument(
        "--rel-cutoff",
        type=float,
        default=50.0,
        help="Relative cutoff in Ry (default: 50)",
    )
    parser.add_argument(
        "--kpts",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("KX", "KY", "KZ"),
        help="Monkhorst-Pack k-grid (default: 4 4 4)",
    )
    parser.add_argument(
        "--xc",
        default="PBE",
        help="Exchange-correlation functional (default: PBE)",
    )
    parser.add_argument(
        "--basis-set-file",
        default=None,
        help="CP2K basis set file (default: inferred from $CP2K_DATA_DIR)",
    )
    parser.add_argument(
        "--potential-file",
        default=None,
        help="CP2K potential file (default: inferred from $CP2K_DATA_DIR)",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="CP2K command (default: $ASE_CP2K_COMMAND or $CP2K_COMMAND)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion for relaxations in eV/Å (default: 0.05)",
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

    atoms = _load_atoms(args.structure)
    print(f"Loaded structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

    spin_polarized = args.spin_polarized
    initial_magmoms = None
    if args.initial_magmoms is not None:
        initial_magmoms = _parse_magmoms(args.initial_magmoms, len(atoms))
        atoms.set_initial_magnetic_moments(initial_magmoms)
        spin_polarized = True

    kwargs: Dict[str, Any] = {
        "command": args.command,
        "cutoff": args.cutoff,
        "rel_cutoff": args.rel_cutoff,
        "kpts": tuple(args.kpts),
        "xc": args.xc,
        "basis_set_file": args.basis_set_file,
        "potential_file": args.potential_file,
        "spin_polarized": spin_polarized,
        "initial_magmoms": initial_magmoms,
    }

    print(f"\nRunning CP2K magnetic calculation (type={args.calc_type}) ...")
    if args.calc_type == "relax":
        result = run_cp2k_relax(atoms, args.output_dir, fmax=args.fmax, **kwargs)
    else:
        result = run_cp2k_static(atoms, args.output_dir, **kwargs)

    magnetic_moments = result.get("magnetic_moments")

    print("\n" + "=" * 60)
    print("CP2K MAGNETIC CALCULATION SUMMARY")
    print("=" * 60)
    print(f"  Converged: {result['converged']}")
    print(f"  Energy: {result.get('energy')} eV")
    if magnetic_moments is not None:
        total = sum(magnetic_moments)
        print(f"  Total magnetization: {total:.3f} μB")
        print(f"  Site moments: {magnetic_moments}")
    print("=" * 60)

    dft_result = DFTResult(
        engine="cp2k",
        energy=result.get("energy"),
        forces=_to_array(result.get("forces")),
        stress=_to_array(result.get("stress")),
        final_atoms=result.get("final_atoms"),
        converged=bool(result.get("converged", False)),
        magnetic_moments=magnetic_moments,
        raw_output_dir=result.get("raw_output_dir"),
        metadata={"source": "cp2k", "spin_polarized": spin_polarized},
    )
    results_json = Path(args.output_dir) / "results.json"
    write_dft_results(dft_result, str(results_json))
    print(f"\n✓ Results saved to {results_json}")

    # Save input configs for reproducibility.
    from src.utils.config_utils import save_skill_inputs

    save_skill_inputs(args, str(results_json))


def _to_array(value):
    if value is None:
        return None
    return np.asarray(value)


if __name__ == "__main__":
    main()
