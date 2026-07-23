"""
Run Quantum ESPRESSO band structure / DOS calculations via ASE.

This script replaces the atomate2 + VASP path for mat-electronic-structure
with an ASE + Quantum ESPRESSO local runner.

Usage:
    python run_qe_band_structure.py structure.cif \
        --mode both \
        --output-dir ./qe_bands \
        --ecutwfc 50 \
        --kpts 6 6 6 \
        --pseudo-dir $ESPRESSO_PSEUDO

Requirements:
    - Pixi environment: qe (ase, seekpath, numpy)
    - Quantum ESPRESSO pseudopotentials in ESPRESSO_PSEUDO
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Allow running from repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.utils.dft import run_qe_band_structure
from src.utils.dft.dft_common import _numpy_to_json

logger = logging.getLogger(__name__)


def _load_atoms(structure_path: str):
    """Load an ASE Atoms object from a structure file."""
    from ase.io import read

    atoms = read(structure_path)
    if atoms is None:
        raise ValueError(f"Could not read structure from {structure_path}")
    return atoms


def _save_results(result: Dict[str, Any], output_path: str) -> None:
    """Serialize band structure results to JSON with numpy conversion."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _numpy_to_json(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Quantum ESPRESSO band structure / DOS calculations."
    )
    parser.add_argument(
        "structure",
        help="Input structure file (CIF, POSCAR, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        default="./qe_band_structure",
        help="Output directory (default: ./qe_band_structure)",
    )
    parser.add_argument(
        "--mode",
        choices=["line", "uniform", "both"],
        default="line",
        help="Calculation mode: line (band structure), uniform (DOS), or both",
    )
    parser.add_argument(
        "--ecutwfc",
        type=float,
        default=50.0,
        help="Plane-wave cutoff in Ry (default: 50)",
    )
    parser.add_argument(
        "--ecutrho",
        type=float,
        default=None,
        help="Charge-density cutoff in Ry (default: 4 * ecutwfc)",
    )
    parser.add_argument(
        "--kpts",
        type=int,
        nargs=3,
        default=[4, 4, 4],
        metavar=("KX", "KY", "KZ"),
        help="SCF Monkhorst-Pack k-grid (default: 4 4 4)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=100,
        help="Number of k-points along the high-symmetry path (default: 100)",
    )
    parser.add_argument(
        "--pseudo-dir",
        default=None,
        help="Quantum ESPRESSO pseudopotential directory (default: $ESPRESSO_PSEUDO)",
    )
    parser.add_argument(
        "--command",
        default=None,
        help="pw.x command (default: $ASE_ESPRESSO_COMMAND or $ESPRESSO_COMMAND or pw.x)",
    )
    parser.add_argument(
        "--smearing",
        default="mv",
        help="Smearing type (default: mv)",
    )
    parser.add_argument(
        "--degauss",
        type=float,
        default=0.01,
        help="Smearing width in Ry (default: 0.01)",
    )
    parser.add_argument(
        "--conv-thr",
        type=float,
        default=1e-8,
        help="SCF convergence threshold in Ry (default: 1e-8)",
    )
    parser.add_argument(
        "--nbnd",
        type=int,
        default=None,
        help="Number of bands for non-SCF calculations (default: QE automatic)",
    )
    parser.add_argument(
        "--magmoms",
        type=str,
        default=None,
        help="Initial magnetic moments, e.g. '2.2' or '5*2.2,0.0'",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print ASE / QE debug output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    atoms = _load_atoms(args.structure)
    print(f"Loaded structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

    if args.magmoms is not None:
        magmoms = _parse_magmoms(args.magmoms, len(atoms))
        atoms.set_initial_magnetic_moments(magmoms)

    kwargs: Dict[str, Any] = {
        "pseudo_dir": args.pseudo_dir,
        "command": args.command,
        "ecutwfc": args.ecutwfc,
        "ecutrho": args.ecutrho,
        "kpts": tuple(args.kpts),
        "smearing": args.smearing,
        "degauss": args.degauss,
        "conv_thr": args.conv_thr,
        "nbnd": args.nbnd,
    }

    print(f"\nRunning QE band structure (mode={args.mode}) ...")
    result = run_qe_band_structure(
        atoms,
        output_dir=args.output_dir,
        mode=args.mode,
        n_points=args.n_points,
        **kwargs,
    )

    # Print a concise summary.
    print("\n" + "=" * 60)
    print("QE BAND STRUCTURE SUMMARY")
    print("=" * 60)
    print(f"  Converged: {result['converged']}")
    if result.get("fermi_energy") is not None:
        print(f"  Fermi energy: {result['fermi_energy']:.4f} eV")
    if result.get("band_gap") is not None:
        print(f"  Band gap: {result['band_gap']:.4f} eV")
    if result.get("eigenvalues_line") is not None:
        shape = np.asarray(result["eigenvalues_line"]).shape
        print(f"  Line eigenvalues shape: {shape}")
    if result.get("eigenvalues_uniform") is not None:
        shape = np.asarray(result["eigenvalues_uniform"]).shape
        print(f"  Uniform eigenvalues shape: {shape}")
    print("=" * 60)

    results_json = Path(args.output_dir) / "results.json"
    _save_results(result, str(results_json))

    # Save input configs for reproducibility.
    from src.utils.config_utils import save_skill_inputs

    save_skill_inputs(args, str(results_json))


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


if __name__ == "__main__":
    main()
