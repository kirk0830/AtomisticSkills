"""
Generate coincidence site lattice (CSL) grain boundary structures using pymatgen.

For a given bulk structure and rotation axis, enumerates all unique tilt grain
boundaries up to a maximum Σ value and writes each to a CIF file.

Usage:
    python create_grain_boundary.py \
        --bulk bulk.cif \
        --rotation-axis 0 0 1 \
        --max-sigma 29 \
        --min-slab-size 10.0 \
        --output-dir gb_structures/

Requirements:
    - Conda environment: base-agent
    - Required packages: pymatgen
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.research_utils import get_current_research_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("GBGenerator")


def generate_gb_structures(
    bulk_path: str,
    rotation_axis: list,
    max_sigma: int,
    min_slab_size: float,
    vacuum: float,
    output_dir: str,
    symprec: float = 0.1,
) -> list:
    """Generate all unique CSL grain boundaries for a given axis and Σ range.

    Args:
        bulk_path      : path to relaxed bulk structure file
        rotation_axis  : Miller indices of rotation axis, e.g. [0, 0, 1]
        max_sigma      : maximum Σ (coincidence index) to enumerate
        min_slab_size  : minimum thickness of each grain in Å
        vacuum         : vacuum thickness in Å (should be 0 for GBs)
        output_dir     : directory to write generated CIF files
        symprec        : symmetry precision for pymatgen (Å)

    Returns:
        List of metadata dicts, one per generated GB.
    """
    from pymatgen.analysis.grain_boundary import GrainBoundaryGenerator
    from pymatgen.core import Structure

    bulk = Structure.from_file(bulk_path)
    formula = bulk.composition.reduced_formula
    logger.info(f"Loaded bulk structure: {formula} ({len(bulk)} atoms/unit cell)")

    gen = GrainBoundaryGenerator(bulk)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    axis_str = "".join(str(x) for x in rotation_axis)
    axis_label = f"[{rotation_axis[0]}{rotation_axis[1]}{rotation_axis[2]}]"

    metadata = []
    n_generated = 0

    for sigma in range(1, max_sigma + 1):
        # Get all unique rotation angles for this Σ value along the chosen axis
        try:
            angles = gen.get_rotation_angle_from_sigma(sigma, rotation_axis)
        except Exception:
            continue

        for angle in angles:
            try:
                gb = gen.gb_from_parameters(
                    rotation_axis=rotation_axis,
                    rotation_angle=angle,
                    expand_times=4,
                    vacuum_thickness=vacuum,
                    ab_shift=[0, 0],
                    normal=True,
                    ratio=None,
                    plane=None,
                )
                # Skip if slab is too thin
                c_length = gb.lattice.c
                if c_length < 2 * min_slab_size:
                    logger.debug(
                        f"  Σ{sigma} angle={angle:.2f}° skipped: c={c_length:.1f} Å < "
                        f"2×{min_slab_size} Å"
                    )
                    continue

                hkl = "".join(str(x) for x in gb.miller_index) if hasattr(gb, "miller_index") else axis_str
                filename = f"sigma{sigma:03d}_{angle:.2f}deg_{axis_str}.cif"
                filepath = output_path / filename
                gb.to(fmt="cif", filename=str(filepath))

                n_atoms = len(gb)
                area = gb.lattice.a * gb.lattice.b  # Å²
                entry = {
                    "sigma": sigma,
                    "rotation_angle_deg": round(angle, 4),
                    "rotation_axis": rotation_axis,
                    "axis_label": axis_label,
                    "n_atoms": n_atoms,
                    "interface_area_A2": round(area, 4),
                    "cell_c_A": round(c_length, 4),
                    "cif_file": filename,
                }
                metadata.append(entry)
                n_generated += 1
                logger.info(
                    f"  Σ{sigma:3d}  {angle:6.2f}°  {n_atoms:4d} atoms  "
                    f"A={area:.1f} Å²  → {filename}"
                )

            except Exception as exc:
                logger.warning(f"  Σ{sigma} angle={angle:.2f}° failed: {exc}")
                continue

    logger.info(f"\nGenerated {n_generated} grain boundary structures in {output_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSL grain boundary structures using pymatgen."
    )
    parser.add_argument(
        "--bulk",
        required=True,
        help="Path to relaxed bulk structure file (CIF, POSCAR, etc.).",
    )
    parser.add_argument(
        "--rotation-axis",
        nargs=3,
        type=int,
        default=[0, 0, 1],
        metavar=("H", "K", "L"),
        help="Rotation axis as Miller indices, e.g. 0 0 1 (default: [001]).",
    )
    parser.add_argument(
        "--max-sigma",
        type=int,
        default=29,
        help="Maximum Σ (coincidence index) to enumerate (default: 29).",
    )
    parser.add_argument(
        "--min-slab-size",
        type=float,
        default=10.0,
        help="Minimum thickness of each grain in Å (default: 10.0).",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=0.0,
        help="Vacuum thickness in Å. Use 0.0 for grain boundaries (default: 0.0).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <research_dir>/gb_structures.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="Symmetry precision for pymatgen in Å (default: 0.1).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        research_dir = get_current_research_dir()
        args.output_dir = str(research_dir / "gb_structures")

    metadata = generate_gb_structures(
        bulk_path=args.bulk,
        rotation_axis=args.rotation_axis,
        max_sigma=args.max_sigma,
        min_slab_size=args.min_slab_size,
        vacuum=args.vacuum,
        output_dir=args.output_dir,
        symprec=args.symprec,
    )

    # Save metadata
    meta_path = Path(args.output_dir) / "gb_structure_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "bulk_path": args.bulk,
                "rotation_axis": args.rotation_axis,
                "max_sigma": args.max_sigma,
                "min_slab_size_A": args.min_slab_size,
                "n_generated": len(metadata),
                "structures": metadata,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved structure metadata to {meta_path}")

    print(f"\nSummary: {len(metadata)} GB structures saved to {args.output_dir}")
    print(f"Metadata: {meta_path}")
    print("Next step: relax with MLIP using relax_cell=False, then run calculate_gb_energy.py")


if __name__ == "__main__":
    main()
