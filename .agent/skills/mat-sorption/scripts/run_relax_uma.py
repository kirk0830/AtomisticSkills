"""
Relax a framework CIF/XYZ with UMA (FairChem); write relaxed xyz/cif and optional traj.

Usage:
    python run_relax_uma.py --structure path/to/framework.cif --name MYCOF --weights path/to/uma.pt --output-dir ./out

Requirements:
    - Conda environment: fairchem-agent
    - Required packages: ase, torch, fairchem
"""

from pathlib import Path
import argparse
import logging
import os
import sys

# Add project root so src.utils and same-dir helpers are importable
_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from relax_common import (
    select_device,
    read_atoms,
    output_paths,
    run_fire_relaxation,
    build_supercell,
    write_relax_results_json,
    prepare_relax_run_dir,
)


LOGGER = logging.getLogger(__name__)


def normalize_charge_spin(atoms) -> None:
    info = atoms.info if isinstance(atoms.info, dict) else {}
    if atoms.info is not info:
        atoms.info = info
    try:
        charge = int(info.get("charge", info.get("chg", 0)))
    except Exception:
        charge = 0
    try:
        spin_mult = int(
            info.get("spin_multiplicity", info.get("multiplicity", info.get("spin", 1)))
        )
    except Exception:
        spin_mult = 1
    atoms.info["charge"] = charge
    atoms.info["spin_multiplicity"] = spin_mult
    atoms.info["spin"] = spin_mult


def build_uma_calculator(
    model_weights: Path | str,
    task_name: str,
    device: str,
    inference_settings: str,
):
    predictor = load_predict_unit(
        path=str(model_weights),
        device=device,
        inference_settings=inference_settings,
    )
    return FAIRChemCalculator(predictor, task_name=task_name)


def _parse_supercell(s: str | None) -> tuple[int, int, int] | None:
    if not s:
        return None
    parts = [p.strip() for p in str(s).replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--supercell must be 'nx,ny,nz' (got: {s!r})")
    sc = tuple(int(p) for p in parts)
    if any(v < 1 for v in sc):
        raise ValueError(f"--supercell entries must be >= 1 (got: {sc})")
    return sc  # type: ignore[return-value]


def relax_with_uma(
    structure: Path,
    name: str,
    weights: Path,
    task_name: str,
    steps: int,
    fmax: float,
    inference_settings: str,
    device: str,
    relax_cell: bool,
    supercell: tuple[int, int, int] | None,
    min_plane_dist: float,
    output_dir: Path,
    run_name: str | None,
    hf_cache: Path | None,
) -> int:
    if hf_cache is not None:
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)

    device = select_device(device)
    reference_atoms = read_atoms(structure)

    run_tag = f"uma__{task_name}__{Path(weights).stem}"
    run_name, target_dir = prepare_relax_run_dir(
        output_dir,
        name,
        run_tag,
        run_name=run_name,
    )
    calc = build_uma_calculator(weights, task_name, device, inference_settings)

    atoms_for_relax = reference_atoms.copy()
    normalize_charge_spin(atoms_for_relax)
    if supercell is not None and any(v > 1 for v in supercell):
        LOGGER.info("Applying requested pre-relax supercell: %s", supercell)
        atoms_for_relax = atoms_for_relax.repeat(supercell)

    paths = output_paths(target_dir, name, is_supercell=False)
    optimized_atoms = run_fire_relaxation(
        atoms_for_relax,
        calc,
        paths["out_xyz"],
        paths["out_cif"],
        None,
        relax_cell,
        steps,
        fmax,
    )

    relaxed_cifs = [paths["out_cif"]]
    relaxed_xyzs = [paths["out_xyz"]]

    made_super, supercell_atoms = build_supercell(
        optimized_atoms, cutoff_distance=min_plane_dist
    )
    if made_super:
        calc = build_uma_calculator(weights, task_name, device, inference_settings)
        sc_paths = output_paths(target_dir, name, is_supercell=True)
        run_fire_relaxation(
            supercell_atoms,
            calc,
            sc_paths["out_xyz"],
            sc_paths["out_cif"],
            None,
            relax_cell,
            steps,
            fmax,
        )
        relaxed_cifs.append(sc_paths["out_cif"])
        relaxed_xyzs.append(sc_paths["out_xyz"])

    write_relax_results_json(
        target_dir,
        cof_name=name,
        source_cif=structure,
        relaxed_cifs=relaxed_cifs,
        relaxed_xyzs=relaxed_xyzs,
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Relax a CIF or XYZ with UMA via FairChem; write relaxed xyz/cif/traj"
    )
    p.add_argument("--structure", type=Path, required=True, help="Path to input structure (.cif or .xyz)")
    p.add_argument("--name", type=str, required=True, help="Name or id of the framework")
    p.add_argument("--weights", type=Path, required=True, help="Path to UMA checkpoint (.pt)")
    p.add_argument(
        "--task-name",
        type=str,
        default="omol",
        choices=["oc20", "omol", "omat", "odac", "omc"],
        help="FairChem task name",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="'auto' picks CUDA",
    )
    p.add_argument(
        "--inference-settings",
        type=str,
        default="default",
        choices=["default", "turbo"],
        help="'turbo' can be faster for fixed-size batches",
    )
    p.add_argument("--fmax", type=float, default=0.05, help="FIRE force threshold (eV/Å)")
    p.add_argument("--steps", type=int, default=1000, help="Maximum FIRE steps")
    p.add_argument("--max-steps", dest="steps", type=int, help="Alias for --steps")
    p.add_argument("--hf-cache", type=Path, default=None, help="Optional HuggingFace cache dir")
    p.add_argument(
        "--min-plane-dist",
        type=float,
        default=12.0,
        help="Minimum interplanar distance (Å) before building supercell",
    )
    p.add_argument("--supercell", type=str, default=None, help="Optional pre-relax supercell e.g. '2,2,2'")
    p.add_argument("--fixed-cell", action="store_true", help="Optimize atoms only (no cell relaxation)")
    p.add_argument("--relax-cell", action="store_true", help="Relax cell (default unless --fixed-cell)")
    p.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for outputs (CIF/XYZ + relax_results.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    relax_cell = True
    if args.fixed_cell:
        relax_cell = False
    if getattr(args, "relax_cell", False):
        relax_cell = True
    return relax_with_uma(
        structure=args.structure,
        name=args.name,
        weights=args.weights,
        task_name=args.task_name,
        steps=args.steps,
        fmax=args.fmax,
        inference_settings=args.inference_settings,
        device=args.device,
        relax_cell=relax_cell,
        supercell=_parse_supercell(args.supercell),
        min_plane_dist=args.min_plane_dist,
        output_dir=args.output_dir,
        run_name=None,
        hf_cache=args.hf_cache,
    )


if __name__ == "__main__":
    raise SystemExit(main())
