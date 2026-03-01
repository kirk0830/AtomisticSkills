"""
Run Widom insertion with UMA (FairChem) for Henry coefficient and heat of adsorption.

Usage:
    python run_widom_uma.py --structure path/to/relaxed.cif --name MYCOF --weights path/to/uma.pt \\
        --gas CO2 --temperature 298 --save-dir ./out

Requirements:
    - Conda environment: fairchem-agent
    - Required packages: ase, torch, fairchem, pydantic, pymatgen, tqdm
"""

from pathlib import Path
import argparse
import os
import sys

_project_root = Path(__file__).resolve().parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

from widom_common import (
    add_common_widom_args,
    read_structure,
    run_widom_job,
    select_device,
)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Widom insertion with a UMA (FairChem) calculator.")
    add_common_widom_args(p)
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
        help="'auto' picks CUDA if available",
    )
    p.add_argument(
        "--inference-settings",
        type=str,
        default="default",
        choices=["default", "turbo"],
        help="'turbo' can be faster for fixed-size batches",
    )
    p.add_argument("--hf-cache", type=Path, default=None, help="Optional HuggingFace cache dir")
    p.add_argument("--model-tag", type=str, default=None, help="Model tag for metadata")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path (default: output_dir/widom_results.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.hf_cache is not None:
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.hf_cache)

    device = select_device(args.device)
    atoms = read_structure(args.structure)
    normalize_charge_spin(atoms)

    model_tag = args.model_tag or f"uma_{args.task_name}"
    calc = build_uma_calculator(
        args.weights,
        args.task_name,
        device,
        args.inference_settings,
    )

    output_path = args.output
    if output_path is None:
        output_path = args.output_dir / "widom_results.json"

    run_widom_job(
        calculator=calc,
        structure=atoms,
        gas=args.gas,
        temperature=args.temperature,
        model_outputs_interaction_energy=False,
        num_insertions=args.num_insertions,
        optimize_structures=args.optimize_structures,
        cutoff_distance=args.cutoff_distance,
        cutoff_to_com=args.cutoff_to_com,
        min_interplanar_distance=args.min_interplanar_distance,
        random_seed=getattr(args, "random_seed", 42),
        min_interaction_energy=args.min_interaction_energy,
        output_path=output_path,
        name=args.name,
        model_tag=model_tag,
        extra_config={
            "weights": str(args.weights),
            "device": device,
            "task_name": args.task_name,
            "inference_settings": args.inference_settings,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
