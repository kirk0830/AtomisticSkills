"""
Shared relaxation logic for UMA framework relaxation (port from COFclean relax/relax_common.py).
Outputs: CIF/XYZ structure files plus relax_results.json only.
"""

from pathlib import Path
import json
import os
from typing import Any, Dict
import logging

import numpy as np
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter


LOGGER = logging.getLogger(__name__)


def select_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def read_atoms(structure_path: Path | str):
    return read(str(structure_path))


def output_paths(base_dir: Path, cof_name: str, is_supercell: bool) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_supercell" if is_supercell else ""
    out_xyz = base_dir / f"{cof_name}{suffix}.relaxed.xyz"
    out_cif = base_dir / f"{cof_name}{suffix}.relaxed.cif"
    out_traj = base_dir / f"{cof_name}{suffix}.relaxed.traj"
    save_prefix = base_dir / cof_name
    return {
        "out_xyz": out_xyz,
        "out_cif": out_cif,
        "out_traj": out_traj,
        "save_prefix": save_prefix,
    }


def run_fire_relaxation(
    input_atoms,
    calc,
    out_xyz: Path,
    out_cif: Path,
    out_traj: Path | None,
    relax_cell: bool,
    steps: int,
    fmax: float,
):
    atoms = input_atoms.copy()
    atoms.calc = calc

    traj = None
    obj = FrechetCellFilter(atoms) if relax_cell else atoms
    try:
        if out_traj is not None:
            traj = Trajectory(str(out_traj), "w", atoms)
            opt = FIRE(obj, trajectory=traj)
        else:
            opt = FIRE(obj)
        opt.run(fmax=fmax, steps=steps)
    finally:
        if traj is not None:
            try:
                traj.close()
            except Exception:
                pass

    write(str(out_xyz), atoms)
    write(str(out_cif), atoms)

    try:
        e = atoms.get_potential_energy()
        LOGGER.info("Relaxation complete. Final energy: %.6f eV", e)
    except Exception:
        LOGGER.info("Relaxation complete.")

    LOGGER.info("Wrote: %s", out_xyz)
    LOGGER.info("Wrote: %s", out_cif)
    if out_traj is not None:
        LOGGER.info("Wrote: %s", out_traj)
    return atoms


def build_supercell(source_atoms, cutoff_distance: float = 6.0) -> tuple[bool, object]:
    structure = source_atoms.copy()

    cell_vectors = np.array(structure.cell)
    cell_volume = structure.get_volume()
    dist_a = cell_volume / np.linalg.norm(np.cross(cell_vectors[1], cell_vectors[2]))
    dist_b = cell_volume / np.linalg.norm(np.cross(cell_vectors[2], cell_vectors[0]))
    dist_c = cell_volume / np.linalg.norm(np.cross(cell_vectors[0], cell_vectors[1]))
    plane_distances = np.array([dist_a, dist_b, dist_c])

    supercell = np.ceil(cutoff_distance / plane_distances).astype(int)
    if np.any(supercell > 1):
        LOGGER.info(
            "Making supercell: %s to enforce min interplanar distance ≥ %.3f Å",
            supercell,
            cutoff_distance,
        )
        structure = structure.repeat(supercell)
        return True, structure
    return False, structure


def compare_structures(reference_atoms, optimized_atoms, save_prefix: Path) -> Dict[str, Any]:
    """Return comparison metrics (for inclusion in relax_results.json). No file output."""
    def pct(ref, val):
        return 100.0 * (val - ref) / ref if ref != 0 else float("nan")

    Lr = reference_atoms.get_cell()
    Lo = optimized_atoms.get_cell()
    a_r, b_r, c_r = Lr.lengths()
    ar, br, gr = Lr.angles()
    a_o, b_o, c_o = Lo.lengths()
    ao, bo, go = Lo.angles()
    Vr = reference_atoms.get_volume()
    Vo = optimized_atoms.get_volume()

    a_pct = pct(a_r, a_o)
    b_pct = pct(b_r, b_o)
    c_pct = pct(c_r, c_o)
    da = ao - ar
    db = bo - br
    dg = go - gr
    V_pct = pct(Vr, Vo)

    f_ref = reference_atoms.get_scaled_positions(wrap=True)
    f_opt = optimized_atoms.get_scaled_positions(wrap=True)
    if len(f_ref) != len(f_opt):
        raise ValueError(f"Atom count differs: ref={len(f_ref)} vs opt={len(f_opt)}")

    df = f_opt - f_ref
    df -= np.round(df)
    disp_cart = df @ Lr.array
    dists = np.linalg.norm(disp_cart, axis=1)
    rmsd = float(np.sqrt(np.mean(dists**2))) if len(dists) else 0.0
    maxd = float(np.max(dists)) if len(dists) else 0.0

    report = (
        f"a (Å)       : {a_r:10.4f} → {a_o:10.4f}   Δ% = {a_pct:+7.3f}\n"
        f"b (Å)       : {b_r:10.4f} → {b_o:10.4f}   Δ% = {b_pct:+7.3f}\n"
        f"c (Å)       : {c_r:10.4f} → {c_o:10.4f}   Δ% = {c_pct:+7.3f}\n"
        f"α (°)       : {ar:10.4f} → {ao:10.4f}   Δ = {da:+7.3f}\n"
        f"β (°)       : {br:10.4f} → {bo:10.4f}   Δ = {db:+7.3f}\n"
        f"γ (°)       : {gr:10.4f} → {go:10.4f}   Δ = {dg:+7.3f}\n"
        f"Volume (Å³) : {Vr:10.4f} → {Vo:10.4f}   Δ% = {V_pct:+7.3f}\n"
        f"ΔVolume% = {V_pct:+.3f}%\n\n"
        f"RMSD = {rmsd:.4f} Å\n"
        f"Max displacement = {maxd:.4f} Å"
    )
    LOGGER.info("\n%s", report)

    return {
        "cell_lengths_angstrom": {"a": (float(a_r), float(a_o)), "b": (float(b_r), float(b_o)), "c": (float(c_r), float(c_o))},
        "volume_angstrom3": (float(Vr), float(Vo)),
        "volume_change_pct": float(V_pct) if np.isfinite(V_pct) else None,
        "rmsd_angstrom": float(rmsd),
        "max_displacement_angstrom": float(maxd),
    }


def write_relax_results_json(
    base_dir: Path,
    cof_name: str,
    source_cif: Path,
    relaxed_cifs: list[Path],
    relaxed_xyzs: list[Path],
    comparison: Dict[str, Any] | None = None,
) -> Path:
    """Write relax_results.json with paths to all CIF/XYZ outputs (and optional comparison)."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "cof_name": cof_name,
        "source_cif": str(source_cif),
        "relaxed_cifs": [str(p) for p in relaxed_cifs],
        "relaxed_xyzs": [str(p) for p in relaxed_xyzs],
        "relaxed_cif": str(relaxed_cifs[-1]) if relaxed_cifs else None,
        "relaxed_xyz": str(relaxed_xyzs[-1]) if relaxed_xyzs else None,
    }
    if comparison is not None:
        out["comparison"] = comparison
    path = base_dir / "relax_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def prepare_relax_run_dir(
    output_dir: Path,
    cof_name: str,
    run_tag: str,
    *,
    run_name: str | None = None,
) -> tuple[str, Path]:
    """Prepare output directory for a relax run. Returns (run_name, out_dir)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_name or cof_name, out_dir
