"""
Common DFT utilities and data structures for MLIP Agent.

This module provides engine-agnostic helpers for DFT workflows,
including k-path generation, k-grid estimation, and result serialization.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _import_ase_atoms():
    """Lazy import ASE Atoms to keep the module top-level importable."""
    try:
        from ase import Atoms

        return Atoms
    except ImportError as exc:
        raise ImportError(
            "ASE is required for DFT utilities but is not installed."
        ) from exc


@dataclass
class DFTResult:
    """
    Engine-agnostic container for a single DFT calculation result.

    Units:
        energy: eV
        forces: eV/Å
        stress: eV/Å³
        band_gap: eV
        eigenvalues: eV
        kpoints: fractional reciprocal coordinates
        magnetic_moments: Bohr magneton
    """

    engine: str = "unknown"
    energy: Optional[float] = None
    forces: Optional[np.ndarray] = None
    stress: Optional[np.ndarray] = None
    final_atoms: Optional[Any] = None
    converged: bool = False
    band_gap: Optional[float] = None
    eigenvalues: Optional[np.ndarray] = None
    kpoints: Optional[np.ndarray] = None
    magnetic_moments: Optional[List[float]] = None
    raw_output_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _atoms_to_dict(atoms: Any) -> Dict[str, Any]:
    """Convert an ASE Atoms object to a JSON-serializable dict."""
    if atoms is None:
        return {}
    return {
        "numbers": atoms.get_atomic_numbers().tolist(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist(),
        "pbc": atoms.get_pbc().tolist(),
        "tags": (atoms.get_tags().tolist() if atoms.has("tags") else None),
        "momenta": (atoms.get_momenta().tolist() if atoms.has("momenta") else None),
        "magmoms": (atoms.get_initial_magnetic_moments().tolist()
                    if atoms.has("magmoms") else None),
        "charges": (atoms.get_initial_charges().tolist()
                    if atoms.has("charges") else None),
        "info": atoms.info,
    }


def _atoms_from_dict(data: Dict[str, Any]) -> Any:
    """Reconstruct an ASE Atoms object from a dict."""
    Atoms = _import_ase_atoms()
    numbers = data.get("numbers", [])
    positions = data.get("positions")
    cell = data.get("cell")
    pbc = data.get("pbc", True)

    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    if data.get("tags") is not None:
        atoms.set_tags(data["tags"])
    if data.get("momenta") is not None:
        atoms.set_momenta(data["momenta"])
    if data.get("magmoms") is not None:
        atoms.set_initial_magnetic_moments(data["magmoms"])
    if data.get("charges") is not None:
        atoms.set_initial_charges(data["charges"])
    if data.get("info"):
        atoms.info.update(data["info"])
    return atoms


def _numpy_to_json(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _numpy_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_json(v) for v in obj]
    return obj


def write_dft_results(result: DFTResult, output_path: str) -> None:
    """
    Serialize a DFTResult to JSON.

    Args:
        result: DFTResult instance to save.
        output_path: Destination JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(result)
    data["final_atoms"] = _atoms_to_dict(result.final_atoms)
    data = _numpy_to_json(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"DFT results written to {output_path}")


def read_dft_results(json_path: str) -> DFTResult:
    """
    Deserialize a DFTResult from JSON.

    Args:
        json_path: Path to the JSON file.

    Returns:
        Reconstructed DFTResult instance.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"DFT results file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_atoms_data = data.pop("final_atoms", {})
    if final_atoms_data:
        data["final_atoms"] = _atoms_from_dict(final_atoms_data)

    # Convert list fields back to numpy arrays when applicable.
    for key in ("forces", "stress", "eigenvalues", "kpoints"):
        if data.get(key) is not None:
            data[key] = np.array(data[key])

    return DFTResult(**data)


def get_kpath_seekpath(
    atoms: Any, n_points: int = 100
) -> Tuple[np.ndarray, List[int], List[str], Dict[str, np.ndarray]]:
    """
    Generate a high-symmetry k-point path using seekpath or ASE fallback.

    Args:
        atoms: ASE Atoms object with a periodic cell.
        n_points: Total number of k-points along the path.

    Returns:
        Tuple of (kpoints, path_indices, labels, special_points).
        - kpoints: array of shape (n_points, 3) in reciprocal fractional coords.
        - path_indices: indices in kpoints where high-symmetry labels sit.
        - labels: high-symmetry point labels (e.g. ["GAMMA", "X", "GAMMA"]).
        - special_points: dict mapping label -> coordinate.
    """
    Atoms = _import_ase_atoms()
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ASE Atoms object")

    cell = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    try:
        import seekpath

        structure = (cell, positions, numbers)
        path = seekpath.get_path(structure)
        point_coords = path["point_coords"]
        path_segments = path["path"]

        labels = []
        kpoints = []
        path_indices = []
        segment_lengths = []

        # Compute segment lengths in reciprocal space.
        rec_cell = cell.reciprocal()
        for seg in path_segments:
            start = point_coords[seg[0]]
            end = point_coords[seg[1]]
            start_cart = rec_cell.T @ np.array(start)
            end_cart = rec_cell.T @ np.array(end)
            segment_lengths.append(np.linalg.norm(end_cart - start_cart))

        total_length = sum(segment_lengths)
        if total_length == 0:
            total_length = 1.0

        for i, seg in enumerate(path_segments):
            start_label, end_label = seg
            start = np.array(point_coords[start_label])
            end = np.array(point_coords[end_label])
            n_seg = max(2, int(round(n_points * segment_lengths[i] / total_length)))
            seg_kpts = np.linspace(start, end, n_seg)

            if i == 0:
                labels.append(start_label)
                path_indices.append(len(kpoints))
                kpoints.extend(seg_kpts.tolist())
            else:
                # Skip duplicate endpoint of previous segment.
                labels.append(start_label)
                path_indices.append(len(kpoints))
                kpoints.extend(seg_kpts[1:].tolist())

        labels.append(end_label)
        path_indices.append(len(kpoints) - 1)

        kpoints = np.array(kpoints)
        special_points = {label: np.array(point_coords[label]) for label in point_coords}
        logger.info(f"Generated k-path with seekpath: {len(labels)} labels, {len(kpoints)} points")
        return kpoints, path_indices, labels, special_points

    except ImportError:
        logger.warning("seekpath not available; falling back to ASE special points.")

    # ASE fallback.
    from ase.dft.kpoints import get_special_points

    special_points = get_special_points(cell)
    # Use a sensible default path for primitive cells.
    default_labels = ["GAMMA", "X", "M", "GAMMA"]
    available_labels = [label for label in default_labels if label in special_points]
    if len(available_labels) < 2:
        available_labels = list(special_points.keys())[:2]

    labels = []
    kpoints = []
    path_indices = []
    for i, label in enumerate(available_labels):
        labels.append(label)
        path_indices.append(len(kpoints))
        kpoints.append(special_points[label].tolist())

    # Linear interpolation between high-symmetry points.
    if len(available_labels) >= 2:
        n_between = max(1, (n_points - len(available_labels)) // (len(available_labels) - 1))
        interpolated = []
        interp_indices = []
        interp_labels = []
        for i in range(len(available_labels) - 1):
            start = np.array(special_points[available_labels[i]])
            end = np.array(special_points[available_labels[i + 1]])
            seg = np.linspace(start, end, n_between + 2)
            if i == 0:
                interpolated.extend(seg.tolist())
                interp_labels.append(available_labels[i])
                interp_indices.append(0)
            else:
                interpolated.extend(seg[1:].tolist())
                interp_labels.append(available_labels[i])
                interp_indices.append(len(interpolated) - len(seg) + 1)
        interp_labels.append(available_labels[-1])
        interp_indices.append(len(interpolated) - 1)
        kpoints = np.array(interpolated)
        labels = interp_labels
        path_indices = interp_indices

    special_points = {label: np.array(coord) for label, coord in special_points.items()}
    logger.info(f"Generated k-path with ASE fallback: {len(labels)} labels, {len(kpoints)} points")
    return kpoints, path_indices, labels, special_points


def estimate_kpoints(atoms: Any, kppa: float = 1000.0) -> Tuple[int, int, int]:
    """
    Estimate a Monkhorst-Pack k-point grid from k-points per reciprocal atom (kppa).

    Args:
        atoms: ASE Atoms object.
        kppa: Target k-points per reciprocal atom.

    Returns:
        Tuple (nkx, nky, nkz) of integer k-point mesh divisions.
    """
    Atoms = _import_ase_atoms()
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ASE Atoms object")

    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ValueError("Cannot estimate k-points for an empty Atoms object")

    total_kpoints = kppa * n_atoms
    cell = atoms.get_cell()
    rec_lengths = np.linalg.norm(cell.reciprocal(), axis=1)
    inv_lengths = rec_lengths
    inv_sum = inv_lengths.sum()

    if inv_sum == 0:
        raise ValueError("Invalid unit cell: reciprocal lattice vectors are zero")

    # Determine per-direction density such that product matches total_kpoints.
    scale = total_kpoints ** (1.0 / 3.0)
    kpoints = np.maximum(
        1,
        np.round(scale * inv_lengths / inv_sum * 3).astype(int),
    )

    # Ensure product is close to target while keeping at least one k-point.
    kpoints = tuple(int(max(1, k)) for k in kpoints)
    logger.info(f"Estimated k-grid {kpoints} for kppa={kppa}")
    return kpoints
