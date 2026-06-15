"""Core utilities for NValchemi GPU-accelerated MLIP batch operations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

try:
    import torch
    from nvalchemi.data import AtomicData, Batch  # noqa: F401

    NVALCHEMI_AVAILABLE = True
except ImportError:
    NVALCHEMI_AVAILABLE = False

if TYPE_CHECKING:
    from ase import Atoms


def check_nvalchemi_available() -> bool:
    """Return True if nvalchemi-toolkit is importable."""
    return NVALCHEMI_AVAILABLE


def atoms_to_atomic_data(
    atoms: "Atoms",
    device: str = "cpu",
    dtype: Optional[Any] = None,
) -> Any:
    """Convert an ASE Atoms object to a NValchemi AtomicData.

    Parameters
    ----------
    atoms : ase.Atoms
    device : str
        Target torch device string.
    dtype : torch.dtype, optional
        Floating-point dtype; defaults to torch.float32.
    """
    if not NVALCHEMI_AVAILABLE:
        raise ImportError(
            "nvalchemi-toolkit is required. "
            "Install with: pip install nvalchemi-toolkit"
        )
    if dtype is None:
        dtype = torch.float32
    return AtomicData.from_atoms(atoms, device=device, dtype=dtype)


def atomic_data_to_atoms(data: Any) -> "Atoms":
    """Reconstruct an ASE Atoms object from a NValchemi AtomicData.

    Attaches a SinglePointCalculator with energy/forces/stress if those
    fields are populated in the AtomicData.

    Parameters
    ----------
    data : AtomicData
        A single-graph NValchemi AtomicData (from Batch.get_data()).
    """
    if not NVALCHEMI_AVAILABLE:
        raise ImportError("nvalchemi-toolkit is required.")

    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    pos = data.positions.detach().cpu().numpy()
    numbers = data.atomic_numbers.detach().cpu().numpy()

    cell_tensor = getattr(data, "cell", None)
    if cell_tensor is not None:
        cell = cell_tensor.squeeze().detach().cpu().numpy()
        pbc = True
    else:
        cell = None
        pbc = False

    atoms = Atoms(numbers=numbers, positions=pos, cell=cell, pbc=pbc)

    # Attach last-step energetics if available
    calc_results: dict[str, Any] = {}

    energy_tensor = getattr(data, "energy", None)
    if energy_tensor is not None:
        calc_results["energy"] = float(energy_tensor.detach().cpu().sum().item())

    forces_tensor = getattr(data, "forces", None)
    if forces_tensor is not None:
        calc_results["forces"] = forces_tensor.detach().cpu().numpy()

    stress_tensor = getattr(data, "stress", None)
    if stress_tensor is not None:
        # NValchemi stress [1,3,3] eV/Å³ (Cauchy, tensile convention)
        # → ASE Voigt 6-vector [xx, yy, zz, yz, xz, xy] eV/Å³
        s = stress_tensor.squeeze(0).detach().cpu().numpy()
        calc_results["stress"] = [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]

    if calc_results:
        atoms.calc = SinglePointCalculator(atoms, **calc_results)

    return atoms


def extract_batch_results(
    final_batch: Any,
    structure_names: list[str],
    output_dirs: list[str],
) -> list[dict[str, Any]]:
    """Extract per-structure results from a NValchemi Batch after dynamics.

    Writes final_structure.cif and energy.txt to each output directory.

    Parameters
    ----------
    final_batch : Batch
        The batch returned by BaseDynamics.run().
    structure_names : list[str]
        Names for each structure (length must match final_batch.num_graphs).
    output_dirs : list[str]
        Per-structure output directories (one per graph).

    Returns
    -------
    list of result dicts, each with keys:
        structure_name, status, energy, cif_path, output_dir
        (or status="failed" + error on exception)
    """
    if not NVALCHEMI_AVAILABLE:
        raise ImportError("nvalchemi-toolkit is required.")

    from pymatgen.io.ase import AseAtomsAdaptor

    results: list[dict[str, Any]] = []
    num_graphs: int = final_batch.num_graphs

    data_list = final_batch.to_data_list()

    for i in range(num_graphs):
        struct_name = (
            structure_names[i] if i < len(structure_names) else f"structure_{i}"
        )
        out_dir = output_dirs[i] if i < len(output_dirs) else output_dirs[0]
        os.makedirs(out_dir, exist_ok=True)

        try:
            atoms = atomic_data_to_atoms(data_list[i])

            structure = AseAtomsAdaptor.get_structure(atoms)
            cif_path = os.path.join(out_dir, "final_structure.cif")
            structure.to(filename=cif_path)

            energy: Optional[float] = None
            if atoms.calc is not None and "energy" in atoms.calc.results:
                energy = atoms.calc.results["energy"]
                with open(os.path.join(out_dir, "energy.txt"), "w") as f:
                    f.write(f"{energy}\n")

            results.append(
                {
                    "structure_name": struct_name,
                    "status": "success",
                    "energy": energy,
                    "cif_path": cif_path,
                    "output_dir": out_dir,
                }
            )
        except Exception as e:
            results.append(
                {
                    "structure_name": struct_name,
                    "status": "failed",
                    "error": str(e),
                    "output_dir": out_dir,
                }
            )

    return results
