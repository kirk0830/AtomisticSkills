"""Core utilities for NValchemi GPU-accelerated MLIP batch operations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

try:
    import torch
    from nvalchemi.data import AtomicData
    from nvalchemi.dynamics.sinks import HostMemory

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
    mode: str = "static",
    memory_sink: Optional[Any] = None,
    log_interval: int = 10,
    timestep: float = 1.0,
    temperature: float = 300.0,
    ensemble: str = "nvt",
) -> list[dict[str, Any]]:
    """Extract per-structure results from a NValchemi Batch or memory sink after dynamics.

    Supports static, relax, and md modes.
    """
    if not NVALCHEMI_AVAILABLE:
        raise ImportError("nvalchemi-toolkit is required.")

    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import write
    import numpy as np

    results: list[dict[str, Any]] = []

    if mode == "static":
        num_graphs = final_batch.num_graphs
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
                energy = None
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
    elif mode == "relax":
        if memory_sink is not None:
            stored_batch = memory_sink.read()
            data_list = stored_batch.to_data_list()
            n_total_frames = len(data_list)
            n_structures = len(structure_names)
            n_snapshots = n_total_frames // n_structures
        else:
            data_list = final_batch.to_data_list()
            n_structures = len(structure_names)
            n_snapshots = 1

        for i in range(n_structures):
            struct_name = structure_names[i]
            out_dir = output_dirs[i]
            os.makedirs(out_dir, exist_ok=True)
            try:
                struct_atoms_list = []
                if memory_sink is not None:
                    for s in range(n_snapshots):
                        idx = s * n_structures + i
                        if idx < n_total_frames:
                            struct_atoms_list.append(
                                atomic_data_to_atoms(data_list[idx])
                            )
                else:
                    struct_atoms_list.append(atomic_data_to_atoms(data_list[i]))

                if not struct_atoms_list:
                    raise RuntimeError("No trajectory snapshots recorded.")

                traj_file = None
                log_file = None
                if memory_sink is not None:
                    traj_file = os.path.join(out_dir, "relax.traj")
                    write(traj_file, struct_atoms_list, format="traj")

                    log_file = os.path.join(out_dir, "relax.log")
                    with open(log_file, "w") as f:
                        f.write("           Step          Energy         fmax\n")
                        for step_idx, atoms in enumerate(struct_atoms_list):
                            energy = atoms.get_potential_energy()
                            forces = atoms.get_forces()
                            fmax = np.sqrt((forces**2).sum(axis=-1)).max()
                            f.write(
                                f"FIRE:  {step_idx:9d}  {energy:14.6f}  {fmax:12.6f}\n"
                            )

                final_atoms = struct_atoms_list[-1]
                structure = AseAtomsAdaptor.get_structure(final_atoms)
                cif_path = os.path.join(out_dir, "relaxed_structure.cif")
                structure.to(filename=cif_path)

                final_energy = final_atoms.get_potential_energy()
                with open(os.path.join(out_dir, "relaxed_energy.txt"), "w") as f:
                    f.write(f"{final_energy}\n")

                res_dict = {
                    "structure_name": struct_name,
                    "status": "success",
                    "energy": final_energy,
                    "cif_path": cif_path,
                    "output_dir": out_dir,
                }
                if traj_file is not None:
                    res_dict["trajectory_path"] = traj_file
                if log_file is not None:
                    res_dict["log_path"] = log_file

                results.append(res_dict)
            except Exception as e:
                results.append(
                    {
                        "structure_name": struct_name,
                        "status": "failed",
                        "error": str(e),
                        "output_dir": out_dir,
                    }
                )
    elif mode == "md":
        if memory_sink is not None:
            stored_batch = memory_sink.read()
            data_list = stored_batch.to_data_list()
            n_total_frames = len(data_list)
            n_structures = len(structure_names)
            n_snapshots = n_total_frames // n_structures
        else:
            data_list = final_batch.to_data_list()
            n_structures = len(structure_names)
            n_snapshots = 1

        for i in range(n_structures):
            struct_name = structure_names[i]
            out_dir = output_dirs[i]
            os.makedirs(out_dir, exist_ok=True)
            try:
                struct_atoms_list = []
                if memory_sink is not None:
                    for s in range(n_snapshots):
                        idx = s * n_structures + i
                        if idx < n_total_frames:
                            data = data_list[idx]
                            atoms = atomic_data_to_atoms(data)
                            velocities_tensor = getattr(data, "velocities", None)
                            if velocities_tensor is not None:
                                atoms.set_velocities(
                                    velocities_tensor.detach().cpu().numpy()
                                )
                            struct_atoms_list.append(atoms)
                else:
                    data = data_list[i]
                    atoms = atomic_data_to_atoms(data)
                    velocities_tensor = getattr(data, "velocities", None)
                    if velocities_tensor is not None:
                        atoms.set_velocities(velocities_tensor.detach().cpu().numpy())
                    struct_atoms_list.append(atoms)

                if not struct_atoms_list:
                    raise RuntimeError("No trajectory snapshots recorded.")

                formula = struct_atoms_list[-1].get_chemical_formula()
                filename_base = f"{formula}_{temperature}K_{ensemble}"

                traj_file = None
                log_file = None
                if memory_sink is not None:
                    traj_file = os.path.join(out_dir, f"{filename_base}.traj")
                    log_file = os.path.join(out_dir, f"{filename_base}.log")
                    write(traj_file, struct_atoms_list, format="traj")

                    with open(log_file, "w") as f:
                        f.write(
                            "Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]\n"
                        )
                        for step_idx, atoms in enumerate(struct_atoms_list):
                            time_ps = (step_idx * log_interval * timestep) / 1000.0
                            ep = atoms.get_potential_energy()
                            ek = atoms.get_kinetic_energy()
                            etot = ep + ek
                            temp = atoms.get_temperature()
                            f.write(
                                f"{time_ps:.4f}      {etot:11.4f}  {ep:11.4f}  {ek:11.4f}   {temp:5.1f}\n"
                            )

                final_atoms = struct_atoms_list[-1]
                structure = AseAtomsAdaptor.get_structure(final_atoms)
                cif_path = os.path.join(out_dir, "final_structure.cif")
                structure.to(filename=cif_path)

                final_energy = final_atoms.get_potential_energy()
                with open(os.path.join(out_dir, "energy.txt"), "w") as f:
                    f.write(f"{final_energy}\n")

                res_dict = {
                    "structure_name": struct_name,
                    "status": "success",
                    "cif_path": cif_path,
                    "output_dir": out_dir,
                }
                if traj_file is not None:
                    res_dict["trajectory_path"] = traj_file
                if log_file is not None:
                    res_dict["log_path"] = log_file

                results.append(res_dict)
            except Exception as e:
                results.append(
                    {
                        "structure_name": struct_name,
                        "status": "failed",
                        "error": str(e),
                        "output_dir": out_dir,
                    }
                )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return results


# ---------------------------------------------------------------------------
# Inflight batching helpers
# ---------------------------------------------------------------------------


def estimate_max_batch_atoms(device: str = "cuda", safety_factor: float = 0.5) -> int:
    """Estimate max total atoms for a single fixed GPU batch from free VRAM.

    Used as the threshold for switching from fixed-batch to inflight mode:
    if the total atom count across all structures exceeds this, inflight
    batching is used so only a subset occupies GPU memory at any time.

    Parameters
    ----------
    device : str
        Torch device string. Non-CUDA devices return a conservative fallback.
    safety_factor : float
        Fraction of free VRAM to budget for the live batch. Default 0.5.

    Returns
    -------
    int
        Estimated atom budget. Minimum 200.
    """
    if not NVALCHEMI_AVAILABLE:
        return 2000
    if torch.cuda.is_available() and device not in ("cpu", "mps"):
        free_bytes, _ = torch.cuda.mem_get_info()
        # ~2 MB/atom: conservative estimate covering neighbor lists,
        # gradient buffers, and intermediate activations for large GNNs.
        return max(200, int(free_bytes * safety_factor / 2_000_000))
    return 2000  # CPU / unknown device fallback


class AtomsDataset:
    """Dataset adapter wrapping pre-loaded ASE Atoms for SizeAwareSampler.

    SizeAwareSampler requires three methods:
      * ``__len__()``
      * ``get_metadata(idx) -> (num_atoms, num_edges)``  — must be O(1)
      * ``__getitem__(idx) -> (AtomicData, dict)``

    ``get_metadata`` is called for every sample at sampler construction to
    build atom-count bins for bin-packing.  Sizes are pre-computed at init
    so the call is a trivial list lookup.

    Parameters
    ----------
    atoms_list : list[Atoms]
        Pre-loaded ASE Atoms objects.
    names : list[str]
        Corresponding structure names (same order as atoms_list).
    device : str
        Target torch device for AtomicData conversion.
    dtype : torch.dtype, optional
        Floating-point dtype. Defaults to torch.float32.
    """

    def __init__(
        self,
        atoms_list: list,
        names: list[str],
        device: str = "cuda",
        dtype: Any = None,
        relax_cell: bool = False,
    ) -> None:
        self._atoms = atoms_list
        self._names = names
        self._device = device
        self._dtype = dtype
        self._relax_cell = relax_cell
        # Pre-computed for O(1) get_metadata — no tensor allocation at query time.
        self._sizes: list[tuple[int, int]] = [(len(a), 0) for a in atoms_list]

    def __len__(self) -> int:
        return len(self._atoms)

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` without constructing AtomicData."""
        return self._sizes[idx]

    def __getitem__(self, idx: int) -> tuple:
        """Return ``(AtomicData, metadata_dict)`` for sample *idx*."""
        if not NVALCHEMI_AVAILABLE:
            raise ImportError("nvalchemi-toolkit is required.")
        dtype = self._dtype if self._dtype is not None else torch.float32
        data = AtomicData.from_atoms(self._atoms[idx], device=self._device, dtype=dtype)
        # Pre-allocate model output tensors as zeros.  SegmentedLevelStorage.concatenate()
        # only keeps keys common to both batches, so replacements added during inflight
        # refill must carry the same keys as the live batch or forces/energy will be dropped.
        n_atoms = len(self._atoms[idx])
        data["forces"] = torch.zeros(n_atoms, 3, device=self._device, dtype=dtype)
        # Energy shape (1, 1) satisfies AtomicData pydantic validation Float[Tensor, 'B 1']
        data["energy"] = torch.zeros(1, 1, device=self._device, dtype=dtype)
        if self._relax_cell:
            data["stress"] = torch.zeros(1, 3, 3, device=self._device, dtype=dtype)
        return data, {"name": self._names[idx], "index": idx}


class HostMemoryWithSystemId(HostMemory):
    """HostMemory that preserves ``system_id`` across the unbatch/rebatch round-trip.

    When FusedStage writes graduated structures to sinks via
    ``_overflow_to_sinks``, the batch is unbatched with ``to_data_list()``.
    During reconstruction the ``__system_keys__`` registration on each
    ``AtomicData`` is reset to defaults, so ``system_id`` (stamped by
    ``SizeAwareSampler``) is silently dropped.  This subclass extracts
    ``system_id`` *before* unbatching and re-registers it on each individual
    ``AtomicData`` so that it survives ``read()`` / ``drain()``.
    """

    def write(self, batch: Any, mask: Any = None) -> None:  # type: ignore[override]
        if not NVALCHEMI_AVAILABLE:
            raise ImportError("nvalchemi-toolkit is required.")

        num_total = batch.num_graphs or 0
        if num_total == 0:
            return

        if mask is not None:
            mask = mask.to(device=batch.device, dtype=torch.bool)
            if mask.shape[0] != num_total:
                raise ValueError(
                    f"mask length {mask.shape[0]} != num_graphs {num_total}"
                )
            num_selected = int(mask.sum().item())
            if num_selected == 0:
                return
            if num_selected < num_total:
                indices = torch.nonzero(mask, as_tuple=True)[0]
                _ = batch.batch_ptr  # trigger lazy init for SegmentedLevelStorage
                batch = batch.index_select(indices)

        # Extract system_id before to_data_list() drops it.
        system_ids = batch["system_id"].detach().to("cpu")
        data_list = batch.to_data_list()

        if len(self._data_list) + len(data_list) > self._capacity:
            raise RuntimeError(
                f"HostMemoryWithSystemId is full: cannot add {len(data_list)} "
                f"samples to buffer with {len(self._data_list)}/{self._capacity}."
            )

        for i, data in enumerate(data_list):
            data_cpu = data.to(self._device)
            data_cpu.add_system_property("system_id", system_ids[i : i + 1])
            self._data_list.append(data_cpu)
