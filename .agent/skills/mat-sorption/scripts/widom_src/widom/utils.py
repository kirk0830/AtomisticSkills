# Copyright (c) 2025 CuspAI. Vendored from COFclean external/widom.

from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.filters import FrechetCellFilter
from ase.io import Trajectory
from ase.optimize import FIRE
from pymatgen.optimization.neighbors import find_points_in_spheres


def sample_gas_positions(
    structure: Atoms,
    gas: Atoms,
    num_insertions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    random_positions = rng.random((num_insertions, 3))
    random_angles = rng.random(num_insertions) * 360
    random_axes = rng.random((num_insertions, 3))
    random_axes /= np.linalg.norm(random_axes, axis=1, keepdims=True)
    cartesian_positions = structure.cell.cartesian_positions(random_positions)
    gas_positions = np.zeros((num_insertions, len(gas), 3))
    for i in range(num_insertions):
        added_gas = gas.copy()
        added_gas.cell = structure.cell
        added_gas.pbc = structure.pbc
        added_gas.rotate(v=random_axes[i], a=random_angles[i])
        added_gas.translate(cartesian_positions[i])
        added_gas.wrap()
        gas_positions[i] = added_gas.get_positions()
    return gas_positions


def check_accessibility(
    gas_positions: np.ndarray,
    framework_coords: np.ndarray,
    lattice_matrix: np.ndarray,
    cutoff_distance: float,
    cutoff_to_com: bool,
) -> np.ndarray:
    num_insertions, num_gas_atoms = gas_positions.shape[:2]
    if num_insertions == 0:
        return np.array([], dtype=bool)
    pbc_array = np.array([1, 1, 1], dtype=np.int64)
    if cutoff_to_com:
        gas_coords = np.mean(gas_positions, axis=1)
    else:
        gas_coords = gas_positions.reshape(-1, 3)
    center_indices, all_coords_indices, offset_vectors, distances = find_points_in_spheres(
        all_coords=gas_coords,
        center_coords=framework_coords.astype(np.float64),
        r=float(cutoff_distance),
        pbc=pbc_array,
        lattice=lattice_matrix.astype(np.float64),
        tol=1e-8,
    )
    if cutoff_to_com:
        overlapping_insertions = set(all_coords_indices)
    else:
        overlapping_insertions = set(all_coords_indices // num_gas_atoms)
    is_accessible = np.array([i not in overlapping_insertions for i in range(num_insertions)])
    return is_accessible


def optimize_atoms(
    calculator: Calculator,
    atoms: Atoms,
    num_total_optimization: int = 30,
    num_internal_steps: int = 50,
    num_cell_steps: int = 50,
    fmax: float = 0.05,
    cell_relax: bool = True,
    trajectory_file: Optional[str] = None,
) -> Optional[Atoms]:
    if trajectory_file is not None:
        trajectory = Trajectory(trajectory_file, "w", atoms)
    opt_atoms = atoms.copy()
    convergence = False
    filter = None
    for _ in range(int(num_total_optimization)):
        opt_atoms = opt_atoms.copy()
        opt_atoms.calc = calculator
        if cell_relax:
            filter = FrechetCellFilter(opt_atoms)
            optimizer = FIRE(filter)
            convergence = optimizer.run(fmax=fmax, steps=num_cell_steps)
            opt_atoms.wrap()
            if trajectory_file is not None:
                optimizer.attach(trajectory.write, interval=1)
            convergence = optimizer.run(fmax=fmax, steps=num_internal_steps)
            if convergence:
                break
        optimizer = FIRE(opt_atoms)
        convergence = optimizer.run(fmax=fmax, steps=num_internal_steps)
        if trajectory_file is not None:
            optimizer.attach(trajectory.write, interval=1)
        if convergence and not cell_relax:
            break
        forces = (filter if filter is not None else opt_atoms).get_forces()
        _fmax = np.sqrt((forces**2).sum(axis=1).max())
        if _fmax > 1000:
            return None
    if not convergence:
        return None
    return opt_atoms


def create_supercell_if_needed(structure: Atoms, min_interplanar_distance: float = 6.0) -> Atoms:
    structure = structure.copy()
    cell_volume = structure.get_volume()
    cell_vectors = np.array(structure.cell)
    dist_a = cell_volume / np.linalg.norm(np.cross(cell_vectors[1], cell_vectors[2]))
    dist_b = cell_volume / np.linalg.norm(np.cross(cell_vectors[2], cell_vectors[0]))
    dist_c = cell_volume / np.linalg.norm(np.cross(cell_vectors[0], cell_vectors[1]))
    plane_distances = np.array([dist_a, dist_b, dist_c])
    supercell = np.ceil(min_interplanar_distance / plane_distances).astype(int)
    if np.any(supercell > 1):
        print(f"Making supercell: {supercell} to prevent interplanar distance < {min_interplanar_distance}")
        structure = structure.repeat(supercell)
    return structure


def bootstrap_ratio_std(
    numerator: np.ndarray,
    denominator: np.ndarray,
    n_bootstrap: int,
    random_seed: int,
) -> float:
    n = len(numerator)
    ratios = np.zeros(n_bootstrap)
    rng = np.random.default_rng(random_seed)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        num_sample = numerator[indices]
        denom_sample = denominator[indices]
        ratios[i] = num_sample.mean() / denom_sample.mean()
    return float(np.std(ratios))


def calculate_atomic_density(atoms: Atoms) -> float:
    volume = atoms.get_volume() * 1e-30
    total_mass = np.sum(atoms.get_masses()) * units._amu
    return total_mass / volume
