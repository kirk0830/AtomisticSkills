# Copyright (c) 2025 CuspAI. Vendored from COFclean external/widom.

import numpy as np
from ase import Atoms

from .utils import check_accessibility, create_supercell_if_needed, sample_gas_positions


def create_combined_structure(
    structure: Atoms,
    gas: Atoms,
    gas_position: np.ndarray,
) -> Atoms:
    combined = structure.copy()
    gas_copy = gas.copy()
    gas_copy.positions = gas_position
    combined.extend(gas_copy)
    combined.wrap()
    return combined


def prepare_structures_for_insertion(
    structure: Atoms,
    gas: Atoms,
    num_insertions: int,
    cutoff_distance: float,
    cutoff_to_com: bool,
    min_interplanar_distance: float,
    random_seed: int,
) -> tuple[Atoms, np.ndarray, np.ndarray]:
    structure_supercell = create_supercell_if_needed(structure, min_interplanar_distance)
    rng = np.random.default_rng(random_seed)
    gas_positions = sample_gas_positions(structure_supercell, gas, num_insertions, rng)
    framework_coords = structure_supercell.get_positions()
    lattice_matrix = np.array(structure_supercell.cell)
    is_accessible = check_accessibility(
        gas_positions,
        framework_coords,
        lattice_matrix,
        cutoff_distance,
        cutoff_to_com
    )
    return structure_supercell, gas_positions, is_accessible
