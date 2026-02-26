# Copyright (c) 2025 CuspAI. Vendored from COFclean external/widom.

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

try:
    from tqdm_loggable.auto import tqdm
except ImportError:
    from tqdm import tqdm

from .structure_preparation import (
    create_combined_structure,
    prepare_structures_for_insertion,
)


def sample_compute_energies(
    calculator: Calculator,
    structure: Atoms,
    gas: Atoms,
    num_insertions: int,
    cutoff_distance: float,
    cutoff_to_com: bool,
    min_interplanar_distance: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    structure = structure.copy()
    gas = gas.copy()
    structure_supercell, gas_positions, is_accessible = prepare_structures_for_insertion(
        structure=structure,
        gas=gas,
        num_insertions=num_insertions,
        cutoff_distance=cutoff_distance,
        cutoff_to_com=cutoff_to_com,
        min_interplanar_distance=min_interplanar_distance,
        random_seed=random_seed,
    )
    print(f"Number of accessible positions: {np.sum(is_accessible)} out of {num_insertions}")
    energies = np.zeros(num_insertions)
    energies[~is_accessible] = 1e10
    accessible_indices = np.where(is_accessible)[0]
    for i in tqdm(accessible_indices):
        combined = create_combined_structure(
            structure_supercell,
            gas,
            gas_positions[i]
        )
        combined.calc = calculator
        energies[i] = combined.get_potential_energy()
    return energies, is_accessible, gas_positions
