# Copyright (c) 2025 CuspAI. Vendored from COFclean external/widom.

from io import BytesIO

import numpy as np
from ase import Atoms, units
from pydantic import BaseModel

from .utils import (
    bootstrap_ratio_std,
    calculate_atomic_density,
)


class WidomInsertionResults(BaseModel):
    henry_coefficient: float
    henry_coefficient_std: float
    averaged_interaction_energy: float
    averaged_interaction_energy_std: float
    heat_of_adsorption: float
    heat_of_adsorption_std: float
    atomic_density: float
    total_energies: list[float]
    energy_gas: float
    energy_structure: float
    interaction_energies: list[float]
    is_accessible: list[bool]
    is_valid: list[bool]
    gas_positions: list[list[list[float]]]
    optimized_structure_cif: str


def analyze_widom_insertions(
    energies: np.ndarray,
    is_accessible: np.ndarray,
    gas_positions: np.ndarray,
    energy_structure: float,
    energy_gas: float,
    temperature: float,
    structure: Atoms,
    energies_are_interaction: bool,
    min_interaction_energy: float,
    random_seed: int,
) -> WidomInsertionResults:
    if energies_are_interaction:
        interaction_energies = energies
    else:
        interaction_energies = energies - energy_structure - energy_gas
    num_samples = len(interaction_energies)
    is_valid = interaction_energies > min_interaction_energy
    interaction_energies_valid = np.where(is_valid, interaction_energies, 1e10)
    boltzmann_factor = np.exp(-interaction_energies_valid / (temperature * units._k / units._e))
    atomic_density = calculate_atomic_density(structure)
    kh = (
        boltzmann_factor.mean()
        / (units._k * units._Nav)
        / temperature
        / atomic_density
    )
    kh_std = (
        boltzmann_factor.std()
        / (units._k * units._Nav)
        / temperature
        / atomic_density
        / np.sqrt(num_samples)
    )
    interaction_energies_shift = interaction_energies_valid - interaction_energies_valid.min()
    boltzmann_factor_shift = np.exp(
        -interaction_energies_shift / (temperature * units._k / units._e)
    )
    u = (interaction_energies_valid * boltzmann_factor_shift).sum() / boltzmann_factor_shift.sum()
    u_std = bootstrap_ratio_std(
        interaction_energies_valid * boltzmann_factor_shift,
        boltzmann_factor_shift,
        n_bootstrap=100,
        random_seed=random_seed,
    )
    qst = (u * units._e - units._k * temperature) * units._Nav * 1e-3
    qst_std = u_std * units._e * units._Nav * 1e-3
    cif_writer = BytesIO()
    structure.write(cif_writer, format="cif")
    cif_writer.seek(0)
    results = WidomInsertionResults(
        henry_coefficient=float(kh),
        henry_coefficient_std=float(kh_std),
        averaged_interaction_energy=float(u),
        averaged_interaction_energy_std=float(u_std),
        heat_of_adsorption=float(qst),
        heat_of_adsorption_std=float(qst_std),
        atomic_density=float(atomic_density),
        total_energies=energies.tolist(),
        energy_gas=float(energy_gas),
        energy_structure=float(energy_structure),
        interaction_energies=interaction_energies.tolist(),
        is_accessible=is_accessible.tolist(),
        is_valid=is_valid.tolist(),
        gas_positions=gas_positions.tolist(),
        optimized_structure_cif=cif_writer.read().decode("ascii"),
    )
    return results
