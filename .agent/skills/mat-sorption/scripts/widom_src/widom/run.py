# Copyright (c) 2025 CuspAI. Vendored from COFclean external/widom.

import logging

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator

from .analyze import (
    WidomInsertionResults,
    analyze_widom_insertions,
)
from .sample_compute_energies import sample_compute_energies
from .utils import optimize_atoms

logger = logging.getLogger(__name__)


def run_widom_insertion(
    calculator: Calculator,
    structure: Atoms,
    gas: str,
    temperature: float,
    model_outputs_interaction_energy: bool,
    num_insertions: int = 10000,
    optimize_structures: bool = False,
    cutoff_distance: float = 1.00,
    cutoff_to_com: bool = False,
    min_interplanar_distance: float = 6.0,
    random_seed: int = 0,
    min_interaction_energy: float = -1.25,
) -> WidomInsertionResults:
    gas_atoms = molecule(gas)
    optimized_gas = optimize_atoms(
        calculator=calculator,
        atoms=gas_atoms,
        cell_relax=False,
    )
    if optimize_structures:
        logger.info("Optimizing structure...")
        optimized_structure = optimize_atoms(
            calculator=calculator,
            atoms=structure,
        )
        if optimized_structure is None:
            raise ValueError("Structure optimization failed.")
        logger.info("Optimizing gas molecule...")
        optimized_gas = optimize_atoms(
            calculator=calculator,
            atoms=gas_atoms,
            cell_relax=False,
        )
        if optimized_gas is None:
            raise ValueError("Gas molecule optimization failed.")
    else:
        optimized_structure = structure
        optimized_gas = gas_atoms

    logger.info(f"Running Widom insertion with {num_insertions} insertions...")
    energies, is_accessible, gas_positions = sample_compute_energies(
        calculator=calculator,
        structure=optimized_structure,
        gas=optimized_gas,
        num_insertions=num_insertions,
        cutoff_distance=cutoff_distance,
        cutoff_to_com=cutoff_to_com,
        min_interplanar_distance=min_interplanar_distance,
        random_seed=random_seed,
    )

    energy_structure = calculator.get_potential_energy(optimized_structure)
    energy_gas = calculator.get_potential_energy(optimized_gas)
    logger.info(f"Energy of structure: {energy_structure} eV")
    logger.info(f"Energy of gas: {energy_gas} eV")

    assert energy_structure is not None
    assert energy_gas is not None

    logger.info("Analyzing results...")
    results = analyze_widom_insertions(
        energies=energies,
        is_accessible=is_accessible,
        gas_positions=gas_positions,
        temperature=temperature,
        structure=optimized_structure,
        energy_structure=energy_structure,
        energy_gas=energy_gas,
        energies_are_interaction=model_outputs_interaction_energy,
        random_seed=random_seed,
        min_interaction_energy=min_interaction_energy,
    )

    logger.info(
        "Results: henry=%.6e +/- %.6e mol/kg/Pa | heat=%.3f +/- %.3f kJ/mol | "
        "avg_interaction=%.6f +/- %.6f eV | atomic_density=%.3f kg/m^3",
        results.henry_coefficient,
        results.henry_coefficient_std,
        results.heat_of_adsorption,
        results.heat_of_adsorption_std,
        results.averaged_interaction_energy,
        results.averaged_interaction_energy_std,
        results.atomic_density,
    )
    return results
