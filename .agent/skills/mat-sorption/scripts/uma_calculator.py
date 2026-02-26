"""
UMA (FairChem) calculator for GCMC (port from COFclean gcmc/uma_calculator.py).
"""

from ase import Atoms
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


def load_uma_calculators(model: str, device: str, task_name: str):
    predictor = load_predict_unit(
        path=str(model),
        device=device,
        inference_settings="default",
    )
    calc_mol = FAIRChemCalculator(predictor, task_name=task_name)
    calc_host = FAIRChemCalculator(predictor, task_name=task_name)
    return predictor, calc_mol, calc_host


def set_uma_spin_info(atoms: Atoms) -> None:
    atoms.info["charge"] = 0
    atoms.info["spin_multiplicity"] = 1
    atoms.info["spin"] = 1
