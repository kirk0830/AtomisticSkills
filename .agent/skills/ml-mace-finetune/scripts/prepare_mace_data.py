#!/usr/bin/env python
"""
Convert fine-tuning JSON data into MACE .xyz training files and generate a configuration YAML.

Usage:
    python prepare_mace_data.py --data path/to/data.json --model MACE-MP-small \\
        --epochs 100 --lr 1e-4 --output-dir ./fine_tuning
"""

import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import yaml
import ast
from ase.io import write
from ase import Atoms

def prepare_training_data(training_data, has_stress_forced=False, vasp_stress_conversion=False):
    """
    Prepare training data in MACE format.
    Converts dicts/pymatgen objects to ASE Atoms and extracts targets.
    """
    structures = []
    energies = []
    forces = []
    stresses = []
    
    # 1 eV/A^3 = 160.21766208 GPa (or kB, wait 1 eV/A^3 = 160.21766208 GPa? No, 1 eV/A^3 = 160.21766208 GPa. VASP stress is kB so 1 kB = 0.1 GPa. So 1 eV/A^3 = 1602.1766208 kB)
    # Wait: VASP stress in kB. 1 eV/A^3 = 1.6021766208e-19 J / (1e-30 m^3) = 1.6021766208e11 Pa = 160.21766208 GPa.
    # 1 kBar = 100 MPa = 0.1 GPa.
    # Therefore, 1 eV/A^3 = 1602.1766208 kBar.
    # So if VASP gives stress in kB, we divide by 1602.1766208 to get eV/A^3.
    stress_conversion = -1.0 / 1602.1766208 if vasp_stress_conversion else 1.0
    
    for data in training_data:
        structure = data.get('atoms', data.get('structure'))
        if structure is None:
            raise KeyError("Data item must contain either 'atoms' or 'structure' key")
        
        if isinstance(structure, Atoms):
            atoms = structure
        elif isinstance(structure, dict):
            if 'positions' in structure and 'symbols' in structure:
                atoms = Atoms(
                    symbols=structure['symbols'],
                    positions=structure['positions'],
                    cell=structure.get('cell'),
                    pbc=structure.get('pbc')
                )
            elif '@module' in structure or '@class' in structure:
                from pymatgen.io.ase import AseAtomsAdaptor
                from pymatgen.core import Structure
                pmg_structure = Structure.from_dict(structure)
                atoms = AseAtomsAdaptor.get_atoms(pmg_structure)
            else:
                atoms = Atoms(**structure)
        elif hasattr(structure, 'lattice') and hasattr(structure, 'species'):
            from pymatgen.io.ase import AseAtomsAdaptor
            atoms = AseAtomsAdaptor.get_atoms(structure)
        else:
            raise ValueError(f"Cannot convert structure to ASE Atoms: {type(structure)}")
        
        if 'config_type' in data:
            atoms.info['config_type'] = data['config_type']
        
        structures.append(atoms)
        energies.append(data.get('energy', data.get('vasp_e')))
        
        forces_val = data.get('forces', data.get('vasp_f'))
        forces.append(np.array(forces_val) if forces_val is not None else None)
        
        stress = data.get('stress', data.get('vasp_s'))
        if stress is not None:
            stresses.append(np.array(stress) * stress_conversion)
        elif has_stress_forced:
            # Pad with zeros if forced
            stresses.append(np.zeros((3,3)))
        else:
            stresses.append(None)
            
    return structures, np.array(energies), forces, stresses

def write_xyz_file(structures, energies, forces, stresses, filepath):
    """Write training data to XYZ file format for MACE."""
    atoms_list = []
    for atoms, energy, force, stress in zip(structures, energies, forces, stresses):
        atoms_copy = atoms.copy()
        atoms_copy.info['REF_energy'] = float(energy)
        atoms_copy.arrays['REF_forces'] = np.array(force)
        
        if stress is not None:
            stress_array = np.array(stress)
            if stress_array.shape == (3, 3):
                stress_flat = stress_array.flatten()
            elif stress_array.shape == (6,):
                stress_3x3 = np.array([
                    [stress_array[0], stress_array[5], stress_array[4]],
                    [stress_array[5], stress_array[1], stress_array[3]],
                    [stress_array[4], stress_array[3], stress_array[2]]
                ])
                stress_flat = stress_3x3.flatten()
            elif stress_array.shape == (9,):
                stress_flat = stress_array
            else:
                stress_flat = None
            
            if stress_flat is not None:
                atoms_copy.info['REF_stress'] = stress_flat
                
        atoms_list.append(atoms_copy)
        
    write(str(filepath), atoms_list, format='extxyz')

def main():
    parser = argparse.ArgumentParser(description="Prepare custom data for MACE training and generate a finetune_config.yaml")
    # Basic arguments
    parser.add_argument("--data", required=True, help="Path to JSON file containing ASE/pymatgen structure dictionaries with 'energy', 'forces', 'stress' keys")
    parser.add_argument("--model", default="MACE-MP-small", help="Base model name or path to a checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Peak learning rate for training")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--output-dir", default="./fine_tuning", help="Directory to save the configuration and data")
    
    # Validation and Regularization
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data to set aside for validation (0.0 to use everything for training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting validation data")
    
    # Model configuration
    parser.add_argument("--device", default="cuda", help="Device to use for training (cuda or cpu). Defaults to cuda.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Sets MACE to freeze its layers up to the readout")
    parser.add_argument("--reinit-head", action="store_true", help="Discard pre-trained readout and initialize a new one")
    parser.add_argument("--multiheads", action="store_true", help="Add a new head for fine-tuning while preserving existing ones")
    
    # Loss weights
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Loss weight for energy")
    parser.add_argument("--forces-weight", type=float, default=10.0, help="Loss weight for forces")
    parser.add_argument("--stress-weight", type=float, default=1.0, help="Loss weight for stress (automatically used if stress data is present)")

    # Data units
    parser.add_argument("--vasp-stress-conversion", action="store_true", help="If flag is present, multiplies stress arrays by -1/1602.1766208 to convert from kB (VASP) to eV/Å³")

    args = parser.parse_args()

    device = args.device
    
    print(f"Loading data from {args.data}...")
    with open(args.data, 'r') as f:
        data_raw = json.load(f)
        
    # Handle dictionary of subsets (WBM formats) or flat lists
    if isinstance(data_raw, dict):
        all_data = []
        for key, items in data_raw.items():
            if isinstance(items, list):
                all_data.extend(items)
            elif isinstance(items, dict):
                if 'atoms' in items or 'structure' in items:
                    all_data.append(items)
                else:
                    all_data.extend(list(items.values()))
            else:
                all_data.append(items)
    else:
        all_data = data_raw

    if args.val_split > 0.0:
        random.seed(args.seed)
        random.shuffle(all_data)
        split_idx = int(len(all_data) * (1.0 - args.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        print(f"Split data into {len(train_data)} training and {len(val_data)} validation samples.")
    else:
        train_data = all_data
        val_data = None
        print(f"Using all {len(train_data)} samples for training.")

    output_path = Path(args.output_dir).absolute()
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_path / "checkpoints"
    results_dir = output_path / "results"
    log_dir = output_path / "logs"
    
    train_structures, train_energies, train_forces, train_stresses = prepare_training_data(train_data, vasp_stress_conversion=args.vasp_stress_conversion)
    has_stress = any(s is not None for s in train_stresses)
    
    val_structures, val_energies, val_forces, val_stresses = None, None, None, None
    if val_data:
        val_structures, val_energies, val_forces, val_stresses = prepare_training_data(val_data, vasp_stress_conversion=args.vasp_stress_conversion)
        
    train_xyz_path = output_path / "train.xyz"
    val_xyz_path = output_path / "valid.xyz" if val_structures else None

    write_xyz_file(train_structures, train_energies, train_forces, train_stresses, train_xyz_path)
    if val_xyz_path:
        write_xyz_file(val_structures, val_energies, val_forces, val_stresses, val_xyz_path)
    
    effective_batch_size = min(args.batch_size, len(train_structures))
    
    if not val_xyz_path and len(train_structures) < 10:
        import shutil
        val_xyz_path = output_path / "valid.xyz"
        shutil.copy2(train_xyz_path, val_xyz_path)
        print(f"Small dataset without validation: copied train.xyz as valid.xyz ({len(train_structures)} structures)")

    try:
        import sys
        # Add project root to path for imports
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from src.utils.mlips.plot_utils import collect_label_distributions, plot_label_distributions
        
        # We need to reconstruct the dictionaries for collect_label_distributions
        train_dicts = [{"structure": s, "energy": e, "forces": f, "stress": st} for s, e, f, st in zip(train_structures, train_energies, train_forces, train_stresses)]
        distributions = collect_label_distributions(train_dicts)
        
        plot_path = output_path / "label_distributions.png"
        plot_label_distributions(distributions, save_path=str(plot_path), show=False)
        print(f"Generated label distribution plot at {plot_path}")
    except Exception as e:
        print(f"Could not generate label distributions plot: {e}")

    # Model resolution logic
    model_name_upper = args.model.upper()
    if "MATPES-R2SCAN" in model_name_upper or "MATPES-r2SCAN" in model_name_upper:
        foundation_model_name = "mace-matpes-r2scan-0"
    elif "MATPES-PBE" in model_name_upper:
        foundation_model_name = "mace-matpes-pbe-0"
    elif "MPA" in model_name_upper:
        foundation_model_name = "mace-mpa-0"
    elif "OMAT" in model_name_upper:
        if "SMALL" in model_name_upper:
            foundation_model_name = "small-omat-0"
        else:
            foundation_model_name = "medium-omat-0"
    else:
        foundation_model_name = args.model if os.path.exists(args.model) else "small"
        
    e0s_dict = "average"
    atomic_numbers_str = None
    # Extract E0s directly from MACE internals if possible
    try:
        from mace.calculators import mace_mp
        foundation_calc = mace_mp(model=foundation_model_name, dispersion=False)
        if hasattr(foundation_calc, 'models') and len(foundation_calc.models) > 0:
            model = foundation_calc.models[0]
            if hasattr(model, "atomic_energies_fn"):
                atomic_energies_fn = model.atomic_energies_fn
                atomic_numbers = model.atomic_numbers
                e0s_tensor = atomic_energies_fn.atomic_energies
                temp_dict = {}
                for i, z in enumerate(atomic_numbers):
                    if e0s_tensor.dim() == 1:
                        val = e0s_tensor[i].item()
                    else:
                        val = e0s_tensor[0, i].item()
                    temp_dict[int(z.item())] = float(val)
                if temp_dict:
                    e0s_dict = str(temp_dict)
                    atomic_numbers_str = str(list(sorted(temp_dict.keys())))
    except Exception as e:
        print(f"Could not extract E0s automatically ({e}), defaulting to 'average'")

    config = {
        "name": f"{args.model.lower().replace('/', '_')}_fine_tuned",
        "train_file": str(train_xyz_path),
        "valid_file": str(val_xyz_path) if val_xyz_path else None,
        "max_num_epochs": args.epochs,
        "lr": args.lr,
        "batch_size": effective_batch_size,
        "valid_batch_size": effective_batch_size,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "device": device,
        "seed": args.seed,
        "checkpoints_dir": str(checkpoints_dir),
        "results_dir": str(results_dir),
        "log_dir": str(log_dir),
        "plot": True,
        "plot_frequency": 1,
        "multiheads_finetuning": args.multiheads,
        "energy_weight": args.energy_weight,
        "forces_weight": args.forces_weight,
        "foundation_model": foundation_model_name,
        "E0s": e0s_dict,
    }

    if atomic_numbers_str:
        config["atomic_numbers"] = atomic_numbers_str

    if args.freeze_backbone:
        config["freeze"] = 2  # As per MACE conventions to freeze interaction blocks but keep readout free. If 2 fails, use "-1" or positive block count.

    if args.reinit_head:
        config["foundation_model_readout"] = False 
    else:
        config["foundation_model_readout"] = True

    if has_stress:
        config.update({
            "loss": "universal",
            "stress_weight": args.stress_weight,
            "stress_key": "REF_stress",
            "compute_stress": True,
            "error_table": "PerAtomRMSEstressvirials"
        })

    config_path = output_path / "finetune_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print("\\n=======================================================")
    print("Data successfully prepared!")
    print(f"Training structures: {len(train_structures)}")
    if val_data:
        print(f"Validation structures: {len(val_structures)}")
    print(f"Configuration written to: {config_path}")
    print("=======================================================\\n")
    print("To start training, simply run:")
    print(f"  conda activate mace-agent")
    print(f"  mace_run_train --config {config_path.absolute()}")

if __name__ == "__main__":
    main()
