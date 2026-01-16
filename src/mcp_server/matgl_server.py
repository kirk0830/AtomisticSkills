import sys
import os
import warnings
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress all warnings to prevent protocol pollution
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Silence common blabbermouth libraries
logging.getLogger("matgl").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import json
import logging
from pathlib import Path
import numpy as np
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, List, Union

def recursive_tolist(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: recursive_tolist(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_tolist(x) for x in obj]
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    else:
        return obj

# Import the migrated MatGL wrapper
from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper
from src.utils.mlips.feature_calculators import MatGLCrystalFeatureCalculator
from src.utils.data_augmenter.sampler import StructureSampler

# Initialize FastMCP server
mcp = FastMCP("MatGL")

# Global variables to hold state
wrapper: Optional[MatGLWrapper] = None
sampler: Optional[StructureSampler] = None

@mcp.tool()
def load_model(model_name: str = 'CHGNet-MatPES-PBE-2025.2.10-2.7M-PES', device: str = "auto") -> str:
    """
    Load a MatGL model.
    
    Supported models include:
    - M3GNet: 'M3GNet-MP-2021.2.8-PES'
    - CHGNet: 'CHGNet-MatPES-PBE-2025.2.10-2.7M-PES'
    - TensorNet: 'TensorNet-MatPES-r2SCAN-v2025.1-PES' (DGL-based)
    
    Args:
        model_name: Name of the model to load (e.g., "M3GNet", "CHGNet").
        device: Device to use ("auto", "cpu", "cuda").
        
    Returns:
        Confirmation message.
    
    CRITICAL: This tool must be called before using any other tool to load the model into memory.
    """
    global wrapper, sampler
    try:
        wrapper = MatGLWrapper(model_name=model_name, device=device)
        wrapper.load()
        return f"Successfully loaded MatGL model: {model_name}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

@mcp.tool()
def predict_structure(structure_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Predict energy and forces for a structure.
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
    
    Returns:
        Dictionary containing:
        - "energy": Total potential energy (eV)
        - "forces": Atomic forces (eV/A)
        - "stress": (Optional) Stress tensor if supported by model
        - "charges": (Optional) Atomic charges if supported (e.g., CHGNet)
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    

    return wrapper.static_calculation(structure_data)

@mcp.tool()
def sample_off_equilibrium(
    structure_data: Union[Dict[str, Any], str],
    total_steps: int = 1000,
    temperature: float = 300.0,
    output_dir: str = "sampled_structures",
    target_atoms: int = 75
) -> Dict[str, Any]:
    """
    Sample structures for off-equilibrium calculations (MD, melting, diffusion).
    
    Uses MD simulation with supercell expansion and clustering.
    
    Args:
        structure_data: Initial structure.
        total_steps: Number of MD steps.
        temperature: Temperature in Kelvin.
        output_dir: Directory to save sampled structures.
        target_atoms: Target number of atoms for supercell (50-100).
        
    Returns:
        Dictionary with sampling results and output path.
    """
    global wrapper, sampler
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if sampler is None:
        # Initialize sampler if not already available
        if wrapper is None:
             raise RuntimeError("Model must be loaded first using load_model")
        
        # Create calculator from wrapper
        calc = MatGLCrystalFeatureCalculator(potential=wrapper.model, device=wrapper.device)
        sampler = StructureSampler(calculator=calc)
    
    try:
        # Map model name to specific version if available/alias
        model_name = wrapper.model_name
        from ..utils.mlips.matgl.matgl_wrapper import AVAILABLE_MATGL_MODELS
        for key, val in AVAILABLE_MATGL_MODELS.items():
            if model_name.upper() == key.upper():
                model_name = val
                break

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
        # Validate and convert structure
        # Need to ensure wrapper is loaded
        if not wrapper.is_loaded:
             raise RuntimeError("Wrapper not loaded properly")
             
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms

        result_tuple = sampler.sample_off_equilibrium(
            atoms=atoms,
            total_steps=total_steps,
            output_dir=output_dir,
            target_atoms=target_atoms,
            temperature=temperature
        )
        
        # Debugging return value mismatch
        if not isinstance(result_tuple, (list, tuple)) or len(result_tuple) != 2:
             print(f"DEBUG: sample_off_equilibrium returned {type(result_tuple)} with len {len(result_tuple) if hasattr(result_tuple, '__len__') else 'N/A'}")
             if hasattr(result_tuple, '__len__') and len(result_tuple) > 0:
                 print(f"DEBUG: First element type: {type(result_tuple[0])}")
        
        structures, metadata = result_tuple
        
        return {
            "status": "success",
            "count": len(structures),
            "output_dir": output_dir,
            "metadata": metadata
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Off-equilibrium sampling failed: {str(e)}"}

@mcp.tool()
def sample_near_equilibrium(
    structure_data: Union[Dict[str, Any], str],
    fmax: float = 0.01,
    max_steps: int = 200
) -> Dict[str, Any]:
    """
    Sample structures for near-equilibrium calculations (ground state).
    
    Uses ionic relaxation to find energy minima.
    
    Args:
        structure_data: Initial structure.
        fmax: Force convergence criterion.
        max_steps: Maximum relaxation steps.
        
    Returns:
        Dictionary with sampling results.
    """
    global wrapper, sampler
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if sampler is None:
         calc = wrapper.create_calculator()
         sampler = StructureSampler(calculator=calc)
        
    try:
        # Validate and convert structure
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms

        # Run sampling (expects list of structures)
        structures = sampler.sample_near_equilibrium(
            initial_structures=[atoms],
            fmax=fmax,
            max_steps=max_steps
        )
        
        # Convert result to predictable format (properties not calculated yet on result without calc attached)
        # But we can return the structure
        from pymatgen.io.ase import AseAtomsAdaptor
        result_structs = [AseAtomsAdaptor.get_structure(s).as_dict() for s in structures]
        
        return {
            "status": "success",
            "count": len(structures),
            "structures": result_structs
        }
    except Exception as e:
        return {"error": f"Near-equilibrium sampling failed: {str(e)}"}

@mcp.tool()
def sample_order_disorder(
    structure_data: Union[Dict[str, Any], str],
    n_structures: int = 10,
    target_atoms: int = 50,
    output_dir: str = "sampled_structures/order_disorder"
) -> Dict[str, Any]:
    """
    Sample ordered structures from a disordered input structure.
    
    Args:
        structure_data: Initial disordered structure (must have partial occupancies).
        n_structures: Number of ordered structures to generate.
        target_atoms: Target number of atoms for supercell.
        output_dir: Output directory.
        
    Returns:
        Dictionary with sampling results.
    """
    global wrapper, sampler
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if sampler is None:
         calc = wrapper.create_calculator()
         sampler = StructureSampler(calculator=calc)
        
    try:
        # For order-disorder, we preferably need pymatgen structure with partial occupancies
        # check_structure_data returns ASE atoms which loses partial occupancy info typically
        # unless handled carefully.
        # However, check_structure_data handles dict -> ASE. 
        # If input is dict with pymatgen structure, we should try to keep it as pymatgen structure
        # for this specific tool.
        # BUT wrapper.check_structure_data returns ASE atoms.
        
        # We'll try to reconstruct or use raw input if it looks like pymatgen dict
        from pymatgen.core import Structure
        try:
             # Try assuming input is pymatgen dict directly or wrapped
             if "lattice" in structure_data and "sites" in structure_data:
                 pmg_structure = Structure.from_dict(structure_data)
             else:
                 # Fallback to ASE conversion then back to PMG (might lose disorder info)
                 atoms = wrapper.check_structure_data(structure_data)
                 if isinstance(atoms, dict) and "error" in atoms:
                     return atoms
                 from pymatgen.io.ase import AseAtomsAdaptor
                 pmg_structure = AseAtomsAdaptor.get_structure(atoms)
        except Exception:
             return {"error": "Invalid structure format for order-disorder sampling."}

        # Run sampling
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        structures = sampler.sample_order_disorder(
            atoms=pmg_structure,
            n_structures=n_structures,
            target_atoms=target_atoms,
            output_dir=output_dir
        )
        
        # Save results
        from ase.io import write
        out_file = os.path.join(output_dir, "ordered_structures.xyz")
        write(out_file, structures)
        
        return {
            "status": "success",
            "count": len(structures),
            "output_file": out_file
        }
    except Exception as e:
        return {"error": f"Order-disorder sampling failed: {str(e)}"}

@mcp.tool()
def get_info() -> Dict[str, Any]:
    """
    Get information about the current loaded model.
    """
    global wrapper
    if wrapper is None:
        return {"status": "no_model_loaded"}
    return wrapper.get_model_info()


@mcp.tool()
def relax_structure(
    structure_data: Union[Dict[str, Any], str],
    fmax: float = 0.01,
    steps: int = 500,
    optimizer: str = "FIRE",
    output_dir: str = "./results/matgl/relaxation"
) -> Dict[str, Any]:
    """
    Relax a structure using the loaded MatGL model and MatCalc.
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
        fmax: Force convergence criterion (eV/Ang).
        steps: Maximum number of optimization steps.
        optimizer: Optimizer to use ("FIRE", "BFGS", "LBFGS").
        output_dir: Directory to save results.
        
    Returns:
        Dictionary with relaxation results (energy, final_structure, trajectory_path).
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
        
    try:
        from matcalc import RelaxCalc
        from pymatgen.io.ase import AseAtomsAdaptor
        import os
        
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        os.makedirs(output_dir, exist_ok=True)
        traj_file = f"{output_dir}/relax.traj"
        
        calc = wrapper.create_calculator()
        
        relaxer = RelaxCalc(
            calculator=calc,
            optimizer=optimizer,
            fmax=fmax,
            max_steps=steps,
            traj_file=traj_file,
            interval=1
        )
        
        result = relaxer.calc(atoms)
        
        # Save relaxed structure to CIF
        final_struct = result["final_structure"]
        cif_path = os.path.join(output_dir, "relaxed_structure.cif")
        
        # Handle ASE Atoms vs Pymatgen Structure
        if hasattr(final_struct, "write"): # ASE
            final_struct.write(cif_path)
        elif hasattr(final_struct, "to"): # Pymatgen
            final_struct.to(filename=cif_path)
            
        # Save energy and results to JSON
        json_path = os.path.join(output_dir, "relaxation_results.json")
        energy_val = float(result.get("energy")) if result.get("energy") is not None else None
        results_data = {
            "energy": energy_val,
            "trajectory_path": traj_file,
            "cif_path": cif_path
        }
        
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        return {
            "energy": energy_val,
            "trajectory_path": traj_file,
            "cif_path": cif_path,
            "json_path": json_path
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Relaxation failed: {str(e)}"}

@mcp.tool()
def run_md(
    structure_data: Union[Dict[str, Any], str],
    temperature: float = 300,
    steps: int = 1000,
    timestep: float = 1.0,
    ensemble: str = "nvt",
    log_interval: int = 100,
    output_dir: str = "./results/matgl/md"
) -> Dict[str, Any]:
    """
    Run molecular dynamics simulation using MatCalc.
    
    Args:
        structure_data: Structure in partial dictionary format.
        temperature: Temperature in Kelvin.
        steps: Number of steps.
        timestep: Timestep in fs.
        ensemble: Ensemble "nve", "nvt" (Nose-Hoover) or "npt" (NPT).
                  Also supports variants like "nvt_langevin", "nvt_andersen", "npt_berendsen", "npt_mtk".
        log_interval: Interval for logging to trajectory and logfile.
        output_dir: Directory to save results.
        
    Returns:
        Dictionary with MD results (trajectory_path, final_structure).
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
        
    try:
        from matcalc import MDCalc
        from pymatgen.io.ase import AseAtomsAdaptor
        import os
        
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        os.makedirs(output_dir, exist_ok=True)
        
        formula = atoms.get_chemical_formula()
        filename_base = f"{formula}_{temperature}K_{ensemble}"
        traj_path = os.path.join(output_dir, f"{filename_base}.traj")
        log_path = os.path.join(output_dir, f"{filename_base}.log")
        
        calc = wrapper.create_calculator()
        
        md_runner = MDCalc(
            calculator=calc,
            ensemble=ensemble.lower(),
            temperature=temperature,
            timestep=timestep,
            steps=steps,
            trajfile=traj_path,
            logfile=log_path,
            loginterval=log_interval,
            relax_structure=False
        )
        
        result = md_runner.calc(atoms)
        
        def get_structure_dict(struct_or_atoms):
            if hasattr(struct_or_atoms, "as_dict"):
                return struct_or_atoms.as_dict()
            return AseAtomsAdaptor.get_structure(struct_or_atoms).as_dict()
        
        return {
            "trajectory_path": traj_path,
            "log_path": log_path,
            "final_structure": get_structure_dict(result["final_structure"]),
            "final_energy": result.get("energy")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"MD simulation failed: {str(e)}"}

@mcp.tool()
def calculate_neb(
    start_structure: Union[Dict[str, Any], str],
    end_structure: Union[Dict[str, Any], str],
    n_images: int = 7,
    output_dir: str = "./results/matgl/neb",
    fmax: float = 0.1,
    climb: bool = True
) -> Dict[str, Any]:
    """
    Calculate NEB barrier between two structures using MatCalc.
    
    Args:
        start_structure: Initial structure dict.
        end_structure: Final structure dict.
        n_images: Number of intermediate images.
        output_dir: Directory to save results.
        fmax: Force convergence.
        climb: Use CI-NEB.
        
    Returns:
        Dictionary with barrier and reaction coordinates.
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
        
    try:
        from matcalc import NEBCalc
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core import Structure
        import os
        
        start_atoms = wrapper.check_structure_data(start_structure)
        end_atoms = wrapper.check_structure_data(end_structure)
        
        if (isinstance(start_atoms, dict) and "error" in start_atoms) or \
           (isinstance(end_atoms, dict) and "error" in end_atoms):
           return {"error": "Invalid start or end structure."}
           
        start_struct = AseAtomsAdaptor.get_structure(start_atoms)
        end_struct = AseAtomsAdaptor.get_structure(end_atoms)
        
        os.makedirs(output_dir, exist_ok=True)
        
        calc = wrapper.create_calculator()
        
        neb_calc = NEBCalc(
            calculator=calc,
            traj_folder=output_dir,
            fmax=fmax,
            climb=climb
        )
        
        result = neb_calc.calc_images(
            start_struct=start_struct,
            end_struct=end_struct,
            n_images=n_images
        )
        
        mep = result.get("mep")
        mep_dict = mep.as_dict() if mep else {}
        
        return {
            "barrier": result.get("barrier"),
            "max_force": result.get("force"),
            "mep": mep_dict
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"NEB failed: {str(e)}"}

@mcp.tool()
def calculate_phonon(
    structure_data: Union[Dict[str, Any], str],
    supercell_matrix: List[List[int]] = ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
    t_step: float = 10,
    t_max: float = 1000,
    t_min: float = 0,
    output_dir: str = "./results/matgl/phonon"
) -> Dict[str, Any]:
    """
    Calculate Phonon properties using MatCalc.
    
    Args:
        structure_data: Structure dict.
        supercell_matrix: Supercell matrix (3x3).
        t_step, t_max, t_min: Temperature range.
        output_dir: Directory to save results.
        
    Returns:
        Dictionary with phonon results.
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
        
    try:
        from matcalc import PhononCalc
        import os
        
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        os.makedirs(output_dir, exist_ok=True)
        
        calc = wrapper.create_calculator()
        
        phonon_calc = PhononCalc(
            calculator=calc,
            supercell_matrix=supercell_matrix,
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            write_phonon=os.path.join(output_dir, "phonon.yaml"),
            write_band_structure=os.path.join(output_dir, "band_structure.yaml"),
            write_total_dos=os.path.join(output_dir, "total_dos.dat")
        )
        
        result = phonon_calc.calc(atoms)
        
        thermal_props = result.get("thermal_properties", {})
        
        return {
            "thermal_properties": thermal_props,
            "output_dir": output_dir
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Phonon calculation failed: {str(e)}"}

@mcp.tool()
def calculate_qha(
    structure_data: Union[Dict[str, Any], str],
    t_step: float = 10,
    t_max: float = 1000,
    t_min: float = 0,
    eos: str = "vinet",
    output_dir: str = "./results/matgl/qha"
) -> Dict[str, Any]:
    """
    Calculate QHA thermal properties using MatCalc.
    
    Args:
        structure_data: Structure dict.
        t_step, t_max, t_min: Temperature range.
        eos: Equation of state ("vinet", "birch_murnaghan", "murnaghan").
        output_dir: Directory to save results.
        
    Returns:
        Dictionary with QHA results.
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    try:
        from matcalc import QHACalc
        import os
        
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        os.makedirs(output_dir, exist_ok=True)
        
        calc = wrapper.create_calculator()
        
        qha_calc = QHACalc(
            calculator=calc,
            t_step=t_step,
            t_max=t_max,
            t_min=t_min,
            eos=eos,
            write_gibbs_temperature=os.path.join(output_dir, "gibbs_temperature.dat"),
            write_thermal_expansion=os.path.join(output_dir, "thermal_expansion.dat")
        )
        
        result = qha_calc.calc(atoms)
        
        output = {
            "temperatures": result.get("temperatures", []).tolist() if hasattr(result.get("temperatures"), "tolist") else result.get("temperatures"),
            "volumes": result.get("volumes"),
            "gibbs_free_energies": result.get("gibbs_free_energies").tolist() if hasattr(result.get("gibbs_free_energies"), "tolist") else result.get("gibbs_free_energies"),
            "thermal_expansion_coefficients": result.get("thermal_expansion_coefficients").tolist() if hasattr(result.get("thermal_expansion_coefficients"), "tolist") else result.get("thermal_expansion_coefficients"),
            "output_dir": output_dir
        }
        return output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"QHA calculation failed: {str(e)}"}

@mcp.tool()
def fine_tune_model(
    training_data: List[Dict[str, Any]],
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    output_dir: Optional[str] = None,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fine-tune the current MatGL model.
    
    Args:
        training_data: List of training samples. Each sample must have:
                       - 'structure': Dict representation (ASE atoms dict or pymatgen dict)
                       - 'energy': Total potential energy (float, eV)
                       - 'forces': Atomic forces (list/array, eV/A)
                       - 'stress': (Optional) Stress tensor (list/array)
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        batch_size: Batch size.
        output_dir: Directory to save results.
        training_config: Optional dictionary for advanced training configuration.
                         See .agent/rules/fine-tuning-guide.md for available keys (e.g., {"freeze_backbone": True}).
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        
        # Pre-process training data to convert dict structures to ASE Atoms
        processed_data = []
        for item in training_data:
            struct_data = item['structure']
            atoms = wrapper.check_structure_data(struct_data)
            if isinstance(atoms, dict) and "error" in atoms:
                return atoms
            
            new_item = item.copy()
            new_item['structure'] = atoms
            processed_data.append(new_item)
            
        config = {
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
        if training_config:
            config.update(training_config)
        
        result = wrapper.fine_tune(
            training_data=processed_data,
            training_config=config,
            output_dir=output_dir
        )
        
        # Add extra info
        result["is_fine_tuned"] = True
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Fine-tuning failed: {str(e)}"}

if __name__ == "__main__":
    # Run the server
    mcp.run()
