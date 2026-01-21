import sys
import os
import io
import logging

# --- ROBUST STDOUT ISOLATION ---
# 1. Save the REAL stdout (the one used for MCP communication)
try:
    # Duplicate original stdout (FD 1) to a private handle
    mcp_stdout_fd = os.dup(1)
    
    # 2. Redirect system-level FD 1 to /dev/null
    # This silences library calls that write directly to stdout (the source of the 'G' error)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)

    # 3. Patch Python's sys.stdout to use the saved handle
    # stdio_server uses sys.stdout.buffer to write JSON-RPC messages.
    # We wrap the saved FD in a binary buffer and a TextIOWrapper.
    sys.stdout = io.TextIOWrapper(
        os.fdopen(mcp_stdout_fd, 'wb', buffering=0), 
        encoding='utf-8', 
        line_buffering=True
    )
except Exception:
    pass
# -------------------------------

import warnings
import warnings
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, Union, List

# Suppress all warnings to prevent protocol pollution
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MACE-Server")

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize FastMCP server
mcp = FastMCP("MACE")
from src.utils.research_utils import get_current_research_dir

# Global variables to hold state
wrapper: Optional[Any] = None
sampler: Optional[Any] = None

def recursive_tolist(obj):
    import numpy as np
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

# ... (removed top-level wrapper/sampler imports) ...


# Initialize FastMCP server
mcp = FastMCP("MACE")
from src.utils.research_utils import get_current_research_dir

# Global variables to hold state
wrapper: Optional[MACEWrapper] = None
sampler: Optional[StructureSampler] = None

@mcp.tool()
def load_model(model_name: str = "MACE-OMAT-0-small", device: str = "auto", task_name: Optional[str] = None) -> str:
    """
    Load a MACE model.
    
    Supported models include:
    - MACE-MH (Multi-Head): 'MACE-MH-1' (latest), 'MACE-MH-0'
    - MACE-MP (Materials Project): 'MACE-MP-small', 'MACE-MP-medium', 'MACE-MP-large'
    - MACE-OMAT (OMAT dataset): 'MACE-OMAT-0-small', 'MACE-OMAT-0-medium'
    - MACE-MATPES: 'MACE-MATPES-PBE-0', 'MACE-MATPES-R2SCAN-0'
    - MACE-OFF (Organic): 'MACE-OFF23-small', 'MACE-OFF23-medium', 'MACE-OFF23-large'
    - Special: 'MACE-ANI-CC', 'MACE-OMOL-extra-large'
    
    Args:
        model_name: Name of the model to load (e.g., 'MACE-MH-1').
        device: Device to use ("auto", "cpu", "cuda").
        task_name: Optional task name that sets the model's head. 
                  Supported options for multi-head models (MACE-MH): 
                  'omat_pbe' (default), 'matpes_r2scan', 'omol', 'spice_wB97M', 'oc20_usemppbe'.
        
    Returns:
        Confirmation message.
    
    CRITICAL: This tool must be called before using any other tool to load the model into memory.
    """
    global wrapper
    try:
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        wrapper = MACEWrapper(model_name=model_name, device=device, head=task_name)
        wrapper.load(model_path=model_name if os.path.exists(model_name) else None)
        
        return f"Successfully loaded MACE model: {model_name}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

@mcp.tool()
def predict_structure(structure_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Predict energy, forces, and stress for a structure.
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
    
    Returns:
        Dictionary containing 'energy', 'forces', and optionally 'stress'.
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    return wrapper.static_calculation(structure_data)

@mcp.tool()
def fine_tune_model(
    training_data_path: str,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    output_dir: Optional[str] = None,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fine-tune the current MACE model.
    
    Performs transfer learning using the `mace-agent` environment.
    Generates training history plots and CSV files in the output directory.
    
    Args:
        training_data_path: Path to a JSON file containing the training data list.
                             Each sample must have:
                               - 'structure': Dict (ASE atoms or pymatgen format)
                               - 'energy': Total potential energy (float, eV)
                               - 'forces': Atomic forces (list/array, eV/A)
                               - 'stress': (Optional) Stress tensor for bulk systems
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        output_dir: Directory to save the fine-tuned model and results.
                    If None, a temporary directory is used.
        training_config: Optional dictionary for advanced training configuration.
                         See .agent/rules/fine-tuning-guide.md for available keys (e.g., {"freeze_backbone": True, "multiheads_finetuning": False}).
        
    Returns:
        Dictionary containing:
        - "is_fine_tuned": Boolean status
        - "training_history": Training metrics history
        - "final_metrics": Final epoch metrics
        - "model_saved_to": Path to the saved model
        - "plot_path": Path to the generated training plot
        - "csv_path": Path to the training history CSV
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
        
    try:
        with open(training_data_path, 'r') as f:
            data = json.load(f)
            # Handle if wrapped in "results" key (from materials_tools)
            if isinstance(data, dict) and "results" in data:
                training_data = data["results"]
            elif isinstance(data, list):
                training_data = data
            else:
                return {"error": "Invalid JSON format in training_data_path. Expected list or dict with 'results'."}
        
        if not training_data:
            return {"error": f"Training data file {training_data_path} is empty."}

        # Normalize keys (handle dft_server output format)
        normalized_data = []
        for sample in training_data:
            new_sample = sample.copy()
            # Map keys
            if "final_structure" in sample and "structure" not in sample:
                new_sample["structure"] = sample["final_structure"]
            if "final_energy" in sample and "energy" not in sample:
                new_sample["energy"] = sample["final_energy"]
            
            normalized_data.append(new_sample)
            
        # Combine basic args with extended config
        final_config = {
            "max_epochs": epochs,
            "learning_rate": learning_rate
        }
        if training_config:
            final_config.update(training_config)
        
        result = wrapper.fine_tune(
            training_data=normalized_data,
            training_config=final_config,
            output_dir=output_dir if output_dir else str(get_current_research_dir() / "mace" / "fine_tuning")
        )
        
        return result
    except Exception as e:
        return {"error": f"Fine-tuning failed: {str(e)}"}

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
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Relax a structure using the loaded MACE model and MatCalc.
    
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
        
        # Helper to get ASE Atoms (reuse base check/convert)
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        if not output_dir:
            output_dir = str(get_current_research_dir() / "mace" / "relaxation")
            
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
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run molecular dynamics simulation using MatCalc.
    
    Args:
        structure_data: Structure in partial dictionary format.
        temperature: Temperature in Kelvin.
        steps: Number of steps.
        timestep: Timestep in fs.
        ensemble: Ensemble "nve", "nvt" (Nose-Hoover), or "npt" (NPT). 
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
            
        if not output_dir:
            output_dir = str(get_current_research_dir() / "mace" / "md")
            
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
    output_dir: Optional[str] = None,
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
        
        # Convert start/end to Pymatgen Structure (NEBCalc prefers Structure for calc_images)
        start_atoms = wrapper.check_structure_data(start_structure)
        end_atoms = wrapper.check_structure_data(end_structure)
        
        if (isinstance(start_atoms, dict) and "error" in start_atoms) or \
           (isinstance(end_atoms, dict) and "error" in end_atoms):
           return {"error": "Invalid start or end structure."}
           
        start_struct = AseAtomsAdaptor.get_structure(start_atoms)
        end_struct = AseAtomsAdaptor.get_structure(end_atoms)
        
        if not output_dir:
            output_dir = str(get_current_research_dir() / "mace" / "neb")

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
        
        # result contains barrier, force, mep
        # mep object needs serialization
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
    output_dir: Optional[str] = None
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
            
        if not output_dir:
            output_dir = str(get_current_research_dir() / "mace" / "phonon")

        os.makedirs(output_dir, exist_ok=True)
        
        calc = wrapper.create_calculator()
        
        # Paths for output
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
        
        # phonopy object is not JSON serializable.
        # result keys: phonon, thermal_properties
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
    output_dir: Optional[str] = None
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
            
        if not output_dir:
            output_dir = str(get_current_research_dir() / "mace" / "qha")

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
        
        # QHA result keys: qha, volumes, electronic_energies, temperatures, thermal_expansion_coefficients, ...
        # Filter serializable ones
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
def sample_off_equilibrium(
    structure_data: Union[Dict[str, Any], str],
    total_steps: int = 1000,
    temperature: float = 300.0,
    output_dir: Optional[str] = None,
    target_atoms: int = 75,
    num_samples: int = 20,
    time_step: Optional[float] = None
) -> Dict[str, Any]:
    """
    Sample structures for off-equilibrium calculations (MD, diffusion).
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
        total_steps: Number of MD steps.
        temperature: Temperature in Kelvin.
        output_dir: Directory to save sampled structures.
        target_atoms: Target number of atoms for supercell (50-100).
        num_samples: Number of structures to sample (default: 20).
        time_step: Time step in fs (default: None, auto-detected based on elements).
        
    Returns:
        Dictionary with sampling results and output path.
    """
    global wrapper, sampler
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    # Lazily initialize sampler if needed
    if sampler is None:
         from src.utils.mlips.mace.mace_wrapper import MACEWrapper
         from src.utils.mlips.feature_calculators import MaceCrystalFeatureCalculator
         from src.utils.data_augmenter.sampler import StructureSampler
         
         base_calc = wrapper.create_calculator()
         calc = MaceCrystalFeatureCalculator(mace_calculator=base_calc)
         sampler = StructureSampler(calculator=calc)
        
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        import os
        
        # Helper to get ASE Atoms
        atoms = wrapper.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        # Run sampling
        # Note: model_name is no longer needed/used by sample_off_equilibrium
        if not output_dir:
             output_dir = str(get_current_research_dir() / "mace" / "sampled_structures")

        structures, metadata = sampler.sample_off_equilibrium(
            atoms=atoms,
            total_steps=total_steps,
            output_dir=output_dir,
            target_atoms=target_atoms,
            temperature=temperature,
            num_samples=num_samples,
            time_step=time_step
        )
        
        # Serialize structures to dicts and save to files
        serialized_structures = []
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for i, s in enumerate(structures):
             # Save to file
             filename = f"structure_{i:03d}.cif"
             filepath = os.path.join(output_dir, filename)
             try:
                 # Use ASE write for CIF
                 if hasattr(s, "write"):
                     s.write(filepath)
                 else:
                     # Adapt to ASE if needed (unlikely if coming from sampler)
                     # But sampler might return Atoms
                     s.write(filepath)
                 saved_files.append(filepath)
             except Exception as e:
                 print(f"Error saving structure {i}: {e}")
                 
             serialized_structures.append(AseAtomsAdaptor.get_structure(s).as_dict())
             
        return {
            "sampled_structures_count": len(structures),
            "output_dir": output_dir,
            "saved_files": saved_files,
            "metadata": metadata,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Sampling failed: {str(e)}"}


if __name__ == "__main__":
    mcp.run()
