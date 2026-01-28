import sys
import os
import io

# --- ROBUST STDOUT/STDERR ISOLATION ---
try:
    # 1. Save the REAL stdout (the one used for MCP communication)
    mcp_stdout_fd = os.dup(1)
    
    # 2. Redirect EVERYTHING to a persistent log file
    log_file_path = "/tmp/mace_mcp_server.log"
    log_f = open(log_file_path, "a", buffering=1)

    # 3. Redirect system-level FD 1 and FD 2
    os.dup2(log_f.fileno(), 1)
    os.dup2(log_f.fileno(), 2)

    # 4. Patch Python sys.stdout to the original pipe handle
    sys.stdout = io.TextIOWrapper(
        os.fdopen(mcp_stdout_fd, 'wb', buffering=0), 
        encoding='utf-8', 
        line_buffering=True
    )
    
    # 5. Patch Python sys.stderr and others to the log file
    sys.stderr = log_f
except Exception:
    pass 
# --------------------------------------

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import warnings
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, Union, List
from src.utils.serialization_utils import recursive_tolist
from src.utils.research_utils import get_current_research_dir
import traceback

# Suppress all warnings to prevent protocol pollution
warnings.filterticks = 0 # Dummy to test if we can edit
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="matplotlib")
os.environ["PYTHONWARNINGS"] = "ignore"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MACE-Server")

# Initialize FastMCP server
mcp = FastMCP("MACE")

# Global variables to hold state
wrapper: Optional[Any] = None
sampler: Optional[Any] = None

from src.utils.serialization_utils import recursive_tolist

# ---------------------------------------------------------------------
# Worker Process for MD Isolation
# ---------------------------------------------------------------------



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
        model_name: Name of the model to load (default: "MACE-OMAT-0-small").
        device: Device to use ("auto", "cpu", "cuda").
        task_name: Optional task name that sets the model's head. 
                  Supported options for multi-head models (MACE-MH): 
                  'omat_pbe' (default), 'matpes_r2scan', 'omol', 'spice_wB97M', 'oc20_usemppbe'.
        
    Returns:
        Confirmation message.
    
    CRITICAL: This tool must be called before using any other tool to load the model into memory.
    """
    global wrapper
    import contextlib
    
    # Isolate execution from stdout/stderr to prevent MCP protocol violation
    # We redirect stdout to sys.stderr (which logs to file) so we don't lose the output,
    # but we keep the actual stdout (MCP pipe) clean.
    # Note: sys.stderr is already redirected to a log file in the header of this script.
    try:
        # We use sys.stderr as the target for stdout. 
        # Since sys.stderr is a file object (log_file), this merges stdout into the log.
        with contextlib.redirect_stdout(sys.stderr):
            from src.utils.mlips.mace.mace_wrapper import MACEWrapper
            wrapper = MACEWrapper(model_name=model_name, device=device, head=task_name)
            wrapper.load(model_path=model_name if os.path.exists(model_name) else None)
            msg = f"Successfully loaded MACE model: {model_name}"
    except Exception as e:
        import traceback
        # Log full traceback to the log file (sys.stderr)
        traceback.print_exc(file=sys.stderr)
        msg = f"Error loading model: {str(e)}"
    
    return msg

@mcp.tool()
def predict_structure(structure_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Predict energy, forces, and stress for a structure.
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
    
    Returns:
        Dict: {'energy': eV, 'forces': eV/A, 'stress': eV/A^3}
    """
    global wrapper
    import contextlib
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    with contextlib.redirect_stdout(sys.stderr):
        return recursive_tolist(wrapper.static_calculation(structure_data))

@mcp.tool()
def predict_atomic_features(structure_data: Union[Dict[str, Any], str], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Predict atomic latent features (descriptors) for a structure.
    Automatically saves features to the current research directory.
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
        output_path: Optional custom path for saving features. If not provided, auto-generates
                     based on structure filename (e.g., 'solid.cif' -> 'solid_features.json').
    
    Returns:
        Dict: {
            'atomic_features': [[...], ...], 
            'feature_dim': int, 
            'num_atoms': int,
            'saved_path': str  # Path to saved JSON file
        }
    """
    global wrapper
    import contextlib
    import json
    from pathlib import Path
    from src.utils.research_utils import get_current_research_dir
    
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    # Get features
    with contextlib.redirect_stdout(sys.stderr):
        result = wrapper.predict_atomic_features(structure_data)
    
    if "error" in result:
        return result
    
    # Save to research directory
    try:
        # Use custom path if provided, otherwise auto-generate
        if output_path:
            save_path = Path(output_path)
        else:
            research_dir = get_current_research_dir()
            
            # Generate filename based on structure path or use generic name
            if isinstance(structure_data, str):
                struct_path = Path(structure_data)
                feature_filename = f"{struct_path.stem}_features.json"
            else:
                feature_filename = "atomic_features.json"
            
            save_path = research_dir / feature_filename
        
        # Save features
        with open(save_path, 'w') as f:
            json.dump(recursive_tolist(result), f)
        
        # Add saved path to result
        result["saved_path"] = str(save_path)
        
        # Remove heavy data from return value
        if "atomic_features" in result:
            del result["atomic_features"]
        
        return result
        
    except Exception as e:
        # If saving fails, still return the features but with error info
        result["save_error"] = f"Failed to save features: {str(e)}"
        return result

@mcp.tool()
def relax_structure(
    structure_data: Union[Dict[str, Any], str],
    fmax: float = 0.01,
    steps: int = 500,
    optimizer: str = "FIRE",
    relax_cell: bool = True,
    output_dir: Optional[str] = None,
    fixed_atoms: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Relax a structure using the loaded MACE model.
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
        fmax: Force convergence criterion (eV/Ang).
        steps: Maximum number of optimization steps.
        optimizer: Optimizer to use ("FIRE", "BFGS", "LBFGS").
        output_dir: Directory to save results.
        fixed_atoms: List of indices of atoms to keep fixed during relaxation.
        
    Returns:
        Dictionary with relaxation results (energy, final_structure, trajectory_path).
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    import contextlib
    import json
    import os
    from src.utils.research_utils import get_current_research_dir

    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "relaxation")
    os.makedirs(output_dir, exist_ok=True)

    with contextlib.redirect_stdout(sys.stderr):
        result = wrapper.relax_structure(
            structure_data=structure_data,
            fmax=fmax,
            steps=steps,
            optimizer=optimizer,
            relax_cell=relax_cell,
            output_dir=output_dir,
            fixed_atoms=fixed_atoms
        )

    # Sanitize output
    if "final_structure" in result:
        # Save to file
        final_struct_path = os.path.join(output_dir, "relaxed_structure.json")
        
        # Convert to dict if needed
        struct_data = result["final_structure"]
        # Handle various structure objects safely
        try:
            if hasattr(struct_data, "as_dict"):
                struct_dict = struct_data.as_dict()
            else:
                from pymatgen.io.ase import AseAtomsAdaptor
                # Check if it's already a dict
                if isinstance(struct_data, dict):
                    struct_dict = struct_data
                else:
                    struct_dict = AseAtomsAdaptor.get_structure(struct_data).as_dict()
            
            with open(final_struct_path, "w") as f:
                json.dump(struct_dict, f)
                
            result["final_structure_path"] = final_struct_path
            del result["final_structure"]
        except Exception as e:
            result["save_error"] = str(e)
            # Keep original structure if save failed, or remove it?
            # Better to remove it to prevent crash, user has error message
            if "final_structure" in result:
                 del result["final_structure"]

    return recursive_tolist(result)

@mcp.tool()
def run_md(
    structure_data: Union[Dict[str, Any], str],
    temperature: float = 300.0,
    steps: int = 1000,
    timestep: float = 1.0,
    ensemble: str = "nvt",
    log_interval: int = 10,
    pressure: float = 0.0,
    pressure_mask: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    monitor: bool = False,
    monitor_type: str = "melting"
) -> Dict[str, Any]:
    """
    Run molecular dynamics simulation using MatCalc.
    
    Args:
        structure_data: Structure data.
        temperature: Temperature in Kelvin.
        steps: Number of steps.
        timestep: Timestep in fs.
        ensemble: Ensemble "nve", "nvt" (Nose-Hoover), or "npt" (NPT).
                  Also supports variants like "nvt_langevin", "nvt_andersen", "npt_berendsen", "npt_mtk".
        log_interval: Interval for logging to trajectory and logfile.
        pressure: Target pressure in bar (for NPT).
        pressure_mask: Mask for anisotropic NPT (e.g., [1, 0, 0] for 1D).
        output_dir: Directory to save results.
        monitor: Whether to enable automatic MD monitoring.
        monitor_type: Type of monitor ("melting").
        
    Returns:
        Dictionary with MD results.
    """
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}

    # Setup Directory
    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "md")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Run MD via wrapper's unified run_md (supports monitoring callbacks)
        result = wrapper.run_md(
            structure_data=structure_data,
            temperature=temperature,
            steps=steps,
            timestep=timestep,
            ensemble=ensemble,
            log_interval=log_interval,
            pressure=pressure,
            pressure_mask=pressure_mask,
            output_dir=output_dir,
            monitor=monitor,
            monitor_type=monitor_type
        )
        
        if "error" in result:
            return result
            
        return recursive_tolist(result)
            
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": f"MD execution failed: {str(e)}", "traceback": traceback.format_exc()}

@mcp.tool()
def calculate_phonon(
    structure_data: Union[Dict[str, Any], str],
    supercell_matrix: Optional[List[List[int]]] = None,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 10.0,
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
    import contextlib
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    try:
        from matcalc import PhononCalc
        import os
        
        with contextlib.redirect_stdout(sys.stderr):
            atoms = wrapper.check_structure_data(structure_data)
            if isinstance(atoms, dict) and "error" in atoms:
                return atoms
                
            if not output_dir:
                output_dir = str(get_current_research_dir() / "mace" / "phonon")

            os.makedirs(output_dir, exist_ok=True)
            
            calc = wrapper.create_calculator()
            
            phonon_calc = PhononCalc(
                calculator=calc,
                supercell_matrix=supercell_matrix or [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                t_step=t_step,
                t_max=t_max,
                t_min=t_min,
                write_phonon=os.path.join(output_dir, "phonon.yaml"),
                write_band_structure=os.path.join(output_dir, "band_structure.yaml"),
                write_total_dos=os.path.join(output_dir, "total_dos.dat")
            )
            
            result = phonon_calc.calc(atoms)
        
        return {
            "thermal_properties": result.get("thermal_properties", {}),
            "output_dir": output_dir
        }
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": f"Phonon calculation failed: {str(e)}"}

@mcp.tool()
def calculate_qha(
    structure_data: Union[Dict[str, Any], str],
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 10.0,
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
    import contextlib
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    try:
        from matcalc import QHACalc
        import os
        
        with contextlib.redirect_stdout(sys.stderr):
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
        
        return {
            "temperatures": recursive_tolist(result.get("temperatures", [])),
            "volumes": recursive_tolist(result.get("volumes", [])),
            "gibbs_free_energies": recursive_tolist(result.get("gibbs_free_energies", [])),
            "thermal_expansion_coefficients": recursive_tolist(result.get("thermal_expansion_coefficients", [])),
            "output_dir": output_dir
        }
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": f"QHA calculation failed: {str(e)}"}

@mcp.tool()
def calculate_neb(
    start_structure: Union[Dict[str, Any], str],
    end_structure: Union[Dict[str, Any], str],
    n_images: int = 5,
    fmax: float = 0.05,
    climb: bool = True,
    output_dir: Optional[str] = None
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
    import contextlib
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    try:
        from matcalc import NEBCalc
        from pymatgen.io.ase import AseAtomsAdaptor
        import os
        
        with contextlib.redirect_stdout(sys.stderr):
            start_atoms = wrapper.check_structure_data(start_structure)
            end_atoms = wrapper.check_structure_data(end_structure)
            
            if (isinstance(start_atoms, dict) and "error" in start_atoms) or \
               (isinstance(end_atoms, dict) and "error" in end_atoms):
               return {"error": "Invalid start or end structure."}
               
            start_pmg = AseAtomsAdaptor.get_structure(start_atoms)
            end_pmg = AseAtomsAdaptor.get_structure(end_atoms)
            
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
                start_struct=start_pmg,
                end_struct=end_pmg,
                n_images=n_images
            )
            
            mep = result.get("mep")
            mep_dict = mep.as_dict() if mep else {}
        
        # Save MEP to file
        import json
        mep_path = os.path.join(output_dir, "neb_mep.json")
        with open(mep_path, "w") as f:
             json.dump(mep_dict, f)

        return {
            "barrier": result.get("barrier"),
            "max_force": result.get("force"),
            "mep_path": mep_path
        }
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": f"NEB failed: {str(e)}"}

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
    
    if sampler is None:
         from src.utils.mlips.mace.mace_wrapper import MaceCrystalFeatureCalculator
         from src.utils.sampler import StructureSampler
         
         # Initialize calculator and sampler
         base_calc = wrapper.get_calculator()
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
        if not output_dir:
             output_dir = str(get_current_research_dir() / "mace" / "sampled_structures")

        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
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
                 from src.utils.structure_utils import save_structure
                 save_structure(s, filepath)
                 saved_files.append(filepath)
             except Exception as e:
                 logger.error(f"Error saving structure {i}: {e}")
                 
             serialized_structures.append(AseAtomsAdaptor.get_structure(s).as_dict())
             
        return {
            "sampled_structures_count": len(structures),
            "output_dir": output_dir,
            "saved_files": saved_files,
            "metadata": metadata,
        }
        
    except Exception as e:
        import traceback
        logger.error(f"Sampling failed: {traceback.format_exc()}")
        return {"error": f"Sampling failed: {str(e)}"}

@mcp.tool()
def fine_tune_model(
    training_data_path: str,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    output_dir: Optional[str] = None,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fine-tune the current MACE model.
    
    Args:
        training_data_path: Path to a JSON file containing the training data list.
                             Each sample must have:
                               - 'structure': Dict (ASE atoms or pymatgen format)
                               - 'energy': Total potential energy (float, eV)
                               - 'forces': Atomic forces (list/array, eV/A)
                               - 'stress': (Optional) Stress tensor (list/array) in eV/A^3.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        output_dir: Directory to save the fine-tuned model.
        training_config: Optional dictionary for advanced configuration.
        
    Returns:
        Dictionary with fine-tuning results.
    """
    import json
    import contextlib
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "fine_tuning")
        
    # Load training data
    try:
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load training data from {training_data_path}: {e}"}
        
    # Prepare config
    config = training_config or {}
    config.update({
        "max_epochs": epochs,
        "learning_rate": learning_rate
    })
        
    with contextlib.redirect_stdout(sys.stderr):
        result = wrapper.fine_tune(
            training_data=training_data,
            output_dir=output_dir,
            training_config=config
        )
    return recursive_tolist(result)

@mcp.tool()
def get_info() -> Dict[str, Any]:
    """
    Get information about the current loaded model.
    
    Returns:
        Dictionary with model status, name, device, and head.
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"status": "No model loaded"}
    
    return {
        "status": "Model loaded",
        "model_name": wrapper.model_name,
        "device": wrapper.device,
        "head": wrapper.head
    }

if __name__ == "__main__":
    mcp.run()
