import sys
import os
import io

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

import logging
import warnings
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, Union, List

# Suppress all warnings to prevent protocol pollution
warnings.filterticks = 0 # Dummy to test if we can edit
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="matplotlib")
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

from src.utils.serialization_utils import recursive_tolist


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
        return wrapper.static_calculation(structure_data)

@mcp.tool()
def relax_structure(
    structure_data: Union[Dict[str, Any], str],
    fmax: float = 0.01,
    steps: int = 500,
    optimizer: str = "FIRE",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Relax a structure using the loaded MACE model.
    
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
    import contextlib
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "relaxation")
    
    with contextlib.redirect_stdout(sys.stderr):
        result = wrapper.relax_structure(
            structure_data=structure_data,
            fmax=fmax,
            steps=steps,
            optimizer=optimizer,
            output_dir=output_dir
        )
    return recursive_tolist(result)

@mcp.tool()
def run_md(
    structure_data: Union[Dict[str, Any], str],
    temperature: float = 300.0,
    steps: int = 1000,
    timestep: float = 1.0,
    ensemble: str = "nvt",
    log_interval: int = 10,
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
    import contextlib
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "md")
        
    with contextlib.redirect_stdout(sys.stderr):
        result = wrapper.run_md(
            structure_data=structure_data,
            temperature=temperature,
            steps=steps,
            timestep=timestep,
            ensemble=ensemble,
            log_interval=log_interval,
            output_dir=output_dir
        )
    return recursive_tolist(result)

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
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "phonon")
        
    result = wrapper.calculate_phonon(
        structure_data=structure_data,
        supercell_matrix=supercell_matrix,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        output_dir=output_dir
    )
    return recursive_tolist(result)

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
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "qha")
        
    result = wrapper.calculate_qha(
        structure_data=structure_data,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        eos=eos,
        output_dir=output_dir
    )
    return recursive_tolist(result)

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
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    if not output_dir:
        output_dir = str(get_current_research_dir() / "mace" / "neb")
        
    result = wrapper.calculate_neb(
        start_structure=start_structure,
        end_structure=end_structure,
        n_images=n_images,
        fmax=fmax,
        climb=climb,
        output_dir=output_dir
    )
    return recursive_tolist(result)

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
                 s.write(filepath)
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
