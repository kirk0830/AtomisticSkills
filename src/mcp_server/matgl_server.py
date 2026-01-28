import os
import sys

# set MATGL_BACKEND to DGL by default for better performance and TensorNet support
if "MATGL_BACKEND" not in os.environ:
    os.environ["MATGL_BACKEND"] = "DGL"

# Silence Wandb and suppress warnings to prevent protocol pollution
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"

import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.mcp_utils import setup_mcp_stdout, run_fastmcp_server

# Setup stdout redirection for MCP
mcp_pipe_binary = setup_mcp_stdout()

import contextlib
import warnings
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, List, Union
from src.utils.serialization_utils import recursive_tolist
from src.utils.research_utils import get_current_research_dir

# Suppress all warnings to prevent protocol pollution
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("MatGL-Server")

# Initialize FastMCP server
mcp = FastMCP("MatGL")

# Global variables to hold state
wrapper: Optional[Any] = None
sampler: Optional[Any] = None


@mcp.tool()
def load_model(model_name: str = 'CHGNet-MatPES-PBE-2025.2.10-2.7M-PES', device: str = "auto") -> str:
    """
    Load a MatGL model.
    
    Supported models include:
    - PES Models: 'CHGNet-MatPES-PBE-2025.2.10-2.7M-PES', 'CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES', 'CHGNet-MPtrj-2024.2.13-11M-PES', 'CHGNet-MPtrj-2023.12.1-2.7M-PES'
    - PES Models: 'M3GNet-MP-2021.2.8-PES', 'M3GNet-MatPES-PBE-v2025.1-PES', 'M3GNet-MatPES-r2SCAN-v2025.1-PES', 'M3GNet-MP-2021.2.8-DIRECT-PES'
    - PES Models: 'TensorNet-MatPES-PBE-v2025.1-PES', 'TensorNet-MatPES-r2SCAN-v2025.1-PES', 'M3GNet-ANI-1x-Subset-PES', 'SO3Net-ANI-1x-Subset-PES'
    
    Args:
        model_name: Name of the model to load (e.g., "M3GNet", "CHGNet").
        device: Device to use ("auto", "cpu", "cuda").
        
    Returns:
        Confirmation message.
    
    CRITICAL: This tool must be called before using any other tool (except predict_bandgap) to load the model into memory.
    """
    global wrapper, sampler
    try:
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper
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
        - "stress": (Optional) Stress tensor in eV/Å³
        - "charges": (Optional) Atomic charges if supported (e.g., CHGNet)
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
    
    return wrapper.static_calculation(structure_data)

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
        
        return result
        
    except Exception as e:
        # If saving fails, still return the features but with error info
        result["save_error"] = f"Failed to save features: {str(e)}"
        return result

# Local variable to cache bandgap predictor separate from global PES wrapper
_bandgap_wrapper: Optional[Any] = None

@mcp.tool()
def predict_bandgap(structure_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Predict the bandgap for a structure using MEGNet.
    Uses an isolated model instance to avoid conflicts with PES calculations.
    
    Args:
        structure_data: Structure data (dict, ASE Atoms, pymatgen Structure, or file path).
    
    Returns:
        Dictionary containing "bandgap" in eV.
    """
    global _bandgap_wrapper
    try:
        if _bandgap_wrapper is None:
            from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper
            _bandgap_wrapper = MatGLWrapper(model_name="MEGNet-MP-2019.4.1-BandGap-mfi")
            _bandgap_wrapper.load()
        
        return _bandgap_wrapper.static_calculation(structure_data)
    except Exception as e:
        return {"error": f"Bandgap prediction failed: {str(e)}"}

@mcp.tool()
def sample_off_equilibrium(
    structure_data: Union[Dict[str, Any], str],
    total_steps: int = 1000,
    temperature: float = 300.0,
    output_dir: Optional[str] = None,
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
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper
        from src.utils.mlips.feature_calculators import MatGLCrystalFeatureCalculator
        from src.utils.data_augmenter.sampler import StructureSampler
        
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
        else:
            output_dir = str(get_current_research_dir() / "matgl" / "sampled_structures")
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
        
        structures, metadata = result_tuple
        
        return {
            "status": "success",
            "count": len(structures),
            "output_dir": output_dir,
            "metadata": metadata
        }
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
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
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper
        from src.utils.data_augmenter.sampler import StructureSampler
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
    output_dir: Optional[str] = None
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
        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper
        from src.utils.data_augmenter.sampler import StructureSampler
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
        if not output_dir:
            output_dir = str(get_current_research_dir() / "matgl" / "sampled_structures" / "order_disorder")
            
        os.makedirs(output_dir, exist_ok=True)
        
        structures = sampler.sample_order_disorder(
            atoms=pmg_structure,
            n_structures=n_structures,
            target_atoms=target_atoms,
            output_dir=output_dir
        )
        
        # Save results
        from src.utils.structure_utils import save_structure
        out_file = os.path.join(output_dir, "ordered_structures.xyz")
        save_structure(structures, out_file)
        
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
    output_dir: Optional[str] = None
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
            
        if not output_dir:
            output_dir = str(get_current_research_dir() / "matgl" / "relaxation")
            
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
        
        from src.utils.structure_utils import save_structure
        save_structure(final_struct, cif_path)
            
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
        traceback.print_exc(file=sys.stderr)
        return {"error": f"Relaxation failed: {str(e)}"}



@mcp.tool()
def run_md(
    structure_data: Union[Dict[str, Any], str],
    temperature: float = 300,
    steps: int = 1000,
    timestep: float = 1.0,
    ensemble: str = "nvt",
    log_interval: int = 10,
    pressure: float = 0.0,
    pressure_mask: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    monitor: bool = False,
    monitor_type: Optional[Union[str, List[str]]] = None,
    monitor_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run molecular dynamics simulation using MatCalc.
    
    Args:
        structure_data: Structure in partial dictionary format.
        log_interval: Interval for logging to trajectory and logfile.
        pressure: Target pressure in bar (for NPT).
        pressure_mask: Mask for anisotropic NPT (e.g., [1, 0, 0] for 1D).
        output_dir: Directory to save results.
        monitor: Whether to enable automatic MD monitoring.
        monitor_type: Type of monitor ("melting", "explosion", "overshoot", "volume") or list of types.
        monitor_params: Optional dictionary of parameters for the monitors (e.g., {"upper_limit_ratio": 4.0}).
    Returns:
        Dictionary with MD results (trajectory_path, final_structure).
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
        
    # Setup Directory
    if not output_dir:
        output_dir = str(get_current_research_dir() / "matgl" / "md")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare configuration for worker
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
            monitor_type=monitor_type,
            monitor_params=monitor_params
        )
        
        return recursive_tolist(result)
            
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": f"MD execution failed: {str(e)}", "traceback": traceback.format_exc()}

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
        
        start_atoms = wrapper.check_structure_data(start_structure)
        end_atoms = wrapper.check_structure_data(end_structure)
        
        if (isinstance(start_atoms, dict) and "error" in start_atoms) or \
           (isinstance(end_atoms, dict) and "error" in end_atoms):
           return {"error": "Invalid start or end structure."}
           
        start_struct = AseAtomsAdaptor.get_structure(start_atoms)
        end_struct = AseAtomsAdaptor.get_structure(end_atoms)
        
        if not output_dir:
            output_dir = str(get_current_research_dir() / "matgl" / "neb")

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
        traceback.print_exc(file=sys.stderr)
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
            output_dir = str(get_current_research_dir() / "matgl" / "phonon")

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
        traceback.print_exc(file=sys.stderr)
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
            output_dir = str(get_current_research_dir() / "matgl" / "qha")

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
        traceback.print_exc(file=sys.stderr)
        return {"error": f"QHA calculation failed: {str(e)}"}

@mcp.tool()
def fine_tune_model(
    training_data_path: str,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    output_dir: Optional[str] = None,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fine-tune the current MatGL model.
    
    Args:
        training_data_path: Path to a JSON file containing the training data list.
                             Each sample must have:
                               - 'structure': Dict representation (ASE atoms dict or pymatgen dict)
                               - 'energy': Total potential energy (float, eV)
                               - 'forces': Atomic forces (list/array, eV/A)
                               - 'stress': (Optional) Stress tensor (list/array) in eV/Å³.
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
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        if not training_data:
            return {"error": f"Training data file {training_data_path} is empty."}

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
            output_dir=output_dir if output_dir else str(get_current_research_dir() / "matgl" / "fine_tuning")
        )
        
        # Add extra info
        result["is_fine_tuned"] = True
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"error": f"Fine-tuning failed: {str(e)}"}

if __name__ == "__main__":
    run_fastmcp_server(mcp, mcp_pipe_binary)
