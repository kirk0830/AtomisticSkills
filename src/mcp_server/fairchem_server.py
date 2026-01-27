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

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, List, Union

# Configure logging to go to the real stderr
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize FastMCP server
mcp = FastMCP("FAIRCHEM")
from src.utils.research_utils import get_current_research_dir

# Global variables to hold state
wrapper: Optional[Any] = None

@mcp.tool()
def load_model(
    model_name: str = "uma-s-1p1", 
    device: str = "auto", 
    task_name: Optional[str] = None,
    inference_settings: str = "default"
) -> str:
    """
    Load a FAIRCHEM model.
    
    Supported models include:
    - UMA (Universal): 'uma-s-1p1', 'uma-m-1p1', 'uma-s-1'
    - ESEN (Organic/Molecular): 'esen-md-direct-all-omol', 'esen-sm-conserving-all-omol', 'esen-sm-direct-all-omol'
    - ESEN (Catalysis/OC25): 'esen-sm-conserving-all-oc25', 'esen-md-direct-all-oc25'
    """
    global wrapper
    try:
        # Lazy import to speed up server startup and avoid timeout
        from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper
        
        wrapper = FAIRCHEMWrapper(
            model_name=model_name, 
            device=device, 
            task_name=task_name,
            inference_settings=inference_settings
        )
        wrapper.load()
        return f"Successfully loaded FAIRCHEM model: {model_name}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

@mcp.tool()
def predict_structure(structure_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Predict energy, forces, and stress for a structure.
    
    Returns:
        Dict: {'energy': eV, 'forces': eV/A, 'stress': eV/Å³}
    """
    global wrapper
    if wrapper is None:
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
    Fine-tune the current FAIRCHEM model.

    Args:
        training_data_path: Path to a JSON file containing the training data list.
                             Each sample must have:
                               - 'structure': Dict (ASE atoms or pymatgen format)
                               - 'energy': Total potential energy (float, eV)
                               - 'forces': Atomic forces (list/array, eV/A)
                               - 'stress': (Optional) Stress tensor (list/array) in eV/Å³. 
                                 NOTE: Provide stress in eV/Å³. (Standard ASE unit)
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        output_dir: Directory to save the fine-tuned model.
        training_config: Optional dictionary for advanced configuration.
    """
    global wrapper
    if wrapper is None:
        return {"error": "Model not loaded. Please call load_model first."}
        
    try:
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
        
        final_config = {
            "max_epochs": epochs,
            "learning_rate": learning_rate
        }
        if training_config:
            final_config.update(training_config)
        
        result = wrapper.fine_tune(
            training_data=training_data,
            training_config=final_config,
            output_dir=output_dir if output_dir else str(get_current_research_dir() / "fairchem" / "fine_tuning")
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
    Relax a structure using the loaded FAIRCHEM model and MatCalc.
    
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
            output_dir = str(get_current_research_dir() / "fairchem" / "relaxation")
            
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
            output_dir = str(get_current_research_dir() / "fairchem" / "md")
            
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(atoms, "get_chemical_formula"):
            formula = atoms.get_chemical_formula()
        else:
            from pymatgen.core import Structure
            if isinstance(atoms, Structure):
                formula = atoms.composition.reduced_formula
            else:
                formula = "unknown"
                
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
        
        start_atoms = wrapper.check_structure_data(start_structure)
        end_atoms = wrapper.check_structure_data(end_structure)
        
        if (isinstance(start_atoms, dict) and "error" in start_atoms) or \
           (isinstance(end_atoms, dict) and "error" in end_atoms):
           return {"error": "Invalid start or end structure."}
           
        start_struct = AseAtomsAdaptor.get_structure(start_atoms)
        end_struct = AseAtomsAdaptor.get_structure(end_atoms)
        
        if not output_dir:
            output_dir = str(get_current_research_dir() / "fairchem" / "neb")

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
            output_dir = str(get_current_research_dir() / "fairchem" / "phonon")

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
            output_dir = str(get_current_research_dir() / "fairchem" / "qha")

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
def mock_dft(
    dft_input_dir: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run mock DFT calculations using the loaded FAIRCHEM model.
    Mimics VASP output format.
    
    Args:
        dft_input_dir: Directory containing VASP inputs (POSCAR) or structure files.
        output_dir: Directory to save mock VASP results (vasprun.xml-like JSON, OUTCAR-like info).
    
    Returns:
        Summary of mock DFT run.
    """
    global wrapper
    if wrapper is None or not wrapper.is_loaded:
        return {"error": "Model not loaded. Please call load_model first."}
        
    try:
        input_path = Path(dft_input_dir)
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = get_current_research_dir() / "fairchem" / "mock_dft"
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find structure files
        # Priority: POSCAR > CIF > XYZ
        poscar_files = list(input_path.rglob("POSCAR"))
        cif_files = list(input_path.rglob("structure_*.cif")) if not poscar_files else []
        xyz_files = list(input_path.rglob("*.xyz")) if not poscar_files and not cif_files else []
        
        structure_files = poscar_files + cif_files + xyz_files
        
        if not structure_files:
            return {"error": f"No structure files found in {dft_input_dir}"}
            
        results = []
        calc = wrapper.create_calculator()
        
        for struct_file in structure_files:
            try:
                # Determine output subdirectory
                if struct_file.parent.name.startswith("structure_"):
                     result_dir = output_path / struct_file.parent.name
                else:
                    # Create directory based on index
                    struct_idx = len(results)
                    result_dir = output_path / f"structure_{struct_idx}"
                
                result_dir.mkdir(parents=True, exist_ok=True)
                
                # Load and calculate
                atoms = read(str(struct_file))
                atoms.calc = calc
                
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                stress = atoms.get_stress() # Might fail if not supported by model, wrapper usually handles it
                
                # Save as 'result.json' which is the UMA mock format our parser supports
                result_data = {
                    "energy": float(energy),
                    "forces": forces.tolist(),
                    "stress": stress.tolist() if stress is not None else None,
                    "structure": atoms.todict() if hasattr(atoms, 'todict') else None # Ase atoms to dict might need handling
                    # Pymatgen structure as_dict is standard, ASE has logic but lets keep it simple
                    # Actually parser expects pymatgen dict structure or we save valid file.
                }
                
                # For compatibility with parser, save CONTCAR
                from src.utils.structure_utils import save_structure
                save_structure(atoms, str(result_dir / "CONTCAR"))
                
                # Save result JSON
                with open(result_dir / "result.json", "w") as f:
                    # We skip the complex structure serialization here and rely on CONTCAR for structure reading
                    # Parser logic: if CONTCAR exists, read it.
                    json.dump(result_data, f, indent=2, cls=NumpyEncoder)
                    
                results.append({
                    "structure_id": result_dir.name,
                    "status": "completed",
                    "energy": energy
                })
                
            except Exception as e:
                import traceback
                logger.error(f"Failed mock DFT for {struct_file}: {e}")
                results.append({
                    "structure_id": struct_file.name,
                    "status": "failed",
                    "error": str(e)
                })
                
        return {
            "success": True,
            "num_structures": len(structure_files),
            "results": results,
            "message": f"Mock DFT completed for {len(results)} structures."
        }
        
    except Exception as e:
        return {"error": f"Mock DFT failed: {str(e)}"}


if __name__ == "__main__":
    mcp.run()
