import sys
import os
import warnings
import logging

# Suppress all warnings to prevent protocol pollution
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Silence common blabbermouth libraries
logging.getLogger("mp-api").setLevel(logging.ERROR)
logging.getLogger("pymatgen").setLevel(logging.ERROR)

import json
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional
from pathlib import Path

from src.utils.dft.vasp_writer import write_vasp_input_files
from src.utils.dft.vasp_parser import VASPParser
from src.utils.structure_utils import (
    load_structure_from_file, 
    get_structure_by_formula, 
    get_structure_by_chemsys, 
    get_structure_by_id
)
from src.utils.api_keys import get_mp_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MaterialsServer")

# Create MCP server
mcp = FastMCP("materials_tools")

@mcp.tool()
def search_materials_project_by_formula(
    formula: str,
    api_key: Optional[str] = None,
    save_to_file: Optional[str] = None
) -> str:
    """
    Search for materials in Materials Project by chemical formula (e.g., "Fe2O3").
    Supports wildcards (e.g., "Fe2O3", "Si*").
    
    Args:
        formula: Chemical formula (e.g., "Si", "Fe2O3").
        api_key: Optional Materials Project API key. If not provided, tries to load from environment.
        save_to_file: Optional path to save the structure file.
        
    Returns:
        Path to the saved structure file (CIF format).
    """
    try:
        mp_key = api_key or get_mp_key()
        if not mp_key:
            return "Error: Materials Project API key not found. Please provide api_key or set MP_API_KEY environment variable."
            
        try:
            from mp_api.client import MPRester
        except ImportError:
            return "Error: mp-api package not installed. Please install it with 'pip install mp-api'."

        with MPRester(mp_key) as mprester:
            atoms = get_structure_by_formula(formula, mprester)
                
        if atoms is None:
            return f"Error: No structure found for formula {formula} in Materials Project."
            
        return _save_atoms(atoms, formula, save_to_file)

    except Exception as e:
        logger.error(f"Error searching materials by formula: {e}")
        return f"Error searching materials by formula: {str(e)}"

@mcp.tool()
def search_materials_project_by_chemsys(
    chemsys: str,
    api_key: Optional[str] = None,
    save_to_file: Optional[str] = None
) -> str:
    """
    Search for materials in Materials Project by chemical system (e.g., "Li-O").
    Returns the most stable structure (lowest energy above hull).
    
    Args:
        chemsys: Chemical system (e.g., "Li-O", "Si-Fe-O").
        api_key: Optional Materials Project API key. If not provided, tries to load from environment.
        save_to_file: Optional path to save the structure file.
        
    Returns:
        Path to the saved structure file (CIF format).
    """
    try:
        mp_key = api_key or get_mp_key()
        if not mp_key:
            return "Error: Materials Project API key not found. Please provide api_key or set MP_API_KEY environment variable."
            
        try:
            from mp_api.client import MPRester
        except ImportError:
            return "Error: mp-api package not installed. Please install it with 'pip install mp-api'."

        with MPRester(mp_key) as mprester:
            atoms = get_structure_by_chemsys(chemsys, mprester)
                
        if atoms is None:
            return f"Error: No structure found for chemical system {chemsys} in Materials Project."
            
        return _save_atoms(atoms, chemsys, save_to_file)

    except Exception as e:
        logger.error(f"Error searching materials by chemsys: {e}")
        return f"Error searching materials by chemsys: {str(e)}"

@mcp.tool()
def search_materials_project_by_id(
    material_id: str,
    api_key: Optional[str] = None,
    save_to_file: Optional[str] = None
) -> str:
    """
    Search for materials in Materials Project by Material ID (e.g., "mp-149").
    
    Args:
        material_id: Material ID (e.g., "mp-149").
        api_key: Optional Materials Project API key. If not provided, tries to load from environment.
        save_to_file: Optional path to save the structure file.
        
    Returns:
        Path to the saved structure file (CIF format).
    """
    try:
        mp_key = api_key or get_mp_key()
        if not mp_key:
            return "Error: Materials Project API key not found. Please provide api_key or set MP_API_KEY environment variable."
            
        try:
            from mp_api.client import MPRester
        except ImportError:
            return "Error: mp-api package not installed. Please install it with 'pip install mp-api'."

        with MPRester(mp_key) as mprester:
            atoms = get_structure_by_id(material_id, mprester)
                
        if atoms is None:
            return f"Error: No structure found for ID {material_id} in Materials Project."
            
        return _save_atoms(atoms, material_id, save_to_file)

    except Exception as e:
        logger.error(f"Error searching materials by ID: {e}")
        return f"Error searching materials by ID: {str(e)}"

def _save_atoms(atoms: Any, name_hint: str, save_to_file: Optional[str] = None) -> str:
    """Helper to save ASE atoms to file."""
    # Determine save path
    if save_to_file:
        save_path = Path(save_to_file)
    else:
        # Create a safe filename
        safe_name = name_hint.replace(" ", "").replace("*", "_star")
        save_path = Path(f"{safe_name}_structure.cif")
        
    # Save structure
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(atoms)
    structure.to(filename=str(save_path), fmt="cif")
    
    return f"Structure for {name_hint} saved to {save_path.absolute()}"

@mcp.tool()
def prepare_vasp_inputs(
    structure_path: str,
    output_dir: str,
    calculation_type: str = "relaxation",
    preset_type: str = "omat",
    config: Optional[Dict[str, Any]] = None,
    vasp_settings: Optional[Dict[str, Any]] = None
) -> str:
    """
    Prepare VASP input files (POSCAR, INCAR, KPOINTS, POTCAR) for a given structure.
    
    Args:
        structure_path: Path to the input structure file (CIF, POSCAR, XYZ, etc.)
        output_dir: Directory where input files will be generated.
        calculation_type: Type of VASP calculation: "relaxation", "static", "md".
        preset_type: Preset calculation type ("omat", "mp", "matpes-pbe", "matpes-r2scan"). Default is "omat".
                     - "omat": Optimized for OC20/OMat24 style calculations.
                     - "mp": Materials Project standard (static or relaxation).
                     - "matpes-pbe": MatPES project settings with PBE functional.
                     - "matpes-r2scan": MatPES project settings with r2SCAN functional.
        config: Custom VASP settings (INCAR tags) to override preset.
        vasp_settings: Deprecated, use config instead.
                        
    Returns:
        Summary message indicating where files were written.
    """
    try:
        input_path = Path(structure_path)
        out_path = Path(output_dir)
        
        # Prepare VASP inputs
        
        # Check if input is a directory (Batch processing)
        if input_path.is_dir():
            structure_files = list(input_path.rglob("*.cif")) + \
                              list(input_path.rglob("*.xyz")) + \
                              list(input_path.rglob("POSCAR"))
            
            if not structure_files:
                return f"Error: No structure files found in directory {structure_path}"
                
            summary = []
            for i, struct_file in enumerate(sorted(structure_files)):
                # Create subdirectory for each structure
                # Use filename stem or index if generic
                sub_name = struct_file.stem
                if sub_name == "POSCAR":
                    sub_name = struct_file.parent.name
                    
                # If names are generic like "structure_0", preserve them
                # Otherwise ensure uniqueness
                sub_dir = out_path / sub_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    structure = load_structure_from_file(str(struct_file))
                    if structure:
                        write_vasp_input_files(
                            atoms=structure,
                            output_dir=str(sub_dir),
                            preset_type=preset_type,
                            calculation_type=calculation_type,
                            config=config or vasp_settings
                        )
                        summary.append(sub_name)
                except Exception as e:
                    logger.warning(f"Failed to process {struct_file}: {e}")
                    
            return f"Successfully prepared VASP inputs for {len(summary)} structures in {output_dir}. Subdirectories: {summary[:5]}..."
            
        else:
            # Single file processing
            structure = load_structure_from_file(structure_path)
            if structure is None:
                return f"Error: Could not load structure from {structure_path}"
            
            # Write files
            files = write_vasp_input_files(
                atoms=structure,
                output_dir=output_dir,
                preset_type=preset_type,
                calculation_type=calculation_type,
                config=config or vasp_settings
            )
            
            return f"Successfully wrote VASP input files to {output_dir}. Files: {list(files.keys())}"
        
    except Exception as e:
        logger.error(f"Error preparing VASP inputs: {e}")
        return f"Error preparing VASP inputs: {str(e)}"

@mcp.tool()
def parse_vasp_results(output_dir: str, save_to_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse VASP output files (vasprun.xml, OUTCAR) or UMA mock DFT results.
    
    Can parse a single calculation directory or a root directory containing 
    multiple subdirectories (e.g., 'structure_0', 'structure_1').
    
    Args:
        output_dir: Directory containing VASP output files (vasprun.xml, OUTCAR) 
                    or UMA result.json.
        save_to_file: Optional path to save the parsed results as a JSON file.
        
    Returns:
        Dictionary containing:
        - For single calculation:
            - 'final_energy': Total energy (eV)
            - 'forces': Atomic forces (eV/A)
            - 'stress': Stress tensor (eV/A^3)
            - 'final_structure': Pymatgen/ASE dict structure
        - For multiple calculations:
            - 'results': List of result dictionaries for each structure
    """
    try:
        parser = VASPParser(output_dir)
        
        # Detect result type (single or multiple)
        # For simplicity, we expose the single parser logic via parsing all and returning list
        # or checking if it's a single calculation
        
        # Check if it has direct result files
        has_direct_result = (
            (parser.output_dir / "vasprun.xml").exists() or 
            (parser.output_dir / "result.json").exists()
        )
        
        if has_direct_result:
            try:
                # Try UMA format first
                if parser.uma_result_path.exists():
                    return parser.parse_uma_result()
                # Try VASP format
                elif parser.vasprun_path.exists():
                    result = parser.parse_vasprun()
                    outcar_result = parser.parse_outcar()
                    result.update(outcar_result)
                    
                    # Serialize for JSON
                    results = parser._prepare_for_json(result)
                    
                    if save_to_file:
                        with open(save_to_file, 'w') as f:
                            json.dump(results, f, indent=2)
                            
                    return results
            except Exception as e:
                return {"error": f"Failed to parse results in {output_dir}: {str(e)}"}
        
        # If not direct, try parsing all subdirectories
        all_results = parser.parse_all()
        if not all_results:
            return {"error": f"No valid VASP or UMA results found in {output_dir}"}
            
        # Return summary or full list? Returning full list might be large.
        # Let's return serialized list.
        results = {"results": parser._prepare_for_json(all_results)}
        
        if save_to_file:
            with open(save_to_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        return results
        
    except Exception as e:
        logger.error(f"Error parsing VASP results: {e}")
        return {"error": f"Error parsing VASP results: {str(e)}"}

if __name__ == "__main__":
    mcp.run()
