"""
Utility functions for structure creation and manipulation.
"""

import logging
from typing import Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def create_hypothetical_structure(formula: str, crystal_system: str = "cubic", lattice_parameter: float = 5.0) -> Optional[Any]:
    """
    Create a hypothetical material structure using pymatgen Composition and ASE.
    
    Args:
        formula: Chemical formula (e.g., "Ag0.5Li0.5", "Li")
        crystal_system: Crystal system ("cubic", "hexagonal", "tetragonal", etc.)
        lattice_parameter: Lattice parameter in Angstroms
    
    Returns:
        ASE Atoms object or None if creation fails.
    """
    try:
        from pymatgen.core import Composition
        from ase.build import bulk
        from ase.data import atomic_numbers
        
        # Parse formula using pymatgen
        comp = Composition(formula)
        elements = [str(e) for e in comp.elements]
        
        if not elements:
            logger.error(f"Could not parse formula: {formula}")
            return None
        
        # Create base structure based on crystal system
        if crystal_system.lower() == "cubic":
            atoms = bulk(elements[0], crystalstructure='fcc', a=lattice_parameter)
        elif crystal_system.lower() == "hexagonal":
            atoms = bulk(elements[0], crystalstructure='hcp', a=lattice_parameter)
        else:
            # Default to cubic
            atoms = bulk(elements[0], crystalstructure='fcc', a=lattice_parameter)
        
        # Create alloy if multiple elements
        if len(elements) > 1:
            atoms = atoms.repeat((2, 2, 2))
            # Get atomic fractions
            fractions = [comp.get_atomic_fraction(e) for e in comp.elements]
            
            # Substitute atoms based on composition
            total_atoms = len(atoms)
            
            # Calculate number of each element
            num_atoms_per_element = [int(f * total_atoms) for f in fractions]
            # Adjust to ensure total matches
            diff = total_atoms - sum(num_atoms_per_element)
            if diff > 0:
                num_atoms_per_element[0] += diff
            
            # Create new atomic numbers array
            new_numbers = []
            for i, num in enumerate(num_atoms_per_element):
                atomic_num = atomic_numbers[elements[i]]
                new_numbers.extend([atomic_num] * num)
            
            atoms.set_atomic_numbers(new_numbers[:len(atoms)])
        
        logger.info(f"Created hypothetical {formula} structure with {len(atoms)} atoms")
        return atoms
    except Exception as e:
        logger.error(f"Failed to create hypothetical structure: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def create_disordered_structure(formula: str, crystal_system: str = "cubic", lattice_parameter: float = 3.6) -> Optional[Any]:
    """
    Create a disordered structure with fractional occupancies using pymatgen.
    
    Args:
        formula: Chemical formula with fractions (e.g., "Cr0.33Ni0.33Co0.33")
        crystal_system: Crystal system ("cubic", "bcc", "fcc")
        lattice_parameter: Lattice parameter in Angstroms
    
    Returns:
        pymatgen Structure object (not ASE Atoms, as ASE doesn't support fractional occupancy)
    """
    try:
        from pymatgen.core import Structure, Lattice, Composition
        
        # Create lattice
        if crystal_system.lower() in ["cubic", "fcc"]:
            # Face-centered cubic
            lattice = Lattice.cubic(lattice_parameter)
            coords = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
            # For FCC, we have 4 sites per unit cell
            # All sites have the same composition (disordered)
            species = [Composition(formula)] * 4
        elif crystal_system.lower() == "bcc":
            # Body-centered cubic
            lattice = Lattice.cubic(lattice_parameter)
            coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
            species = [Composition(formula)] * 2
        else:
            # Simple cubic default
            lattice = Lattice.cubic(lattice_parameter)
            coords = [[0, 0, 0]]
            species = [Composition(formula)]
            
        structure = Structure(lattice, species, coords)
        logger.info(f"Created disordered structure for {formula} ({crystal_system})")
        return structure
        
    except Exception as e:
        logger.error(f"Failed to create disordered structure: {e}")
        return None


def load_structure_from_file(file_path: str) -> Optional[Any]:
    """
    Load material structure from file. 
    Tries ASE first, then pymatgen as fallback.
    
    Args:
        file_path: Path to structure file (POSCAR, CIF, XYZ, etc.)
    
    Returns:
        ASE Atoms object or None if loading fails.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None
    
    # Try ASE first
    try:
        from ase.io import read
        atoms = read(str(file_path))
        if atoms is not None and len(atoms) > 0:
            logger.info(f"Loaded structure from {file_path} using ASE")
            return atoms
    except Exception as e:
        logger.debug(f"ASE failed to load {file_path}: {e}")
        pass
        
    # Try pymatgen fallback
    try:
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        
        struct = Structure.from_file(str(file_path))
        atoms = AseAtomsAdaptor.get_atoms(struct)
        logger.info(f"Loaded structure from {file_path} using Pymatgen")
        return atoms
    except Exception as e:
        logger.error(f"Failed to load structure from {file_path} (tried ASE and Pymatgen): {e}")
        return None


def get_structure_by_id(material_id: str, mprester: Any) -> Optional[Any]:
    """
    Load structure from Materials Project by Material ID (e.g., "mp-149").
    
    Args:
        material_id: Material ID string.
        mprester: MPRester instance.
        
    Returns:
        ASE Atoms object or None if not found.
    """
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core import Structure as PymatgenStructure
        
        try:
            structure_dict = mprester.get_structure_by_material_id(material_id)
            if isinstance(structure_dict, dict):
                structure = PymatgenStructure.from_dict(structure_dict)
            else:
                structure = structure_dict
            
            atoms = AseAtomsAdaptor.get_atoms(structure)
            logger.info(f"Successfully loaded structure from Materials Project (material_id: {material_id})")
            return atoms
        except Exception as e:
            logger.warning(f"Failed to load structure by material_id {material_id}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error in get_structure_by_id: {e}")
        return None

def get_structure_by_chemsys(chemsys: str, mprester: Any) -> Optional[Any]:
    """
    Load most stable structure from Materials Project by chemical system (e.g., "Li-O").
    
    Args:
        chemsys: Chemical system string.
        mprester: MPRester instance.
    
    Returns:
        ASE Atoms object or None if not found.
    """
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core import Structure as PymatgenStructure
        
        try:
            # Search by chemical system
            results = mprester.materials.summary.search(
                chemsys=chemsys,
                fields=["material_id", "formula_pretty", "structure", "energy_above_hull"]
            )
            
            results_list = list(results) if hasattr(results, '__iter__') else []
            
            if results_list:
                # Sort by energy above hull (most stable first)
                results_list.sort(key=lambda x: (x.energy_above_hull if hasattr(x, 'energy_above_hull') and x.energy_above_hull is not None else float('inf')))
                
                best_match = results_list[0]
                if best_match.structure:
                    structure = best_match.structure
                    # structure from summary search is already a Pymatgen Structure object usually
                    # but let's be safe and ensure correct type if needed, usually it is fine.
                    # The search API returns SummaryDoc, which has a structure field of type Structure.
                    
                    atoms = AseAtomsAdaptor.get_atoms(structure)
                    material_id = best_match.material_id if hasattr(best_match, 'material_id') else best_match.get('material_id')
                    logger.info(f"Loaded most stable structure for system {chemsys} from Materials Project (material_id: {material_id})")
                    return atoms
        except Exception as e:
            logger.warning(f"Failed to search by chemical system {chemsys}: {e}")
            return None
            
        logger.debug(f"No matching structure found in Materials Project for system {chemsys}")
        return None
    except Exception as e:
        logger.error(f"Error in get_structure_by_chemsys: {e}")
        return None

def get_structure_by_formula(formula: str, mprester: Any) -> Optional[Any]:
    """
    Load most stable structure from Materials Project by formula (e.g., "Fe2O3").
    
    Args:
        formula: Chemical formula string (supports wildcards).
        mprester: MPRester instance.
    
    Returns:
        ASE Atoms object or None if not found.
    """
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core import Structure as PymatgenStructure
        
        try:
            # Search using formula argument
            results = mprester.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "structure", "energy_above_hull"]
            )
            
            results_list = list(results) if hasattr(results, '__iter__') else []
            
            if results_list:
                # Sort by energy above hull (most stable first)
                results_list.sort(key=lambda x: (x.energy_above_hull if hasattr(x, 'energy_above_hull') and x.energy_above_hull is not None else float('inf')))
                
                best_match = results_list[0]
                if best_match.structure:
                    structure = best_match.structure
                    
                    atoms = AseAtomsAdaptor.get_atoms(structure)
                    material_id = best_match.material_id if hasattr(best_match, 'material_id') else best_match.get('material_id')
                    logger.info(f"Loaded most stable structure for formula {formula} from Materials Project (material_id: {material_id})")
                    return atoms
        except Exception as e:
            logger.debug(f"MP search API error: {e}")
            
        logger.debug(f"No matching structure found in Materials Project for formula {formula}")
        return None
    except Exception as e:
        logger.warning(f"Error loading structure from Materials Project: {e}")
        return None
