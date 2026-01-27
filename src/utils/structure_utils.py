
import os
from typing import Any, Union, Optional, List
from pathlib import Path

def save_structure(structure: Any, filename: Union[str, Path]):
    """
    Standardized function to save an atomic structure to a file.
    Handles ASE Atoms and Pymatgen Structure objects.
    
    Args:
        structure: The structure object to save (ASE Atoms or Pymatgen Structure).
        filename: Path to the output file (e.g., "relaxed_structure.cif").
    """
    from ase import Atoms
    from pymatgen.core import Structure
    
    filename_str = str(filename)
    
    if isinstance(structure, Structure):
        # Pymatgen Structure
        structure.to(filename=filename_str)
    elif isinstance(structure, Atoms):
        # ASE Atoms
        structure.write(filename_str)
    elif hasattr(structure, "to"):
        # Generic 'to' method (usually Pymatgen-like)
        structure.to(filename=filename_str)
    elif hasattr(structure, "write"):
        # Generic 'write' method (usually ASE-like)
        structure.write(filename_str)
    elif isinstance(structure, list) and len(structure) > 0:
        # Batch of atoms (e.g. trajectory)
        from ase.io import write
        write(filename_str, structure)
    else:
        # Fallback using Adaptor
        from pymatgen.io.ase import AseAtomsAdaptor
        try:
            # Try converting to ASE Atoms first for broad format support
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.write(filename_str)
        except Exception:
            # Last ditch effort: Pymatgen
            pmg_struct = AseAtomsAdaptor.get_structure(structure)
            pmg_struct.to(filename=filename_str)

def load_structure_from_file(filename: Union[str, Path]):
    """
    Load a structure from a file and return it as a Pymatgen Structure.
    """
    from pymatgen.core import Structure
    return Structure.from_file(str(filename))

def get_structure_by_formula(formula: str, mprester: Any) -> Any:
    """
    Search for the most stable structure by formula in Materials Project.
    Returns ASE Atoms object.
    """
    docs = mprester.summary.search(formula=formula, fields=["structure", "energy_above_hull"])
    if not docs:
        return None
    # Sort by stability
    stable_doc = min(docs, key=lambda x: x.energy_above_hull)
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_atoms(stable_doc.structure)

def get_structure_by_chemsys(chemsys: str, mprester: Any) -> Any:
    """
    Search for the most stable structure by chemical system in Materials Project.
    Returns ASE Atoms object.
    """
    docs = mprester.summary.search(chemsys=chemsys, fields=["structure", "energy_above_hull"])
    if not docs:
        return None
    # Sort by stability
    stable_doc = min(docs, key=lambda x: x.energy_above_hull)
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_atoms(stable_doc.structure)

def get_structure_by_id(material_id: str, mprester: Any) -> Any:
    """
    Search for a structure by Material ID in Materials Project.
    Returns ASE Atoms object.
    """
    doc = mprester.materials.get_structure_by_material_id(material_id)
    if not doc:
        return None
    from pymatgen.io.ase import AseAtomsAdaptor
    return AseAtomsAdaptor.get_atoms(doc)
