
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

def load_structures(inputs: Union[str, Path, dict, List, Any]) -> List[Any]:
    """
    Universal function to load structures into a list of Pymatgen Structures.
    
    Args:
        inputs: Can be:
            - Path string or Path object (file or directory)
            - Structure/Atoms object
            - Dict representation
            - List of any of the above
            
    Returns:
        List[Structure]: List of Pymatgen Structure objects.
    """
    from pymatgen.core import Structure
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    import glob
    
    structures = []
    
    # Handle single items by wrapping in list, unless it's a directory path
    if isinstance(inputs, (str, Path)):
        path = Path(inputs)
        if path.is_dir():
            # Directory: expand to list of files
            exts = ["cif", "CIF", "poscar", "POSCAR", "vasp", "xyz", "json"]
            files = []
            for ext in exts:
                files.extend(glob.glob(str(path / f"**/*.{ext}"), recursive=True))
            files = sorted(list(set(files)))
            # Recursively call with list of files
            return load_structures(files)
        elif "*" in str(inputs):
            # Glob string
            files = sorted(glob.glob(str(inputs)))
            return load_structures(files)
        else:
            # Single file
            raw_items = [inputs]
    elif isinstance(inputs, list):
        raw_items = inputs
    else:
        raw_items = [inputs]
        
    for item in raw_items:
        try:
            struct = None
            if isinstance(item, (str, Path)):
                # Load from file
                struct = Structure.from_file(str(item))
            elif isinstance(item, dict):
                # Load from dict
                struct = Structure.from_dict(item)
            elif isinstance(item, Structure):
                # Already Structure
                struct = item
            elif isinstance(item, Atoms):
                # ASE Atoms
                struct = AseAtomsAdaptor.get_structure(item)
            else:
                # Try generic conversion or fail gracefully
                if hasattr(item, "as_dict"):
                    struct = Structure.from_dict(item.as_dict())
                else:
                    print(f"Warning: Could not convert item of type {type(item)} to Structure.")
            
            if struct:
                structures.append(struct)
        except Exception as e:
            print(f"Error loading structure from {item}: {e}")
            
    return structures

