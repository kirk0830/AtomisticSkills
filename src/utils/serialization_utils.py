import numpy as np
from pathlib import Path

def recursive_tolist(obj):
    """
    Recursively convert objects to list/dict structures compatible with JSON.
    Handles numpy arrays, Path objects, and objects with as_dict() method.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (Path,)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: recursive_tolist(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [recursive_tolist(x) for x in obj]
    elif hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    elif hasattr(obj, "as_dict"): # pymatgen or others
        return recursive_tolist(obj.as_dict())
    else:
        return obj
