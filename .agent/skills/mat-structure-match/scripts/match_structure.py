"""
Match a target structure against a candidate structure or a directory of candidate structures.

Usage:
    python match_structure.py <target_structure> <candidates> [--ltol 0.2] [--stol 0.3] [--angle_tol 5]

Requirements:
    - Conda environment: base-agent
    - Required packages: pymatgen, ase, argparse
"""
import argparse
import sys
import json
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

import os
import sys

# Add project root to sys.path
import inspect

# Use inspect to get the absolute path of the script robustly
script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Project root is four levels up: .agent/skills/mat-structure-match/scripts
project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.structure_utils import load_structures

def main():
    parser = argparse.ArgumentParser(description="Match a target structure against candidate structures.")
    parser.add_argument("target", help="Path to the target structure file.")
    parser.add_argument("candidates", help="Path to a candidate structure file or a directory of candidate structures.")
    parser.add_argument("--ltol", type=float, default=0.2, help="Fractional length tolerance for structure matching.")
    parser.add_argument("--stol", type=float, default=0.3, help="Site tolerance for structure matching.")
    parser.add_argument("--angle_tol", type=float, default=5.0, help="Angle tolerance for structure matching.")
    parser.add_argument("--output", type=str, default="match_results.json", help="Output JSON file for match results.")
    
    args = parser.parse_args()
    
    # Load target structure
    print(f"Loading target structure from {args.target}...")
    try:
        targets = load_structures(args.target)
        if not targets:
            print(f"Error: Could not load target structure from {args.target}")
            sys.exit(1)
        target_struct = targets[0]
    except Exception as e:
        print(f"Error reading target structure: {e}")
        sys.exit(1)
        
    print(f"Target structure formula: {target_struct.composition.reduced_formula}")
    
    # Load candidate structure(s)
    print(f"Loading candidate structure(s) from {args.candidates}...")
    candidate_structs = []
    cand_path = Path(args.candidates)
    try:
        from pymatgen.core import Structure
        if cand_path.is_dir():
            for f in cand_path.glob("*"):
                if f.is_file() and f.suffix in [".cif", ".xyz", ".poscar", ".vasp"]:
                    try:
                        stm = Structure.from_file(str(f))
                        stm.properties = stm.properties or {}
                        stm.properties["filename"] = f.name
                        candidate_structs.append(stm)
                    except:
                        pass
        else:
            stm = Structure.from_file(str(cand_path))
            stm.properties = stm.properties or {}
            stm.properties["filename"] = cand_path.name
            candidate_structs.append(stm)
            
        if not candidate_structs:
            print(f"Warning: No candidate structures found in {args.candidates}")
    except Exception as e:
        print(f"Error reading candidate structures: {e}")
        
    print(f"Loaded {len(candidate_structs)} candidate structure(s).")
    
    # Initialize StructureMatcher
    matcher = StructureMatcher(
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False
    )
    
    results = {
        "target": str(args.target),
        "target_formula": target_struct.composition.reduced_formula,
        "matches": [],
        "num_candidates_checked": len(candidate_structs),
        "match_found": False
    }

    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sa = SpacegroupAnalyzer(target_struct)
        sg_symbol = sa.get_space_group_symbol()
        crystal_system = sa.get_crystal_system()
        print(f"Target structure symmetry: {sg_symbol} ({crystal_system})")
        results["target_spacegroup"] = sg_symbol
        results["target_crystal_system"] = crystal_system
    except Exception as e:
        print(f"Warning: Could not determine spacegroup: {e}")
        sg_symbol = ""
        crystal_system = ""
        results["target_spacegroup"] = None
        results["target_crystal_system"] = None
    
    # Perform matching
    print("Matching structures...")
    
    for cand_struct in candidate_structs:
        # Check if they match
        match = matcher.fit(target_struct, cand_struct)
        if match:
            cand_id = "unknown"
            
            # Extract from properties if available
            if hasattr(cand_struct, "properties") and "material_id" in cand_struct.properties:
                cand_id = cand_struct.properties["material_id"]
            elif hasattr(cand_struct, "properties") and "filename" in cand_struct.properties:
                cand_id = cand_struct.properties["filename"].split(".")[0]
            else:
                pass
                
            match_info = {
                "candidate_formula": cand_struct.composition.reduced_formula,
                "material_id_hint": cand_id
            }
            results["matches"].append(match_info)
            results["match_found"] = True
            print(f"  -> Match found with candidate: {cand_id} (formula={match_info['candidate_formula']})")
            
    if not results["match_found"]:
        print("  -> No structural matches found.")
        
        if sg_symbol and crystal_system:
            query_str = f'"{results["target_formula"]}" AND ("{sg_symbol}" OR "{crystal_system}")'
            print(f"\nChecking literature database for the composition '{results['target_formula']}' with symmetry '{sg_symbol}' ({crystal_system})...")
        else:
            query_str = f'"{results["target_formula"]}"'
            print(f"\nChecking literature database for the composition '{results['target_formula']}'...")
            
        try:
            from src.utils.literature_utils import query_openalex
            lit_results = query_openalex(query_str, limit=5)
            
            results["literature_reported"] = len(lit_results) > 0
            results["literature_matches"] = []
            
            if lit_results:
                print(f"  -> Success: Found {len(lit_results)} recent papers reporting '{results['target_formula']}' with matching symmetry.")
                for i, r in enumerate(lit_results, 1):
                    title = r.get("title", "Unknown Title")
                    doi = r.get("doi", "Unknown DOI")
                    print(f"     [{i}] {title} ({doi})")
                    results["literature_matches"].append({"title": title, "doi": doi})
            else:
                print(f"  -> Failure: No literature found for '{results['target_formula']}' with matching symmetry. Truly novel composition/polymorph.")
        except ImportError:
            print("  -> Could not import literature_utils. Skipping literature check.")
        except Exception as e:
            print(f"  -> Error executing literature search: {e}")
        
    # Write output
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
