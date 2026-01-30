
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixEntry, PourbaixPlotter
from pymatgen.entries.compatibility import MU_H2O
from mp_api.client import MPRester

def load_relaxed_solids(relaxed_dir: Path, exclude_formulas: List[str] = None) -> Dict[str, Dict]:
    """Load total energies from relaxed structure result files."""
    energies = {}
    exclude_formulas = exclude_formulas or []
    
    for subdir in relaxed_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        energy_file = subdir / "relaxed_energy.txt"
        struct_file = subdir / "relaxed_structure.cif"
        if not energy_file.exists():
            continue
            
        with open(energy_file) as f:
            total_energy = float(f.read().strip())
            
        # Extract formula and subdir name
        name_parts = subdir.name.split('_')
        formula = name_parts[0]
        if formula in exclude_formulas or subdir.name in exclude_formulas:
            continue
            
        from pymatgen.core import Structure
        struct = Structure.from_file(str(struct_file)) if struct_file.exists() else None
        comp = struct.composition if struct else Composition(formula)
        
        energies[subdir.name] = {
            'energy': total_energy,
            'composition': comp,
            'formula': formula
        }
    return energies

def main():
    parser = argparse.ArgumentParser(description='Strict Pymatgen Pourbaix Diagram Script')
    parser.add_argument('--relaxed_solids', type=Path, required=True, help='Dir containing relaxed solid results')
    parser.add_argument('--water_correction', type=Path, required=True, help='JSON from calculate_water_correction.py')
    parser.add_argument('--target', type=str, required=True, help='Target metal element')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--ion_concentration', type=float, default=1e-6, help='Ion concentration in M')
    parser.add_argument('--exclude_phases', nargs='*', help='Formulas to exclude (e.g. Zn(HO)2)')
    parser.add_argument('--mlip_name', type=str, default="MLIP", help='MLIP name for plot title')
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine Chemical Potentials
    # Check if we can use pre-computed elemental energies
    # Standard path to elemental-energies skill resources
    skill_dir = Path(__file__).resolve().parent.parent.parent
    ee_resources = skill_dir / "elemental-energies/resources"
    
    # Try to load elemental energies JSON for this MLIP
    # Handle different naming conventions or just try exact match first
    # The user should pass the exact checkpoing name used in elemental-energies (e.g. MACE-OMAT-0-small)
    ee_json_path = ee_resources / f"{args.mlip_name}_energies.json"
    ee_data = {}
    if ee_json_path.exists():
        print(f"Loading pre-computed energies from {ee_json_path}")
        with open(ee_json_path) as f:
            ee_data = json.load(f)
    else:
        print(f"Warning: No pre-computed energies found for {args.mlip_name}. Will rely on local relaxations.")

    def get_mu_element(element_symbol: str) -> float:
        """Get chemical potential (eV/atom) for an element."""
        # 1. Try Elemental Energies Library
        if element_symbol in ee_data:
            return ee_data[element_symbol]

        
        # 2. Try Local Relaxation (relaxed_solids)
        # Look for {Element}_* or just {Element}
        # Special case: H might be H2, O might be O2
        candidates = []
        possible_names = [element_symbol, f"{element_symbol}2"] # e.g. H, H2
        
        for p in possible_names:
             candidates.extend(list(args.relaxed_solids.glob(f"{p}_*")))
             candidates.extend(list(args.relaxed_solids.glob(f"{p}")))
             
        # Filter to actual elements
        valid_candidates = []
        for cand in candidates:
             if not cand.is_dir(): continue
             try:
                from pymatgen.core import Structure
                s = Structure.from_file(str(cand / "relaxed_structure.cif"))
                # Verify composition
                if s.composition.is_element and s.composition.elements[0].symbol == element_symbol:
                     with open(cand / "relaxed_energy.txt") as f:
                        e = float(f.read().strip())
                     valid_candidates.append(e / len(s))
             except Exception:
                 continue
                 
        if valid_candidates:
            return min(valid_candidates) # Return most stable
            
        raise ValueError(f"Could not find chemical potential for {element_symbol}. "
                         "Please ensure it is in elemental-energies OR relaxed locally.")

    print("Determining reference chemical potentials...")
    try:
        mu_H = get_mu_element("H")
        mu_metal = get_mu_element(args.target)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # 3. Calculate mu_O from H2O
    # We always need H2O from local relaxation because it's not an element
    h2o_candidates = list(args.relaxed_solids.glob("H2O*"))
    if not h2o_candidates:
        print("Error: Could not find H2O in relaxed_solids. H2O is required for water correction.")
        return 1
    
    # Use the most stable H2O found
    e_h2o_min = float('inf')
    n_h2o_atoms = 3 # Expecting H2O molecule or unit
    # Check actual structure to be safe
    best_h2o_path = None
    
    for cand in h2o_candidates:
        try:
             # Read energy
             with open(cand / "relaxed_energy.txt") as f:
                e = float(f.read().strip())
             # Read structure to get composition
             from pymatgen.core import Structure
             s = Structure.from_file(str(cand / "relaxed_structure.cif"))
             comp = s.composition
             
             # Check if it's H2O (H:2, O:1 ratio)
             if abs(comp.get_atomic_fraction("H") - 0.666) < 0.05:
                 # Normalize to per H2O formula unit
                 # Count number of O atoms
                 n_O_in_cell = comp["O"]
                 e_per_formula = e / n_O_in_cell
                 if e_per_formula < e_h2o_min:
                     e_h2o_min = e_per_formula
                     best_h2o_path = cand
        except Exception:
            continue
            
    if e_h2o_min == float('inf'):
        print("Error: Found H2O directories but could not validate energies/composition.")
        return 1

    print(f"Using H2O from: {best_h2o_path}")
    
    # Calculate mu_O
    # G_f_exp(H2O) = -2.4583 eV
    # G_f_calc = E(H2O) - 2*mu_H - mu_O
    # => mu_O = E(H2O) - 2*mu_H - G_f_exp
    DGF_H2O_EXP = -2.4583
    mu_O = e_h2o_min - 2 * mu_H - DGF_H2O_EXP
    
    print(f"Standard Referencing:")
    print(f"  mu_{args.target}: {mu_metal:.4f} eV/atom")
    print(f"  mu_H:  {mu_H:.4f} eV/atom")
    print(f"  mu_O:  {mu_O:.4f} eV/atom (derived from H2O)")
    print()

    # 2. Load and shift Solids
    mlip_solids = load_relaxed_solids(args.relaxed_solids, args.exclude_phases)
    all_entries = []
    
    print("Solid Entries (Calculated Formation Energies):")
    for name, data in mlip_solids.items():
        comp = data['composition']
        e_total = data['energy']
        
        n_metal = comp.get(args.target, 0)
        n_O = comp.get("O", 0)
        n_H = comp.get("H", 0)
        
        # Formation energy G_f = E - sum(n_i * mu_i)
        # This is the standard formation energy relative to elements (H2, O2, Metal)
        g_f = e_total - (n_metal * mu_metal + n_O * mu_O + n_H * mu_H)
        
        # In pymatgen, PourbaixEntry(ComputedEntry(comp, g_f)) 
        # will internally calculate: e_pbx = g_f - n_O * MU_H2O
        # This matches Persson's E = G_f - n_O * G_f(H2O) since MU_H2O is ~ -2.4583 eV.
        
        entry = ComputedEntry(comp, g_f, entry_id=f"mlip_{name}")
        pb_entry = PourbaixEntry(entry)
        all_entries.append(pb_entry)
        print(f"  {name:20} | G_f(elemental)={g_f:8.3f} | G_pbx(SHE)={pb_entry.energy:8.3f}")

    # 3. Load MP Ions
    print("\nQuerying MP for aqueous ions...")
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        print("Error: MP_API_KEY not set.")
        return 1
        
    with MPRester(api_key) as mpr:
        # get_pourbaix_entries returns entries already wrapped in PourbaixEntry
        # and their energies are already referenced to MP's internal scale.
        # Since we are following the MP/Persson scheme, we can use them directly.
        mp_pb_entries = mpr.get_pourbaix_entries([args.target, 'O', 'H'])
    
    for pb_entry in mp_pb_entries:
        if pb_entry.phase_type == "Ion":
            # Set desired concentration
            pb_entry.concentration = args.ion_concentration
            all_entries.append(pb_entry)
            print(f"  {pb_entry.name:20} | PB_E={pb_entry.energy:8.3f} (Concentration={args.ion_concentration})")

    # 4. Construct Diagram
    comp_dict = {args.target: 1.0}
    pb = PourbaixDiagram(all_entries, comp_dict=comp_dict, filter_solids=True)
    
    print(f"\nStable Entries:")
    for e in pb.stable_entries:
        print(f"  {e.name} ({e.phase_type})")

    # 5. Plotting
    plotter = PourbaixPlotter(pb)
    fig = plotter.get_pourbaix_plot(limits=[[0, 14], [-2, 2]])
    plt.title(f"Pourbaix Diagram: {args.target} ({args.mlip_name})")
    
    plot_path = args.output / "pourbaix_diagram.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save stable entries
    stable_data = []
    for e in pb.stable_entries:
        stable_data.append({
            "name": e.name,
            "phase_type": e.phase_type,
            "energy": e.energy,
            "entry_id": e.entry_id
        })
        
    with open(args.output / "stable_entries.json", "w") as f:
        json.dump(stable_data, f, indent=2)

    print(f"\n✓ Saved results to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
