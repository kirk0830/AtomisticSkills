
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
    
    # 1. Load water correction and elemental potentials
    with open(args.water_correction) as f:
        corr = json.load(f)
    
    mu_H = corr['mu_H_ref']
    mu_O = corr['mu_O']
    
    # Find most stable elemental metal reference
    mu_metal = float('inf')
    metal_candidates = list(args.relaxed_solids.glob(f"{args.target}_mp-*"))
    for cand in metal_candidates:
        # Check if it's actually an element
        # (Simple check: name shouldn't contain O or H or other capitals after symbol)
        # Better: Load composition
        try:
            with open(cand / "relaxed_energy.txt") as f:
                e = float(f.read().strip())
            from pymatgen.core import Structure
            s = Structure.from_file(str(cand / "relaxed_structure.cif"))
            if s.composition.is_element:
                e_per_atom = e / len(s)
                if e_per_atom < mu_metal:
                    mu_metal = e_per_atom
        except Exception:
            continue
            
    if mu_metal == float('inf'):
        print(f"Error: Could not find any elemental metal reference for {args.target}")
        return 1
    
    print(f"Standard Referencing:")
    print(f"  mu_{args.target}: {mu_metal:.4f} eV/atom")
    print(f"  mu_H:  {mu_H:.4f} eV/atom")
    print(f"  mu_O:  {mu_O:.4f} eV/atom")
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
