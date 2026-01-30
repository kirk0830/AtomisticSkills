"""
Construct Pourbaix diagram using water-corrected MLIP energies.

This script implements the Persson et al. (2012) methodology for thermodynamically
consistent Pourbaix diagrams by:
1. Using MLIP total energies for solids
2. Applying water correction to align computational and experimental scales  
3. Combining with MP aqueous ion data

Reference:
    K. A. Persson, B. Waldwick, P. Lazic, G. Ceder
    Phys. Rev. B 85, 235438 (2012)

Usage:
    python calculate_pourbaix_custom.py \
        --relaxed_solids ./relaxed_solids \
        --water_correction ./water_correction.json \
        --target Ta \
        --output ./pourbaix_results
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.analysis.pourbaix_diagram import (
    PourbaixDiagram, PourbaixEntry, PourbaixPlotter
)
from pymatgen.entries.computed_entries import ComputedEntry  
from pymatgen.core import Composition, Element
from mp_api.client import MPRester
import os


# Constants
K_B = 8.617333262e-5  # eV/K
T_STANDARD = 298.15   # K
E_CHARGE = 1.0        # Elementary charge (for voltage terms)


def load_mlip_energies(relaxed_dir: Path, exclude_formulas: List[str] = None) -> Dict[str, Dict]:
    """
    Load MLIP total energies from relaxed structures.
    
    Args:
        relaxed_dir: Directory containing subdirectories with result.json files
        exclude_formulas: List of chemical formulas to exclude (e.g., ['Mn29H2'])
        
    Returns:
        Dictionary mapping structure names to energy data
    """
    if exclude_formulas is None:
        exclude_formulas = []
    
    energies = {}
    excluded_count = 0
    
    for subdir in relaxed_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        result_file = subdir / 'result.json'
        energy_file = subdir / 'relaxed_energy.txt'
        cif_file = subdir / 'relaxed_structure.cif'
        
        data = {}
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
        elif energy_file.exists():
            with open(energy_file) as f:
                try:
                    data['energy'] = float(f.read().strip())
                except ValueError:
                    print(f"  Warning: Could not parse energy from {energy_file}")
                    continue
        else:
            continue
        
        # Get composition from structure
        if cif_file.exists():
            from pymatgen.core import Structure
            struct = Structure.from_file(str(cif_file))
            comp = struct.composition
        else:
            # Parse from directory name
            name = subdir.name.replace('_mp-', ' ').split()[0]
            comp = Composition(name)
        
        # Check if this phase should be excluded
        formula = comp.reduced_formula
        # Check against exclude list (simple string match)
        if any(excl in subdir.name or excl == formula for excl in exclude_formulas):
            print(f"  Excluding: {subdir.name} ({formula})")
            excluded_count += 1
            continue
        
        energies[subdir.name] = {
            'energy': data['energy'],
            'composition': comp.as_dict(),
            'num_atoms': comp.num_atoms
        }
    
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} phase(s)")
    
    return energies


def create_pourbaix_entries_with_correction(
    mlip_energies: Dict[str, Dict],
    mu_H_ref: float,
    mu_O: float,
    pH: float = 0.0,
    voltage: float = 0.0,
    temperature: float = T_STANDARD
) -> List[PourbaixEntry]:
    """
    Create PourbaixEntry objects from MLIP energies.
    
    Args:
        mlip_energies: Dictionary of MLIP energies
        mu_H_ref: Reference H chemical potential from water correction (eV/H) - NOT USED HERE
        mu_O: O chemical potential (eV/O) - NOT USED HERE
        pH: pH value (for voltage/pH dependent correction) - NOT USED HERE
        voltage: Applied voltage vs SHE (V) - NOT USED HERE
        temperature: Temperature (K) - NOT USED HERE
        
    Returns:
        List of PourbaixEntry objects
    """
    entries = []
    
    for name, data in mlip_energies.items():
        comp = Composition.from_dict(data['composition'])
        E_total = data['energy']
        
        # PourbaixEntry expects an energy that is formation energy relative to elements
        # OR total energy if we provide the corrections.
        # But here we are creating ComputedEntry with the ALIGNED energy (calculated in main)
        # So it acts as the formation energy relative to the reference states we defined.
        
        entry = ComputedEntry(
            composition=comp,
            energy=E_total,  # This is actually the formation energy calculated in main()
            entry_id=f"mlip_{name}"
        )
        
        pb_entry = PourbaixEntry(entry)
        entries.append(pb_entry)
    
    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Pourbaix diagram with water correction'
    )
    parser.add_argument('--relaxed_solids', type=Path, required=True,
                       help='Directory with MLIP-relaxed solid structures')
    parser.add_argument('--water_correction', type=Path, required=True,
                       help='Water correction JSON file from calculate_water_correction.py')
    parser.add_argument('--target', type=str, required=True,
                       help='Target material formula (e.g., ZnO)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--ion_concentration', type=float, default=1e-6,
                       help='Ion concentration (M)')
    parser.add_argument('--ph_range', nargs=2, type=float, default=[0, 14],
                       help='pH range for diagram')
    parser.add_argument('--voltage_range', nargs=2, type=float, default=[-3, 3],
                       help='Voltage range (V vs SHE)')
    parser.add_argument('--exclude_phases', nargs='*', default=[],
                       help='Phases to exclude (e.g., Mn29H2). Matches formula in directory name.')
    
    parser.add_argument('--mlip_name', type=str, default=None,
                       help='Name of the MLIP model to display on the plot')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Pourbaix Diagram with Water Correction (Persson 2012)")
    print("="*70)
    print(f"Target: {args.target}")
    print()
    
    # Identify metal element
    target_comp = Composition(args.target)
    # Get the metal element (excluding O and H)
    metal_el = [el for el in target_comp.elements if el.symbol not in ["O", "H"]]
    if len(metal_el) != 1:
        print(f"Error: Target {args.target} must contain exactly one metal element (besides O/H).")
        # Fallback: if target is just O or H?
        return 1
    metal_symbol = metal_el[0].symbol
    print(f"Identified metal element: {metal_symbol}")

    
    # Load water correction
    print("Loading water correction...")
    with open(args.water_correction) as f:
        correction = json.load(f)
    
    mu_H_ref = correction['mu_H_ref']
    mu_O = correction['mu_O']
    
    print(f"  μ_H^ref = {mu_H_ref:.6f} eV/H")
    print(f"  μ_O     = {mu_O:.6f} eV/O")
    print(f"  ΔGf(H2O)_corrected = {correction['DGf_H2O_calc']:.6f} eV")
    print()
    
    # Load MLIP solid energies
    print("Loading MLIP solid energies...") 
    mlip_energies = load_mlip_energies(args.relaxed_solids, exclude_formulas=args.exclude_phases)
    print(f"  Loaded {len(mlip_energies)} solid structures")
    print()
    
    # ---------------------------------------------------------
    # SCALE ALIGNMENT: Convert Absolute Energies to Formation Energies
    # This is critical to match the scale of MP Ions (which use Ef)
    # ---------------------------------------------------------
    
    # 1. Find Metal metal reference (lowest energy per atom)
    metal_ref_energy = None
    min_e_per_atom = float('inf')
    
    for name, data in mlip_energies.items():
        comp = Composition.from_dict(data['composition'])
        # Check if it is the pure metal
        if comp.reduced_formula == metal_symbol:
            e_per_atom = data['energy'] / data['num_atoms']
            if e_per_atom < min_e_per_atom:
                min_e_per_atom = e_per_atom
                metal_ref_energy = e_per_atom
    
    if metal_ref_energy is None:
        print(f"Error: No {metal_symbol} metal found in relaxed solids! Cannot calculate reference scale.")
        return 1
        
    print(f"  Reference Mu({metal_symbol}) = {metal_ref_energy:.4f} eV/atom")
    
    
    # 2. Get O and H references
    # CRITICAL: We utilize the corrected chemical potentials directly from the JSON
    # These values now include entropy corrections for H2 gas and water cycle consistency.
    
    # We already loaded these earlier:
    # mu_H_ref = correction['mu_H_ref']
    # mu_O = correction['mu_O']
    
    # Just assign them to variables used in the formation energy loop
    mu_H_mlip = mu_H_ref
    mu_O_mlip = mu_O
    
    print(f"  Reference Mu(H)  = {mu_H_mlip:.4f} eV/atom (Entropy corrected)")
    print(f"  Reference Mu(O)  = {mu_O_mlip:.4f} eV/atom (Derived from H2O cycle)")

    
    # 3. Apply reference correction to all solids
    # E_form = E_total - Sum(N_i * mu_i)
    # We use these Formation Energies for Pourbaix, which aligns 
    # the MLIP scale (Elements=0) with the MP Ion scale (Elements=0)
    aligned_energies = {}
    for name, data in mlip_energies.items():
        comp = Composition.from_dict(data['composition'])
        e_total = data['energy']
        
        n_metal = comp.get(metal_symbol, 0)
        if n_metal == 0:
            continue
            
        n_O = comp.get("O", 0)
        n_H = comp.get("H", 0)
        
        # Calculate formation energy
        # Note: This assumes only Metal, O, H system
        e_form = e_total - (n_metal * metal_ref_energy + n_O * mu_O_mlip + n_H * mu_H_mlip)
        
        aligned_energies[name] = {
            'energy': e_form,  # Now Formation Energy!
            'composition': data['composition']
        }
    
    print("  Converted all MLIP energies to Formation Energies relative to MLIP elements.")
    print()

    # Create entries with ALIGNED (Formation) energies
    solid_entries = create_pourbaix_entries_with_correction(
        aligned_energies,
        mu_H_ref, # Not used in updated function
        mu_O     # Not used in updated function
    )
    
    # Load MP aqueous ions
    print("Querying MP for aqueous ions...")
    api_key = os.getenv("MP_API_KEY")
    if not api_key:
        print("Warning: MP_API_KEY not set. Using only solids which might be incorrect for Pourbaix.")
    
    # We want ions for the metal + O + H
    chemsys_elements = [metal_symbol, 'O', 'H']
    
    ion_entries = []
    if api_key:
        with MPRester(api_key) as mpr:
            # Get Pourbaix Entries to ensure we get proper ion energies
            # Note: We filter for ions only, as we use our own solids
            mp_pb_entries = mpr.get_pourbaix_entries(chemsys_elements)
        
        ion_entries = [e for e in mp_pb_entries if e.phase_type == "Ion"]
        print(f"  Retrieved {len(ion_entries)} ions from MP")
    else:
        print("  Skipping MP ions (no API key)")
    print()
    
    # Combine entries
    all_entries = solid_entries + ion_entries
    print(f"Total entries: {len(all_entries)} ({len(solid_entries)} MLIP solids + {len(ion_entries)} MP ions)")
    print()
    
    # Construct Pourbaix diagram
    print("Constructing Pourbaix diagram...")
    # comp_dict defines the element of interest concentration (usually 1.0 for solid, or user specified)
    # But PourbaixDiagram takes comp_dict to filter? 
    # "The dictionary of composition of entries. If None, it is inferred from the entries."
    # Wait, PourbaixDiagram signature:
    # def __init__(self, entries, comp_dict=None, filter_solids=False, n_jobs=None)
    # If comp_dict is provided, it sets the composition of the aqueous component?
    
    # Actually, for standard Pourbaix, we usually just pass entries.
    # But usually one specifies the metal element concentration.
    # In pymatgen, comp_dict is {Element: concentration}.
    comp_dict = {metal_symbol: args.ion_concentration}
    
    try:
        pb = PourbaixDiagram(
           all_entries,
            comp_dict=comp_dict,
            filter_solids=True
        )
    except Exception as e:
        print(f"Error constructing diagram: {e}")
        return 1
    
    print(f"  Stable entries: {len(pb.stable_entries)}")
    for entry in pb.stable_entries:
        print(f"    {entry.name} ({entry.phase_type})")
    print()
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Plot
    try:
        plotter = PourbaixPlotter(pb)
        # get_pourbaix_plot returns a matplotlib.pyplot object if no ax is provided? 
        # Actually checking pymatgen source: it typically returns the 'plt' module or the figure.
        # But the error "'Axes' object has no attribute 'gca'" implies 'plt' (the variable) is an Axes object.
        # Let's verify. If it is an axes, we use it directly.
        
        # Pymatgen 2024+ might return Axes.
        returned_object = plotter.get_pourbaix_plot(
            limits=[args.ph_range, args.voltage_range],
            label_domains=True
        )
        
        import matplotlib.pyplot as pyplot
        
        # Check type
        if hasattr(returned_object, 'gca'):
            # It's a structure like plt module or Figure
            ax = returned_object.gca()
            fig = returned_object.gcf()
        elif hasattr(returned_object, 'lines'):
            # likely an Axes object
            ax = returned_object
            fig = ax.get_figure()
        else:
            # Fallback
            ax = pyplot.gca()
            fig = pyplot.gcf()
            
        # Remove central lines logic
        lines_to_remove = []
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            
            # Identify horizontal line at y=0
            if len(ydata) > 1 and all(abs(y) < 1e-6 for y in ydata):
                lines_to_remove.append(line)
            # Identify vertical line at x=0
            if len(xdata) > 1 and all(abs(x) < 1e-6 for x in xdata):
                lines_to_remove.append(line)
            # Identify vertical line at x=7 (neutral pH)
            if len(xdata) > 1 and all(abs(x - 7.0) < 1e-6 for x in xdata):
                lines_to_remove.append(line)
        
        for line in lines_to_remove:
            line.remove()
            
        # Add MLIP Name Label if provided
        if args.mlip_name:
            ax.text(0.95, 0.05, f"MLIP: {args.mlip_name}", 
                    transform=ax.transAxes, 
                    ha='right', va='bottom', 
                    fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
        plot_path = args.output / "pourbaix_diagram.png"
        fig.savefig(plot_path, dpi=300)
        print(f"✓ Saved plot to {plot_path}")
    except Exception as e:
        print(f"⚠ Plotting failed: {e}")
    
    # Save stable entries
    stable = []
    for e in pb.stable_entries:
        stable.append({
            "formula": e.name,
            "energy": e.energy,
            "phase_type": e.phase_type,
            "entry_id": e.entry_id
        })
    
    with open(args.output / "stable_entries.json", 'w') as f:
        json.dump(stable, f, indent=2)
    
    print(f"✓ Saved results to {args.output}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
