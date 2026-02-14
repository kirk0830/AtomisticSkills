#!/usr/bin/env python3
"""
Materials Project Database Query Tool (MP API)

Query Materials Project for inorganic crystal structures with computed properties.
Supports retrieval of:
  - Structures (CIF format)
  - Thermodynamic properties (energy_above_hull, formation_energy_per_atom)
  - Electronic properties (band_gap, is_metal)
  - Structural properties (density, volume, nsites)
  - Magnetic properties (total_magnetization)

Query options:
  - Chemical system (e.g., "Li-S", "Li-Fe-P-O")
  - Chemical formula (e.g., "LiFePO4")
  - List of elements (e.g., Li Fe P O)
  - Property filtering (e.g., energy_above_hull < 0.1 eV/atom for stable materials)

Output formats:
  - JSON: Full structure + property data (compatible with MatterGen training pipeline)
  - CSV: Tabular format for quick inspection (properties only, no structures)

Usage:
    # Query Li-S structures with stability data
    python query_mp.py --chemsys "Li-S" --properties energy_above_hull formation_energy_per_atom --limit 50 --output li_s.json
    
    # Query stable Li-O materials (e_hull < 0.05 eV/atom)
    python query_mp.py --chemsys "Li-O" --properties energy_above_hull --e_above_hull_max 0.05 --limit 20 --output li_o_stable.json
    
    # Query by formula
    python query_mp.py --formula "LiFePO4" --properties energy_above_hull band_gap --output lifepo4.json
    
    # Export to CSV for inspection
    python query_mp.py --chemsys "Li-S" --properties energy_above_hull --limit 10 --output li_s.csv

Requirements:
    - Conda environment: base-agent
    - Required packages: mp-api, pymatgen
    - MP_API_KEY environment variable must be set
"""

import argparse
import json
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


def query_materials_project(
    chemsys: Optional[str] = None,
    formula: Optional[str] = None,
    elements: Optional[List[str]] = None,
    properties: Optional[List[str]] = None,
    e_above_hull_max: Optional[float] = None,
    formation_energy_max: Optional[float] = None,
    limit: int = 100,
    endpoint: str = "summary",
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Query Materials Project for structures with properties.
    
    Args:
        chemsys: Chemical system (e.g., "Li-S", "Li-O-P")
        formula: Chemical formula (e.g., "LiFePO4")
        elements: List of elements to include
        properties: List of properties to retrieve (default: ["energy_above_hull", "formation_energy_per_atom"])
        e_above_hull_max: Maximum energy above hull (eV/atom) for filtering
        formation_energy_max: Maximum formation energy (eV/atom) for filtering
        limit: Maximum number of results
        endpoint: API endpoint to use ("summary" or "thermo", default: "summary")
        api_key: MP API key (defaults to environment)
        
    Returns:
        List of dictionaries with structure and property data
    """
    from mp_api.client import MPRester
    from pymatgen.io.cif import CifWriter
    import io
    
    # Get API key
    mp_key = api_key or os.environ.get('MP_API_KEY')
    if not mp_key:
        raise ValueError("Materials Project API key not found. Set MP_API_KEY environment variable.")
    
    # Default properties if not specified
    if properties is None:
        properties = ["energy_above_hull", "formation_energy_per_atom"]
    
    # Build query fields (thermo endpoint doesn't support structure field)
    if endpoint == "thermo":
        fields = ["material_id", "formula_pretty"] + properties
    else:
        fields = ["material_id", "formula_pretty", "structure"] + properties
    
    print(f"Querying Materials Project...")
    print(f"  Endpoint: {endpoint}")
    print(f"  Chemical system: {chemsys if chemsys else 'N/A'}")
    print(f"  Formula: {formula if formula else 'N/A'}")
    print(f"  Elements: {elements if elements else 'N/A'}")
    print(f"  Properties: {properties}")
    print(f"  Limit: {limit}")
    
    with MPRester(mp_key) as mpr:
        # Build query criteria
        criteria = {}
        
        if chemsys:
            criteria["chemsys"] = chemsys
        if formula:
            criteria["formula"] = formula
        if elements:
            criteria["elements"] = elements
        if e_above_hull_max is not None:
            criteria["energy_above_hull"] = (None, e_above_hull_max)
        if formation_energy_max is not None:
            criteria["formation_energy_per_atom"] = (None, formation_energy_max)
        
        # Query materials using specified endpoint
        if endpoint == "thermo":
            docs = mpr.materials.thermo.search(
                **criteria,
                fields=fields,
                num_chunks=1,
                chunk_size=limit
            )
        else:  # summary endpoint (default)
            docs = mpr.materials.summary.search(
                **criteria,
                fields=fields,
                num_chunks=1,
                chunk_size=limit
            )
    
    print(f"Found {len(docs)} materials")
    
    # Process results
    results = []
    for doc in docs:
        # Build result dict
        result = {
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
        }
        
        # Convert structure to CIF string (only available from summary endpoint)
        if hasattr(doc, 'structure') and doc.structure:
            structure = doc.structure
            cif_writer = CifWriter(structure)
            cif_string = str(cif_writer)
            result["cif"] = cif_string
            result["structure"] = structure.as_dict()  # For programmatic use
        
        # Add requested properties
        for prop in properties:
            if hasattr(doc, prop):
                value = getattr(doc, prop)
                result[prop] = float(value) if value is not None else None
            else:
                result[prop] = None
        
        results.append(result)
    
    return results


def save_to_json(data: List[Dict], output_path: str) -> None:
    """Save data to JSON file."""
    output_path = Path(output_path)
    
    # Remove 'structure' field for cleaner JSON (keep CIF)
    clean_data = []
    for item in data:
        clean_item = {k: v for k, v in item.items() if k != "structure"}
        clean_data.append(clean_item)
    
    with open(output_path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    print(f"Saved {len(data)} materials to {output_path}")


def save_to_csv(data: List[Dict], output_path: str, properties: List[str]) -> None:
    """Save data to CSV file."""
    output_path = Path(output_path)
    
    # Get all unique keys except structure and CIF (too large for CSV)
    fieldnames = ["material_id", "formula"] + properties
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in data:
            row = {k: item.get(k) for k in fieldnames}
            writer.writerow(row)
    
    print(f"Saved {len(data)} materials to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Query Materials Project for structures with properties"
    )
    
    # Query options
    parser.add_argument("--chemsys", type=str, help="Chemical system (e.g., 'Li-S', 'Li-O-P')")
    parser.add_argument("--formula", type=str, help="Chemical formula (e.g., 'LiFePO4')")
    parser.add_argument("--elements", nargs="+", help="Elements to include (e.g., Li S O)")
    
    # Property options
    parser.add_argument(
        "--properties", 
        nargs="+", 
        default=["energy_above_hull", "formation_energy_per_atom"],
        help="Properties to retrieve (default: energy_above_hull formation_energy_per_atom)"
    )
    
    # Filters
    parser.add_argument("--endpoint", type=str, choices=["summary", "thermo"], default="summary",
                        help="API endpoint to use (default: summary, use 'thermo' for more detailed thermodynamic data)")
    parser.add_argument("--e_above_hull_max", type=float, help="Maximum energy above hull (eV/atom)")
    parser.add_argument("--formation_energy_max", type=float, help="Maximum formation energy (eV/atom)")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of results (default: 100)")
    
    # Output
    parser.add_argument("--output", type=str, required=True, help="Output file path (.json or .csv)")
    parser.add_argument("--api_key", type=str, help="Materials Project API key (defaults to MP_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not any([args.chemsys, args.formula, args.elements]):
        parser.error("Must specify at least one of: --chemsys, --formula, --elements")
    
    # Query Materials Project
    try:
        results = query_materials_project(
            chemsys=args.chemsys,
            formula=args.formula,
            elements=args.elements,
            properties=args.properties,
            e_above_hull_max=args.e_above_hull_max,
            formation_energy_max=args.formation_energy_max,
            limit=args.limit,
            endpoint=args.endpoint,
            api_key=args.api_key
        )
        
        if not results:
            print("No materials found matching criteria")
            sys.exit(1)
        
        # Save results
        output_path = Path(args.output)
        if output_path.suffix == ".csv":
            save_to_csv(results, args.output, args.properties)
        else:
            save_to_json(results, args.output)
        
        print(f"\n✓ Successfully retrieved {len(results)} materials")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
