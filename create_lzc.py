import ase.io
import random
try:
    atoms = ase.io.read('/home/bdeng/Li3InCl6.cif')
    print("Read CIF")
    li_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'Li']
    in_indices = [i for i, atom in enumerate(atoms) if atom.symbol == 'In']
    print(f"Found {len(li_indices)} Li, {len(in_indices)} In")
    for i in in_indices:
        atoms[i].symbol = 'Zr'
    to_remove = random.sample(li_indices, 6)
    del atoms[to_remove]
    out_path = '/home/bdeng/projects/simulation_mcp/research/2026-01-28_Li2ZrCl6_melting_point/Li2ZrCl6_initial.cif'
    ase.io.write(out_path, atoms)
    print(f"Saved to {out_path}")
except Exception as e:
    print(f"Error: {e}")
