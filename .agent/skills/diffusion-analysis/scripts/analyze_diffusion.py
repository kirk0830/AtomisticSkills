import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
try:
    from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer
except ImportError:
    from pymatgen.analysis.diffusion_analyzer import DiffusionAnalyzer
import os

def analyze_diffusion(traj_path, species, temperature, time_step=1.0, log_interval=100, start_skip=50, output_dir="."):
    """
    Analyze diffusion using pymatgen's DiffusionAnalyzer.
    
    Args:
        traj_path: Path to the ASE trajectory file.
        species: Diffusing species element (e.g., 'Li').
        temperature: Simulation temperature in K.
        time_step: MD time step in fs (default: 1.0).
        log_interval: Number of steps between each frame in the trajectory (default: 100).
        start_skip: Number of initial frames to skip (equilibration) (default: 50).
        output_dir: Directory to save results (default: ".").
    """
    print(f"Loading trajectory from {traj_path}...")
    atoms_list = read(traj_path, index=":")
    
    if len(atoms_list) <= start_skip:
        print(f"Error: Trajectory has only {len(atoms_list)} frames, but start_skip is {start_skip}.")
        return

    # Exclude initial equilibration segment
    print(f"Skipping first {start_skip} frames (equilibration)...")
    atoms_list = atoms_list[start_skip:]
    
    # Convert ASE atoms to pymatgen structures
    print(f"Converting {len(atoms_list)} frames to pymatgen structures...")
    adaptor = AseAtomsAdaptor()
    structures = [adaptor.get_structure(atoms) for atoms in atoms_list]
    
    # Convert MD time step from fs to ps
    time_step_ps = time_step / 1000.0
    
    print(f"Running DiffusionAnalyzer for species {species} at {temperature}K...")
    # Map time_step and log_interval directly to DiffusionAnalyzer
    # DiffusionAnalyzer handles the multiplication: dt = step * time_step * step_skip
    analyzer = DiffusionAnalyzer.from_structures(
        structures=structures,
        specie=species,
        temperature=temperature,
        time_step=time_step_ps,
        step_skip=log_interval, 
    )
    
    # Print results
    d = analyzer.diffusivity
    print(f"Diffusivity (D): {d:.5e} cm^2/s")
    
    # Save plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    analyzer.get_msd_plot().savefig(os.path.join(output_dir, f"msd_{species}.png"))
    print(f"Saved MSD plot to {output_dir}/msd_{species}.png")
    
    # Save results to file
    with open(os.path.join(output_dir, "diffusion_results.txt"), "a") as f:
        f.write(f"T: {temperature}K, Species: {species}, D: {d:.5e} cm^2/s\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze diffusion from an ASE trajectory using pymatgen.")
    parser.add_argument("traj_path", help="Path to ASE trajectory file (.traj)")
    parser.add_argument("--species", required=True, help="Diffusing element (e.g., Li)")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature in K")
    parser.add_argument("--time_step", type=float, default=1.0, help="MD time step in fs (default: 1.0)")
    parser.add_argument("--log_interval", type=int, default=100, help="Steps between frames in trajectory (default: 100)")
    parser.add_argument("--start_skip", type=int, default=0, help="Number of initial frames to skip for equilibration (default: 0)")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    analyze_diffusion(
        args.traj_path,
        args.species,
        args.temperature,
        args.time_step,
        args.log_interval,
        args.start_skip,
        args.output_dir
    )
