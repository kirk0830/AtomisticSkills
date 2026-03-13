"""
Load and visualize an experimental XRD pattern from the opXRD dataset.

Usage:
    python load_sample.py --index 0 --output opxrd_sample.png

Requirements:
    - Conda environment: xrd-agent
    - Required packages: opxrd, matplotlib
"""

import argparse
from typing import Dict, Any

from opxrd import OpXRD as OpxrdDataset
from xrdpattern.crystal import CrystalStructure
import matplotlib.pyplot as plt

# Monkeypatch CrystalStructure for backwards compatibility with 0.9.0 Zenodo exports
_old_init = CrystalStructure.__init__
def _new_init(self, lattice=None, basis=None, **kwargs):
    _old_init(self, lattice=lattice, basis=basis, **kwargs)
CrystalStructure.__init__ = _new_init

def load_and_plot(index: int, output_path: str) -> Dict[str, Any]:
    """
    Load a pattern from opXRD and plot it, then return its metadata.
    
    Args:
        index: The index of the sample in the dataset.
        output_path: Path to save the plot.
        
    Returns:
        The dataset sample dictionary.
    """
    # Instantiate    # 2. Get dataset (PyTorch style wrapper for the loaded experimental diffractograms)
    dataset = OpxrdDataset.load('./research/opxrd_data', download=True)
    
    # Load the specific sample
    sample = dataset.patterns[index]
    
    # Extract data for plotting
    two_theta = sample.two_theta_values
    intensity = sample.intensities
    
    # Simple plot
    plt.figure(figsize=(10, 5))
    plt.plot(two_theta, intensity, color='black', linewidth=1)
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    
    # We can fetch sample information natively
    sample_id = f"Pattern {index}"
    plt.title(f'opXRD Experimental Pattern: {sample_id}')
    plt.grid(True, alpha=0.3)
    
    # Just print the metadata string if needed instead of looking for specific fields
    plt.text(0.95, 0.95, f'Metadata Present', 
             transform=plt.gca().transAxes, 
             ha='right', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                 
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Successfully saved plot for index {index} to {output_path}")
    print("\nMetadata Object:")
    print(f" - {sample.metadata}")
        
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot a sample from opXRD dataset.")
    parser.add_argument("--index", type=int, default=0, help="Index of the diffractogram to load.")
    parser.add_argument("--output", type=str, default="opxrd_sample.png", help="Path to save the plotted XRD pattern.")
    
    args = parser.parse_args()
    
    load_and_plot(args.index, args.output)
