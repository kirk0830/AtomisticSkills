import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

def collect_label_distributions(training_data: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    Collect distributions of energy, force, and stress labels from training data.
    
    Args:
        training_data: List of training samples, each containing:
            - 'structure': ASE Atoms or pymatgen Structure
            - 'energy': Total energy (eV)
            - 'forces': Forces array (eV/Å)
            - 'stress': Stress tensor (eV/Å³)
    
    Returns:
        Dictionary with keys:
            - 'energy_distribution': List of per-atom energies (eV/atom)
            - 'force_distribution': List of all force components (eV/Å)
            - 'stress_distribution': List of all stress components (eV/Å³)
    """
    from ase import Atoms
    from pymatgen.core import Structure
    
    energies = []
    forces = []
    stresses = []
    
    for data in training_data:
        # Extract structure to get number of atoms
        structure = data.get('structure')
        if structure is None:
            continue
        
        # Convert to ASE Atoms if needed
        if isinstance(structure, dict):
            # Try to convert from dict representation
            if '@module' in structure and '@class' in structure:
                # pymatgen Structure dict
                from pymatgen.core import Structure as PymatgenStructure
                from pymatgen.io.ase import AseAtomsAdaptor
                struct = PymatgenStructure.from_dict(structure)
                atoms = AseAtomsAdaptor.get_atoms(struct)
            else:
                # Assume it's already an ASE Atoms dict or skip
                continue
        elif isinstance(structure, Structure):
            from pymatgen.io.ase import AseAtomsAdaptor
            atoms = AseAtomsAdaptor.get_atoms(structure)
        elif isinstance(structure, Atoms):
            atoms = structure
        else:
            continue
        
        num_atoms = len(atoms)
        
        # Extract energy and convert to per-atom
        if 'energy' in data and data['energy'] is not None:
            energy = float(data['energy'])
            energy_per_atom = energy / num_atoms if num_atoms > 0 else energy
            energies.append(energy_per_atom)
        
        # Extract forces and flatten
        if 'forces' in data and data['forces'] is not None:
            f = np.array(data['forces']).flatten()
            forces.extend(f.tolist())
            
        # Extract stress and flatten
        if 'stress' in data and data['stress'] is not None:
            s = np.array(data['stress']).flatten()
            stresses.extend(s.tolist())
            
    return {
        'energy_distribution': energies,
        'force_distribution': forces,
        'stress_distribution': stresses
    }

def plot_label_distributions(distributions: Dict[str, List[float]], save_path: Optional[str] = None, show: bool = True) -> None:
    """
    Plot the distributions of energies, forces, and stresses.
    """
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    fig.suptitle('Training Data Label Distributions', fontsize=16, fontweight='bold')
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    axes = [ax1, ax2, ax3]
    
    if 'energy_distribution' in distributions and len(distributions['energy_distribution']) > 0:
        energies = distributions['energy_distribution']
        axes[0].hist(energies, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_xlabel('Energy (eV/atom)')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('Energy Distribution')
    
    if 'force_distribution' in distributions and len(distributions['force_distribution']) > 0:
        forces = distributions['force_distribution']
        axes[1].hist(forces, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1].set_xlabel('Force (eV/Å)')
        axes[1].set_ylabel('Count')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No force data', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Force Distribution')
    
    if 'stress_distribution' in distributions and len(distributions['stress_distribution']) > 0:
        stresses = distributions['stress_distribution']
        axes[2].hist(stresses, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[2].set_xlabel('Stress (eV/Å³)')
        axes[2].set_ylabel('Count')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No stress data', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('Stress Distribution')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Label distribution plot saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_training_history(training_history: Dict[str, Any], save_path: Optional[str] = None, show: bool = True, model_name: str = 'Model') -> None:
    """
    Plot training history showing Energy/Force/Stress MAE and Loss.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[0, 0])  # Energy MAE
    ax5 = fig.add_subplot(gs[0, 1])  # Force MAE
    ax6 = fig.add_subplot(gs[1, 0])  # Stress MAE
    ax_loss = fig.add_subplot(gs[1, 1]) # Loss
    
    axes = [ax4, ax5, ax6]
    
    # Energy MAE (meV/atom)
    energy_mae_train = [x for x in training_history.get('energy_mae_train', []) if x is not None]
    energy_mae_val = [x for x in training_history.get('energy_mae_val', []) if x is not None]
    if len(energy_mae_train) > 0 or len(energy_mae_val) > 0:
        if len(energy_mae_train) > 0:
            epochs_train = range(1, len(energy_mae_train) + 1)
            axes[0].plot(epochs_train, np.array(energy_mae_train), 'b-', label='Train', linewidth=2, marker='o')
        if len(energy_mae_val) > 0:
            epochs_val = range(1, len(energy_mae_val) + 1)
            axes[0].plot(epochs_val, np.array(energy_mae_val), 'r-', label='Validation', linewidth=2, marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Energy MAE (meV/atom)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No energy MAE data', ha='center', va='center', transform=axes[0].transAxes)
    axes[0].set_title('Energy MAE')
    
    # Force MAE (meV/Å)
    force_mae_train = [x for x in training_history.get('force_mae_train', []) if x is not None]
    force_mae_val = [x for x in training_history.get('force_mae_val', []) if x is not None]
    if len(force_mae_train) > 0 or len(force_mae_val) > 0:
        if len(force_mae_train) > 0:
            epochs_train = range(1, len(force_mae_train) + 1)
            axes[1].plot(epochs_train, np.array(force_mae_train), 'b-', label='Train', linewidth=2, marker='o')
        if len(force_mae_val) > 0:
            epochs_val = range(1, len(force_mae_val) + 1)
            axes[1].plot(epochs_val, np.array(force_mae_val), 'r-', label='Validation', linewidth=2, marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Force MAE (meV/Å)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No force MAE data', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_title('Force MAE')
    
    # Stress MAE (meV/Å³)
    stress_mae_train = [x for x in training_history.get('stress_mae_train', []) if x is not None]
    stress_mae_val = [x for x in training_history.get('stress_mae_val', []) if x is not None]
    if len(stress_mae_train) > 0 or len(stress_mae_val) > 0:
        if len(stress_mae_train) > 0:
            epochs_train = range(1, len(stress_mae_train) + 1)
            axes[2].plot(epochs_train, np.array(stress_mae_train), 'b-', label='Train', linewidth=2, marker='o')
        if len(stress_mae_val) > 0:
            epochs_val = range(1, len(stress_mae_val) + 1)
            axes[2].plot(epochs_val, np.array(stress_mae_val), 'r-', label='Validation', linewidth=2, marker='s')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Stress MAE (meV/Å³)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No stress MAE data', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title('Stress MAE')
    
    # Loss plot
    loss_train = [x for x in training_history.get('loss_train', []) if x is not None]
    loss_val = [x for x in training_history.get('loss_val', []) if x is not None]
    if len(loss_train) > 0 or len(loss_val) > 0:
        if len(loss_train) > 0:
            epochs_train = range(1, len(loss_train) + 1)
            ax_loss.plot(epochs_train, loss_train, 'b-', label='Train', linewidth=2, marker='o')
        if len(loss_val) > 0:
            epochs_val = range(1, len(loss_val) + 1)
            ax_loss.plot(epochs_val, loss_val, 'r-', label='Validation', linewidth=2, marker='s')
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_loss.legend(fontsize=12)
        ax_loss.grid(True, alpha=0.3)
    else:
        ax_loss.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax_loss.transAxes)
    ax_loss.set_title('Training Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0, w_pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
        
    if show:
        plt.show()
    else:
        plt.close()
