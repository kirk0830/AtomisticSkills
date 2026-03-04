import json
import numpy as np
import ase.units
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import sys
import os

# Ensure we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))))
from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

def main():
    val_data_path = "output/val_data.json"
    checkpoint_path = "output/fine_tuned_model.pth"
    
    print(f"Loading val data from {val_data_path}...")
    with open(val_data_path, "r") as f:
        val_data = json.load(f)
        
    print(f"Loading MatGL wrapper from {checkpoint_path}...")
    wrapper = MatGLWrapper(checkpoint_path)
    
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    wrapper.model_name = ckpt["model_name"]
    wrapper.load(model_path=None) # Load architecture
    wrapper.model.load_state_dict(ckpt["weights"])
    wrapper.is_loaded = True
    
    stresses_true = []
    stresses_pred = []
    
    for i, item in enumerate(val_data):
        struct = Structure.from_dict(item["structure"])
        atoms = AseAtomsAdaptor().get_atoms(struct)
        
        # True stress in eV/A^3
        true_stress = np.array(item["stress"])
        if true_stress.shape == (3, 3):
            true_stress = np.array([
                true_stress[0,0], true_stress[1,1], true_stress[2,2],
                true_stress[1,2], true_stress[0,2], true_stress[0,1]
            ])
        elif true_stress.shape != (6,):
            true_stress = np.zeros(6)
        
        # Predict natively to force stress computation
        if not wrapper.calculator:
            wrapper.calculator = wrapper.create_calculator()
        atoms.calc = wrapper.calculator
        try:
            pred_stress = atoms.get_stress()
        except Exception as e:
            print(f"Failed to get stress for item {i}: {e}")
            pred_stress = np.zeros(6)
        
        stresses_true.append(true_stress)
        stresses_pred.append(pred_stress)
        
    stresses_true = np.array(stresses_true)
    stresses_pred = np.array(stresses_pred)
    
    # Check max and mean values to see standard scale
    print(f"True stress mean std: {np.std(stresses_true):.6f} eV/A^3")
    print(f"Pred stress mean std: {np.std(stresses_pred):.6f} eV/A^3")
    
    mae_ev = np.mean(np.abs(stresses_true - stresses_pred))
    mae_mev = mae_ev * 1000
    print(f"\n=============================================")
    print(f"Validation Stress MAE: {mae_mev:.3f} meV/A^3")
    print(f"Validation Stress MAE: {mae_ev:.6f} eV/A^3")
    print(f"Validation Stress MAE (GPa): {mae_ev / ase.units.GPa:.6f} GPa")
    print(f"=============================================\n")

if __name__ == "__main__":
    main()
