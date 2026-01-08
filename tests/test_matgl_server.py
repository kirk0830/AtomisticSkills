
import pytest
import os
import shutil
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from mlip_mcp_wrappers.mcp_server import matgl_server

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    if os.path.exists("./results/matgl_test"):
        shutil.rmtree("./results/matgl_test")

@pytest.fixture(scope="module")
def loaded_server():
    # Use M3GNet as it is usually fastest/easiest
    res = matgl_server.load_model("M3GNet", device="cpu")
    if "error" in res:
        pytest.fail(f"Failed to load model: {res}")
    return matgl_server

@pytest.fixture
def cu_structure():
    atoms = bulk("Cu")
    return AseAtomsAdaptor.get_structure(atoms).as_dict()

def test_predict_structure(loaded_server, cu_structure):
    res = loaded_server.predict_structure(cu_structure)
    assert "energy" in res
    assert "forces" in res

def test_relax_structure(loaded_server, cu_structure):
    output_dir = os.path.abspath("./results/matgl_test/relax")
    res = loaded_server.relax_structure(
        cu_structure, 
        steps=5, 
        output_dir=output_dir
    )
    assert "error" not in res
    assert "final_structure" in res

def test_run_md(loaded_server, cu_structure):
    output_dir = os.path.abspath("./results/matgl_test/md")
    res = loaded_server.run_md(
        cu_structure,
        steps=5,
        output_dir=output_dir
    )
    assert "error" not in res # Trajectory path handled
    assert "trajectory_path" in res

def test_calculate_neb(loaded_server):
    atoms_start = bulk("Cu")
    atoms_end = atoms_start.copy()
    atoms_end.positions[0] += 0.5
    start_dict = AseAtomsAdaptor.get_structure(atoms_start).as_dict()
    end_dict = AseAtomsAdaptor.get_structure(atoms_end).as_dict()
    
    output_dir = os.path.abspath("./results/matgl_test/neb")
    res = loaded_server.calculate_neb(
        start_dict,
        end_dict,
        n_images=3,
        fmax=1.0,
        output_dir=output_dir
    )
    assert "error" not in res
    assert "barrier" in res

def test_sampler_near_equilibrium(loaded_server, cu_structure):
    # This uses relaxation internally
    res = loaded_server.sample_near_equilibrium(
        cu_structure,
        max_steps=5
    )
    assert "error" not in res
    assert res.get("count") > 0
    assert "structures" in res

def test_sampler_off_equilibrium(loaded_server, cu_structure):
    output_dir = os.path.abspath("./results/matgl_test/sampler_off")
    res = loaded_server.sample_off_equilibrium(
        cu_structure,
        total_steps=10,
        output_dir=output_dir,
        target_atoms=1 # small
    )
    assert "error" not in res
    assert res.get("count") > 0

def test_sampler_order_disorder(loaded_server):
    # Construct disordered CuAu structure (FCC)
    # Pymatgen structure manual construction
    from pymatgen.core import Structure, Lattice
    lattice = Lattice.cubic(3.6)
    # Check proper species dict format for disorder: {"Cu": 0.5, "Au": 0.5}
    struct = Structure(lattice, [{"Cu": 0.5, "Au": 0.5}], [[0,0,0]])
    struct_inv = struct.as_dict()
    
    output_dir = os.path.abspath("./results/matgl_test/sampler_od")
    res = loaded_server.sample_order_disorder(
        struct_inv,
        n_structures=2,
        target_atoms=2, # Creates supercell
        output_dir=output_dir
    )
    assert "error" not in res
    assert res.get("count") == 4 # 2 structures * 2 (original + pertrubed)
    
def test_fine_tune_model(loaded_server, cu_structure):
    # 1. Initial Prediction
    res_pre = loaded_server.predict_structure(cu_structure)
    assert "error" not in res_pre
    e_pre = res_pre["energy"]
    
    # 2. Prepare Training Data with different energy
    # MatGL fine-tuning tool expects list of dicts with structure, energy, forces, stress
    import numpy as np
    atoms_len = len(cu_structure["sites"])
    training_data = [
        {
            "structure": cu_structure,
            "energy": -100.0, # Massive shift
            "forces": np.zeros((atoms_len, 3)).tolist(),
            "stress": np.zeros(6).tolist()
        },
        {
            "structure": cu_structure,
            "energy": -100.1,
            "forces": [[0.1, 0.0, 0.0]] * atoms_len,
            "stress": np.zeros(6).tolist()
        }
    ]
    
    output_dir = os.path.abspath("./results/matgl_test/finetune")
    
    # 3. Fine-tune
    res = loaded_server.fine_tune_model(
        training_data=training_data,
        epochs=1,
        learning_rate=0.01,
        batch_size=2,
        output_dir=output_dir
    )
    
    assert "error" not in res
    assert res.get("is_fine_tuned") is True
    
    # 4. Verify Checkpoints
    assert os.path.exists(output_dir)
    # MatGL saves "fine_tuned_model.pth" typically or similar
    files = os.listdir(output_dir)
    model_files = [f for f in files if f.endswith(".pth") or f.endswith(".pt")]
    assert len(model_files) > 0, f"No model files found in {output_dir}. Files: {files}"
    
    model_path = os.path.join(output_dir, model_files[0])
    
    # 5. Reload Model
    load_res = loaded_server.load_model(model_name=model_path)
    assert "Successfully loaded" in load_res
    
    # 6. Predict Again
    res_post = loaded_server.predict_structure(cu_structure)
    assert "error" not in res_post
    e_post = res_post["energy"]
    
    # 7. Compare
    print(f"Pre-train Energy: {e_pre}, Post-train Energy: {e_post}")
    assert e_pre != e_post, "Energy should change after fine-tuning"
