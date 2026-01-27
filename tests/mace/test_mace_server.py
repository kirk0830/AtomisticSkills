
import pytest
import os
import shutil
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from src.mcp_server import mace_server

# Helper to clean up output
@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    if os.path.exists("./results/mace_test"):
        shutil.rmtree("./results/mace_test")

@pytest.fixture(scope="module")
def loaded_server():
    # Use small model for speed
    res = mace_server.load_model("MACE-MP-small", device="cpu")
    if "error" in res:
        pytest.fail(f"Failed to load model: {res}")
    return mace_server

@pytest.fixture
def cu_structure():
    atoms = bulk("Cu")
    return AseAtomsAdaptor.get_structure(atoms).as_dict()

def test_get_info(loaded_server):
    info = loaded_server.get_info()
    assert info.get("model_name") == "MACE-MP-small"

def test_predict_structure(loaded_server, cu_structure):
    res = loaded_server.predict_structure(cu_structure)
    assert "energy" in res
    assert "forces" in res
    assert "stress" in res # MACE usually returns stress

def test_relax_structure(loaded_server, cu_structure):
    # Perturb
    import numpy as np
    atoms_obj = bulk("Cu")
    atoms_obj.positions[0] += 0.1
    struct_dict = AseAtomsAdaptor.get_structure(atoms_obj).as_dict()
    
    output_dir = os.path.abspath("./results/mace_test/relax")
    res = loaded_server.relax_structure(
        struct_dict, 
        steps=5, 
        fmax=0.1, 
        output_dir=output_dir
    )
    
    assert "error" not in res
    assert "final_structure" in res
    assert os.path.exists(output_dir)

def test_run_md(loaded_server, cu_structure):
    output_dir = os.path.abspath("./results/mace_test/md")
    res = loaded_server.run_md(
        cu_structure,
        temperature=300,
        steps=5,
        log_interval=1,
        output_dir=output_dir
    )
    assert "error" not in res
    assert "trajectory_path" in res
    assert os.path.exists(res["trajectory_path"])

def test_calculate_neb(loaded_server):
    atoms_start = bulk("Cu")
    atoms_end = atoms_start.copy()
    # Move an atom to a clearly different position (e.g. interstitial or vacancy hop)
    # For bulk Cu (fcc), maybe just a small shift
    atoms_end.positions[0] += 0.5
    
    start_dict = AseAtomsAdaptor.get_structure(atoms_start).as_dict()
    end_dict = AseAtomsAdaptor.get_structure(atoms_end).as_dict()
    
    output_dir = os.path.abspath("./results/mace_test/neb")
    res = loaded_server.calculate_neb(
        start_dict,
        end_dict,
        n_images=3,
        fmax=1.0, # Loose fmax for speed
        output_dir=output_dir
    )
    assert "error" not in res
    assert "barrier" in res
    assert "mep" in res

def test_calculate_phonon(loaded_server, cu_structure):
    output_dir = os.path.abspath("./results/mace_test/phonon")
    res = loaded_server.calculate_phonon(
        cu_structure,
        supercell_matrix=[[2,0,0],[0,2,0],[0,0,2]],
        t_step=100,
        t_max=300,
        output_dir=output_dir
    )
    assert "error" not in res
    assert "thermal_properties" in res
    
def test_calculate_qha(loaded_server, cu_structure):
    output_dir = os.path.abspath("./results/mace_test/qha")
    res = loaded_server.calculate_qha(
        cu_structure,
        t_step=100,
        t_max=300,
        output_dir=output_dir
    )
    assert "error" not in res
    assert "thermal_expansion_coefficients" in res


def test_fine_tune_model(loaded_server):
    # Construct dummy training data
    atoms = bulk("Cu")
    struct_dict = AseAtomsAdaptor.get_structure(atoms).as_dict()
    
    # 1. Initial Prediction
    res_pre = loaded_server.predict_structure(struct_dict)
    assert "error" not in res_pre
    e_pre = res_pre["energy"]
    
    # Create training data with VERY different energy to force large update
    training_data = [
        {
            "structure": struct_dict,
            "energy": -100.0, # Large shift from typical Cu energy (~-3.5 eV)
            "forces": [[0.0, 0.0, 0.0]] * len(atoms),
            "stress": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
             "structure": struct_dict,
             "energy": -100.1, 
             "forces": [[0.1, 0.0, 0.0]] * len(atoms),
             "stress": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    ]
    
    output_dir = os.path.abspath("./results/mace_test/finetune")
    
    # 2. Fine-tune
    res = loaded_server.fine_tune_model(
        training_data=training_data,
        epochs=1, # Just one epoch needed to move weights
        learning_rate=0.1, # High LR to ensure move
        output_dir=output_dir,
        training_config={
            "use_foundation_model": False,
            "batch_size": 1
        }
    )
    
    if "error" in res:
         # Fail if fine-tuning itself fails
         pytest.fail(f"Fine-tuning failed: {res['error']}")
        
    assert res.get("is_fine_tuned") is True
    
    # 3. Verify Checkpoints
    assert os.path.exists(output_dir)
    # Check for MACE specific output files
    # MACE saves model as something like "model_swa.model" or "model_stochastic.model" or ending in .pt or .model
    # The wrapper saves "fine_tuned_mace.model" usually (need to check wrapper implementation if explicit)
    # Or MACE default output.
    # checking directory content
    files = os.listdir(output_dir)
    model_files = [f for f in files if f.endswith(".model") or f.endswith(".pt")]
    assert len(model_files) > 0, f"No model files found in {output_dir}. Files: {files}"
    
    # 4. Reload Model
    # MACE wrapper might need updated load logic or we use the server's load_model with path
    # The tool load_model accepts model_name. If we pass absolute path, it should work.
    
    # Find the model file (MACE often outputs multiple, we pick the final one)
    # Usually `output_dir/model_swa.model` is the best one if SWA used, else Checkpoint
    # For now, let's just pick one.
    model_path = os.path.join(output_dir, model_files[0])
    # Prefer one ending in _compiled.model if exists, or just .model
    
    load_res = loaded_server.load_model(model_name=model_path) 
    # load_model logic: if path exists, load it.
    
    assert "Successfully loaded" in load_res or "error" not in load_res
    
    # 5. Predict Again
    res_post = loaded_server.predict_structure(struct_dict)
    assert "error" not in res_post
    e_post = res_post["energy"]
    
    # 6. Compare
    print(f"Pre-train Energy: {e_pre}, Post-train Energy: {e_post}")
    assert e_pre != e_post, "Energy should change after fine-tuning"

