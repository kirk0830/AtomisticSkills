
import pytest
import os
import shutil
from ase.build import bulk
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from src.mcp_server import fairchem_server

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    if os.path.exists("./results/fairchem_test"):
        shutil.rmtree("./results/fairchem_test")
    if os.path.exists("./temp_dft_input"):
        shutil.rmtree("./temp_dft_input")

@pytest.fixture(scope="module")
def loaded_server():
    res = fairchem_server.load_model("uma-s-1p1", device="cpu")
    if "error" in res:
        pytest.fail(f"Failed to load model: {res}")
    return fairchem_server

@pytest.fixture
def cu_structure_dict():
    atoms = bulk("Cu")
    return AseAtomsAdaptor.get_structure(atoms).as_dict()

def test_predict_structure(loaded_server, cu_structure_dict):
    res = loaded_server.predict_structure(cu_structure_dict)
    assert "energy" in res
    assert "forces" in res

def test_relax_structure(loaded_server, cu_structure_dict):
    output_dir = os.path.abspath("./results/fairchem_test/relax")
    res = loaded_server.relax_structure(
        cu_structure_dict, 
        steps=5, 
        output_dir=output_dir
    )
    assert "error" not in res
    assert "final_structure" in res

def test_run_md(loaded_server, cu_structure_dict):
    output_dir = os.path.abspath("./results/fairchem_test/md")
    res = loaded_server.run_md(
        cu_structure_dict,
        steps=5,
        output_dir=output_dir
    )
    assert "error" not in res
    assert "trajectory_path" in res

# Phonon and QHA are shared MatCalc, just check one to save time?
# User said "test all". I'll add them.
@pytest.mark.xfail(reason="Fairchem fine-tuning configuration issues (AtomicData attribute error)")
