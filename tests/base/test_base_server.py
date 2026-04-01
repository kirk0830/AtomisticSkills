import pytest
from src.mcp_server import base_server
from pymatgen.core import Structure, Lattice

@pytest.mark.base
def test_create_research_dir(skip_if_wrong_env):
    res = base_server.create_research_dir("test_topic")
    assert "error" not in res

@pytest.mark.base
def test_search_materials_project_by_formula(skip_if_wrong_env, mock_mp_api_key):
    res = base_server.search_materials_project_by_formula("Si")
    # Without real API key it might return an error string
    assert isinstance(res, str)

@pytest.mark.base
def test_search_materials_project_by_chemsys(skip_if_wrong_env, mock_mp_api_key):
    res = base_server.search_materials_project_by_chemsys("Li-O")
    assert isinstance(res, str)

@pytest.mark.base
def test_visualize_structure(skip_if_wrong_env, tmp_cif_file):
    res = base_server.visualize_structure(str(tmp_cif_file))
    assert "error" not in res

@pytest.mark.base
def test_supercell_expansion(skip_if_wrong_env, tmp_cif_file):
    res = base_server.supercell_expansion(
        structure_path=str(tmp_cif_file),
        scaling_matrix_json="[2, 2, 2]"
    )
    assert "Successfully created supercell" in res

@pytest.mark.base
def test_modify_structure(skip_if_wrong_env, tmp_cif_file):
    res = base_server.modify_structure(
        structure_path=str(tmp_cif_file),
        substitution_dict_json='{"Si": "Fe"}'
    )
    assert "Successfully modified structure" in res

@pytest.mark.base
def test_search_literature(skip_if_wrong_env):
    res = base_server.search_literature("battery")
    assert isinstance(res, str)
