import pytest
from src.mcp_server import atomate2_server

@pytest.mark.atomate2
def test_run_atomate2_vasp_calculation(skip_if_wrong_env):
    res = atomate2_server.run_atomate2_vasp_calculation(
        structures_path="dummy.cif",
        output_dir="dummy_out",
        check_only=True
    )
    assert "error" not in res

@pytest.mark.atomate2
def test_get_atomate2_results_by_id(skip_if_wrong_env):
    res = atomate2_server.get_atomate2_results_by_id(job_ids=["dummy"])
    assert "error" not in res # Might be empty or gracefully fail

@pytest.mark.atomate2
def test_get_atomate2_results_by_formula(skip_if_wrong_env):
    res = atomate2_server.get_atomate2_results_by_formula(formula="Si")
    assert "error" not in res

@pytest.mark.atomate2
def test_get_atomate2_summary(skip_if_wrong_env):
    res = atomate2_server.get_atomate2_summary()
    assert "error" not in res

@pytest.mark.atomate2
def test_get_atomate2_recent_jobs(skip_if_wrong_env):
    res = atomate2_server.get_atomate2_recent_jobs(limit=1)
    assert "error" not in res

@pytest.mark.atomate2
def test_get_atomate2_job_status(skip_if_wrong_env):
    res = atomate2_server.get_atomate2_job_status(job_id="dummy")
    assert "error" in res # or "job not found"
