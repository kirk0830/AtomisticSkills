"""
Test that all pretrained models available via matgl.load_model() can be loaded successfully.

Run in: matgl-agent conda environment
Command: conda activate matgl-agent && pytest tests/matgl/test_matgl_load_all_models.py -v
"""

import pytest
import matgl


ALL_MODELS = matgl.get_available_pretrained_models()


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_load_pretrained_model(model_name):
    """Verify every pretrained model can be loaded without error."""
    model = matgl.load_model(model_name)
    assert model is not None, f"load_model('{model_name}') returned None"
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, f"Model '{model_name}' has no parameters"
