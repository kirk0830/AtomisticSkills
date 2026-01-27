"""
Tests for MACE wrapper functionality

These tests validate:
- predict_atomic_features() implementation
- MACE calculator creation
- MACE model loading

Run in: mace-agent conda environment
Command: conda activate mace-agent && pytest -m mace
"""

import pytest
import numpy as np
from ase import Atoms


@pytest.mark.mace
class TestMACEPredictAtomicFeatures:
    """Test MACE atomic feature extraction"""
    
    def test_predict_atomic_features_returns_correct_format(self, sample_ase_atoms, skip_if_wrong_env):
        """Test that predict_atomic_features returns expected dictionary format"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        result = wrapper.predict_atomic_features(sample_ase_atoms)
        
        # Should not have error
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        
        # Check required keys
        assert "atomic_features" in result
        assert "feature_dim" in result
        assert "num_atoms" in result
        
        # Validate shapes
        features = np.array(result["atomic_features"])
        assert features.shape[0] == result["num_atoms"]
        assert features.shape[1] == result["feature_dim"]
        assert features.shape[0] == len(sample_ase_atoms)
    
    def test_predict_atomic_features_with_file_path(self, tmp_cif_file, skip_if_wrong_env):
        """Test feature extraction from file path"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        result = wrapper.predict_atomic_features(str(tmp_cif_file))
        
        assert "error" not in result
        assert "atomic_features" in result
        assert result["num_atoms"] == 2  # Si2 structure
    
    def test_feature_dimensions_consistent(self, skip_if_wrong_env):
        """Test that feature dimensions are consistent across different structures"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        # Create two different sized structures
        atoms1 = Atoms('Si2', positions=[[0, 0, 0], [1.35, 1.35, 1.35]], cell=[5.43]*3, pbc=True)
        atoms2 = Atoms('Si4', positions=[[0, 0, 0], [1.35, 1.35, 1.35], [2.7, 0, 0], [0, 2.7, 0]], 
                       cell=[5.43]*3, pbc=True)
        
        result1 = wrapper.predict_atomic_features(atoms1)
        result2 = wrapper.predict_atomic_features(atoms2)
        
        # Feature dimension should be same, num_atoms should differ
        assert result1["feature_dim"] == result2["feature_dim"], \
            "Feature dimension should be consistent across structures"
        assert result1["num_atoms"] == 2
        assert result2["num_atoms"] == 4


@pytest.mark.mace
class TestMACECalculator:
    """Test MACE calculator creation and basic functionality"""
    
    def test_create_calculator(self, skip_if_wrong_env):
        """Test that create_calculator returns a valid ASE calculator"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        from ase.calculators.calculator import Calculator
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        calc = wrapper.create_calculator()
        
        assert calc is not None
        assert isinstance(calc, Calculator), \
            f"Expected ASE Calculator, got {type(calc)}"
    
    def test_calculator_computes_energy_and_forces(self, sample_ase_atoms, skip_if_wrong_env):
        """Test that calculator can compute energy and forces"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        calc = wrapper.create_calculator()
        sample_ase_atoms.calc = calc
        
        # Should be able to compute energy and forces
        energy = sample_ase_atoms.get_potential_energy()
        forces = sample_ase_atoms.get_forces()
        
        assert isinstance(energy, (float, np.floating))
        assert forces.shape == (len(sample_ase_atoms), 3)


@pytest.mark.mace
class TestMACEModelLoading:
    """Test MACE model loading"""
    
    def test_load_small_model(self, skip_if_wrong_env):
        """Test loading MACE-OMAT-0-small model"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        assert wrapper.is_loaded
        assert wrapper.model is not None
    
    def test_load_with_custom_device(self, skip_if_wrong_env):
        """Test loading with explicit CPU device"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        assert wrapper.device == "cpu"
        assert wrapper.is_loaded
    
    def test_static_calculation(self, sample_ase_atoms, skip_if_wrong_env):
        """Test static_calculation returns energy, forces, and stress"""
        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        
        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        wrapper.load()
        
        result = wrapper.static_calculation(sample_ase_atoms)
        
        assert "error" not in result
        assert "energy" in result
        assert "forces" in result
        assert "stress" in result
        
        # Validate types
        assert isinstance(result["energy"], (float, np.floating))
        forces = np.array(result["forces"])
        assert forces.shape == (len(sample_ase_atoms), 3)
