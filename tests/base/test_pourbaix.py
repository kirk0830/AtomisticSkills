"""
Unit tests for Pourbaix diagram calculation using pymatgen.

Tests the core functionality of the pourbaix-diagram skill.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import functions to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".agent" / "skills" / "pourbaix-diagram" / "scripts"))

from calculate_pourbaix import (
    load_mlip_energies,
    create_solid_entries,
    construct_pourbaix_diagram
)


class TestLoadMLIPEnergies:
    """Test loading MLIP energies from JSON."""
    
    def test_load_valid_energies(self, tmp_path):
        """Test loading valid energy file."""
        energies_file = tmp_path / "energies.json"
        test_data = {
            "ZnO": -10.5,
            "Zn": -5.0,
            "ZnO2": -12.3
        }
        
        with open(energies_file, 'w') as f:
            json.dump(test_data, f)
        
        result = load_mlip_energies(energies_file)
        
        assert result == test_data
        assert len(result) == 3
        assert result["ZnO"] == -10.5


class TestCreateSolidEntries:
    """Test creating PourbaixEntry objects from MLIP energies."""
    
    def test_create_entries_from_energies(self):
        """Test creating Pourbaix entries."""
        mlip_energies = {
            "ZnO": -10.5,
            "Zn": -5.0
        }
        
        entries = create_solid_entries(mlip_energies)
        
        assert len(entries) == 2
        # Check that entries are PourbaixEntry objects
        for entry in entries:
            assert hasattr(entry, 'phase_type')
            assert entry.phase_type == "Solid"
            assert hasattr(entry, 'energy')


class TestConstructPourbaixDiagram:
    """Test Pourbaix diagram construction."""
    
    @patch('calculate_pourbaix.PourbaixDiagram')
    def test_diagram_construction(self, mock_diagram_class):
        """Test that diagram is constructed with correct parameters."""
        from pymatgen.analysis.pourbaix_diagram import PourbaixEntry
        from pymatgen.entries.computed_entries import ComputedEntry
        from pymatgen.core import Composition
        
        # Create mock entries
        mock_entries = []
        for formula, energy in [("ZnO", -10.5), ("Zn", -5.0)]:
            entry = ComputedEntry(Composition(formula), energy)
            pb_entry = PourbaixEntry(entry)
            mock_entries.append(pb_entry)
        
        # Mock the PourbaixDiagram class
        mock_diagram = MagicMock()
        mock_diagram_class.return_value = mock_diagram
        
        # Call function
        result = construct_pourbaix_diagram(
            mock_entries,
            "ZnO",
            ion_concentration=1e-6
        )
        
        # Verify PourbaixDiagram was called
        mock_diagram_class.assert_called_once()
        call_kwargs = mock_diagram_class.call_args[1]
        
        assert 'entries' in call_kwargs
        assert 'comp_dict' in call_kwargs
        assert 'conc_dict' in call_kwargs
        assert call_kwargs['filter_solids'] == True


class TestPrepareS olidEnergies:
    """Test the prepare_solid_energies.py helper script."""
    
    def test_extract_from_mcp_output(self, tmp_path):
        """Test extracting energies from MCP output structure."""
        from prepare_solid_energies import extract_energies_from_mcp_output
        
        # Create mock MCP output structure
        relaxed_dir = tmp_path / "relaxed"
        relaxed_dir.mkdir()
        
        # Create subdirectories with result.json files
        for formula, energy in [("ZnO", -10.5), ("Zn", -5.0)]:
            struct_dir = relaxed_dir / formula
            struct_dir.mkdir()
            
            result_data = {
                "energy": energy,
                "formula": formula
            }
            
            with open(struct_dir / "result.json", 'w') as f:
                json.dump(result_data, f)
        
        # Extract energies
        energies = extract_energies_from_mcp_output(relaxed_dir)
        
        assert len(energies) == 2
        assert energies["ZnO"] == -10.5
        assert energies["Zn"] == -5.0


@pytest.mark.integration
class TestFullWorkflow:
    """Integration test for complete Pourbaix workflow."""
    
    @pytest.mark.skip(reason="Requires MP_API_KEY and network access")
    def test_full_pourbaix_calculation(self, tmp_path):
        """Test complete workflow from energies to diagram."""
        # This would require actual MP API access
        # Skip by default, run manually with valid API key
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
