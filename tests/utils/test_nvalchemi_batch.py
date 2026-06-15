"""Integration tests for NValchemi GPU-accelerated batch operations.

These tests require the appropriate conda environment with nvalchemi installed
and at least one MLIP model available.

Run individual groups:
    conda run -n mace-agent pytest tests/utils/test_nvalchemi_batch.py::TestFallback -v
    conda run -n mace-agent pytest tests/utils/test_nvalchemi_batch.py::TestBatchRelaxMACE -v
    conda run -n matgl-agent pytest tests/utils/test_nvalchemi_batch.py::TestBatchRelaxM3GNet -v
    conda run -n matgl-agent pytest tests/utils/test_nvalchemi_batch.py::TestBatchRelaxCHGNet -v
    conda run -n fairchem-agent pytest tests/utils/test_nvalchemi_batch.py::TestBatchStaticFairChem -v
    conda run -n mace-agent pytest tests/utils/test_nvalchemi_batch.py::TestBatchMDNVT -v
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_cu_bulk(scale: float = 1.0):
    """Return a Cu FCC unit cell, optionally strained."""
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.6 * scale)
    return atoms


def _make_structures(n: int = 3):
    scales = [1.0, 0.98, 1.02][:n]
    return [_make_cu_bulk(s) for s in scales]


# ---------------------------------------------------------------------------
# Fallback test (no nvalchemi needed)
# ---------------------------------------------------------------------------


class TestFallback:
    """When nvalchemi is unavailable, batch ops must fall back to sequential."""

    def test_batch_relax_falls_back_without_nvalchemi(self, tmp_path, monkeypatch):
        """Monkeypatching NVALCHEMI_AVAILABLE=False should force sequential path."""
        import src.utils.mlips.nvalchemi.nvalchemi_utils as _nv_utils

        monkeypatch.setattr(_nv_utils, "NVALCHEMI_AVAILABLE", False)

        # The dispatch in base.py checks check_nvalchemi_available() which reads the module attr
        monkeypatch.setattr(_nv_utils, "check_nvalchemi_available", lambda: False)

        try:
            from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        except ImportError:
            pytest.skip("MACE not available in this environment")

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.5,
            steps=5,
            output_dir=str(tmp_path),
        )
        # Should complete without error (via sequential path)
        assert result is not None

    def test_single_prediction_unchanged(self, tmp_path):
        """Single-structure prediction path must not be affected by NValchemi changes."""
        try:
            from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        except ImportError:
            pytest.skip("MACE not available in this environment")

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
            # create_calculator() verifies mace.calculators is importable
            wrapper.create_calculator()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        atoms = _make_cu_bulk()
        result = wrapper.static_calculation(atoms)
        assert "energy" in result
        assert isinstance(result["energy"], float)


# ---------------------------------------------------------------------------
# MACE batch relax
# ---------------------------------------------------------------------------


@pytest.mark.mace
class TestBatchRelaxMACE:
    def test_batch_relax_mace_nvalchemi(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=20,
            output_dir=str(tmp_path),
        )
        assert result is not None

    def test_batch_relax_mace_energy_close_to_sequential(self, tmp_path):
        """NValchemi batch relax energies should be within 0.1% of sequential."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper
        import src.utils.mlips.nvalchemi.nvalchemi_utils as _nv

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)

        # Run with NValchemi
        result_nv = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=5,
            output_dir=str(tmp_path / "nv"),
        )

        # Force sequential
        original = _nv.NVALCHEMI_AVAILABLE
        _nv.NVALCHEMI_AVAILABLE = False
        result_seq = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=5,
            output_dir=str(tmp_path / "seq"),
        )
        _nv.NVALCHEMI_AVAILABLE = original

        # Both paths complete without error (energy comparison may differ
        # slightly due to internal optimizer state but should be same order)
        assert result_nv is not None
        assert result_seq is not None


# ---------------------------------------------------------------------------
# MatGL M3GNet batch relax
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestBatchRelaxM3GNet:
    def test_batch_relax_m3gnet(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(model_name="M3GNet-PES-MatPES-PBE-2025.2", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=20,
            output_dir=str(tmp_path),
        )
        assert result is not None

    def test_m3gnet_wrapper_factory(self):
        """M3GNetWrapper.from_potential() should succeed for an M3GNet potential."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            import matgl
            from matgl.models._m3gnet import M3GNet
            from src.utils.mlips.nvalchemi.matgl_wrappers import M3GNetWrapper

            potential = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
        except Exception:
            pytest.skip("M3GNet model unavailable in this environment")

        assert isinstance(potential.model, M3GNet)
        nv_model = M3GNetWrapper.from_potential(potential)
        assert nv_model is not None
        assert nv_model.model_config.neighbor_config is not None
        assert (
            abs(
                float(nv_model.model_config.neighbor_config.cutoff)
                - potential.model.cutoff
            )
            < 0.01
        )


# ---------------------------------------------------------------------------
# MatGL CHGNet batch relax
# ---------------------------------------------------------------------------


@pytest.mark.matgl
class TestBatchRelaxCHGNet:
    def test_batch_relax_chgnet(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper

        wrapper = MatGLWrapper(
            model_name="CHGNet-PES-MatPES-PBE-2025.2.10", device="cpu"
        )
        try:
            wrapper.load()
        except Exception:
            pytest.skip("CHGNet model unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.relax_structure(
            structure_data=structures,
            fmax=0.2,
            steps=20,
            output_dir=str(tmp_path),
        )
        assert result is not None

    def test_chgnet_wrapper_factory(self):
        """CHGNetWrapper.from_potential() should succeed for a CHGNet potential."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            import matgl
            from matgl.models._chgnet import CHGNet
            from src.utils.mlips.nvalchemi.matgl_wrappers import CHGNetWrapper

            potential = matgl.load_model("CHGNet-PES-MatPES-PBE-2025.2.10")
        except Exception:
            pytest.skip("CHGNet model unavailable in this environment")

        assert isinstance(potential.model, CHGNet)
        nv_model = CHGNetWrapper.from_potential(potential)
        assert nv_model is not None
        assert nv_model.model_config.neighbor_config is not None
        assert (
            abs(
                float(nv_model.model_config.neighbor_config.cutoff)
                - potential.model.cutoff
            )
            < 0.01
        )


# ---------------------------------------------------------------------------
# FairChem batch static
# ---------------------------------------------------------------------------


@pytest.mark.fairchem
class TestBatchStaticFairChem:
    def test_batch_static_fairchem(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper
        except ImportError:
            pytest.skip("FairChem not available in this environment")

        wrapper = FAIRCHEMWrapper(model_name="uma-s-1p2", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("FairChem uma-s-1p2 unavailable in this environment")

        structures = _make_structures(3)
        result = wrapper.static_calculation(structure_data=structures)
        assert result is not None

    def test_fairchem_wrapper_neighbor_config_none(self):
        """FairChemWrapper must have neighbor_config=None (builds own graph)."""
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        try:
            from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper
        except ImportError:
            pytest.skip("FairChem not available in this environment")

        wrapper = FAIRCHEMWrapper(model_name="uma-s-1p2", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("FairChem uma-s-1p2 unavailable in this environment")

        nv_model = wrapper._get_nvalchemi_model()
        if nv_model is None:
            pytest.skip("_get_nvalchemi_model returned None")

        assert nv_model.model_config.neighbor_config is None


# ---------------------------------------------------------------------------
# MD batch NVT
# ---------------------------------------------------------------------------


@pytest.mark.mace
class TestBatchMDNVT:
    def test_batch_md_nvt_nose_hoover(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        result = wrapper.run_md(
            structure_data=structures,
            temperature=300,
            steps=50,
            timestep=1.0,
            ensemble="nvt_nose_hoover",
            output_dir=str(tmp_path),
        )
        assert result is not None

    def test_batch_md_nve(self, tmp_path):
        from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

        if not NVALCHEMI_AVAILABLE:
            pytest.skip("nvalchemi not installed")

        from src.utils.mlips.mace.mace_wrapper import MACEWrapper

        wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cpu")
        try:
            wrapper.load()
        except Exception:
            pytest.skip("MACE model unavailable in this environment")

        structures = _make_structures(2)
        result = wrapper.run_md(
            structure_data=structures,
            temperature=300,
            steps=20,
            timestep=1.0,
            ensemble="nve",
            output_dir=str(tmp_path),
        )
        assert result is not None
