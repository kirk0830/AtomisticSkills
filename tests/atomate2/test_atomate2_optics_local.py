import shutil
import sys
from pathlib import Path

from pymatgen.core import Lattice, Structure

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.dft.atomate2_utils import Atomate2Handler  # noqa: E402


def test_atomate2_local_si_optics():
    """Smoke-test Atomate2 OpticsMaker construction without running VASP."""
    test_dir = Path("tests/tmp_atomate2_si_optics")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    try:
        lattice = Lattice.cubic(5.43)
        structure = Structure(
            lattice,
            ["Si", "Si"],
            [[0, 0, 0], [0.25, 0.25, 0.25]],
        )
        structure_path = test_dir / "Si.cif"
        structure.to(filename=str(structure_path))

        handler = Atomate2Handler(str(test_dir))

        env = handler.check_environment()
        assert env["atomate2"] is True

        config = {
            "NBANDS": 32,
            "NEDOS": 400,
            "CSHIFT": 0.1,
            "EDIFF": 1e-4,
        }

        maker = handler.get_flow_maker(
            preset_type="mp",
            calculation_type="optics",
            config=config,
        )

        flow = maker.make(structure)
        assert len(flow.jobs) == 2
        assert [job.name for job in flow.jobs] == ["static", "optics"]

        static_incar = maker.static_maker.input_set_generator.user_incar_settings
        for key, value in config.items():
            assert static_incar[key] == value
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_atomate2_local_si_optics()
