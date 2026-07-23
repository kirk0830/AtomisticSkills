"""
DFT Utilities (VASP, Quantum ESPRESSO, CP2K).
"""

from .dft_common import (
    DFTResult,
    estimate_kpoints,
    get_kpath_seekpath,
    read_dft_results,
    write_dft_results,
)
from .cp2k_utils import (
    build_cp2k_calculator,
    parse_cp2k_mulliken_spins,
    parse_cp2k_results,
    run_cp2k_relax,
    run_cp2k_static,
)
from .qe_utils import (
    build_qe_calculator,
    parse_qe_results,
    run_qe_band_structure,
    run_qe_relax,
    run_qe_static,
)

__all__ = [
    # Common
    "DFTResult",
    "write_dft_results",
    "read_dft_results",
    "get_kpath_seekpath",
    "estimate_kpoints",
    # Quantum ESPRESSO
    "build_qe_calculator",
    "run_qe_static",
    "run_qe_relax",
    "run_qe_band_structure",
    "parse_qe_results",
    # CP2K
    "build_cp2k_calculator",
    "run_cp2k_static",
    "run_cp2k_relax",
    "parse_cp2k_results",
    "parse_cp2k_mulliken_spins",
]

# VASP submodules require ASE (and optionally pymatgen). Import them atomically so
# that QE/CP2K/common utilities remain usable when ASE is not installed.
try:
    from . import vasp_hpc, vasp_parser, vasp_writer

    _vasp_exports = {
        "VaspHPCRunner": vasp_hpc.VaspHPCRunner,
        "VaspResult": vasp_hpc.VaspResult,
        "generate_vasp_input": vasp_hpc.generate_vasp_input,
        "parse_vasp_output": vasp_hpc.parse_vasp_output,
        "VASPParser": vasp_parser.VASPParser,
        "write_vasp_input_files": vasp_writer.write_vasp_input_files,
    }
except ImportError:
    _vasp_exports = {}

if _vasp_exports:
    globals().update(_vasp_exports)
    __all__.extend(list(_vasp_exports.keys()))
