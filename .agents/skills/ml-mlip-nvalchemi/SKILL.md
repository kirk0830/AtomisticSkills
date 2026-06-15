---
name: ml-mlip-nvalchemi
description: GPU-accelerated batched inference for MACE, MatGL (TensorNet/M3GNet/CHGNet), and FairChem MLIPs using NValchemi, enabling parallel static, relax, and MD workflows across multiple structures simultaneously.
category: machine-learning
---

# ml-mlip-nvalchemi

## Goal

Exploit NVIDIA's NValchemi toolkit to run energy/force/stress predictions, geometry relaxations, and molecular dynamics for a **batch of structures** in a single GPU-parallel forward pass, instead of N sequential CPU loops. This is automatically activated when `nvalchemi-toolkit` is installed in the environment — the existing MCP tool surface (`static_calculation`, `relax_structure`, `run_md`) passes a list of structures and dispatches to the NValchemi backend transparently.

## Background

NValchemi provides batched dynamics integrators (FIRE, NVT Nose-Hoover, NPT, etc.) and a `BaseModelMixin` interface. AtomisticSkills wraps each MLIP in a `BaseModelMixin`-compatible class:

| MLIP | NValchemi wrapper | Location |
|------|------------------|----------|
| MACE | `nvalchemi.models.mace.MACEWrapper` | upstream (nvalchemi-toolkit) |
| MatGL TensorNet | `matgl.ext._alchmtk.TensorNetWrapper` | matgl package |
| MatGL M3GNet | `M3GNetWrapper` | `src/utils/mlips/nvalchemi/matgl_wrappers.py` |
| MatGL CHGNet | `CHGNetWrapper` | `src/utils/mlips/nvalchemi/matgl_wrappers.py` |
| MatGL QET | `QETWrapper` | matgl package |
| FairChem UMA | `FairChemWrapper` | `src/utils/mlips/nvalchemi/fairchem_nv.py` |

The dispatch lives in `src/utils/mlips/base.py`:
- `static_calculation(list)` → `_batch_static_nvalchemi()` → single batched forward
- `relax_structure(list)` → `_batch_relax_nvalchemi()` → batched FIRE
- `run_md(list)` → `_batch_md_nvalchemi()` → batched NVT/NVE/NPT integrator

## Instructions

### Step 1 — Verify NValchemi is Available

```python
# Env: mace-agent  (or matgl-agent, fairchem-agent)
from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE
print(NVALCHEMI_AVAILABLE)  # must be True

from src.utils.mlips.mace.mace_wrapper import MACEWrapper
wrapper = MACEWrapper(model_name="MACE-OMAT-0-small", device="cuda")
wrapper.load()
nv = wrapper._get_nvalchemi_model()
print(nv)  # should be non-None MACEWrapper(nvalchemi)
```

### Step 2 — Batch Static Calculation

Pass a list of ASE Atoms objects to `static_calculation`. The result dict includes a `"backend": "nvalchemi"` key when the batch path was used:

```python
# Env: mace-agent
from ase.build import bulk
import numpy as np

structures = [bulk("Cu", "fcc", a=3.6 * s) for s in np.linspace(0.96, 1.04, 10)]
result = wrapper.static_calculation(structures)
# result["backend"] == "nvalchemi"
# result["total_structures"] == 10
# result["results"][i] == {"energy": ..., "forces": ..., "stress": ...}
```

Identical API for MatGL and FairChem wrappers:

```python
# Env: matgl-agent
from src.utils.mlips.matgl.matgl_wrapper import MatGLWrapper
wrapper = MatGLWrapper(model_name="TensorNet-PES-MatPES-PBE-2025.2", device="cuda")
wrapper.load()
result = wrapper.static_calculation(structures)
```

```python
# Env: fairchem-agent
from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper
wrapper = FAIRCHEMWrapper(model_name="uma-s-1p2", device="cuda")
wrapper.load()
result = wrapper.static_calculation(structures)
```

### Step 3 — Batch Geometry Relaxation

```python
# Env: mace-agent
result = wrapper.relax_structure(
    structure_data=structures,   # list of ASE Atoms
    fmax=0.05,                   # eV/Å convergence
    steps=500,
    output_dir="/path/to/output"
)
```

### Step 4 — Batch Molecular Dynamics

```python
# Env: mace-agent
result = wrapper.run_md(
    structure_data=structures,
    temperature=1000,
    steps=1000,
    timestep=2.0,                # fs
    ensemble="nvt_nose_hoover",
    output_dir="/path/to/output"
)
```

Supported batch ensembles: `nve`, `nvt_nose_hoover`, `nvt_langevin`, `npt`, `npt_nose_hoover`, `npt_mtk`.
Unsupported (Berendsen, Andersen, inhomogeneous NPT) fall back to sequential automatically.

### Step 5 — Disable NValchemi (Sequential Fallback)

To force sequential processing (e.g., debugging):

```python
# Env: any
import src.utils.mlips.nvalchemi.nvalchemi_utils as _nv
_nv.check_nvalchemi_available = lambda: False   # temporary
result = wrapper.static_calculation(structures)  # sequential
_nv.check_nvalchemi_available = lambda: True    # restore
```

### Step 6 — Run the Benchmark Script

To re-run the full accuracy and speed benchmark for any environment:

```bash
# Env: mace-agent
python .agents/skills/ml-mlip-nvalchemi/scripts/run_nvalchemi_benchmark.py \
    --env mace \
    --n-repeat 3 \
    --output results_mace.json

# Env: matgl-agent
python .agents/skills/ml-mlip-nvalchemi/scripts/run_nvalchemi_benchmark.py \
    --env matgl \
    --n-repeat 3 \
    --output results_matgl.json

# Env: fairchem-agent
python .agents/skills/ml-mlip-nvalchemi/scripts/run_nvalchemi_benchmark.py \
    --env fairchem \
    --n-repeat 3 \
    --output results_fairchem.json
```

The script tests N=2, 5, 10, 20 structures and prints a speedup/accuracy table.

## Benchmark Results

See [resources/benchmark_results.md](resources/benchmark_results.md) for the full results table.

### Speedup Summary (GPU, NVIDIA GB10 Blackwell cc12.1, best-of-3, N=20 structures)

| Model | Speedup (N=5) | Speedup (N=20) | ΔE max (eV) |
|-------|:---:|:---:|:---:|
| MACE-OMAT-0-small | 21.7× | **68×** | 9.5e-07 |
| MACE-OMAT-0-medium | 22.9× | **72×** | 9.5e-07 |
| MACE-MH-1/omat_pbe | 14.3× | **34×** | 1.6e-07 |
| MACE-MH-1/matpes_r2scan | 14.4× | **34×** | 1.2e-07 |
| MACE-MP-medium-0b3 | 22.9× | **76×** | 1.4e-06 |
| MACE-MATPES-PBE-0 | 23.6× | **77×** | 7.2e-07 |
| MACE-MATPES-R2SCAN-0 | 24.2× | **76×** | 1.9e-06 |
| TensorNet-PES-MatPES-PBE-2025.2 | 3.6× | **11×** | 1.4e-07 |
| TensorNet-PES-MatPES-r2SCAN-2025.2 | 3.8× | **12×** | 8.0e-07 |
| M3GNet-PES-MatPES-PBE-2025.2 | 3.7× | **11×** | 1.3e-03¹ |
| M3GNet-PES-MatPES-r2SCAN-2025.2 | 3.9× | **11×** | 7.2e-04¹ |
| CHGNet-PES-MatPES-PBE-2025.2.10 | 4.3× | **12×** | 2.4e-07 |
| CHGNet-PES-MatPES-r2SCAN-2025.2.10 | 4.2× | **13×** | 9.5e-07 |
| QET-PES-MatPES-PBE-2025.2 | 4.2× | **13×** | 7.2e-07 |
| QET-PES-MatPES-r2SCAN-2025.2 | 3.9× | **14×** | 9.5e-07 |
| SO3Net-PES-ANI-1x-Subset | — | — | not supported |
| FairChem uma-s-1p2 (omat) | 3.0× | 2.9× | 1.6e-07 |
| FairChem uma-m-1p1 (omat) | 2.9× | 3.5× | 1.7e-07 |
| FairChem uma-s-1p1 (omat) | 3.4× | 5.5× | 2.5e-07 |

¹ M3GNet energy errors (~0.7–1.8×10⁻³ eV) from different neighbor-list graph connectivity (NValchemi GPU warp kernel vs. CPU `radius_graph_pbc`). Forces are exact (ΔF = 0). Within 5×10⁻³ eV tolerance for PES screening.

## Constraints

- **NValchemi required**: `nvalchemi-toolkit` must be installed. Check `NVALCHEMI_AVAILABLE` flag. Falls back to sequential if unavailable.
- **Environment isolation**: Must use the correct conda environment per MLIP:
  - `mace-agent` — MACE models
  - `matgl-agent` — MatGL (TensorNet, M3GNet, CHGNet)
  - `fairchem-agent` — FairChem UMA
- **Stress format**: NValchemi returns 3×3 Cauchy stress tensor (eV/Å³); sequential path returns ASE Voigt-6. Both formats are accepted downstream — `_extract_static()` in tests handles the conversion.
- **FairChem dataset field**: UMA model requires `dataset` (e.g., `"omat"`) passed to `FCAtomicData`. This is handled automatically by `FairChemWrapper`; defaults to `"omat"` when `task_name=None`.
- **CHGNet batch speedup**: CHGNet directed line graph construction parallelizes well on GPU (12–13× at N=20). CPU performance is marginal (<3×); always use `device="cuda"` for batch workloads.
- **SO3Net not supported**: `SO3Net-PES-ANI-1x-Subset` falls back to sequential automatically (`_get_nvalchemi_model()` returns `None`).
- **ANI-1x models with transition metals**: TensorNet-PES-ANI-1x and M3GNet-PES-ANI-1x training sets cover only H/C/N/O. Using them with Cu or other transition metals causes a CUDA index OOB error that corrupts the CUDA context for the session. Run ANI-1x models in a separate process from other models.
- **Unsupported ensembles for batch MD**: `nvt_berendsen`, `nvt_andersen`, `nvt_bussi`, `npt_berendsen`, and `npt_inhomogeneous` have no NValchemi equivalent and always run sequentially.

## References

- NValchemi toolkit: NVIDIA internal package (nvalchemi-toolkit, PyPI: `https://pypi.nvidia.com`)
- MACE: Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields", *NeurIPS 2022*. [arXiv:2206.07697](https://arxiv.org/abs/2206.07697)
- MatGL / TensorNet: Chen & Ong, "A Universal Graph Deep Learning Interatomic Potential for the Periodic Table", *Nature Computational Science 2023*. [DOI:10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3)
- M3GNet: Chen & Ong, "A universal graph deep learning interatomic potential for the periodic table", *Nature Computational Science 2022*.
- CHGNet: Deng et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling", *Nature Machine Intelligence 2023*. [DOI:10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3)
- FairChem UMA: Meta FAIR, "Scaling Universal Molecular Atomistic Machine Learning for Open Catalyst 2024". [arXiv:2411.12234](https://arxiv.org/abs/2411.12234)

---

**Author:** Bowen Deng
**Contact:** [github.com/bowen-bd](https://github.com/bowen-bd)
