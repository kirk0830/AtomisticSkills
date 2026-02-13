# MatGL Agent Environment

This environment supports the MatGL graph neural network models.

## Quick Installation
Most users should use the simplified installation script, which installs only the necessary core packages:

```bash
bash install.sh
```

This installs:
- python 3.10
- matgl
- pymatgen
- dgl

## Full Reproduction
If you need to reproduce the exact environment state (including all pinned dependency versions), use the full example configuration:

```bash
```bash
conda env create -f example_full_env.yaml
```

## Blackwell GPU (GB10) Installation

**For NVIDIA DGX Spark systems with Blackwell GB10 GPUs (ARM/aarch64, CUDA 13.0)**, see the detailed installation guide:

[`INSTALL_BLACKWELL.md`](file:///home/bdeng/projects/AtomisticSkills/conda-envs/matgl-agent/INSTALL_BLACKWELL.md)

This guide includes:
- PyTorch 2.9.1+cu130 installation
- DGL source compilation for compute capability 12.1
- MatGL installation with GPU acceleration
- Verification and benchmarking

> **Note**: Users with older x86_64 hardware or earlier GPUs should use the Quick Installation above or follow the standard [MatGL installation guide](https://matgl.ai/installation.html).

