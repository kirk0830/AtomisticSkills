# ADiT Agent Environment

Environment for the All-atom Diffusion Transformer (ADiT) — a unified latent diffusion model for generating both periodic crystals and non-periodic molecules.

## Prerequisites

- Conda/miniforge3 installed
- Python 3.10
- CUDA 13.0 compatible GPU

## Installation

1. Create the conda environment:
```bash
conda env create -f core_env.yaml
conda activate adit-agent
```

2. Install PyTorch 2.9.1+cu130:
```bash
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
```

3. Install PyG and extensions:
```bash
pip install torch-geometric
```

> **GB10 / ARM / CUDA 13.0 users**: torch-scatter and torch-cluster must be compiled from source. Follow the **"Installation on ARM"** section in [`conda-envs/mattergen-agent/README.md`](../mattergen-agent/README.md) for `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`, and build commands.

4. Install remaining dependencies:
```bash
pip install lightning==2.4.0 hydra-core hydra-colorlog
pip install e3nn==0.5.1 einops rootutils rich omegaconf torchdiffeq huggingface_hub
pip install pymatgen ase rdkit pyxtal tqdm scipy pandas matplotlib torchmetrics
pip install timm lmdb wandb pathos p-tqdm download
pip install smact matminer importlib_resources
pip install mcp fastmcp
```

5. Clone the AADT repository:
```bash
cd /path/to/AtomisticSkills
git clone https://github.com/facebookresearch/all-atom-diffusion-transformer .agent/tmp/adit
```

## MCP Tool

| Tool | Description |
|---|---|
| `generate_structures` | Generate crystals (MP20) or molecules (QM9) unconditionally |

**Parameters**: `generation_type` (`crystals` / `molecules`), `num_structures`, `cfg_scale`, `batch_size`, `device`, `output_dir`

## Checkpoints

Pre-trained checkpoints are auto-downloaded from HuggingFace on first use:
- Repository: `chaitjo/all-atom-diffusion-transformer`
- `ckpts/ldm.ckpt` — Latent diffusion model
- `ckpts/vae.ckpt` — Variational autoencoder

## Key Dependencies

| Package | Version |
|---|---|
| Python | 3.10 |
| torch | 2.9.1+cu130 |
| torch-geometric | 2.7.0 |
| lightning | 2.4.0 |
| e3nn | 0.5.1 |
| pymatgen, ase | latest |

## Notes

- The AADT repository must be on `PYTHONPATH` (configured automatically in `mcp_config.json`)
- First run downloads ~1-2 GB of model checkpoints from HuggingFace
- ADiT supports **dataset-type** (crystals vs molecules) and **spacegroup** conditioning only — no composition or property conditioning
