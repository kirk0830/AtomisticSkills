---
trigger: always_on
---

# MCP Server Environment Rules

When running scripts or debugging code related to specific MCP servers, you MUST use the corresponding environment defined in `pixi.toml` (preferred) or `mcp_config.json`.

For general development logic that cuts across tools, usually the `base` environment is sufficient, but when importing specific libraries that are isolated (like `atomate2`, `jobflow_remote`, `mace`, `fairchem`), you must use the correct environment.

## Environment Management: Pixi (Preferred) vs Conda

This project supports both Pixi and Conda environments. **Pixi is the preferred and recommended approach** because:
- Environments are isolated within `.pixi/envs/` (no global PATH pollution)
- `pixi.lock` guarantees reproducible installs
- Declarative configuration in `pixi.toml`
- No `conda env remove` brute-force reinstalls

Conda is still supported for backward compatibility. If `pixi.toml` exists in the project root, `configure_mcp.py` will automatically use Pixi environments.

## Installation Instructions

### Pixi (Recommended)

Install environments with:
```bash
pixi install -e <env-name>
```

All environments are defined in `pixi.toml`.

| Environment | Description |
| :--- | :--- |
| `base` | Core: pymatgen, ase, rdkit, packmol |
| `mace` | Core: mace-torch, pymatgen, ase (CUDA) |
| `mace-lammps` | mace + LAMMPS (compiled from ACEsuit fork) |
| `matgl` | Core: matgl, pymatgen, dgl (CPU) |
| `matgl-lammps` | matgl + LAMMPS-Kokkos-CUDA (conda-forge) |
| `fairchem` | Core: fairchem-core≥2.18, pymatgen, Python 3.12 (CUDA) |
| `fairchem-lammps` | fairchem + LAMMPS + fairchem-lammps (conda-forge) |
| `atomate2` | Core: atomate2, pymatgen |
| `adit` | Core: ADiT, lightning, hydra, PyG |
| `diffcsp` | Core: DiffCSP++, hydra, PyG, pyxtal |
| `mattergen` | Core: MatterGen, PyG, lightning |
| `smol` | Core: smol, pymatgen |
| `drugdisc` | Core: rdkit, autodock-vina, pdbfixer, meeko |
| `calphad` | Core: pycalphad, pymatgen |
| `phasefield` | Core: fipy, scipy, numpy, imageio |
| `nmr` | Core: nmrsim, rdkit |
| `msms` | Core: ICEBERG/ms-pred, rdkit, torch, dgl |
| `void` | Core: VOID, rdkit, networkx |
| `orca` | Core: scine_utilities, scine_readuct, ase. Requires `ORCA_BINARY_PATH` env var. No MCP server, scripts only. |
| `react-ot` | Core: PyTorch + react-ot. No MCP server, scripts only. |
| `scd` | Core: SelfConditionedDenoisingAtoms, lightning, huggingface. No MCP server, scripts only. |

### Conda (Legacy)

For detailed installation instructions, please refer to the `README.md` and `install.sh` located in each environment's directory under `conda-envs/<env_name>/`.

| MCP Server | Conda Environment | Python Path |
| :--- | :--- | :--- |
| base | `base-agent` | `<conda_base>/envs/base-agent/bin/python` |
| mace | `mace-agent` | `<conda_base>/envs/mace-agent/bin/python` |
| matgl | `matgl-agent` | `<conda_base>/envs/matgl-agent/bin/python` |
| fairchem | `fairchem-agent` | `<conda_base>/envs/fairchem-agent/bin/python` |
| atomate2 | `atomate2-agent` | `<conda_base>/envs/atomate2-agent/bin/python` |
| smol | `smol-agent` | `<conda_base>/envs/smol-agent/bin/python` |
| adit | `adit-agent` | `<conda_base>/envs/adit-agent/bin/python` |
| diffcsp | `diffcsp-agent` | `<conda_base>/envs/diffcsp-agent/bin/python` |
| mattergen | `mattergen-agent` | `<conda_base>/envs/mattergen-agent/bin/python` |
| drugdisc | `drugdisc-agent` | `<conda_base>/envs/drugdisc-agent/bin/python` |

## Running Scripts in the Right Environment

### Pixi

```bash
# Run a command in a specific environment
pixi run -e <env-name> <command>

# Or activate the environment shell
pixi shell -e <env-name>
```

### Conda (Legacy)

```bash
# Run with conda run
conda run -n <env-name> python <script>

# Or activate
conda activate <env-name>
```

## Skill Environment Annotations

Skills use `# Env: <env-name>` comments to indicate which environment is required. For example:
```
# Env: mace
python .agents/skills/ml-mace-finetune/scripts/train_mace.py
```

With Pixi, this translates to:
```bash
pixi run -e mace python .agents/skills/ml-mace-finetune/scripts/train_mace.py
```

Note: Conda environment names use the `-agent` suffix (e.g., `mace-agent`), while Pixi environment names do not (e.g., `mace`).
