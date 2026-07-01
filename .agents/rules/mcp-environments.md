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
| `mace` | MACE MLIP (MP, OMAT, MatPES heads) with CUDA support |
| `mace-lammps` | MACE + LAMMPS (ACEsuit fork, compiled from source) |
| `matgl` | MatGL (CHGNet, M3GNet, TensorNet) with CPU PyTorch |
| `matgl-lammps` | MatGL + LAMMPS-Kokkos-CUDA (conda-forge) |
| `fairchem` | FairChem (UMA/ESEN) with CUDA support |
| `fairchem-lammps` | FairChem + LAMMPS + fairchem-lammps (conda-forge) |
| `atomate2` | Atomate2 + Jobflow-remote + AMSET + Lobsterpy |
| `adit` | ADiT generative model (all-atom diffusion transformer) |
| `diffcsp` | DiffCSP++ crystal structure generation |
| `mattergen` | MatterGen generative crystal design |
| `react-ot` | React-OT transition state generation |
| `scd` | SelfConditionedDenoisingAtoms property prediction |
| `calphad` | CALPHAD phase diagrams (pycalphad) |
| `phasefield` | Phase-field simulations (FiPy) |
| `smol` | Cluster expansion + Monte Carlo (SMOL) |
| `drugdisc` | Drug discovery tools (Vina, fpocket, RDKit, Meeko, PoseBusters) |
| `drugmd` | Drug discovery + OpenMM MD (OpenMM, OpenFF, ParmEd, AmberTools, ProLIF) |
| `nmr` | NMR prediction/analysis (nmrsim, RDKit) |
| `msms` | MS/MS prediction (ICEBERG) |
| `xrd` | XRD analysis (DARA) |
| `void` | Porous materials docking (VOID) |
| `orca` | Molecular DFT via ORCA (SCINE/ReaDuct). Scripts only — no MCP server. |

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
