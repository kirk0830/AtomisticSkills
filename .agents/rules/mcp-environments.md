---
trigger: always_on
---

# MCP Server Environment Rules

When running scripts or debugging code related to specific MCP servers, you MUST use the corresponding environment defined in `pixi.toml` (preferred) or `mcp_config.json`.

For general development logic that cuts across tools, usually the `base` environment is sufficient, but when importing specific libraries that are isolated (like `atomate2`, `jobflow_remote`, `mace`, `fairchem`), you must use the correct environment.

## Environment Management: Pixi

All environments are managed through **Pixi**:
- Environments are isolated within `.pixi/envs/` (no global PATH pollution)
- `pixi.lock` guarantees reproducible installs
- Declarative configuration in `pixi.toml`
- No brutal delete/recreate — incremental updates

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

## Running Scripts in the Right Environment

### Pixi

```bash
# Run a command in a specific environment
pixi run -e <env-name> <command>

# Or activate the environment shell
pixi shell -e <env-name>
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

Note: Pixi environment names do not use the `-agent` suffix — use `mace`, `matgl`, `base`, etc.
