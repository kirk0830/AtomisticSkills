# AtomisticSkills

![AtomisticSkills Logo](site/logo/atomisticskills_logo.svg)

[![arXiv](https://img.shields.io/badge/arXiv-2605.24002-b31b1b.svg)](https://arxiv.org/abs/2605.24002)

## Overview

**AtomisticSkills** is a composable framework for AI-driven atomistic materials research. Built on the **hierarchical decomposition** of complex scientific tasks into **Workflows** → **Skills** → **Tools**, it enables coding AI agents to autonomously conduct multi-stage materials, chemistry, and drug discovery research by combining modular, reusable capabilities.

The framework integrates state-of-the-art Machine Learning Interatomic Potentials (MLIPs), DFT calculations, generative AI, database APIs, and advanced simulation methods through the Model Context Protocol (MCP) tools and Skills, making advanced materials research accessible to AI copilots like [Google Antigravity](https://antigravity.google), [Cursor](https://www.cursor.com/), [Claude Code](https://code.claude.com/docs/en/overview), and [OpenAI Codex](https://openai.com/codex/).

<div align="center">

🌐 **[Documentation Website](https://learningmatter-mit.github.io/AtomisticSkills/)** &nbsp;|&nbsp; 📄 **[Preprint](https://arxiv.org/abs/2605.24002)**

</div>

---

## Hierarchical Research Framework

**AtomisticSkills** constructs complex scientific tasks from three abstraction levels: **Tools** → **Skills** → **Workflows**.

### 📎 Tools (Low-Level Research Primitives)

[**View MCP Tools**](src/mcp_server)

Tools are **strictly structured, fundamental operations** exposed as Python functions through MCP servers. They have **fixed input/output types** and must match function call signatures exactly.

**Key Characteristics:**
- **Strict Type Checking**: Input and output types must match Python function signatures precisely
- **Battle-Tested**: Optimized, reliable implementations for core operations
- **Direct Callable**: The agent invokes tools directly via MCP protocol

**Tool Categories:**
- Structure relaxation (geometry optimization)
- Molecular dynamics (NVT, NPT, NVE ensembles)
- Monte Carlo simulation (cluster expansion)
- MLIP simulation (MACE, MatGL, FairChem)
- DFT input preparation and output parsing
- Database queries (Materials Project, PubChem, ChEMBL, PDB)

### ⚙️ Skills (Mid-Level Research Tutorials)

[**Browse Skills →**](.agents/skills)

Skills are **flexible tutorials** that combine multiple tool calls to solve focused research problems. Unlike tools, skills have **no fixed input/output type constraints**—the agent handles all data conversion and orchestration between steps.

**Key Characteristics:**
- **Flexible Composition**: Tutorials showing "how to combine tools" for specific tasks
- **Agent-Managed**: The agent handles data format conversions between tool calls
- **Self-Documented**: Each skill includes instructions (`SKILL.md`), helper scripts, and examples

**Examples:**
- [**MLIP Benchmark**](.agents/skills/ml-mlip-benchmark/SKILL.md): Benchmark MLIP accuracy against a labeled dataset
- [**Material Stability**](.agents/skills/mat-stability/SKILL.md): Calculate 0K thermodynamic stability and $E_{hull}$
- [**Diffusion Analysis**](.agents/skills/mat-diffusion-analysis/SKILL.md): Compute diffusion coefficients and activation energies
- [**DFT with ORCA**](.agents/skills/chem-dft-orca-singlepoint/SKILL.md): Run molecular DFT calculations (local or HPC)
- [**DFT with VASP (legacy)**](.agents/skills/mat-dft-vasp/SKILL.md): Run periodic DFT calculations (local, HPC, or Atomate2); CP2K/QE are the recommended open-source alternatives for new workflows, while VASP is retained for atomate2 workflows not yet available for QE/CP2K

### 🎯 Workflows (High-Level Research Objectives)

[**Browse Workflows →**](.agents/workflows)

Workflows represent **complete, high-level research goals** that may span multiple skills and require strategic planning. They provide a research roadmap for the agent to follow.

**Examples:**
- Search for novel MOF sorption materials in the Li-N-O chemical space
- Explore solid-state conductors compatible with LiFePO₄ cathodes
- Design thermally stable perovskites for high-temperature applications

---

## Key Features

### 1. Simulation Infrastructure
- Multi-framework MLIP support (MACE, MatGL, FairChem) with unified API
- DFT integration: CP2K / Quantum ESPRESSO (recommended open-source periodic DFT), ORCA (molecular), and legacy VASP retained for atomate2 workflows not yet available for QE/CP2K (dielectric response, ferroelectric, electronic transport)
- HPC job submission via unified Slurm module (local login node or SSH)
- Lattice-level cluster expansion and Monte Carlo via SMOL

### 2. Database APIs
- Materials Project, ChEMBL, PDB, PubChem, QMOF, ArXiv
- Query structures, properties, bioactivity data, and literature

### 3. Property Evaluation
- Stability ($E_{hull}$), phase diagrams, phonons, QHA thermal expansion
- Elastic tensor, melting point, ionic diffusion, NEB barriers
- Surface energy & adsorption, grain boundary energy, intercalation voltage
- Pourbaix diagrams, vibrational spectra, Raman spectra

### 4. Experimental Tools
- Synthesis recommendation from text-mined literature
- XRD spectrum calculation and phase analysis
- Protein preparation, molecular docking (AutoDock Vina)
- ADMET prediction, molecular fingerprints

### 5. Machine Learning Tools
- MatterGen (generative crystal design)
- MLIP fine-tuning & benchmarking
- Foundation potential selection guide
- Cluster expansion training

---

## Installation

> [!CAUTION]
> **🔒 安全声明 — 关于本 Fork 的安全改进**
>
> 本 fork 仓库 (`kirk0830/AtomisticSkills`) 针对 upstream 仓库进行了以下安全性和可维护性改进：
>
> - **🛡️ 存储空间安全**：安装前强制检查磁盘空间，避免因磁盘耗尽导致的系统故障
> - **🛡️ 环境隔离保护**：所有依赖通过 Pixi 严格隔离，杜绝 PATH 污染和全局环境破坏
> - **🛡️ 文件操作安全**：配置脚本采用幂等设计，避免意外覆盖用户已有文件
> - **🛡️ AstrBot 安全适配**：通过软连接机制实现技能映射，避免文件复制带来的权限风险
> - **🛡️ 模块化代码结构**：配置逻辑按 agent 类型拆分，降低代码复杂度和安全审计难度
> - **🛡️ 模板化文档生成**：使用 Jinja2 模板统一管理配置文档，减少手动出错风险
>
> **与 upstream 的主要区别：**
> - 环境管理从 Conda 完全迁移至 Pixi（隔离环境、可复现锁文件）
> - 新增 AstrBot 聊天机器人框架完整支持（技能软连接、人格设定、MCP 配置生成）
> - 统一 CLI 入口 `atomisticskills`（整合 configure / list-agents / list-servers）
> - 配置代码模块化拆分，提高可维护性
> - Jinja2 模板引擎用于配置和文档生成
> - 中文人格设定和文档支持

> ⚠️ **Disk space requirement**: AtomisticSkills environments contain large scientific
> packages (PyTorch, RDKit, OpenMM, CUDA toolkits, etc.). Before installing, check
> available space with `df -h .`.
>
> | Install scope | Approx. disk required |
> |---------------|----------------------:|
> | Minimal (`supercomputing` or `base`) | ~3 GB |
> | Core research (MLIP + DFT) | ~40–60 GB |
> | Full (all environments) | ≥150 GB, prefer 200 GB |
> | Full + optional build tasks (VOID, SCD, react-ot, ICEBERG) | 200 GB+ |
>
> **New**: `supercomputing` environment (~3 GB) is all you need on the login node
> for Slurm job submission. Heavy environments (mace, vasp, cp2k, …) are only needed
> on compute nodes. See `pixi.toml` for per-environment disk estimates.

AtomisticSkills uses **Pixi** for reproducible, isolated environment management:

- **No PATH pollution**: Environments isolated in `.pixi/envs/`
- **Lockfile reproducibility**: `pixi.lock` ensures identical environments
- **No brutal delete/recreate**: Incremental updates
- **Declarative config**: All dependencies in `pixi.toml`
- **Python package**: `atomistic-skills` installed as editable package in all environments
- **Unified CLI**: `atomisticskills configure --agent <name>` for all agent setup

### System Requirements & Disk Space

Before installing, ensure you have enough free disk space. The `.pixi/` directory
contains one isolated environment per research area and grows quickly.

Recommended free space:

- **Login node / supercomputing only**: ~3 GB for `supercomputing` or `base`
- **Core research** (MLIP + DFT, no LAMMPS/generative/specialized): ~40–60 GB
- **Full install** (all 25 environments): ≥150 GB, 200 GB recommended
- **Full install + optional build tasks**: 200 GB+

Check available space before you start:

```bash
df -h .
```

If disk space is limited, install only the environments you need instead of running
`pixi install` for all environments. See the **Available Environments** table below
for per-environment notes and sizes.

### Quick Start

1. **Clone the repository**:
   ```bash
   # 本 fork 仓库 (kirk0830/AtomisticSkills) - 包含安全改进和 AstrBot 支持
   git clone https://github.com/kirk0830/AtomisticSkills.git
   cd AtomisticSkills
   ```

2. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

3. **Install environments**:
   ```bash
   # Install all environments (requires ≥150 GB free disk space)
   pixi install

   # Or install specific environments only to save space
   pixi install -e base
   pixi install -e base -e mace
   pixi install -e matgl
   ```

4. **Configure MCP servers**:
   ```bash
   pixi run -e base atomisticskills configure
   ```

5. **Add to your AI assistant**:
   
   | Client | Project scope | Global scope |
   |--------|--------------|--------------|
   | **Claude Code** | `.mcp.json` | `~/.claude.json` |
   | **Cursor** | `.cursor/mcp.json` | `~/.cursor/mcp.json` |
   | **Gemini CLI** | `.gemini/settings.json` | `~/.gemini/settings.json` |
   | **Codex CLI** | `.codex/config.toml` | `~/.codex/config.toml` |
   | **AstrBot** | n/a — see [AstrBot Integration](docs/astrbot-integration.md) | n/a |

   ```bash
    # IDE-embedded agents (project scope)
    pixi run -e base atomisticskills configure

    # IDE-embedded agents (global scope)
    pixi run -e base atomisticskills configure --scope global

    # AstrBot chatbot framework
    pixi run -e base atomisticskills configure --agent astrbot --data-dir /path/to/astrbot/data
   ```

   > **Which environment?** The `atomisticskills configure` command only needs
   > the `base` environment (~3 GB) — it doesn't require heavy MLIP or DFT
   > environments.  The actual computation (MCP servers, Slurm jobs) runs in
   > the environments specified in each Skill's `# Env:` annotation.

### Available Environments

**Tier: Lightweight** (login-node safe, no CUDA/MLIP/DFT software)

| Environment | Description | Size |
|-------------|-------------|------|
| `supercomputing` | **Slurm job submission MCP server** — all `*_via_slurm` tools. Recommended for login nodes. | ~3 GB |
| `base` | General research: Materials Project, literature, structure tools, plotting. | ~3.5 GB |

**Tier: MLIP** (GPU recommended)

| Environment | Description | Size |
|-------------|-------------|------|
| `mace` | MACE MLIP (OMAT, MatPES heads) with CUDA | ~12 GB |
| `matgl` | MatGL (CHGNet, M3GNet, TensorNet) with CPU PyTorch | ~10 GB |
| `fairchem` | FairChem (UMA/ESEN) with CUDA | ~15 GB |

**Tier: DFT** (open-source unless noted)

| Environment | Description | Size |
|-------------|-------------|------|
| `orca` | Molecular DFT via ORCA ⚠️ external binary required | ~6 GB |
| `vasp` | Periodic DFT via VASP/Atomate2 ⚠️ VASP license required | ~8 GB |
| `cp2k` | Periodic DFT via CP2K — fully open-source | ~6 GB |
| `qe` | Periodic DFT via Quantum ESPRESSO — fully open-source | ~6 GB |

**Tier: MD / LAMMPS**

| Environment | Description | Size |
|-------------|-------------|------|
| `mace-lammps` | LAMMPS + MACE (ACEsuit fork, source build) | ~15 GB |
| `matgl-lammps` | LAMMPS + MatGL (conda-forge, CUDA) | ~13 GB |
| `fairchem-lammps` | LAMMPS + FairChem | ~18 GB |

**Tier: Generative**

| Environment | Description | Size |
|-------------|-------------|------|
| `diffcsp` | DiffCSP++ crystal generation | ~10 GB |
| `mattergen` | MatterGen diffusion-based generation (CUDA 11.8) | ~15 GB |
| `adit` | ADiT all-atom diffusion transformer | ~8 GB |
| `react-ot` | React-OT TS generation (git build) | ~4 GB |
| `scd` | SCD property prediction (TorchMD kernel build) | ~10 GB |

**Tier: Specialized**

| Environment | Description | Size |
|-------------|-------------|------|
| `smol` | Cluster expansion + Monte Carlo | ~4 GB |
| `calphad` | CALPHAD phase diagrams (pycalphad) | ~5 GB |
| `phasefield` | Cahn-Hilliard / Allen-Cahn (FiPy) | ~4 GB |
| `drugdisc` | Drug discovery (docking, ADMET, fingerprints) | ~6 GB |
| `drugmd` | Drug discovery + OpenMM MD | ~10 GB |
| `nmr` | NMR prediction/analysis (nmrsim) | ~5 GB |
| `msms` | MS/MS prediction (ICEBERG, Python 3.10) | ~8 GB |
| `xrd` | XRD analysis (DARA, manual install) | ~4 GB |
| `void` | Porous materials docking (VOID, git build) | ~5 GB |

### Configuration

Create `~/.atomistic_skills.yaml` for API keys and HPC settings:

```yaml
# API Keys
MP_API_KEY: "your_mp_api_key_here"

# HPC Configuration (for Slurm job submission)
hpc:
  profile: "nersc_perlmutter"  # or "generic", "mit_supercloud", etc.
  mode: "auto"                  # "local", "ssh", or "auto"

  # SSH mode (for remote submission)
  ssh_host: "cluster.university.edu"
  ssh_user: "your_username"
  ssh_key: "~/.ssh/id_ed25519"

  # Application-specific modules
  modules:
    vasp: ["vasp/6.4.2-cpu"]
    orca: ["orca/5.0.4", "openmpi/4.1.5"]
```

See [docs/hpc_job_submission.md](docs/hpc_job_submission.md) for full HPC configuration details.

#### API Keys & Credentials

AtomisticSkills integrates several external APIs. Without the corresponding keys, the related features will fail:

| Use case | Required key(s) | What fails without them |
| :--- | :--- | :--- |
| Materials Project structure/property queries | `MP_API_KEY` | All Materials Project lookups fail. |
| MOF / QMOF database queries | `MP_API_KEY` | QMOF queries fail. |
| Literature full-text downloads | `ELSEVIER_API_KEY`, `SPRINGER_API_KEY`, `UNPAYWALL_EMAIL` | Automatic downloads from Elsevier / Springer / Unpaywall fail; only metadata/abstracts remain. |
| Literature search politeness | `OPENALEX_EMAIL` (recommended) | OpenAlex falls back to a shared default email; Unpaywall cannot use `OPENALEX_EMAIL` as fallback unless it is set. |
| IBM RXN retrosynthesis | `RXN_API_KEY` | Retrosynthesis predictions fail. |
| HuggingFace gated models | `HF_TOKEN` | Model / dataset downloads fail. |
| VLM plot digitization | `GOOGLE_API_KEY` or `OPENAI_API_KEY` | Chart metadata extraction fails. |

See [docs/environment_variables.md](docs/environment_variables.md) for the full variable list and [docs/api_key_guide.md](docs/api_key_guide.md) for step-by-step acquisition instructions.

---

## Usage

### Running Skills

Each skill specifies its required environment via `# Env: <name>` in the SKILL.md:

```bash
# Run ORCA single-point calculation
pixi run -e orca python .agents/skills/chem-dft-orca-singlepoint/scripts/run_singlepoint.py \
    --structure molecule.xyz \
    --functional B3LYP \
    --basis_set def2-TZVP

# Run material stability calculation (uses Materials Project data, not VASP)
pixi run -e base python .agents/skills/mat-stability/scripts/calculate_stability.py \
    --structure LiFePO4.cif
```

### Using MCP Tools

MCP tools are automatically available once configured. In your AI assistant:

```
Search Materials Project for LiFePO4 structures with bandgap < 3 eV
```

The assistant will use the `search_materials_project` tool from the base MCP server.

### HPC Job Submission

For computationally expensive calculations (DFT, large-scale MD), submit to HPC:

```python
# From any pixi environment
from src.utils.dft.orca_hpc import OrcaHPCRunner

runner = OrcaHPCRunner(mode="hpc")
result = runner.run_singlepoint(
    structure_path="molecule.xyz",
    functional="B3LYP",
    basis_set="def2-TZVP",
    nprocs=16,
)
print(f"Job ID: {result.job_id}")
print(f"Energy: {result.energy_eV:.6f} eV")
```

---

## Developer Guide

### Architecture & Layer Map

The framework uses a **three-layer hierarchy** — Workflows → Skills → MCP Tools. For a complete cross-reference of which skills use which MCP tools and which workflows compose which skills, see the [Skill / MCP Tool / Workflow Map](docs/skill_mcp_workflow_map.md).

```
┌─────────────────────────────────────────────────────────┐
│                    AI Agent (IDE)                       │
└────────────────────┬────────────────────────────────────┘
                     │ MCP Protocol
          ┌──────────┴──────────┬───────────┬─────────────┐
          │                     │           │             │
     ┌────▼────┐         ┌──────▼──┐   ┌───▼───┐   ┌─────▼─────┐
     │  MACE   │         │ MatGL   │   │  Fair │   │   Base    │
     │ Server  │         │ Server  │   │ Chem  │   │  Server   │
     └────┬────┘         └────┬────┘   └───┬───┘   └─────┬─────┘
          │                   │            │             │
     ┌────▼────┐         ┌────▼────┐  ┌───▼───┐   ┌─────▼─────┐
     │ .pixi/  │         │ .pixi/  │  │ .pixi/│   │  .pixi/   │
     │envs/mace│         │envs/matgl│ │fair   │   │ envs/base │
     └─────────┘         └─────────┘  └───────┘   └───────────┘
```

### Project Structure

```
AtomisticSkills/
├── pixi.toml              # Environment definitions (all dependencies)
├── pyproject.toml         # Python package configuration
├── pixi.lock              # Lockfile (reproducibility)
├── src/
│   ├── cli.py             # Unified CLI entry point
│   ├── config/            # Agent-specific configuration modules
│   ├── mcp_server/        # MCP server implementations
│   └── utils/             # Utility modules
│       ├── hpc/           # HPC job submission module
│       ├── mlips/         # MLIP wrappers (MACE, MatGL, FairChem)
│       ├── dft/           # DFT utilities (VASP, ORCA)
│       ├── drugdisc/      # Drug discovery utilities
│       ├── generative/    # Generative model wrappers
│       └── ...
├── .agents/
│   ├── rules/             # Project-specific standards
│   ├── skills/            # Skill definitions (129+ skills)
│   ├── workflows/         # Workflow definitions
│   ├── templates/         # Jinja2 templates for config/doc generation
│   └── patches/           # Git dependency patches
├── .pixi/
│   ├── envs/              # Isolated Pixi environments
│   └── build/             # Build artifacts for git deps
├── configure_mcp.py       # [deprecated] use atomisticskills configure instead
└── configure_astrbot.py   # [deprecated] use atomisticskills configure --agent astrbot
```

### Adding a New Skill

1. Create `.agents/skills/<skill-name>/SKILL.md`
2. Add scripts to `scripts/` directory
3. Specify required environment: `# Env: <name>`
4. Test with: `pixi run -e <env> python scripts/<script>.py`

See [`.agents/rules/skill-standards.md`](.agents/rules/skill-standards.md) for detailed guidelines.

### Adding a New MCP Tool

1. Choose the appropriate MCP server in `src/mcp_server/`
2. Add the tool function with type hints
3. Test by running the server: `pixi run -e <env> python -m src.mcp_server.<server>`

---

## Best Practices

1. **Leverage Local GPUs**: MLIP tasks run fastest with local GPU resources
2. **Use HPC for DFT**: Submit ORCA, VASP, CP2K, or Quantum ESPRESSO calculations to HPC clusters via the unified module (for ASE-QE/CP2K skills, generate inputs locally and submit the run directory manually)
3. **Configure Once**: Set up `~/.atomistic_skills.yaml` with all your API keys and HPC settings
4. **Contribute Back**: Submit PRs for new skills, tools, or bug fixes

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| MCP tools not showing | Verify JSON syntax in config file, restart IDE |
| `pixi install` fails | Check network connection, try `pixi install --frozen` |
| `pixi install` fails with “No space left on device” | Free up disk or install only needed environments (`pixi install -e <env>`). Lightweight install needs ~80–100 GB; full install needs ≥150 GB. |
| HPC submission fails | Verify SSH key, check `HPC_MODE` and `HPC_SSH_*` env vars |
| Import errors | Run `pixi install` to ensure package is installed |
| MLIP environment conflicts | Use separate pixi environments (each is isolated) |

---

## Citation

If you use AtomisticSkills in your research, please cite:

```bibtex
@article{deng2025atomisticskills,
  title   = {Harnessing AtomisticSkills for Agentic Atomistic Research},
  author  = {Bowen Deng and Bohan Li and Matthew Cox and Hoje Chun and Juno Nam and
             Artur Lyssenko and Sathya Edamadaka and Jurgis Ruza and Xiaochen Du and
             Nofit Segal and Jesus Diaz Sanchez and Mingrou Xie and Ty Perez and
             Yu Yao and Miguel Steiner and Sauradeep Majumdar and Charles B. Musgrave III and
             Anirban Chandra and Abhirup Patra and Detlef Hohl and Connor W. Coley and
             Ju Li and Rafael G{\'{o}}mez-Bombarelli},
  journal = {arXiv preprint arXiv:2605.24002},
  year    = {2025},
  url     = {https://arxiv.org/abs/2605.24002},
  doi     = {10.48550/arXiv.2605.24002}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.