# AtomisticSkills

![AtomisticSkills Logo](site/logo/atomisticskills_logo.svg)

[![arXiv](https://img.shields.io/badge/arXiv-2605.24002-b31b1b.svg)](https://arxiv.org/abs/2605.24002)

## Overview

**AtomisticSkills** is a composable framework for AI-driven atomistic materials research. Built on the **hierarchical decomposition** of complex scientific tasks into **Workflows** → **Skills** → **Tools**, it enables coding AI agents to autonomously conduct multi-stage materials, chemistry, and drug discovery research by combining modular, reusable capabilities.

The framework integrates state-of-the-art Machine Learning Interatomic Potentials (MLIPs), DFT calculations, generative AI, database APIs, and advanced simulation methods through the Model Context Protocol (MCP) tools and Skills, making advanced materials research accessible to AI copilots like [Google Antigravity](https://antigravity.google), [Cursor](https://www.cursor.com/), [Claude Code](https://code.claude.com/docs/en/overview), [OpenAI Codex](https://openai.com/codex/), [Windsurf](https://windsurf.com/), and [AstrBot](https://docs.astrbot.app/).

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
- [**DFT with VASP**](.agents/skills/mat-dft-vasp/SKILL.md): Run periodic DFT calculations (local, HPC, or Atomate2)

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
- DFT integration: VASP (periodic) and ORCA (molecular)
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

> ⚠️ **Disk space requirement**: AtomisticSkills environments contain large scientific
> packages (PyTorch, RDKit, OpenMM, CUDA toolkits, etc.). Before installing, check
> available space with `df -h .`.
>
> | Install scope | Approx. disk required |
> |---------------|----------------------:|
> | Minimal (`base` only) | ~3 GB |
> | Lightweight (no VASP/ORCA/LAMMPS) | ~80–100 GB |
> | Full (all environments) | ≥150 GB, prefer 200 GB |
> | Full + optional build tasks (VOID, SCD, react-ot, ICEBERG) | 200 GB+ |

AtomisticSkills uses **Pixi** for reproducible, isolated environment management. This replaces the previous Conda-based approach with significant improvements:

- **No PATH pollution**: Environments isolated in `.pixi/envs/`
- **Lockfile reproducibility**: `pixi.lock` ensures identical environments
- **Incremental updates**: No brutal delete/recreate cycles
- **Declarative config**: All dependencies defined in `pixi.toml`
- **Editable Python package**: `atomistic-skills` installed as editable in all environments

### System Requirements & Disk Space

Before installing, ensure you have enough free disk space. The `.pixi/` directory
contains one isolated environment per research area and grows quickly.

Recommended free space:

- **Minimal / single environment**: ~3 GB for `base`
- **Lightweight install** (no VASP, ORCA, or LAMMPS): ~80–100 GB
- **Full install** (all `pixi.toml` environments): ≥150 GB, 200 GB recommended
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
   git clone https://github.com/LearningMatter-MIT/AtomisticSkills.git
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
   # Auto-detect installed agents and write project-scoped config
   pixi run -e base python configure_mcp.py

   # List detected agents first
   pixi run -e base python configure_mcp.py --list-agents
   ```

5. **Add to your AI assistant**:

   | Client | Project scope | Global scope |
   |--------|--------------|--------------|
   | **Claude Code** | `.mcp.json` | `~/.claude.json` |
   | **Cursor** | `.cursor/mcp.json` | `~/.cursor/mcp.json` |
   | **Gemini CLI / IDE** | `.gemini/settings.json` | `~/.gemini/settings.json` |
   | **Codex CLI** | `.codex/config.toml` | `~/.codex/config.toml` |
   | **Windsurf** | n/a | `~/.codeium/windsurf/mcp_config.json` |
   | **AstrBot** | n/a — see [AstrBot Integration](docs/astrbot-integration.md) | n/a |

   ```bash
   # Auto-detect and configure all installed agents (project scope, default)
   pixi run -e base python configure_mcp.py

   # Configure specific agent(s) only
   pixi run -e base python configure_mcp.py --agent claude
   pixi run -e base python configure_mcp.py --agent claude cursor gemini

   # Global scope (available across all projects)
   pixi run -e base python configure_mcp.py --scope global

   # Both project and global scope
   pixi run -e base python configure_mcp.py --scope both

   # AstrBot chatbot framework (symlinks skills + generates MCP config)
   pixi run -e base python configure_astrbot.py --data-dir /path/to/astrbot/data
   ```

### Available Environments

AtomisticSkills defines 23 isolated Pixi environments in `pixi.toml`. Below is a categorized reference:

#### MCP Server Environments (Interactive Tools)

These environments expose MCP servers for AI agent interaction:

| Environment | Description | Size / Notes |
|-------------|-------------|--------------|
| `base` | Core tools: Materials Project queries, VASP I/O, structure utils, base MCP server | ~2–3 GB (recommended) |
| `mace` | MACE MLIP (MP, OMAT, MatPES models) + MCP server | ~15 GB, CUDA 12 |
| `matgl` | MatGL MLIP (CHGNet, M3GNet, TensorNet) + MCP server | ~12 GB |
| `fairchem` | FairChem MLIP (UMA, ESEN) + MCP server | ~16 GB, CUDA 12 |
| `smol` | Cluster expansion + Monte Carlo via SMOL + MCP server | ~2 GB |
| `drugdisc` | Drug discovery: docking, ADMET, fingerprints + MCP server | ~2–3 GB |
| `mattergen` | Generative crystal design (MatterGen) + MCP server | ~15 GB, CUDA 11.8 |
| `diffcsp` | DiffCSP++ generative model + MCP server | ~10 GB |
| `adit` | ADiT all-atom diffusion transformer + MCP server | ~12 GB |

#### Script-Only Environments (No MCP Server)

These environments are used for running skill scripts directly via `pixi run -e <env>`:

| Environment | Description | Size / Notes |
|-------------|-------------|--------------|
| `atomate2` | DFT workflow management via Atomate2 + jobflow-remote | ~5–10 GB, VASP workflows |
| `orca` | Molecular DFT via ORCA (local or HPC) | ~3–5 GB, requires external ORCA binary |
| `mace-lammps` | LAMMPS with MACE pair_style (ACEsuit fork, source build) | Very heavy, build via `pixi run -e mace-lammps build-lammps-mace` |
| `matgl-lammps` | LAMMPS with MatGL (conda-forge lammps + Kokkos CUDA) | Heavy, CUDA 12 |
| `fairchem-lammps` | LAMMPS with FairChem (conda-forge lammps CPU + lmp_fc) | Heavy |
| `react-ot` | Transition state generation (React-OT) | ~1 GB, git build via `pixi run -e react-ot install-react-ot` |
| `scd` | Self-Conditioned Denoising property prediction | ~10 GB |
| `calphad` | CALPHAD phase diagram calculations | ~2 GB |
| `phasefield` | Phase-field simulations (Allen-Cahn, Cahn-Hilliard) | ~2 GB |
| `drugmd` | Drug discovery MD: OpenMM, OpenFF, AmberTools, ProLIF | ~5–8 GB |
| `nmr` | NMR spectrum prediction and analysis | ~2–3 GB |
| `msms` | LC-MS/MS prediction via ICEBERG | ~10 GB, Python 3.10 |
| `xrd` | XRD spectrum calculation and phase analysis | ~2 GB |
| `void` | MOF sorption docking via VOID library | ~3–5 GB, install via `pixi run -e void install-void` |

### Configuration

Create a config file for API keys and HPC settings. The framework searches in this order:

1. `~/.atomistic_skills.yaml` (recommended)
2. `~/.atomistic_skills.yml`
3. `~/.config/atomistic_skills/config.yaml`

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
  ssh_port: 22
  ssh_remote_work_dir: "~/hpc_jobs"

  # Application-specific modules (override profile defaults)
  modules:
    vasp: ["vasp/6.4.2-cpu"]
    orca: ["orca/5.0.4", "openmpi/4.1.5"]
```

Environment variables with `MLIP_` prefix override config file values (e.g., `MLIP_MP_API_KEY`).

See [docs/hpc_job_submission.md](docs/hpc_job_submission.md) for full HPC configuration details.

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

# Run VASP stability calculation
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
├── pixi.toml              # Environment definitions (all dependencies, 23 envs)
├── pyproject.toml         # Python package configuration
├── pixi.lock              # Lockfile for reproducibility
├── mcp_config.json        # MCP server definitions (source template with PIXI_PROJECT placeholders)
├── configure_mcp.py       # MCP config generator for IDE agents (Claude, Cursor, Gemini, Codex, Windsurf)
├── configure_astrbot.py   # AstrBot chatbot framework configurator (skill symlinks + MCP config)
├── src/
│   ├── config/            # Modular configuration system
│   │   ├── base.py        # Shared utilities (symlinks, JSON, Jinja2, instruction files)
│   │   ├── mcp_loader.py  # MCP server loader + path rewriting (pixi/conda support)
│   │   └── agents/        # Per-agent config writers
│   │       ├── claude.py  # Claude Code (.mcp.json / ~/.claude.json)
│   │       ├── cursor.py  # Cursor IDE (.cursor/mcp.json + rules)
│   │       ├── gemini.py  # Gemini CLI/IDE (settings.json + plugin + global instructions)
│   │       ├── codex.py   # OpenAI Codex CLI (config.toml + global skills symlinks)
│   │       ├── windsurf.py # Windsurf IDE (global mcp_config.json only)
│   │       └── astrbot.py # AstrBot chatbot (skill symlinks + persona + MCP config)
│   ├── mcp_server/        # MCP server implementations (base, mace, matgl, fairchem, ...)
│   └── utils/             # Utility modules
│       ├── hpc/           # HPC job submission module (Slurm)
│       ├── mlips/         # MLIP wrappers (MACE, MatGL, FairChem)
│       ├── dft/           # DFT utilities (VASP, ORCA)
│       ├── drugdisc/      # Drug discovery utilities
│       ├── generative_models/ # Generative model wrappers
│       └── ...
├── .agents/
│   ├── rules/             # Project-specific standards
│   ├── skills/            # Skill definitions (100+ skills)
│   ├── workflows/         # Workflow definitions (9 workflows)
│   ├── templates/         # Jinja2 templates for config/doc generation
│   └── patches/           # Git dependency patches
├── .pixi/
│   └ envs/                # Isolated environments (one per research area)
│   └ build/               # Build artifacts for git deps
├── docs/                  # Documentation
└── tests/                 # Test suite
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
2. **Use HPC for DFT**: Submit ORCA/VASP calculations to HPC clusters via the unified module
3. **Configure Once**: Set up `~/.atomistic_skills.yaml` with all your API keys and HPC settings
4. **Contribute Back**: Submit PRs for new skills, tools, or bug fixes

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| MCP tools not showing | Verify JSON syntax in config file, restart IDE / agent |
| `pixi install` fails | Check network connection, try `pixi install --frozen` |
| `pixi install` fails with "No space left on device" | Free up disk or install only needed environments (`pixi install -e <env>`). Lightweight install needs ~80–100 GB; full install needs ≥150 GB. |
| `configure_mcp.py` can't find conda/pixi | Ensure `pixi.toml` exists in project root, or pass `--conda /path/to/miniforge3` explicitly |
| HPC submission fails | Verify SSH key, check `hpc` config in `~/.atomistic_skills.yaml` |
| Import errors when running scripts | Run with `pixi run -e <env>` to use the correct environment |
| MLIP environment conflicts | Use separate pixi environments (each is fully isolated) |
| Windsurf MCP not working | Windsurf only supports global config; use `--scope global` |
| AstrBot can't see skills | Verify symlinks in `<data-dir>/skills/`, re-run `configure_astrbot.py` |

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