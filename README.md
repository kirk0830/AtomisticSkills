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

AtomisticSkills uses **Pixi** for reproducible, isolated environment management. This replaces the previous Conda-based approach with significant improvements:

- **No PATH pollution**: Environments isolated in `.pixi/envs/`
- **Lockfile reproducibility**: `pixi.lock` ensures identical environments
- **No brutal delete/recreate**: Incremental updates
- **Declarative config**: All dependencies in `pixi.toml`
- **Python package**: `atomistic-skills` installed as editable package in all environments

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone git@github.com:learningmatter-mit/AtomisticSkills.git
   cd AtomisticSkills
   ```

2. **Install Pixi** (if not already installed):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

3. **Install environments**:
   ```bash
   # Install all environments
   pixi install

   # Or install specific environments only
   pixi install -e base
   pixi install -e mace
   pixi install -e matgl
   ```

4. **Configure MCP servers**:
   ```bash
   pixi run -e base python configure_mcp.py
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
   pixi run -e base python configure_mcp.py

   # IDE-embedded agents (global scope)
   pixi run -e base python configure_mcp.py --scope global

   # AstrBot chatbot framework (uses symlinks into data/skills/ + WebUI MCP config)
   pixi run -e base python configure_astrbot.py --data-dir /path/to/astrbot/data
   ```

### Available Environments

| Environment | Description |
|-------------|-------------|
| `base` | Materials Project queries, VASP I/O, base tools (recommended) |
| `mace` | MACE models (MP, OMAT, MatPES) |
| `matgl` | MatGL models (CHGNet, M3GNet, TensorNet) |
| `fairchem` | FairChem models (UMA, ESEN) |
| `atomate2` | DFT workflow management via Atomate2 + jobflow-remote |
| `smol` | Cluster expansion and Monte Carlo |
| `drugdisc` | Drug discovery tools (docking, ADMET, fingerprints) |
| `mattergen` | Generative crystal design |
| `orca` | Molecular DFT via ORCA (local or HPC) |
| `react-ot` | Transition state generation |
| `lammps-mace` | LAMMPS with MACE backend |
| `lammps-matgl` | LAMMPS with MatGL backend |
| `lammps-fairchem` | LAMMPS with FairChem backend |

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
├── pixi.toml              # Environment definitions (all dependencies)
├── pyproject.toml         # Python package configuration
├── pixi.lock              # Lockfile (reproducibility)
├── src/
│   ├── mcp_server/        # MCP server implementations
│   └── utils/             # Utility modules
│       ├── hpc/           # HPC job submission module
│       ├── mlips/         # MLIP wrappers (MACE, MatGL, FairChem)
│       └ dft/             # DFT utilities (VASP, ORCA)
│       └ drugdisc/        # Drug discovery utilities
│       └ generative/      # Generative model wrappers
│       └ ...
├── .agents/
│   ├── rules/             # Project-specific standards
│   ├── skills/            # Skill definitions (129+ skills)
│   └ workflows/           # Workflow definitions
│   └ patches/             # Git dependency patches
├── .pixi/
│   └ envs/                # Isolated environments
│   └ build/               # Build artifacts for git deps
├── configure_mcp.py       # MCP configuration generator (IDE agents)
└── configure_astrbot.py   # AstrBot chatbot framework configurator
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
| MCP tools not showing | Verify JSON syntax in config file, restart IDE |
| `pixi install` fails | Check network connection, try `pixi install --frozen` |
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