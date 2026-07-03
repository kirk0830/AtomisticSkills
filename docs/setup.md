# AtomisticSkills Setup Guide

Guide the user step-by-step through setting up AtomisticSkills with Pixi.

## Step 1: Clone the Repository

Ask the user to run this in their terminal *(Optional: Ask them to fork the repository first if they plan to contribute, and clone their fork instead)*:

```bash
# 本 fork 仓库 (kirk0830/AtomisticSkills) - 包含安全改进和 AstrBot 支持
git clone https://github.com/kirk0830/AtomisticSkills.git
cd AtomisticSkills
```

## Step 2: Install Pixi

If the user doesn't have Pixi installed:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

*(Remind them to restart their shell or run `source ~/.bashrc` / `source ~/.zshrc` after installation)*

## Step 3: Choose Environments

AtomisticSkills uses separate Pixi environments to manage conflicting MLIP and DFT dependencies. Each environment is isolated in `.pixi/envs/`.

Present using a list:

**MCP Server Environments (Interactive Tools):**
- [ ] **Base** (`base`): Materials Project queries, VASP I/O, base tools (Highly Recommended)
- [ ] **MACE** (`mace`): MACE models (MP, OMAT, MatPES)
- [ ] **MatGL** (`matgl`): MatGL models (CHGNet, M3GNet, TensorNet)
- [ ] **FairChem** (`fairchem`): FairChem models (UMA, ESEN)
- [ ] **Atomate2** (`atomate2`): Remote DFT job management via Jobflow/Jobflow-remote
- [ ] **Smol** (`smol`): Cluster expansion and Monte Carlo
- [ ] **DrugDisc** (`drugdisc`): Drug discovery tools (Fingerprints, Docking, ADMET)
- [ ] **MatterGen** (`mattergen`): Generative crystal design

**Script-Only Environments (No MCP Server):**
- [ ] **ORCA** (`orca`): Molecular DFT via ORCA (local or HPC)
- [ ] **React-OT** (`react-ot`): Transition state generation
- [ ] **SCD** (`scd`): Self-Conditioned Denoising property prediction
- [ ] **VOID** (`void`): MOF sorption docking
- [ ] **LAMMPS-MACE** (`lammps-mace`): LAMMPS with MACE backend
- [ ] **LAMMPS-MatGL** (`lammps-matgl`): LAMMPS with MatGL backend
- [ ] **LAMMPS-FairChem** (`lammps-fairchem`): LAMMPS with FairChem backend

Options: Keep it simple! Ask exactly which frameworks they want to use. "I only need MACE and basic tools" → `base` and `mace`.

## Step 4: Install Environments

Depending on their choices in Step 3, provide the commands:

```bash
# Install all environments (may take several minutes)
pixi install

# Or install specific environments only
pixi install -e base
pixi install -e mace
pixi install -e matgl
pixi install -e fairchem
# ... other selected environments
```

*(Remind the user this might take a few minutes. Wait for them to finish before proceeding)*

> [!NOTE]
> Pixi automatically installs the `atomistic-skills` Python package in all environments, so you can `from src.utils...` import from anywhere.

## Step 5: Configuration & API Keys

Many tools require API keys (like the Materials Project API) or binary paths. Have the user create a global configuration file in their home directory.

**Provide this template (`~/.atomistic_skills.yaml`):**

```yaml
# Materials Project API Key (Required for base-server)
MP_API_KEY: "your_mp_api_key_here"

# HPC Configuration (for Slurm job submission)
hpc:
  # Profile: generic, nersc_perlmutter, mit_supercloud, etc.
  profile: "generic"
  
  # Mode: auto, local, ssh
  mode: "auto"
  
  # SSH configuration (for ssh mode)
  ssh_host: null
  ssh_user: null
  ssh_key: null
  ssh_port: 22
  ssh_remote_work_dir: "~/hpc_jobs"
  
  # Application-specific modules (override profile defaults)
  modules:
    vasp: ["vasp/6.4.2-cpu"]
    orca: ["orca/5.0.4", "openmpi/4.1.5"]
    lammps: ["lammps/2024"]

# Required for running molecular DFT calculations with ORCA (local mode)
ORCA_BINARY_PATH: /path/to/orca_directory/orca

# Required for VASP POTCAR generation
PMG_VASP_PSP_DIR: /path/to/potcar_directory
```

*(Tell them they can also set these as environment variables like `export MP_API_KEY="key"`, but the file is persistent)*

## Step 6: Configure MCP Servers

The project provides an `mcp_config.json` that defines all tools. Run the configuration script to set up paths:

```bash
pixi run -e base python configure_mcp.py
```

*(This auto-detects Pixi environments. If it fails, they can pass the project root manually: `python configure_mcp.py /path/to/AtomisticSkills`)*

## Step 7: Add to AI Assistant

Finally, copy the patched MCP settings into their AI copilot's configuration file.

| Client | Project scope | Global scope (all projects) |
|--------|--------------|----------------------------|
| **Claude Code** | `.mcp.json` (project root) | `~/.claude.json` |
| **Codex CLI** | `.codex/config.toml` | `~/.codex/config.toml` |
| **Cursor** | `.cursor/mcp.json` | `~/.cursor/mcp.json` |
| **Windsurf** | — | `~/.codeium/windsurf/mcp_config.json` |
| **Gemini CLI** | `.gemini/settings.json` | `~/.gemini/settings.json` |

The `configure_mcp.py` script handles path substitution automatically:

```bash
# Project scope only (default) — tools available when inside this repo
pixi run -e base python configure_mcp.py

# Global scope — tools available in every project/session
pixi run -e base python configure_mcp.py --scope global

# Both scopes
pixi run -e base python configure_mcp.py --scope both
```

Restart the assistant after any config changes.

## What's Next? (Guided First Use)

**Run a live test WITH the user.**

**Demo Query (Base agent):** "Search the Materials Project for the stable structure of LiFePO4."
*(Use the `search_materials_project_by_formula` tool)*

If they set up a MLIP (e.g., MACE): "Predict the forces and energy for this LiFePO4 structure using the MACE model."

## Best Practices for Users

- **Leverage Local GPUs**: We highly recommend running the framework on a machine with local GPU resources so MLIP tasks can evaluate quickly without external compute costs.
- **Use HPC for DFT**: Submit ORCA/VASP calculations to HPC clusters via the unified module (see [HPC Job Submission docs](hpc_job_submission.md)).
- **Chatbot frameworks**: To expose AtomisticSkills through a chat platform (QQ, Telegram, etc.), see the [AstrBot Integration Guide](astrbot-integration.md). AstrBot sandboxes the agent to its `data/` directory, so we use per-skill symlinks + MCP server configs.
- **Find the right tools**: Use the [Skill / MCP Tool / Workflow Map](skill_mcp_workflow_map.md) to understand the three-layer hierarchy and find the right tools for your task.
- **Customize**: Add your own specialized SKILLs, MCP tools, and Workflows directly to the project structure to tailor it to your research needs.
- **Contribute Back**: If you develop a robust, generalized tool or SKILL, please submit a PR to the main branch! We actively acknowledge all open-source contributors.

## Common Issues

| Issue | Fix |
|-------|-----|
| MCP Tools not showing up | Verify JSON syntax in the copilot's config file and restart the IDE/copilot. |
| `pixi install` fails | Check network connection. Try `pixi install --frozen` for reproducible install. |
| Tool execution failed / Python not found | Run `pixi run -e base python configure_mcp.py` to update paths. |
| Atomate2 remote worker issues | See `conda-envs/atomate2-agent/atomate2_remote_worker_setup.md` |
| HPC submission fails | Verify SSH key exists, check `HPC_MODE` and `HPC_SSH_*` env vars. |
| Import errors in scripts | Run `pixi install` to ensure `atomistic-skills` package is installed. |