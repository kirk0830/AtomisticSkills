# AtomisticSkills: Claude Code Instructions

## Framework Overview

This project decomposes complex research tasks into three levels:

- **Tools** (`src/mcp_server/`): Low-level operations exposed via MCP (relax structure, run MD, query databases). Strict typed I/O.
- **Skills** (`.agent/skills/`): Mid-level tutorials combining tools and scripts to solve focused tasks (calculate melting point, run docking). Each has a `SKILL.md` with step-by-step instructions.
- **Workflows** (`.agent/workflows/`): High-level research campaigns that chain multiple skills (e.g., screen for stable Li-ion conductors).

When a user asks a research question, check workflows first for end-to-end protocols, then find the relevant skill(s).

## Skill Discovery

Each skill subdirectory contains a `SKILL.md` with YAML frontmatter (`name`, `description`, `category`) and numbered instructions.

**To find a relevant skill**, scan the frontmatter descriptions:
```bash
grep -r "^description:" .agent/skills/*/SKILL.md
```

Then read the full `SKILL.md` for any matching skill and follow its numbered instructions.

Use the `/skill-search` command for interactive discovery: `/skill-search [search term]`

## Executing Skills

### Scripts with `# Env:` annotations
Many skills reference helper scripts with environment annotations like:
```bash
# Env: mace-agent
python .agent/skills/mat-melting-point/scripts/create_interface.py ...
```

Activate the specified conda environment before running:
```bash
mamba activate <env-name>
```

### MCP tool calls
Skills that reference `mcp_*` functions (e.g., `mcp_mace_run_md()`) require MCP servers to be configured. If MCP servers are not available, check whether the underlying operation can be performed directly via scripts in the skill's `scripts/` directory or by importing from `src/utils/`.

## Environment Mapping

| Conda Environment | Purpose |
|---|---|
| `base-agent` | Pymatgen, ASE, structure tools, Materials Project API |
| `mace-agent` | MACE models (MP, OMAT, MATPES, Multi-Head, OFF) |
| `matgl-agent` | MatGL models (CHGNet, M3GNet, TensorNet) |
| `fairchem-agent` | FairChem models (UMA, ESEN) |
| `atomate2-agent` | Remote DFT workflows via Jobflow-remote |
| `smol-agent` | Cluster expansion and Monte Carlo |
| `drugdisc-agent` | RDKit, AutoDock Vina, PDBFixer, Meeko |
| `mattergen-agent` | MatterGen generative crystal design |

## Project Rules

Development standards are documented in `.agent/rules/`:
- `skill-standards.md`: how to create and structure new skills
- `coding-standards.md`: code style, testing, and dependencies
- `mcp-environments.md`: environment-to-server mapping
- `research-standards.md`: research workflow conventions

## MCP Server Setup

Skills that call `mcp_*` functions need MCP servers configured. You only need the servers for the skills you plan to use.

1. Install the conda environment(s) you need (see `conda-envs/*/install.sh`)
2. Open `mcp_config.json` for the server definitions
3. Copy the entries you need into `~/.claude.json` under `"mcpServers"`
4. Update the `command` path to point to your conda env's Python, e.g.:
   ```
   "/Users/you/miniforge3/envs/mace-agent/bin/python"
   ```
5. Update the `PYTHONPATH` in `env` to your local clone path
6. Restart Claude Code

See `README.md` for full installation instructions.
