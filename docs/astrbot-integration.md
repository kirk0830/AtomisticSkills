# AstrBot Integration Guide

[AstrBot](https://docs.astrbot.app/) is an open-source chatbot framework that hosts an LLM agent and exposes it through messaging platforms (QQ, Telegram, etc.). This guide explains how to wire AtomisticSkills into an AstrBot instance so the agent can run atomistic research tasks via chat.

Unlike IDE-embedded agents (Claude Code, Cursor, etc.), AstrBot **restricts the agent's filesystem access to its own `data/` directory**. AtomisticSkills cannot be loaded directly from its repository path. This guide implements the **MCP-first + symlink** strategy to bridge that gap.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       AstrBot Chatbot                            │
│  ┌────────────────────┐    ┌─────────────────────────────────┐  │
│  │   LLM Agent        │    │  MCP Client (stdio / http)      │  │
│  │  (sandboxed to     │    │                                 │  │
│  │   data/ only)      │    │  Connects to:                   │  │
│  │                    │    │   • base  • mace  • matgl       │  │
│  │  Reads:            │    │   • fairchem • atomate2         │  │
│  │   data/skills/*    │    │   • smol • drugdisc ...         │  │
│  │   (symlinks → ────────────► AtomisticSkills MCP servers    │  │
│  │    .agents/skills/)│    │                                 │  │
│  └────────────────────┘    └─────────────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │ MCP protocol (stdio)
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                  AtomisticSkills (host project)                  │
│                                                                  │
│  .agents/skills/        ←── real skill files (read by symlinks)  │
│  src/mcp_server/*.py    ←── MCP server entry points              │
│  src/utils/             ←── shared utilities (HPC, DFT, MLIP)    │
│  .pixi/envs/<name>/     ←── isolated Python envs per MCP server  │
│  ~/.atomistic_skills.yaml ←── API keys + HPC SSH config          │
└──────────────────────────────────────────────────────────────────┘
```

**Why this works**:

1. **Skills as symlinks** — AstrBot scans `data/skills/*/SKILL.md`. We symlink each project skill into that directory, so the agent sees a normal skill tree.
2. **MCP servers run on the host** — MCP servers are launched by AstrBot as subprocesses; they are not sandboxed. Each server runs in its dedicated Pixi env and can read/write anywhere on the host.
3. **Editable Python package** — The `atomistic-skills` package is installed editable in every Pixi env (declared in `pixi.toml`), so MCP servers can `from src.utils...` import without `sys.path` hacks.

---

## Prerequisites

- AtomisticSkills cloned and at least the `base` Pixi environment installed (see [docs/setup.md](setup.md))
- AstrBot v3.5.0+ (MCP support required)
- `uv` installed (`pip install uv`) — AstrBot uses this to launch MCP servers if you choose the `uv` form
- (Optional) `node` + `npm` — only needed if you want to add non-Python MCP servers later

---

## Step 1: Install Pixi Environments

Make sure the Pixi environments you plan to use are installed:

```bash
cd /path/to/AtomisticSkills
pixi install -e base
pixi install -e mace        # optional
pixi install -e matgl       # optional
pixi install -e fairchem    # optional
# ... etc.
```

The MCP server configs generated below point at `.pixi/envs/<name>/bin/python`, so each env you want to expose must exist.

---

## Step 2: Configure API Keys and HPC

Create or edit `~/.atomistic_skills.yaml`:

```yaml
# Materials Project API Key (required for base MCP server)
MP_API_KEY: "your_mp_api_key_here"

# HPC Configuration (for ORCA/VASP via Slurm)
hpc:
  profile: "nersc_perlmutter"     # or "generic", "mit_supercloud", etc.
  mode: "auto"                     # "local" (login node sbatch) or "ssh"

  # SSH mode (if AstrBot runs off-cluster and submits via SSH)
  ssh_host: "cluster.university.edu"
  ssh_user: "your_username"
  ssh_key: "~/.ssh/id_ed25519"

  # Application modules to load on the cluster
  modules:
    vasp: ["vasp/6.4.2-cpu"]
    orca: ["orca/5.0.4", "openmpi/4.1.5"]

# Local binary paths (only if running DFT locally, not via HPC)
ORCA_BINARY_PATH: /path/to/orca
PMG_VASP_PSP_DIR: /path/to/pseudopotentials
```

See [docs/hpc_job_submission.md](hpc_job_submission.md) for full HPC details.

---

## Step 3: Run `configure_astrbot.py`

From the AtomisticSkills repository root:

```bash
# Auto-detect AstrBot data dir (looks in ./data, ../astrbot/data, ~/astrbot/data)
pixi run -e base python configure_astrbot.py

# Or specify the AstrBot data directory explicitly
pixi run -e base python configure_astrbot.py --data-dir /path/to/astrbot/data

# Also save a reference MCP config JSON to <data-dir>/config/
pixi run -e base python configure_astrbot.py --data-dir /path/to/astrbot/data --write-mcp-config
```

This will:

1. **Create symlinks** for each project skill:
   - `<astrbot>/data/skills/<skill_name>` → `<AtomisticSkills>/.agents/skills/<skill_name>`
2. **Write an index skill** at `<astrbot>/data/skills/atomisticskills/SKILL.md` explaining the framework to the agent.
3. **Print MCP server JSON configs** — one block per server, ready to paste into AstrBot's WebUI.

Sample output:

```
[skills]
  Linked 127 new skills, refreshed 0, skipped 0, removed 0 stale symlinks
  Index SKILL.md → /path/to/astrbot/data/skills/atomisticskills/SKILL.md

[mcp]

=== AstrBot MCP Server Configs ===
Paste each block into AstrBot WebUI → MCP → Add MCP Server.

--- base ---
{
  "command": "/abs/path/to/AtomisticSkills/.pixi/envs/base/bin/python",
  "args": ["-m", "src.mcp_server.base_server"],
  "env": {
    "PYTHONPATH": "/abs/path/to/AtomisticSkills"
  }
}

--- mace ---
...
```

> [!TIP]
> Re-run this script any time you pull new commits to AtomisticSkills — it refreshes symlinks idempotently and removes stale links for deleted skills.

---

## Step 4: Add MCP Servers in AstrBot WebUI

Open AstrBot's WebUI (typically `http://localhost:6185`) and navigate to **MCP → Add MCP Server**.

For each server printed by `configure_astrbot.py`:

1. Click **Add MCP Server**
2. Paste the JSON block (e.g. the `--- base ---` block) into the config field
3. Save

### If AstrBot rejects absolute Python paths

AstrBot's stdio launcher restricts the `command` field to `python`, `node`, or `uv` for security. If your build rejects absolute paths like `/abs/path/to/.pixi/envs/base/bin/python`, use the `env` + `uv` form instead:

```json
{
  "command": "env",
  "args": [
    "PYTHONPATH=/abs/path/to/AtomisticSkills",
    "uv",
    "run",
    "--python",
    "/abs/path/to/AtomisticSkills/.pixi/envs/base/bin/python",
    "python",
    "-m",
    "src.mcp_server.base_server"
  ]
}
```

This launches the server via `uv run` with the correct interpreter and the `PYTHONPATH` env var.

---

## Step 5: Restart AstrBot

After adding MCP servers, restart AstrBot so it picks up the new tools:

```bash
# If running from source
Ctrl+C in the AstrBot terminal, then restart

# If running in Docker
docker restart astrbot
```

Verify the MCP servers connected by checking the WebUI's MCP panel — each server should show a green status.

---

## Step 6: Verify with a Test Query

In your messaging platform (QQ, Telegram, etc.), message the bot:

```
What MCP tools are available?
```

The agent should list tools from the connected AtomisticSkills servers (e.g. `search_materials_project`, `predict_energy`, `relax_structure`).

Try a real query:

```
Search the Materials Project for the stable structure of LiFePO4 and report its bandgap.
```

The agent will:
1. Read the `atomisticskills` index SKILL.md to understand the framework
2. Call the `search_materials_project_by_formula` MCP tool from the `base` server
3. Return the result as a chat message

---

## How Skills Work from AstrBot

When the agent receives a research query, it follows this flow:

1. **Skill discovery** — AstrBot scans `data/skills/*/SKILL.md` (which are symlinks to the real skills).
2. **Skill selection** — the agent reads the matching `SKILL.md` to understand the task.
3. **Tool invocation** — the agent calls MCP tools (registered separately in WebUI) to perform the actual computation. The MCP server runs in its Pixi env on the host.
4. **Heavy compute (DFT, MD)** — for ORCA, VASP, or long-running MD, the skill's `SKILL.md` instructs the agent to use the **HPC submission mode**. The MCP server (e.g. `base` or `atomate2`) submits a Slurm job via `src.utils.hpc` and returns the job ID immediately, so the chatbot doesn't block on multi-hour jobs.

> [!IMPORTANT]
> The agent inside AstrBot's sandbox **cannot run shell commands** like `pixi run` or `conda activate`. All heavy lifting must go through MCP tools. This is by design — the MCP servers handle environment isolation correctly.

---

## File Access and Outputs

AstrBot restricts the agent to `data/` only. This affects where research outputs land:

- **MCP tool outputs**: when an MCP tool writes a file (e.g. a relaxed structure), it writes to whatever path the tool was given. By default this is the project's `research/<date>_<topic>/` directory on the host, **outside** the sandbox.
- **Sharing files in chat**: to send a file back to the user via chat, the MCP server must write to a path inside `data/` (e.g. `<astrbot>/data/research_outputs/`). Configure this in `~/.atomistic_skills.yaml` or pass the output path explicitly in the tool call.

Example — using the `base` MCP server's `create_research_dir` tool:

```python
# The agent calls this MCP tool:
create_research_dir(research_topic="lifepo4_stability")
# Returns: /abs/path/to/AtomisticSkills/research/2026-07-01_lifepo4_stability/

# To make files visible to the chatbot, write to data/ instead:
create_research_dir(
    research_topic="lifepo4_stability",
    base_dir="/path/to/astrbot/data/research_outputs"
)
```

---

## HPC Submission from AstrBot

For DFT (ORCA, VASP) or large-scale MD, the chatbot should not block on local execution. Use the HPC submission mode:

1. **Configure SSH in `~/.atomistic_skills.yaml`** (see Step 2 above).
2. The agent reads the relevant `SKILL.md` (e.g. `chem-dft-orca-singlepoint`), which instructs it to ask the user for HPC parameters (partition, cores, wall time).
3. The agent calls the MCP tool (e.g. `orca_run_singlepoint` with `mode="hpc"`) — the server uses `src.utils.hpc.JobManager` to submit a Slurm job.
4. The server returns the job ID immediately; the agent tells the user the job is queued.
5. The user can later ask the bot to check job status (`check_job_status(job_id=...)`).

> [!WARNING]
> SSH keys are read from disk only — never hardcoded. Make sure the AstrBot service account has read access to the SSH key file specified in `~/.atomistic_skills.yaml`.

---

## Updating Skills After Pulling New Commits

When you `git pull` new commits in AtomisticSkills:

```bash
cd /path/to/AtomisticSkills
git pull
pixi install   # in case new deps were added

# Refresh symlinks (also removes stale ones for deleted skills)
pixi run -e base python configure_astrbot.py --data-dir /path/to/astrbot/data --skills-only
```

Restart AstrBot if any MCP server code changed.

---

## Troubleshooting

### MCP server fails to start
- Check that the Pixi env exists: `ls .pixi/envs/<name>/bin/python`
- Run the server manually to see the error:
  ```bash
  /path/to/AtomisticSkills/.pixi/envs/base/bin/python -m src.mcp_server.base_server
  ```
- Verify `PYTHONPATH` is set to the project root in the MCP config.

### Skills not showing up in AstrBot
- Verify symlinks exist: `ls -la /path/to/astrbot/data/skills/`
- Each skill should have a `SKILL.md` reachable through the symlink.
- Re-run `configure_astrbot.py --skills-only` to refresh.

### AstrBot says "command not allowed"
- AstrBot's stdio mode restricts `command` to `python`/`node`/`uv`. Use the `env` + `uv` form in Step 4.

### HPC SSH fails
- Verify the SSH key path in `~/.atomistic_skills.yaml` is readable by the AstrBot service account.
- Test SSH manually: `ssh -i ~/.ssh/id_ed25519 user@cluster.university.edu "sinfo"`
- See [docs/hpc_job_submission.md](hpc_job_submission.md) for full HPC troubleshooting.

### Agent can't write files to share in chat
- AstrBot restricts the agent to `data/`. Configure MCP tool calls to write outputs to `<astrbot>/data/research_outputs/` (see "File Access and Outputs" above).

---

## Quick Reference

| Task | Command |
|------|---------|
| Configure AstrBot (skills + MCP) | `pixi run -e base python configure_astrbot.py --data-dir <astrbot>/data` |
| Configure skills only | `pixi run -e base python configure_astrbot.py --data-dir <astrbot>/data --skills-only` |
| Print MCP configs only | `pixi run -e base python configure_astrbot.py --mcp-only` |
| Save MCP reference JSON | `pixi run -e base python configure_astrbot.py --data-dir <astrbot>/data --write-mcp-config` |
| List available MCP servers | `pixi run -e base python configure_astrbot.py --list-servers` |
| Refresh skills after `git pull` | `pixi run -e base python configure_astrbot.py --data-dir <astrbot>/data --skills-only` |

---

## Related Documentation

- [Setup Guide](setup.md) — initial AtomisticSkills install
- [HPC Job Submission](hpc_job_submission.md) — Slurm SSH/local mode details
- [Developer Guide](developer_guide.md) — project architecture
- [OpenClaw Integration](openclaw-integration.md) — alternative chatbot framework
- [AstrBot MCP Docs](https://docs.astrbot.app/en/use/mcp.html) — official AstrBot MCP reference
