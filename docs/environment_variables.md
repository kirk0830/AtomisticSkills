# Environment Variables Guide

This document lists the environment variables used by AtomisticSkills MCP tools and skills.
Many values can be set once in `~/.atomistic_skills.yaml` and are injected into the
environment automatically when MCP servers start. For step-by-step acquisition
instructions, see [`docs/api_key_guide.md`](./api_key_guide.md).

## Core Variables

| Variable | Required? | Description | Used By | How to obtain |
| :--- | :--- | :--- | :--- | :--- |
| `CURRENT_RESEARCH_DIR` | Auto-set | Tracks the active research directory for the current session. | All MCP Tools | Set automatically by tools. |

## Literature & Downloads

| Variable | Required? | Description | Used By | How to obtain |
| :--- | :--- | :--- | :--- | :--- |
| `ELSEVIER_API_KEY` | **Required** for Elsevier downloads | API key for the ScienceDirect / Elsevier API. | `src/utils/paper_downloader.py`, `search_literature` | https://dev.elsevier.com/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `ELSEVIER_INST_TOKEN` | Optional | Institution token for Elsevier institutional access. | `src/utils/paper_downloader.py` | https://dev.elsevier.com/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `SPRINGER_API_KEY` | **Required** for Springer downloads | API key for Springer Meta / TDM API. | `src/utils/paper_downloader.py`, `search_literature` | https://dev.springernature.com/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `UNPAYWALL_EMAIL` | **Required** for Unpaywall | Email address used for the Unpaywall polite pool. | `src/utils/paper_downloader.py`, `search_literature` | https://unpaywall.org/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `OPENALEX_EMAIL` | Recommended | Email for the OpenAlex polite pool; also used as a fallback for Unpaywall. | `src/utils/literature_utils.py`, `src/utils/paper_downloader.py` | https://openalex.org/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |

## Materials & Chemistry APIs

| Variable | Required? | Description | Used By | How to obtain |
| :--- | :--- | :--- | :--- | :--- |
| `MP_API_KEY` | **Required** for Materials Project | API key for querying Materials Project structures and properties. | `src/mcp_server/base_server.py`, `src/utils/structure_utils.py`, MOF/QMOF skills, many materials skills | https://next-gen.materialsproject.org/api — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `RXN_API_KEY` | **Required** for IBM RXN | API key for IBM RXN for Chemistry retrosynthesis / reaction prediction. | `drug-retrosynthesis` skill | https://rxn.res.ibm.com/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `HF_TOKEN` | **Required** for gated HuggingFace resources | HuggingFace access token for gated models or datasets. | `chem-nmr-analysis` skill | https://huggingface.co/settings/tokens — see [`docs/api_key_guide.md`](./api_key_guide.md) |

## VLM / Plot Digitization

| Variable | Required? | Description | Used By | How to obtain |
| :--- | :--- | :--- | :--- | :--- |
| `GOOGLE_API_KEY` | **Required** for Gemini | API key for Google Gemini vision-language models. | `general-plot-digitizer` skill | https://ai.google.dev/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `OPENAI_API_KEY` | **Required** for OpenAI | API key for OpenAI GPT-4V / GPT-4o vision models. | `general-plot-digitizer` skill | https://platform.openai.com/ — see [`docs/api_key_guide.md`](./api_key_guide.md) |
| `GEMINI_MODEL` | Optional | Override the default Gemini model ID. | `general-plot-digitizer` skill | Set to a valid model ID, e.g. `gemini-2.5-flash-lite`. |

## DFT Engines

| Variable | Required? | Description | Used By | Example |
| :--- | :--- | :--- | :--- | :--- |
| `ORCA_BINARY_PATH` | **Required** for local ORCA | Absolute path to the ORCA executable. ORCA is not available on conda-forge and must be installed separately. | `src/utils/dft/orca_utils.py`, ORCA skills | `/opt/orca/5.0.4/orca` |
| `VASP_CMD` or `ATOMATE2_VASP_CMD` | Local VASP / Atomate2 **Required** | Command to run VASP (legacy periodic DFT; CP2K/QE are the recommended open-source alternatives). | `atomate2`, `src/utils/dft/vasp_hpc.py` | `mpirun -np 16 vasp_std` |
| `PMG_VASP_PSP_DIR` | Local VASP POTCAR **Required** | Path to the VASP POTCAR directory. | `pymatgen`, `atomate2` | `/path/to/vasp/potcar` |
| `ATOMATE2_CONFIG_FILE` | Optional | Path to the Atomate2 configuration file. | `atomate2` | `~/.config/atomate2/config.yaml` |
| `atomate2_remote_project` (or `ATOMATE2_REMOTE_PROJECT`) | Remote Atomate2 **Recommended** | Default project name for remote job submission (auto-detected when only one project exists). | `atomate2` | `remote_perlmutter` |
| `DFT_ENGINE` | Optional (default `cp2k`) | Default periodic DFT engine for skills that support multiple backends: `cp2k`, `qe`, or `vasp`. | ASE-QE/CP2K skills, `src/utils/dft/*` | `cp2k` |
| `ESPRESSO_PSEUDO` | **Required** for QE | Directory containing Quantum ESPRESSO pseudopotential files (UPF format). | ASE ESPRESSO calculator, QE skills | `/path/to/pseudo` |
| `ASE_ESPRESSO_COMMAND` or `ESPRESSO_COMMAND` | Local QE **Required** | Shell command to run `pw.x` (ASE checks `ASE_ESPRESSO_COMMAND` first, then `ESPRESSO_COMMAND`). | ASE ESPRESSO calculator, QE skills | `mpirun -np 16 pw.x` |
| `CP2K_DATA_DIR` | **Required** for CP2K | Directory containing CP2K basis sets and pseudopotentials (e.g. `BASIS_SET`, `POTENTIAL`). | ASE CP2K calculator, CP2K skills | `/path/to/cp2k/data` |
| `ASE_CP2K_COMMAND` or `CP2K_COMMAND` | Local CP2K **Required** | Shell command to run `cp2k.popt`/`cp2k.psmp` (ASE checks `ASE_CP2K_COMMAND` first, then `CP2K_COMMAND`). | ASE CP2K calculator, CP2K skills | `mpirun -np 16 cp2k.popt` |

> **CP2K / Quantum ESPRESSO**: These open-source periodic DFT engines are installed from conda-forge via the `cp2k` and `qe` environments. They require the pseudopotential and binary-path variables listed above for local execution; on HPC these are usually supplied via modules and `CP2K_DATA_DIR` / `ESPRESSO_PSEUDO`. See the [DFT migration report](../.trae/documents/dft_migration_report.md) for details.

> **VASP deprecation note**: VASP support is retained as a legacy path. New periodic DFT workflows should use **CP2K** or **Quantum ESPRESSO**, except for atomate2 workflows (dielectric response, ferroelectric, electronic transport) that are not yet available for QE or CP2K.

## HPC Execution Environment

| Variable | Required? | Description | Used By | How to obtain |
| :--- | :--- | :--- | :--- | :--- |
| `HPC_MODE` | Optional (default `auto`) | HPC backend mode: `local`, `ssh`, or `auto`. | `src/utils/hpc/*`, `src/utils/dft/orca_hpc.py`, `src/utils/dft/vasp_hpc.py` | Set to the desired mode. |
| `HPC_PROFILE` | Optional (default `generic`) | Built-in HPC profile name (e.g. `nersc_perlmutter`, `mit_supercloud`). | `src/utils/hpc/config_loader.py` | Set to a profile name. |
| `HPC_SSH_HOST` | **Required** in SSH mode | Remote HPC login node hostname. | `src/utils/hpc/config_loader.py`, `src/utils/hpc/job_manager.py` | Your cluster login node, e.g. `cluster.university.edu`. |
| `HPC_SSH_USER` | **Required** in SSH mode | Remote SSH username. | `src/utils/hpc/config_loader.py`, `src/utils/hpc/job_manager.py` | Your cluster username. |
| `HPC_SSH_KEY` | **Required** in SSH mode | Path to the SSH private key used for authentication. | `src/utils/hpc/config_loader.py`, `src/utils/hpc/job_manager.py` | Path to your key, e.g. `~/.ssh/id_ed25519`. |
| `HPC_SSH_PORT` | Optional (default `22`) | SSH port for remote HPC connection. | `src/utils/hpc/config_loader.py`, `src/utils/hpc/job_manager.py` | Port number. |
| `HPC_REMOTE_WORK_DIR` | Optional (default `~/hpc_jobs`) | Working directory on the remote HPC. | `src/utils/hpc/job_manager.py` | Remote path. |
| `HPC_WORK_DIR` | Optional | Working directory for local Slurm submissions. | `src/utils/hpc/job_manager.py` | Local path. |
| `HPC_MODULES_<APP>` | Optional | Application-specific module list, e.g. `HPC_MODULES_VASP`, `HPC_MODULES_ORCA`, `HPC_MODULES_CP2K`, `HPC_MODULES_QE`. | `src/utils/hpc/config_loader.py` | Comma-separated module names, e.g. `vasp/6.4.2-cpu,intel/2023`. |

See [`docs/hpc_job_submission.md`](./hpc_job_submission.md) for full HPC configuration details.

## Logging / Optional

| Variable | Required? | Description | Used By | How to obtain |
| :--- | :--- | :--- | :--- | :--- |
| `WANDB_MODE` | Optional (default `online`) | Weights & Biases mode: `online`, `offline`, or `disabled`. | `src/utils/wandb_utils.py`, MatGL wrappers | Set as needed. |
| `WANDB_PROJECT` | Optional (default `base-agent`) | Weights & Biases project name. | `src/utils/wandb_utils.py` | Set your project name. |

## Configuration Files

Instead of setting variables manually, it is recommended to use standard configuration files where possible:

- **AtomisticSkills**: `~/.atomistic_skills.yaml` or `~/.config/atomistic_skills.yaml`
    - This is the main configuration for the agent. Any key defined here will be injected into the environment.
    - Example:
      ```yaml
      ATOMATE2_REMOTE_PROJECT: remote_perlmutter
      MP_API_KEY: abc123def456
      ```
- **Pymatgen**: `~/.pmgrc.yaml` (Manages `PMG_VASP_PSP_DIR`, `MP_API_KEY`)
- **JobFlow Remote**: `~/.jfremote/` (Manages remote clusters and projects like `remote_perlmutter`)
- **Atomate2**: `~/.config/atomate2/config.yaml`

The `.env` file in the project root can be used for project-specific overrides or for variables not covered by the above (like `CURRENT_RESEARCH_DIR`).
