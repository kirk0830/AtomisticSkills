# Skill / MCP Tool / Workflow Map

This document provides a cross-reference map of the three abstraction layers in AtomisticSkills:

1. **Workflows** — high-level research campaigns (`.agents/workflows/`)
2. **Skills** — mid-level task tutorials (`.agents/skills/`)
3. **MCP Tools** — low-level primitives exposed via MCP servers (`src/mcp_server/`)

Use this map to trace the execution path from a user query down to the tools that actually do the work, or to find which skills/workflows use a given MCP tool.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Agent Intent Classification (research-standards.md) │
└───────────────────────┬─────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  Direct Query    Research Task    Literature Review
  (MCP tools)     (Skills + MCP)    (MCP + synthesis)
                        │
                        ▼
              ┌─────────────────────┐
              │ Workflows (optional) │  ── composes multiple skills
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ Skills (127 total)   │  ── orchestrates MCP tool calls
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ MCP Servers (10)     │  ── 49+ tools, isolated envs
              └─────────────────────┘
```

---

## MCP Servers & Tools

| Server | Env | Tools | Description |
|--------|-----|-------|-------------|
| **base** | `base` | `create_research_dir`, `search_materials_project_by_formula`, `search_materials_project_by_chemsys`, `visualize_structure`, `search_literature`, `supercell_expansion`, `modify_structure`, `search_model_registry`, `register_model` | Core utilities, MP queries, literature |
| **mace** | `mace` | `load_model`, `predict_structure`, `predict_atomic_features`, `relax_structure`, `run_md`, `get_info` | MACE MLIP (energy, forces, relax, MD) |
| **matgl** | `matgl` | `load_model`, `predict_structure`, `predict_atomic_features`, `predict_bandgap`, `relax_structure`, `run_md`, `get_info` | MatGL MLIP (CHGNet, M3GNet, TensorNet) + bandgap |
| **fairchem** | `fairchem` | `load_model`, `predict_structure`, `relax_structure`, `run_md`, `get_info` | FairChem MLIP (UMA, ESEN) |
| **atomate2** | `atomate2` | `run_atomate2_vasp_calculation`, `get_atomate2_results_by_id`, `get_atomate2_results_by_formula`, `get_atomate2_summary`, `get_atomate2_recent_jobs`, `get_atomate2_job_status`, `get_atomate2_project_status` | VASP DFT workflows via Atomate2 |
| **smol** | `smol` | `sample_ordered_structures`, `train_cluster_expansion`, `run_monte_carlo`, `compute_feature_vectors`, `get_feature_matrix`, `fit_feature_matrix`, `check_mapping` | Cluster expansion + Monte Carlo |
| **drugdisc** | `drugdisc` | `parse_smiles_input`, `standardize_molecule`, `convert_to_pdbqt`, `compute_molecular_descriptors`, `compute_molecular_fingerprints` | Drug discovery utilities |
| **mattergen** | `mattergen` | `generate_structures` | MatterGen generative crystal design |
| **diffcsp** | `diffcsp` | `generate_structures_with_symmetry` | DiffCSP++ structure generation |
| **adit** | `adit` | `generate_structures` | ADiT all-atom diffusion transformer |

---

## Skills by Category → Primary MCP Tools

### Materials Science (`mat-*`, 43 skills)

| Skill | Primary MCP Tools | Related Skills |
|-------|-------------------|----------------|
| `mat-stability` | `base.search_materials_project_by_*`, MLIP `relax_structure` | [mat-db-mp](../.agents/skills/mat-db-mp/SKILL.md), [ml-foundation-potentials](../.agents/skills/ml-foundation-potentials/SKILL.md), [mat-electrochemical-window](../.agents/skills/mat-electrochemical-window/SKILL.md) |
| `mat-diffusion-analysis` | MLIP `run_md` | [mat-md-monitors](../.agents/skills/mat-md-monitors/SKILL.md), [ml-foundation-potentials](../.agents/skills/ml-foundation-potentials/SKILL.md) |
| `mat-phonon` | MLIP `predict_structure` (for forces), `base.visualize_structure` | [mat-qha-thermal-expansion](../.agents/skills/mat-qha-thermal-expansion/SKILL.md) |
| `mat-elasticity` | MLIP `relax_structure`, `predict_structure` | [mat-equation-of-state](../.agents/skills/mat-equation-of-state/SKILL.md) |
| `mat-surface-energy` | MLIP `relax_structure` | [mat-surface-adsorption](../.agents/skills/mat-surface-adsorption/SKILL.md) |
| `mat-melting-point` | MLIP `run_md` | [mat-md-monitors](../.agents/skills/mat-md-monitors/SKILL.md) |
| `mat-db-mp` | `base.search_materials_project_by_*` | [mat-db-optimade](../.agents/skills/mat-db-optimade/SKILL.md) |
| `mat-dft-vasp` | `atomate2.run_atomate2_vasp_calculation`, `base.*` | [mat-dft-mixing-functionals](../.agents/skills/mat-dft-mixing-functionals/SKILL.md), [mat-dft-electronic-structure](../.agents/skills/mat-electronic-structure/SKILL.md) |
| `mat-xrd-calculator` | `base.*` (pymatgen) | [mat-xrd-phase-analysis](../.agents/skills/mat-xrd-phase-analysis/SKILL.md), [mat-xrd-refinement](../.agents/skills/mat-xrd-refinement/SKILL.md) |
| `mat-phase-diagram` | `base.search_materials_project_by_*` | [mat-stability](../.agents/skills/mat-stability/SKILL.md), [mat-pourbaix-diagram](../.agents/skills/mat-pourbaix-diagram/SKILL.md) |

### Chemistry (`chem-*`, 28 skills)

| Skill | Primary MCP Tools | Related Skills |
|-------|-------------------|----------------|
| `chem-dft-orca-singlepoint` | (scripts only — `orca` env) | [chem-dft-orca-optimization](../.agents/skills/chem-dft-orca-optimization/SKILL.md), [chem-thermochemistry](../.agents/skills/chem-thermochemistry/SKILL.md) |
| `chem-thermochemistry` | MLIP `predict_structure` (Hessian), `chem-vibration` | [chem-vibration](../.agents/skills/chem-vibration/SKILL.md) |
| `chem-vibration` | MLIP `predict_structure` (Hessian) | [chem-thermochemistry](../.agents/skills/chem-thermochemistry/SKILL.md) |
| `chem-neb-barrier` | MLIP `relax_structure`, `run_md` | [chem-ts-optimization](../.agents/skills/chem-ts-optimization/SKILL.md), [chem-irc-verification](../.agents/skills/chem-irc-verification/SKILL.md) |
| `chem-sorption-gcmc` | MLIP `predict_structure` (energy) | [chem-sorption-widom](../.agents/skills/chem-sorption-widom/SKILL.md), [chem-sorption-relax](../.agents/skills/chem-sorption-relax/SKILL.md) |
| `chem-docking-void` | (scripts only — `void` env) | [drug-docking-vina](../.agents/skills/drug-docking-vina/SKILL.md) |
| `chem-nmr-predict` | (scripts only — `nmr` env) | [chem-nmr-analysis](../.agents/skills/chem-nmr-analysis/SKILL.md) |
| `chem-msms-predict` | (scripts only — `msms` env) | [chem-spectrum-matcher](../.agents/skills/chem-spectrum-matcher/SKILL.md) |

### Machine Learning (`ml-*`, 18 skills)

| Skill | Primary MCP Tools | Related Skills |
|-------|-------------------|----------------|
| `ml-mace-finetune` | `mace.*` | [ml-mlip-benchmark](../.agents/skills/ml-mlip-benchmark/SKILL.md), [ml-foundation-potentials](../.agents/skills/ml-foundation-potentials/SKILL.md) |
| `ml-matgl-finetune` | `matgl.*` | [ml-mlip-benchmark](../.agents/skills/ml-mlip-benchmark/SKILL.md) |
| `ml-fairchem-finetune` | `fairchem.*` | [ml-mlip-benchmark](../.agents/skills/ml-mlip-benchmark/SKILL.md) |
| `ml-mlip-benchmark` | `mace.*`, `matgl.*`, `fairchem.*` | [ml-foundation-potentials](../.agents/skills/ml-foundation-potentials/SKILL.md) |
| `ml-foundation-potentials` | `mace.*`, `matgl.*`, `fairchem.*` | [ml-mlip-benchmark](../.agents/skills/ml-mlip-benchmark/SKILL.md), [mat-dft-mixing-functionals](../.agents/skills/mat-dft-mixing-functionals/SKILL.md) |
| `ml-generative-mattergen` | `mattergen.generate_structures` | [mat-stability](../.agents/skills/mat-stability/SKILL.md) |
| `ml-generative-diffcsp` | `diffcsp.generate_structures_with_symmetry` | [mat-stability](../.agents/skills/mat-stability/SKILL.md) |
| `ml-cluster-expansion` | `smol.*` | [mat-disorder](../.agents/skills/mat-disorder/SKILL.md) |
| `ml-property-predictor` | `mace.predict_atomic_features`, `matgl.predict_atomic_features` | [ml-property-predict-scd](../.agents/skills/ml-property-predict-scd/SKILL.md) |

### Drug Discovery (`drug-*`, 21 skills)

| Skill | Primary MCP Tools | Related Skills |
|-------|-------------------|----------------|
| `drug-docking-vina` | `drugdisc.convert_to_pdbqt`, `drugdisc.compute_molecular_descriptors` | [drug-ligand-prep](../.agents/skills/drug-ligand-prep/SKILL.md), [drug-protein-prep](../.agents/skills/drug-protein-prep/SKILL.md) |
| `drug-protein-ligand-md` | (scripts only — `drugmd` env, OpenMM) | [drug-complex-system-builder](../.agents/skills/drug-complex-system-builder/SKILL.md), [drug-trajectory-analysis](../.agents/skills/drug-trajectory-analysis/SKILL.md) |
| `drug-mmpbsa-gbsa` | (scripts only — `drugmd` env, OpenMM + AmberTools) | [drug-protein-ligand-md](../.agents/skills/drug-protein-ligand-md/SKILL.md) |
| `drug-admet-prediction` | `drugdisc.compute_molecular_descriptors` | [drug-molecular-fingerprints](../.agents/skills/drug-molecular-fingerprints/SKILL.md) |
| `drug-db-pubchem` | (scripts only — `base` env, PubChem API) | [drug-db-chembl](../.agents/skills/drug-db-chembl/SKILL.md), [drug-db-pdb](../.agents/skills/drug-db-pdb/SKILL.md) |
| `drug-pose-validation` | `drugdisc.*` | [drug-docking-analysis](../.agents/skills/drug-docking-analysis/SKILL.md) |

### General (`general-*`, 17 skills)

| Skill | Primary MCP Tools | Related Skills |
|-------|-------------------|----------------|
| `general-arxiv-search` | (scripts only — ArXiv API) | [general-deep-research](../.agents/skills/general-deep-research/SKILL.md) |
| `general-deep-research` | `base.search_literature`, `general-arxiv-search` | [general-peer-review](../.agents/skills/general-peer-review/SKILL.md) |
| `general-plot-digitizer` | (scripts only — VLM + CV) | [mat-xrd-digitizer](../.agents/skills/mat-xrd-digitizer/SKILL.md) |
| `general-workflow-planner` | (reference only) | all workflows |
| `general-query-literature-database` | `base.search_literature` | [general-deep-research](../.agents/skills/general-deep-research/SKILL.md) |

---

## Workflows → Skills

| Workflow | Skills Used | Description |
|----------|-------------|-------------|
| [sorption-discovery](../.agents/workflows/sorption-discovery.md) | `chem-sorption-relax`, `chem-sorption-widom`, `chem-sorption-gcmc` | Gas sorption screening in porous frameworks |
| [mof-co2-dac-screening](../.agents/workflows/mof-co2-dac-screening.md) | `chem-db-mof`, `chem-sorption-relax`, `chem-sorption-widom`, `chem-sorption-gcmc` | MOF CO₂ DAC screening pipeline |
| [mlip-benchmark-finetune](../.agents/workflows/mlip-benchmark-finetune.md) | `ml-mlip-benchmark`, `ml-mace-finetune`, `ml-matgl-finetune`, `ml-fairchem-finetune`, `ml-foundation-potentials` | MLIP benchmark + fine-tuning campaign |
| [generative-halide-discovery](../.agents/workflows/generative-halide-discovery.md) | `ml-generative-mattergen`, `mat-db-mp`, `mat-stability`, `mat-dft-vasp`, `mat-electrochemical-window`, `mat-diffusion-analysis`, `mat-md-probability-density`, `mat-structure-novelty` | Generative SSE discovery (halides) |
| [materials-discovery](../.agents/workflows/materials-discovery.md) | `mat-db-mp`, `ml-generative-mattergen`, `ml-generative-diffcsp`, `ml-generative-adit`, `mat-ionic-substitution`, `mat-stability`, `mat-phonon`, `mat-elasticity`, and 18 more | General materials discovery protocol |
| [drug-hit-finding-htvs](../.agents/workflows/drug-hit-finding-htvs.md) | `drug-db-pdb`, `drug-protein-prep`, `drug-binding-site-definition`, `drug-redocking-rmsd`, `drug-docking-analysis`, `drug-docking-vina`, and 10 more | Structure-based virtual screening funnel |
| [nmr-reaction-kinetics](../.agents/workflows/nmr-reaction-kinetics.md) | `general-plot-digitizer`, `chem-nmr-analysis`, `drug-db-pubchem`, `chem-nmr-predict` | Reaction kinetics from NMR spectra |
| [reaction-to-nmr-quantification](../.agents/workflows/reaction-to-nmr-quantification.md) | `chem-nmr-predict`, `chem-nmr-analysis`, `drug-db-pubchem`, `general-plot-digitizer` | Reaction monitoring via NMR |
| [image-to-xrd-phase](../.agents/workflows/image-to-xrd-phase.md) | `mat-xrd-digitizer`, `mat-xrd-phase-analysis`, `mat-xrd-refinement`, `mat-synthesis-recommendation` | XRD phase ID from plot images |

---

## Script-Only Skills (No MCP Server)

These skills rely on standalone Python scripts rather than MCP tools. They still run in Pixi environments but don't expose tools through the MCP protocol.

| Skill | Env | Reason |
|-------|-----|--------|
| `chem-dft-orca-singlepoint` | `orca` | SCINE/ReaDuct + ORCA binary, complex I/O |
| `chem-dft-orca-optimization` | `orca` | SCINE/ReaDuct + ORCA binary |
| `chem-dft-orca-advanced-calculation` | `orca` | SCINE/ReaDuct + ORCA binary |
| `chem-docking-void` | `void` | VOID library (no MCP wrapper yet) |
| `chem-react-ot` | `react-ot` | PyTorch model, complex pipeline |
| `chem-msms-predict` | `msms` | ICEBERG model inference |
| `chem-nmr-predict` | `nmr` | NMRdb.org API + nmrsim |
| `chem-nmr-analysis` | `nmr` | NMR spectrum deconvolution |
| `chem-spectrum-matcher` | `nmr` | Spectrum matching pipeline |
| `mat-xrd-phase-analysis` | `xrd` | DARA BGMN search |
| `mat-xrd-refinement` | `xrd` | DARA Rietveld refinement |
| `mat-calphad-phase-diagram` | `calphad` | pycalphad TDB processing |
| `mat-calphad-property-diagram` | `calphad` | pycalphad property diagrams |
| `mat-phase-field-conservative` | `phasefield` | FiPy Cahn-Hilliard |
| `mat-phase-field-non-conservative` | `phasefield` | FiPy Allen-Cahn |
| `drug-docking-vina` | `drugdisc` | AutoDock Vina (script-based) |
| `drug-protein-ligand-md` | `drugmd` | OpenMM MD simulation |
| `drug-complex-system-builder` | `drugmd` | OpenMM system builder |
| `drug-mmpbsa-gbsa` | `drugmd` | MM-GBSA / MM-PBSA (OpenMM + AmberTools) |
| `drug-trajectory-analysis` | `drugmd` | MDAnalysis + ProLIF trajectory analysis |
| `ml-property-predict-scd` | `scd` | SCD foundation model |
| `general-plot-digitizer` | `base` | VLM + CV pipeline |

> [!NOTE]
> Some script-only skills could be migrated to MCP tools in the future to enable direct tool calling from agents. This is especially true for frequently-used skills like Vina docking and NMR prediction.

---

## How to Use This Map

### From a User Query → Find the Right Path

1. **Check the workflows** — if the query matches a high-level campaign pattern, start there
2. **Find the relevant skills** — workflows list their component skills; for standalone queries, browse by category above
3. **Identify MCP tools** — each skill lists its primary tools; use these directly for quick queries

### Adding a New Skill → Know What to Build

1. **Check if MCP tools exist** — if core primitives are already in an MCP server, use them
2. **If not, consider adding to an MCP server** — for reusable primitives
3. **If too specialized**, keep as a script-only skill with `# Env:` annotations

### Adding a New Workflow → Reference Skills Correctly

1. Use `[skill-name](../skills/skill-name/SKILL.md)` Markdown links for skill references
2. Follow the template in [workflow-standards.md](../.agents/rules/workflow-standards.md)
3. Each workflow step should map to at least one skill or MCP tool

---

## Related Documentation

- [Research Standards](../.agents/rules/research-standards.md) — intent classification, research protocol
- [Skill Standards](../.agents/rules/skill-standards.md) — skill file format, naming conventions
- [Workflow Standards](../.agents/rules/workflow-standards.md) — workflow file format
- [MCP Environment Rules](../.agents/rules/mcp-environments.md) — environment mappings
- [Developer Guide](developer_guide.md) — architecture overview
- [HPC Job Submission](hpc_job_submission.md) — Slurm/HPC integration for DFT/MD
