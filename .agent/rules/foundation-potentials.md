---
trigger: model_decision
description: Rule for choosing foundation machine learning interatomic potential (mlip)
---

# Foundation Potentials Selection Rules

You are an expert MLIP model selection agent. Your task is to recommend the most appropriate foundation MLIP model based on the user's query, simulation type, and structure composition. Review the user's initial request to identify simulating type (e.g., MD, relaxation) and material composition (organic vs inorganic).


# MLIP Model Selection Guide

## MatGL Models
- **CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES**:  Use for r2SCAN-level inorganic materials simulation with charge information and magnetic moments involved (e.g., when simulation task involves calculating transition metal valence states).
- **TensorNet-MatPES-r2SCAN-v2025.1-PES**: Use for r2SCAN-level inorganic materials simulation. This is a smaller, cheaper model suitable for dynamic simulations (MD, NEB, phonons).

### FAIRCHEM Models
- **uma-s-1p1**: Use for organic and inorganic simulations. Note: UMA models are typically slower and more expensive, so avoid for dynamic simulations with system more than 500 atoms.
- **uma-m-1p1**: organic and inorganic simulations with less than 100 atoms.
- **esen-md-direct-all-omol**: organic ionic relaxation for ground state calculations

### MACE Models 
- **MACE-MH-1**: Latest multi-head foundation model. Use as default for most tasks.
  - `omat_pbe` head (default): General materials, balanced performance.
  - `matpes_r2scan` head: High-accuracy materials simulation.
  - `omol` head: Molecular systems, organic chemistry, organometallic.
  - `spice_wB97M` head: Molecular systems and organic chemistry.
  - `oc20_usemppbe` head: Surface catalysis, adsorbates.
- **MACE-MATPES-r2SCAN-0**: specialized for r2SCAN-level inorganic systems. 
- **MACE-OMAT-0-small**: Small, efficient model for materials.

## Selection Criteria (in priority order):
1. **User explicit request**: If the user explicitly mentions a model name or framework (e.g., "MACE model", "fine-tuned MACE", "CHGNet", "UMA"), use that model/framework. Detect the framework from keywords like "MACE", "CHGNet", "TensorNet", "UMA", "ESEN", "FAIRCHEM", "MatGL".

2. **Calculation expense**: If the simulation involves dynamic/expensive calculations (molecular dynamics/MD, NEB, phonons, diffusion, melting temperature), prioritize smaller and cheaper models:
   - Prefer TensorNet-MatPES-r2SCAN-v2025.1-PES or MACE-MATPES-r2SCAN-0 (MACE small)
   - Avoid UMA models for dynamic simulations (they are slower and more expensive)

3. **System composition**: Consider the chemical elements present in the system:
   - If system contains C and other organic elements (H, N, O, P, S), it's organic -> use UMA models with task_name = "omol"
   - If system is inorganic, use MatGL, MACE models, or UMA with task_name = "omat"

4. **Default**: For general materials, use MACE-MH-1 with `omat_pbe` head.


