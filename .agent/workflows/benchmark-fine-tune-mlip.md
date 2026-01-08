---
description: How to benchmark and fine-tune a MLIP with data-augmentation.
---

# Workflow: Calculating Melting Temperature

This workflow describes the steps to benchmark and fine-tune a foundation potential with data-augmentation.

1. **Structure Retrieval**: 
Query the material structure. If unavailable in exisisting database, construct the structure and relax it using a foundation MLIP. Save the relaxed structure as a `.cif` file locally.
If the simulation task involves multiple different structures or a range of chemical compositions, we need to prepare multiple starting structures that covers the simulation task.
2. **Potential Energy Surface(PES) Sampling**:
    - Use a MCP sampler tool to sample the PES related to the simulation task.
    - For off-equilibrium simulation task (deformations, structure transformations, dynamic process, diffusions), use the off-equilibrium sampler with a high MD temperature. The sampler MD temperature needs to be high enough to cover the interatomic chemical environment that is expected in the targeted simulation task.
    - For near-equilibrium simulation task (ionic relaxation, etc), use near-equilibrium sampler.
    - Ensure the simulation cell contains approximately 50-100 atoms, so medium to long range effects are included.
    - Save sampled structures as individual CIF files in a `sampled_structures` directory.
3. **Label Preparation**:
    - For each initial structure, we need approximately 100 sampled structures for labeling.
    - Based on the foundation potential that was chosen for downstream task, the new labels is prefered to be compatible with the pre-training labels of the foundation potential.
    - The prepare_vasp_inputs mcp tool accepts preset_type that writes compatible VASP input given the chosen foundation potential.
    - In case of 1. fast testing this workflow or, 2. distilation of MLIP or, 3. when DFT is unavailable, use 'uma-m-1p1' to "mock" the calculations.
    - Parse the DFT results using parse_vasp_results tool into MLIP training data format
4. **Benchmarking**
    - Use the predict_structure mcp_tool to acquire model predicted energies and forces on the sampled structures, and compare against the labels created in step 3.
    - Make energy and force parity plot, and report the energy MAEs in units of meV/atom, force MAEs in units of meV/A
6. **Fine-tuning**:
    - Fine-tune the selected foundation potential using fine-tune mcp tool
    - The model checkpoints and final error metrics will be saved after the fine-tuning is completed
7. **Deployment**: Provide a Python script to load the fine-tuned model into an ASE calculator or LAMMPS potential for downstream simulations.
8. **Reporting**: Generate an `experiment_report.md` file summarizing the workflow, tools used, arguments, and all saved artifacts.