---
description: How to fine-tune a MLIP for alloy properties (stacking-fault energies, short-range order).
---

# Workflow: Fine-tuning MLIP for Alloy Properties

This workflow outlines the process for fine-tuning a Machine Learning Interatomic Potential (MLIP) to predict composition-dependent properties in alloys, such as stacking-fault energies and short-range order (SRO).

1. **Literature Review**: Conduct a web search for academic papers on the target alloy system (e.g., CrCoNi), focusing on composition-dependent properties, stacking-fault energies, and short-range ordering behavior.
2. **Structure Generation**:
    - Query the Materials Project for existing structures of the target alloy compositions.
    - If specific compositions are missing, generate structures with varying ratios (e.g., 1:1:1, 2:1:1, etc.).
    - Save each unique structure as a `.cif` file.
3. **Chemical Ordering Sampling**:
    - For each composition, generate multiple structures (~100) with different chemical orderings using Lattice Monte Carlo or `pymatgen.transformations.advanced_transformations.OrderDisorderTransformation`.
    - Aggregate all generated structures into `sampled_structures.cif` within a `sampled_structures` directory.
4. **Thermal Fluctuation Sampling**:
    - Generate one perturbed copy for each structure to sample thermal fluctuations.
    - Use a small perturbation length (e.g., 0.1 Å).
5. **DFT Preparation**:
    - Prepare the sampled structures for DFT calculations.
    - Use appropriate DFT settings, such as those from an `atomate2` MatPES flow.
6. **DFT Calculations**:
    - Submit VASP calculations to obtain labels (energy, forces, stress).
    - If actual DFT is not feasible for initial testing, use UMA to mock the results.
    - Save results in a `mock_dft_results` directory.
7. **Fine-tuning**:
    - Fine-tune a foundation potential with the collected data.
    - Save the model in a `{model_name}_fine_tuned` directory.
    - Include training history plots (`training_history.png`) and logs (`training_history.json`).
8. **Validation**:
    - Validate the fine-tuned model by calculating stacking-fault energies and SRO parameters.
    - Compare results with literature values or DFT benchmarks.
9. **Deployment**: provide a script to load the model into ASE or LAMMPS for simulating phase diagrams, heat capacities, and other properties.
10. **Reporting**: Generate an `experiment_report.md` summarizing the workflow, tools, arguments, and artifacts.
