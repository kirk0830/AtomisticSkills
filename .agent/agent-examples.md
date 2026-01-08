---
alwaysApply: false
---

Here are a few examples of user prompts and expected behaviour of the mlip agent.
Note non of the following chemical systems should be hard-coded, the LLM agent is expected to plan a workflow to construct the calculations.
Use these expected behaviour as the "test set" for the agent.

> I want to calculate the melting temperature of FeCl3" 
1. Web search on relevant academic papers, get an approximated melting temperature.
2. Query a relevant structure from Materials Project. If the structure doesn't exist then construct it and relax using UMA. The structure need to be saved as a cif file locally.
3. Use M3GNet sampler to run MD at high temperature and sample the structures, the MD needs to be at least reaching melting of the structures. The MD structure needs to be have around 100 atoms. Sampled structures should be saved as individual CIF files in a `sampled_structures` directory and aggregated into `sampled_structures.cif`.
4. Prepare around 100 sampled structures that is DFT ready. In this case we want to simulate the melting temperature using a MatPES trained foundation potential (can be MACE-MATPES-r2SCAN-0), so we should use pymatgen MatPESStaticSet to generate VASP settings.
5. Submit the VASP calculations and collect results. We can use UMA to mock VASP calculations. The collected labels (energy, forces, stress) are saved in a `mock_dft_results` directory with standard VASP outputs (vasprun.xml/CONTCAR) or JSON summaries.
6. Fine-tune the selected foundation potential. The fine-tuned model must be saved to a directory named `{model_name}_fine_tuned` (e.g., `m3gnet-matpes-r2scan-v2025.1-pes_fine_tuned`). Training history including MAE plots and logs must be saved as `training_history.png` and `training_history.json` in the same directory.
7. Provide simple script to load the fine-tuned model into ASE calculator or LAMMPS potential for downstream simulations.
8. Generate an `experiment_report.md` file that summarizes the entire workflow, including tools called, arguments used, and a list of all saved artifacts.

> I want to fine-tune a MLIP for CrNiCo alloy to predict stacking-fault energies and short-range order across different compositions"
1. Web search on relevant academic papers about CrCoNi alloy, particularly focusing on composition-dependent properties, stacking-fault energies, and short-range ordering behavior.
2. Query structures from Materials Project for CrCoNi compositions. If specific compositions don't exist, generate multiple structures with different Cr:Co:Ni ratios (e.g., equiatomic 1:1:1, and off-equiatomic compositions like 2:1:1, 1:2:1, 1:1:2). Each structure should be saved as a cif file locally.
3. For each composition, generate structures with different chemical orderings through Lattice Monte Carlo or pymatgen OrderDisrorderTransformation. Sample around 100 structures for each composition. All structures should be aggregated into `sampled_structures.cif` in the `sampled_structures` directory.
4. For each structure, create 1 perturbed copy to sample thermal fluctuations. Use a small perturbation length of 0.1 A
5. Prepare structures to be DFT ready. For faster testing, we can use UMA to mock VASP calculations. For DFT setting, we should use atomate2 MatPES flow to generate VASP settings.
6. Submit the VASP calculations and collect results. The collected labels (energy, forces, stress) are saved in a `mock_dft_results` directory.
7. Fine-tune the selected foundation potential. The fine-tuned model must be saved to `{model_name}_fine_tuned` directory. Training history plots (`training_history.png`) and logs (`training_history.json`) must be saved in the same directory.
8. Validate the fine-tuned model by calculating stacking-fault energies and short-range order parameters for different compositions and comparing with literature values or DFT benchmarks.
9. Provide simple script to load the fine-tuned model into ASE calculator or LAMMPS potential for downstream simulations of phase diagrams, heat capacities, and other composition-dependent properties.
10. Generate an `experiment_report.md` file that summarizes the entire workflow, including tools called, arguments used, and a list of all saved artifacts.
