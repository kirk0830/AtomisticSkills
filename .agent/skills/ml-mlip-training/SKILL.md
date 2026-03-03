---
name: ml-mlip-training
description: Benchmark and fine-tune Machine Learning Interatomic Potentials (MLIPs) using data-augmentation.
category: [machine-learning]
---

# MLIP Training

## Goal
To evaluate and improve the accuracy of a foundation potential (MACE, MatGL, FairChem) for a specific chemical system or physical property through automated sampling and fine-tuning.

## Instructions

1.  **PES Sampling**: Generate diverse representative structural configurations. For off-equilibrium state sampling, use the [mat-sample-pes-by-md](../mat-sample-pes-by-md/SKILL.md) skill. For disordered structures, use the `run_ordering` script under the [mat-disorder](../mat-disorder/SKILL.md) skill.
2.  **Labeling**: Obtain high-fidelity energy and forces using DFT (via `prepare_vasp_inputs`) or a expensive high-quality MLIPs (MLIP distillation).
3.  **Benchmarking**: Predict results on the new labels and generate parity plots using [plot_training_results.py](scripts/plot_training_results.py).
    ```bash
    python scripts/plot_training_results.py --results benchmarking.json
    ```
4.  **Fine-Tuning**: Execute `fine_tune_model` with recommended [hyperparams.json](resources/hyperparams.json).
5.  **Validation**: Plot the training history to verify convergence.
    ```bash
    python scripts/plot_training_results.py --history history.json
    ```

## Training Config Guides

Each MLIP has its own supported `training_config` parameters. See the per-MLIP guide for full reference tables:

- **FairChem (UMA)**: [resources/guides/fairchem.md](resources/guides/fairchem.md)
- **MACE**: [resources/guides/mace.md](resources/guides/mace.md)
- **MatGL (CHGNet/M3GNet)**: [resources/guides/matgl.md](resources/guides/matgl.md)

### Quick Comparison

| Feature | FairChem | MACE | MatGL |
|:--------|:---------|:-----|:------|
| Freeze backbone | `freeze_backbone` | `freeze_backbone` | `freeze_backbone` |
| Re-init readout head | `reinit_head` (no-op¹) | `reinit_head` | `reinit_head` |
| LR scheduler | Cosine w/ warmup (fixed) | ReduceLROnPlateau (configurable) | CosineAnnealingLR or ReduceLROnPlateau |
| Early stopping | ❌ Not supported | `patience: 2048` (default off) | `patience` (disabled by default) |
| Grad clipping | `clip_grad_norm: 100` | `clip_grad: 10.0` | N/A |
| EMA | `ema_decay: 0.999` | `ema_decay: 0.99` | N/A |
| Loss weights | Automatic via regression task | `energy/forces/stress_weight` | `energy/force/stress_weight` |

¹ FairChem always re-initializes output heads for new task names — `reinit_head` is accepted for API consistency.

## Examples

- [UMA Si-O fine-tuning with frozen backbone](examples/uma-sio-frozen/) — 95 structures, freeze_backbone verified
- [MACE Si-O fine-tuning with frozen backbone](examples/mace-sio-frozen/) — MACE-OMAT-0-small, 10 epochs
- [CHGNet Si-O fine-tuning with frozen backbone](examples/chgnet-sio-frozen/) — CHGNet-MatPES-PBE, 10 epochs

## Constraints
- **Data Size**: For small datasets (<500 structures), `freeze_backbone=True` is strongly recommended.
- **Units (input)**: Stress labels must be in eV/Å³ as per project standards.
- **Units (output)**: All `training_history.json` files use **meV** units: energy MAE in meV/atom, force MAE in meV/Å, stress MAE in meV/Å³.


---

**Author:** Bowen Deng  
**Contact:** [GitHub @bowen-bd](https://github.com/bowen-bd)
