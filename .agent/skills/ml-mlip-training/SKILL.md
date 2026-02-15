---
name: ml-mlip-training
description: Benchmark and fine-tune Machine Learning Interatomic Potentials (MLIPs) using data-augmentation.
category: machine-learning
---

# MLIP Training

## Goal
To evaluate and improve the accuracy of a foundation potential (MACE, MatGL, FairChem) for a specific chemical system or physical property through automated sampling and fine-tuning.

## Instructions

1.  **PES Sampling**: Generate diverse structural configurations using the near-equilibrium or off-equilibrium samplers.
2.  **Labeling**: Obtain high-fidelity energy and forces using DFT (via `prepare_vasp_inputs`) or a high-quality mock potential (UMA).
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
| LR scheduler | Cosine w/ warmup (fixed) | ReduceLROnPlateau (configurable) | CosineAnnealingLR or ReduceLROnPlateau |
| Early stopping | ❌ Not supported | `patience: 2048` (default off) | `patience` (disabled by default) |
| Grad clipping | `clip_grad_norm: 100` | `clip_grad: 10.0` | N/A |
| EMA | `ema_decay: 0.999` | `ema_decay: 0.99` | N/A |
| Loss weights | Automatic via regression task | `energy/forces/stress_weight` | `energy/force/stress_weight` |

## Examples

- [UMA Si-O fine-tuning with frozen backbone](examples/uma-sio-frozen/) — 95 structures, freeze_backbone verified

## Constraints
- **Data Size**: For small datasets (<500 structures), `freeze_backbone=True` is strongly recommended.
- **Units**: Ensure stress labels are in eV/Å³ as per project standards.


Author: Bowen Deng
Contact: github username <bowen-bd>
