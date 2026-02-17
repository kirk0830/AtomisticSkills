# MACE Fine-Tuning Guide

MACE fine-tuning uses the `mace_run_train` CLI. All parameters not explicitly handled by the wrapper are **passed through** directly as CLI arguments (e.g., `--scheduler ReduceLROnPlateau`).

## Basic Parameters

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `epochs` | int | 100 | ≥1 | Number of training epochs. |
| `learning_rate` | float | 0.01 | >0 | Peak learning rate (passed as `--lr`). |
| `batch_size` | int | 2 | ≥1 (auto-capped to dataset size) | Training batch size. |

## Model Freezing

| Key | Type | Default | Choices | Description |
|:----|:-----|:--------|:--------|:------------|
| `freeze_backbone` | bool | True | `True`, `False` | Freeze backbone (interaction blocks); only readout heads are trained. |
| `reinit_head` | bool | False | `True`, `False` | Re-initialize readout weights. When False (default), pre-trained readout is preserved. |
| `multiheads_finetuning` | bool | False | `True`, `False` | Enable multi-head fine-tuning (adds new head while keeping existing ones). |

## Optimizer & Regularization

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `optimizer` | str | `"adam"` | `"adam"`, `"adamw"`, `"schedulefree"` | Optimizer type. |
| `weight_decay` | float | 5e-7 | ≥0 | L2 weight decay. |
| `amsgrad` | bool | True | `True`, `False` | Use AMSGrad variant of Adam. |
| `clip_grad` | float | 10.0 | >0 or None | Maximum gradient norm for clipping. Set to None to disable. |
| `ema` | bool | True | `True`, `False` | Enable exponential moving average of model weights. |
| `ema_decay` | float | 0.99 | 0–1 | EMA decay rate. Higher = more smoothing. |

## LR Scheduler

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `scheduler` | str | `"ReduceLROnPlateau"` | `"ReduceLROnPlateau"`, `"ExponentialLR"` | LR scheduler type. |
| `lr_factor` | float | 0.8 | 0–1 | Factor by which LR is reduced on plateau (for ReduceLROnPlateau). |
| `scheduler_patience` | int | 50 | ≥1 | Epochs without improvement before reducing LR (for ReduceLROnPlateau). |
| `lr_scheduler_gamma` | float | 0.9993 | 0–1 | Per-epoch multiplicative decay factor (for ExponentialLR). |

**ReduceLROnPlateau** (default): Monitors validation loss. When loss stops improving for `scheduler_patience` epochs, LR is multiplied by `lr_factor`.

**ExponentialLR**: LR decays every epoch by `lr_scheduler_gamma`. At epoch *n*: `LR = learning_rate × lr_scheduler_gamma^n`.

## Early Stopping

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `patience` | int | 2048 | ≥1 | Stop training after this many epochs without improvement. |

> [!NOTE]
> Default `patience=2048` effectively disables early stopping. Set lower (e.g., 100–200) if you want to stop early.

## Loss Function

| Key | Type | Default | Choices | Description |
|:----|:-----|:--------|:--------|:------------|
| `loss` | str | `"weighted"` | `"weighted"`, `"universal"`, `"ef"`, `"forces_only"`, `"stress"`, `"virials"`, `"huber"`, `"dipole"`, `"dipole_polar"`, `"energy_forces_dipole"`, `"l1l2energyforces"` | Loss function type. `"universal"` is auto-set when stress data is detected. |
| `energy_weight` | float | 1.0 | ≥0 | Weight for energy loss. |
| `forces_weight` | float | 100.0 | ≥0 | Weight for forces loss. |
| `stress_weight` | float | 1.0 | ≥0 | Weight for stress loss (only when stress data is present). |
| `compute_forces` | bool | True | `True`, `False` | Include forces in training. |
| `compute_stress` | bool | False | `True`, `False` | Include stress in training (auto-enabled when stress data present). |

> [!WARNING]
> **Learning rate sensitivity for MACE-OMAT**: The official MACE docs recommend `lr=0.01` for MACE-MP-0, but MACE-OMAT-0-small requires `lr=1e-4` to avoid divergence. Higher values (1e-3, 0.01) cause catastrophic forgetting even with frozen backbone + EMA.

> [!IMPORTANT]
> All keys not in the reserved list (`epochs`, `learning_rate`, `batch_size`, `validation_split`, `early_stopping_patience`, `save_best_model`, `use_foundation_model`, `stress_weight`) are passed through verbatim to the MACE CLI. This means any valid `mace_run_train` argument works.

## Example

```python
result = fine_tune_model(
    training_data_path="/path/to/training_data.json",
    epochs=100,
    learning_rate=0.01,
    output_dir="/path/to/output",
    training_config={
        "freeze_backbone": True,
        "optimizer": "adamw",
        "forces_weight": 100.0,
        "stress_weight": 1.0,
        "compute_stress": True,
        "ema": True,
        "ema_decay": 0.99,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": 50,
        "patience": 200,
        "clip_grad": 10.0,
    }
)
```
