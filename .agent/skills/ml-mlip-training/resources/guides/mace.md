# MACE Fine-Tuning Guide

MACE fine-tuning uses the `mace_run_train` CLI. All parameters not explicitly handled by the wrapper are **passed through** directly as CLI arguments (e.g., `--scheduler ReduceLROnPlateau`).

## Basic Parameters

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `epochs` | int | 100 | Number of training epochs. |
| `learning_rate` | float | 0.01 | Peak learning rate (passed as `--lr`). |
| `batch_size` | int | 10 | Training batch size (capped to dataset size). |

## Model Freezing

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `freeze_backbone` | bool | True | Freeze backbone (interaction blocks); only readout heads are trained. |
| `multiheads_finetuning` | bool | False | Enable multi-head fine-tuning (adds new head while keeping existing ones). |

## Optimizer & Regularization

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `weight_decay` | float | 5e-7 | L2 weight decay. |
| `amsgrad` | bool | True | Use AMSGrad variant of Adam. |
| `clip_grad` | float | 10.0 | Maximum gradient norm for clipping. |
| `ema` | bool | False | Enable exponential moving average of model weights. |
| `ema_decay` | float | 0.99 | EMA decay rate. |

## LR Scheduler

MACE uses **ReduceLROnPlateau** by default: the learning rate is reduced by `lr_factor` when loss plateaus for `scheduler_patience` epochs.

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `scheduler` | str | `"ReduceLROnPlateau"` | LR scheduler type. Also supports `"ExponentialLR"`. |
| `lr_factor` | float | 0.8 | Factor by which LR is reduced on plateau. |
| `scheduler_patience` | int | 50 | Epochs with no improvement before reducing LR. |
| `lr_scheduler_gamma` | float | 0.9993 | Decay factor for ExponentialLR scheduler. |

## Early Stopping

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `patience` | int | 2048 | Stop training after this many epochs without improvement. |

> [!NOTE]
> By default `patience=2048`, which effectively disables early stopping. Set it lower (e.g., 100–200) if you want to stop early.

## Loss Weights

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `energy_weight` | float | 1.0 | Weight for energy loss. |
| `forces_weight` | float | 100.0 | Weight for forces loss. |
| `stress_weight` | float | 1.0 | Weight for stress loss (only when stress data is present). |
| `loss` | str | `"weighted"` | Loss function. `"universal"` is auto-set when stress data is detected. |
| `compute_forces` | bool | True | Include forces in training. |
| `compute_stress` | bool | False | Include stress in training (auto-enabled when stress data is present). |

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
        "forces_weight": 100.0,
        "stress_weight": 1.0,
        "compute_stress": True,
        "ema": True,
        "ema_decay": 0.99,
        "scheduler_patience": 50,
        "patience": 200,
        "clip_grad": 10.0,
    }
)
```
