# MACE Fine-Tuning Guide

MACE fine-tuning uses the `mace_run_train` CLI. Key parameters are passed via the `training_config` dict to the `fine_tune_model` MCP tool.

## Basic Parameters

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `epochs` | int | 100 | Number of training epochs (auto-mapped to `max_epochs` internally). |
| `learning_rate` / `lr` | float | 0.01 | Peak learning rate. |
| `batch_size` | int | 10 | Training batch size. |

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
| `ema` | bool | True | Enable exponential moving average of model weights. |
| `ema_decay` | float | 0.99 | EMA decay rate. |

## LR Scheduler

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `scheduler` | str | `"ReduceLROnPlateau"` | LR scheduler type. |
| `lr_factor` | float | 0.8 | Factor by which LR is reduced. |
| `scheduler_patience` | int | 50 | Number of epochs with no improvement before reducing LR. |

## Loss Weights

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `energy_weight` | float | 1.0 | Weight for energy loss. |
| `forces_weight` | float | 100.0 | Weight for forces loss. |
| `stress_weight` | float | 1.0 | Weight for stress loss. |
| `compute_forces` | bool | True | Include forces in training. |
| `compute_stress` | bool | False | Include stress in training. |

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
        "ema_decay": 0.99,
        "scheduler_patience": 50,
    }
)
```
