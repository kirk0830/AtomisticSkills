# MatGL Fine-Tuning Guide

MatGL fine-tuning uses PyTorch Lightning with M3GNet or CHGNet models. Parameters are passed via the `training_config` dict to the `fine_tune_model` MCP tool.

## Basic Parameters

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `epochs` | int | 10 | Number of training epochs. |
| `learning_rate` | float | 1e-3 | Learning rate for Adam optimizer. |
| `batch_size` | int | 4 | Training batch size. |

## Model Freezing

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `freeze_backbone` | bool | True | Freeze all layers except the final readout. |

## LR Scheduler

MatGL supports two LR scheduler options via the `scheduler` key:

### CosineAnnealingLR (default)

The default scheduler. LR oscillates between `lr` and `lr × decay_alpha` with period `decay_steps`.

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `scheduler` | str | `"CosineAnnealingLR"` | Scheduler type. |
| `decay_steps` | int | 1000 | Period (T_max) of the cosine annealing cycle in steps. |
| `decay_alpha` | float | 0.01 | Minimum LR as fraction of initial LR. Final LR = `lr × decay_alpha`. |

### ReduceLROnPlateau

Reduces LR when validation loss stops improving.

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `scheduler` | str | — | Set to `"ReduceLROnPlateau"` to enable. |
| `lr_factor` | float | 0.5 | Factor by which LR is reduced on plateau. |
| `scheduler_patience` | int | 10 | Epochs with no improvement before reducing LR. |

## Early Stopping

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `patience` | int | None | Stop after this many epochs with no improvement in `val_Total_Loss`. Not enabled by default. |

> [!NOTE]
> Early stopping monitors `val_Total_Loss`. Set `patience` to e.g. 20–50 for typical fine-tuning runs. If not set, training runs for the full `epochs` count.

## Loss Weights

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `energy_weight` | float | 1.0 | Weight for energy loss. |
| `force_weight` | float | 1.0 | Weight for forces loss. |
| `stress_weight` | float | auto | Weight for stress loss. Auto-detected: 1.0 if stress data present, 0.0 otherwise. |

## Example

```python
result = fine_tune_model(
    training_data_path="/path/to/training_data.json",
    epochs=200,
    learning_rate=1e-3,
    output_dir="/path/to/output",
    training_config={
        "freeze_backbone": True,
        "batch_size": 32,
        "scheduler": "ReduceLROnPlateau",
        "lr_factor": 0.5,
        "scheduler_patience": 10,
        "patience": 30,
        "force_weight": 10.0,
    }
)
```
