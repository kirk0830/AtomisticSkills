# MatGL Fine-Tuning Guide

MatGL fine-tuning uses PyTorch Lightning with M3GNet or CHGNet models. Parameters are passed via the `training_config` dict to the `fine_tune_model` MCP tool.

## Basic Parameters

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `epochs` | int | 10 | ≥1 | Number of training epochs. |
| `learning_rate` | float | 1e-3 | >0 | Learning rate for Adam optimizer. |
| `batch_size` | int | 4 | ≥1 | Training batch size. |

## Model Freezing

| Key | Type | Default | Choices | Description |
|:----|:-----|:--------|:--------|:------------|
| `freeze_backbone` | bool | True | `True`, `False` | Freeze all layers except the final readout. |
| `reinit_head` | bool | False | `True`, `False` | Re-initialize readout weights (Xavier uniform for weights, zeros for biases). Default preserves pre-trained readout. |

## LR Scheduler

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `scheduler` | str | `"CosineAnnealingLR"` | `"CosineAnnealingLR"`, `"ReduceLROnPlateau"` | Scheduler type. |

### CosineAnnealingLR (default)

LR oscillates between `lr` and `lr × decay_alpha` with period `decay_steps`.

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `decay_steps` | int | 1000 | ≥1 | Period (T_max) of the cosine annealing cycle in steps. |
| `decay_alpha` | float | 0.01 | 0–1 | Minimum LR as fraction of initial LR. Final LR = `lr × decay_alpha`. |

### ReduceLROnPlateau

Reduces LR when validation loss stops improving. Set `scheduler: "ReduceLROnPlateau"` to enable.

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `lr_factor` | float | 0.5 | 0–1 | Factor by which LR is reduced on plateau. |
| `scheduler_patience` | int | 10 | ≥1 | Epochs without improvement before reducing LR. |

## Early Stopping

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `patience` | int | None (disabled) | ≥1 or None | Stop after this many epochs without improvement in `val_Total_Loss`. |

> [!NOTE]
> Early stopping monitors `val_Total_Loss`. Set `patience` to e.g. 20–50 for typical runs. If None, training runs for the full `epochs`.

## Loss Function

| Key | Type | Default | Choices | Description |
|:----|:-----|:--------|:--------|:------------|
| `loss` | str | `"mse_loss"` | `"mse_loss"`, `"huber_loss"`, `"smooth_l1_loss"`, `"l1_loss"` | PyTorch loss function for energy/force/stress terms. |
| `loss_params` | dict | `{}` | Any valid kwargs for the chosen loss | Extra params passed to the loss (e.g., `{"delta": 1.0}` for Huber). |

> [!NOTE]
> `loss` and `loss_params` are not currently wired through our wrapper (they require being passed directly to `PotentialLightningModule`). The parameter tables above document the upstream API for completeness — wiring can be added if needed.

## Loss Weights

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `energy_weight` | float | 1.0 | ≥0 | Weight for energy loss. |
| `force_weight` | float | 1.0 | ≥0 | Weight for forces loss. |
| `stress_weight` | float | auto | ≥0 | Weight for stress loss. Auto: 1.0 if stress data present, 0.0 otherwise. |

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
