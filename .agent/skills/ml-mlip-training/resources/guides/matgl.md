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

## Cosine Annealing LR Scheduler

MatGL uses `CosineAnnealingLR` by default. The learning rate decays from `lr` to `lr × decay_alpha` over `decay_steps` steps, then restarts.

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `decay_steps` | int | 1000 | Period (T_max) of the cosine annealing cycle in steps. |
| `decay_alpha` | float | 0.01 | Minimum LR as fraction of initial LR. Final LR = `lr × decay_alpha`. |

**Schedule visualization:**
```
LR
 ^
 |‾‾‾\      /‾‾‾\
 |    \    /     \
 |     \  /       \___
 |      \/
 +------------------------> steps
   T_max   T_max
   (decay_steps)
```

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
    epochs=100,
    learning_rate=1e-3,
    output_dir="/path/to/output",
    training_config={
        "freeze_backbone": True,
        "batch_size": 32,
        "decay_steps": 500,
        "decay_alpha": 0.001,
        "force_weight": 10.0,
    }
)
```
