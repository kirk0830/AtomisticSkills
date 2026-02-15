# MatGL Fine-Tuning Guide

MatGL fine-tuning uses PyTorch Lightning with M3GNet or CHGNet models. Parameters are passed via the `training_config` dict to the `fine_tune_model` MCP tool.

## Basic Parameters

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `epochs` | int | 10 | Number of training epochs. |
| `learning_rate` / `lr` | float | 1e-3 | Learning rate. |
| `batch_size` | int | 4 | Training batch size. |

## Model Freezing

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `freeze_backbone` | bool | True | Freeze all layers except the final readout. |

## LR Scheduler

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `decay_steps` | int | 1000 | Steps between LR decay. |
| `decay_alpha` | float | 0.01 | Final LR = `decay_alpha × lr`. |

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
        "decay_steps": 1000,
        "decay_alpha": 0.01,
    }
)
```
