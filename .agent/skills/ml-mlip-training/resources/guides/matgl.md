# MatGL Fine-Tuning Guide

MatGL fine-tuning uses PyTorch Lightning with M3GNet or CHGNet models. Parameters are passed via the `training_config` dict to the `fine_tune_model` MCP tool.

## Basic Parameters

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `epochs` | int | 10 | Number of training epochs (auto-mapped to `max_epochs` internally). |
| `learning_rate` | float | 1e-3 | Learning rate. |
| `batch_size` | int | 4 | Training batch size. |

## Model Freezing

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `freeze_backbone` | bool | True | Freeze all layers except the final readout. |

## LR Scheduler

> [!NOTE]
> MatGL currently uses the default PyTorch Lightning LR schedule. No custom scheduler parameters are exposed.

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
    }
)
```
