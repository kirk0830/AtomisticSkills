# FairChem Fine-Tuning Guide

FairChem fine-tuning wraps the official `fairchem` CLI with a Hydra-based YAML configuration. All parameters below are supported via the `training_config` dict passed to the `fine_tune_model` MCP tool.

## Basic Parameters

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `epochs` | int | 10 | Number of training epochs. |
| `learning_rate` | float | 4e-4 | Peak learning rate for AdamW optimizer. |
| `batch_size` | int | 2 | Training batch size per GPU. |
| `task_name` | str | `"omat"` | Task name (dataset key in multi-task config). |
| `base_model` | str | loaded model | Base UMA checkpoint name (e.g., `"uma-s-1p1"`). |

## Model Freezing

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `freeze_backbone` | bool | False | Freeze all backbone parameters; only output heads are trained. Strongly recommended for small datasets (<500 structures). |

When `freeze_backbone=True`, a helper module is generated that wraps `initialize_finetuning_model` and sets `requires_grad=False` on all `model.backbone` parameters. Only the output head layers (energy, force, stress predictions) are updated during training.

**Verified**: All 143 backbone parameters remain identical to the base model after training. See [examples/uma-sio-frozen](../examples/uma-sio-frozen/).

## Optimizer & Regularization

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `weight_decay` | float | 1e-3 | L2 weight decay for AdamW. |
| `clip_grad_norm` | float | 100 | Maximum gradient norm for clipping. Lower values (e.g., 10) can stabilize training on noisy data. |
| `ema_decay` | float | 0.999 | Exponential moving average decay rate. The EMA model is used for evaluation and checkpoint saving. |

## Cosine LR Scheduler

FairChem uses a **cosine annealing LR scheduler with linear warmup**. There is no option to switch to other schedulers via the CLI.

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `warmup_factor` | float | 0.2 | Initial LR during warmup = `warmup_factor × learning_rate`. |
| `warmup_epochs` | float | 0.01 | Fraction of total epochs used for linear warmup. E.g., 0.01 = 1% of training. |
| `lr_min_factor` | float | 0.01 | Minimum LR at end of cosine decay = `lr_min_factor × learning_rate`. |

**Schedule visualization:**
```
LR
 ^
 |  /‾‾‾‾‾‾‾‾‾\
 | /            \
 |/              \___________
 +--------------------------->  steps
   warmup   cosine decay  min_lr
```

## Checkpointing & Evaluation

| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `evaluate_every_n_steps` | int | 100 | Run validation every N training steps. |
| `checkpoint_every_n_steps` | int | 1000 | Save model checkpoint every N training steps. Up to 5 checkpoints are kept. |

## Early Stopping

> [!IMPORTANT]
> FairChem does **not** have built-in early stopping. Monitor validation loss manually and use a sufficient number of epochs with the cosine scheduler.

## Example

```python
result = fine_tune_model(
    training_data_path="/path/to/training_data.json",
    epochs=50,
    learning_rate=1e-4,
    output_dir="/path/to/output",
    training_config={
        "freeze_backbone": True,
        "weight_decay": 1e-3,
        "clip_grad_norm": 10.0,
        "warmup_epochs": 0.05,
        "lr_min_factor": 0.001,
        "evaluate_every_n_steps": 50,
        "checkpoint_every_n_steps": 500,
        "ema_decay": 0.999,
    }
)
```
