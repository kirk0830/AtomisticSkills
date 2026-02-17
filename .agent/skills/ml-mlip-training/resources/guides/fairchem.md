# FairChem Fine-Tuning Guide

FairChem fine-tuning wraps the official `fairchem` CLI with a Hydra-based YAML configuration. All parameters below are supported via the `training_config` dict passed to the `fine_tune_model` MCP tool.

## Basic Parameters

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `epochs` | int | 10 | ≥1 | Number of training epochs. |
| `learning_rate` | float | 4e-4 | >0 | Peak learning rate for AdamW optimizer. |
| `batch_size` | int | 2 | ≥1 | Training batch size per GPU. |
| `task_name` | str | `"omat"` | Any valid UMA task key | Task name (dataset key in multi-task config). |
| `base_model` | str | loaded model | `"uma-s-1p1"`, `"uma-s-1"`, etc. | Base UMA checkpoint name. |

## Model Freezing

| Key | Type | Default | Choices | Description |
|:----|:-----|:--------|:--------|:------------|
| `freeze_backbone` | bool | False | `True`, `False` | Freeze all backbone parameters; only output heads are trained. Strongly recommended for small datasets (<500 structures). |
| `reinit_head` | bool | False | `True`, `False` | Accepted for API consistency. No functional effect — FairChem always re-initializes heads for new task names. |

When `freeze_backbone=True`, a helper module wraps `initialize_finetuning_model` and sets `requires_grad=False` on all `model.backbone` parameters. Only the output head layers are updated.

**Verified**: All 143 backbone parameters remain identical to the base model after training. See [examples/uma-sio-frozen](../examples/uma-sio-frozen/).

## Optimizer & Regularization

| Key | Type | Default | Choices / Range | Description |
|:----|:-----|:--------|:----------------|:------------|
| `weight_decay` | float | 1e-3 | ≥0 | L2 weight decay for AdamW. |
| `clip_grad_norm` | float | 100 | >0 | Maximum gradient norm for clipping. Lower values (e.g., 10) can stabilize noisy data. |
| `ema_decay` | float | 0.999 | 0–1 | Exponential moving average decay rate. EMA model is used for evaluation and checkpoints. |

> [!NOTE]
> FairChem always uses **AdamW** as the optimizer. This is not configurable.

## Cosine LR Scheduler

FairChem uses a **cosine annealing LR scheduler with linear warmup**. This is the only available scheduler — it cannot be swapped for another type.

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `warmup_factor` | float | 0.2 | 0–1 | Initial LR during warmup = `warmup_factor × learning_rate`. |
| `warmup_epochs` | float | 0.01 | 0–1 | Fraction of total epochs used for linear warmup. E.g., 0.01 = 1%. |
| `lr_min_factor` | float | 0.01 | 0–1 | Minimum LR at end of cosine decay = `lr_min_factor × learning_rate`. |

**Schedule visualization:**
```
LR
 ^
 |  /‾‾‾‾‾‾‾‾‾\
 | /            \
 |/              \___________
 +---------------------------> steps
   warmup   cosine decay  min_lr
```

## Checkpointing & Evaluation

| Key | Type | Default | Range | Description |
|:----|:-----|:--------|:------|:------------|
| `evaluate_every_n_steps` | int | 100 | ≥1 | Run validation every N training steps. |
| `checkpoint_every_n_steps` | int | 1000 | ≥1 | Save model checkpoint every N steps. Up to 5 checkpoints are kept. |

## Early Stopping

> [!IMPORTANT]
> FairChem does **not** support early stopping. Training runs as a CLI subprocess with no callback mechanism. Workaround: monitor validation loss in training logs and set `epochs` accordingly based on prior experience.

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
        "warmup_factor": 0.2,
        "warmup_epochs": 0.05,
        "lr_min_factor": 0.001,
        "evaluate_every_n_steps": 50,
        "checkpoint_every_n_steps": 500,
        "ema_decay": 0.999,
    }
)
```
