---
trigger: model_decision
description: Fine-tuning hyperparameters for MACE, FairChem, and MatGL MLIPs
---

# Fine-tuning Configuration Guide

This guide describes the available fine-tuning configuration keys for each Machine Learning Interatomic Potential (MLIP) supported in this environment. These keys can be passed to the `fine_tune_model` MCP tool via the `training_config` dictionary.

## MACE

MACE fine-tuning is primarily controlled by the command-line arguments of `mace.cli.run_train`. Key parameters include:

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | float | 0.01 | Initial learning rate. |
| `max_epochs` | int | 100 | Maximum number of training epochs. |
| `batch_size` | int | 10 | Training batch size. |
| `valid_batch_size` | int | 10 | Validation batch size. |
| `scheduler` | str | "ReduceLROnPlateau" | Learning rate scheduler type. |
| `lr_factor` | float | 0.8 | Factor by which the learning rate will be reduced. |
| `scheduler_patience` | int | 50 | Number of epochs with no improvement after which learning rate will be reduced. |
| `ema` | bool | True | Whether to use Exponential Moving Average. |
| `ema_decay` | float | 0.99 | EMA decay rate. |
| `amsgrad` | bool | True | Whether to use the AMSGrad variant of Adam. |
| `weight_decay` | float | 5e-7 | Weight decay (L2 penalty). |
| `loss` | str | "weighted" | Type of loss function ("ef", "weighted", "forces_only"). |
| `energy_weight` | float | 1.0 | Weight of energy loss. |
| `forces_weight` | float | 100.0 | Weight of forces loss. |
| `stress_weight` | float | 1.0 | Weight of stress/virial loss. |
| `multiheads_finetuning` | bool | False | (Defaulted to False in this wrapper) Whether to use multi-head replay. |
| `freeze_backbone` | bool | True | (Defaulted to True in this wrapper) Whether only the readout layers are trainable. |
| `compute_stress` | bool | False | Whether to compute stress. |
| `compute_forces` | bool | True | Whether to compute forces. |

## FAIRCHEM

FAIRCHEM (supporting UMA, ESEN, etc.) uses a YAML/Hydra configuration. Keys in `training_config` override the default training setup:

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr_initial` | float | 1e-4 | Initial learning rate. |
| `max_epochs` | int | 100 | Maximum number of training epochs. |
| `batch_size` | int | 4 | Training batch size per device. |
| `ema_decay` | float | 0.999 | EMA decay rate. |
| `clip_grad_norm` | float | 10.0 | Maximum allowed gradient norm. |
| `freeze_backbone` | bool | True | (Defaulted to True in this wrapper) Whether to freeze the model backbone. |
| `optimizer` | str | "AdamW" | Optimizer class name. |
| `scheduler` | str | "ReduceLROnPlateau" | Scheduler class name. |
| `energy_weight` | float | 1.0 | Weight for energy loss. |
| `force_weight` | float | 100.0 | Weight for force loss. |
| `stress_weight` | float | 1.0 | Weight for stress loss. |

## MatGL

MatGL (supporting CHGNet, M3GNet, TensorNet) uses PyTorch Lightning. Common keys include:

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lr` | float | 1e-3 | Learning rate. |
| `max_epochs` | int | 100 | Maximum epochs. |
| `batch_size` | int | 32 | Batch size. |
| `scheduler` | str | "ReduceLROnPlateau" | Scheduler type. |
| `freeze_backbone` | bool | True | (Defaulted to True in this wrapper) Whether to freeze non-readout layers. |
| `decay_steps` | int | 1000 | Steps for learning rate decay. |
| `decay_alpha` | float | 0.01 | Minimum learning rate as a fraction of initial. |