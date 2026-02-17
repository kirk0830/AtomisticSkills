# MACE Si-O Fine-Tuning with Frozen Backbone

Fine-tuning MACE-OMAT-0-small on 93 Si-O structures with MatPES-PBE DFT labels.
Backbone parameters are frozen — only the readout head is trained.
Uses official MACE fine-tuning defaults (EMA, amsgrad, float64, rms_forces_scaling).

## Data Preparation

The original 95-structure dataset was filtered to remove 2 outliers:
- 1 structure with max force = 81 eV/Å and max stress = 5.2 eV/Å³
- 1 structure with max force = 4.4 eV/Å

**Filter criteria**: max|F| ≤ 2.0 eV/Å, max|S| ≤ 0.5 eV/Å³

## Setup

- **Model**: `MACE-OMAT-0-small`
- **Dataset**: 93 Si-O structures (filtered, ~84 train / ~9 val)
- **Labels**: MatPES-PBE static calculations via atomate2

## MCP Tool Call

```python
result = fine_tune_model(
    training_data_path="training_data_filtered.json",
    epochs=10,
    learning_rate=1e-4,
    output_dir="output/",
    training_config={
        "freeze_backbone": True,
    }
)
```

> [!NOTE]
> The wrapper automatically applies official MACE fine-tuning defaults:
> `--ema`, `--ema_decay=0.99`, `--amsgrad`, `--default_dtype=float64`,
> `--scaling=rms_forces_scaling`, `--batch_size=2`.

## Expected Results

| Metric | Epoch 0 | Epoch 9 |
|:-------|:--------|:--------|
| Energy MAE (val) | 61.4 meV/atom | 62.8 meV/atom |
| Force MAE (val) | 105.8 meV/Å | 104.3 meV/Å |
| Stress MAE (val) | 4.6 meV/Å³ | 4.6 meV/Å³ |
| Val loss | 0.0741 | 0.0733 |

## Output Files

- `training_history.json` — Convergence metrics per epoch
- `training_history.png` — Training curve plot
