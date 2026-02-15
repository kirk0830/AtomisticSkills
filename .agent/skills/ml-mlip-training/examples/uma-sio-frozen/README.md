# UMA Si-O Fine-Tuning with Frozen Backbone

Fine-tuning UMA-S-1p1 on 95 Si-O structures with MatPES-PBE DFT labels.
Backbone parameters are frozen — only the output head is trained.

## Setup

- **Model**: `uma-s-1p1`
- **Dataset**: 95 Si-O structures (80 train / 15 val)
- **Labels**: MatPES-PBE static calculations via atomate2

## MCP Tool Call

```python
result = fine_tune_model(
    training_data_path="training_data.json",
    epochs=3,
    learning_rate=4e-4,
    output_dir="output/",
    training_config={
        "freeze_backbone": True,
    }
)
```

## Expected Results

| Metric | Epoch 0 | Epoch 2 |
|:-------|:--------|:--------|
| Train loss | 4.585 | 0.806 |
| Val loss | 4.894 | 4.832 |
| Energy MAE (val) | 0.283 eV/atom | 0.279 eV/atom |
| Force MAE (val) | 0.047 eV/Å | 0.047 eV/Å |
| Stress MAE (val) | 0.009 eV/ų | 0.009 eV/ų |

## Freeze Backbone Verification

All 143 backbone parameters remain identical to the base model.
6 output head parameters were updated during training (max diff up to 0.98).

| Check | Result |
|:------|:-------|
| Backbone params unchanged | 143 / 143 |
| Backbone params changed | 0 |
| Head params updated | 6 |
| **Verdict** | **PASS** |

## Output Files

- `training_history.json` — Convergence metrics and verification results
- `training_history.png` — Training curve plot
