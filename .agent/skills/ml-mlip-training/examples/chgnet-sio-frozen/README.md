# CHGNet Si-O Fine-Tuning with Frozen Backbone

Fine-tuning CHGNet-MatPES-PBE on 93 Si-O structures with MatPES-PBE DFT labels.
Backbone parameters are frozen — only the output layers are trained.
Dataset was filtered to remove 2 outlier structures with extreme forces/stresses.

## Data Preparation

The original 95-structure dataset was filtered to remove 2 outliers:
- 1 structure with max force = 81 eV/Å and max stress = 5.2 eV/Å³
- 1 structure with max force = 4.4 eV/Å

**Filter criteria**: max|F| ≤ 2.0 eV/Å, max|S| ≤ 0.5 eV/Å³

## Setup

- **Model**: `CHGNet-MatPES-PBE-2025.2.10-2.7M-PES`
- **Dataset**: 93 Si-O structures (filtered, ~79 train / ~14 val)
- **Labels**: MatPES-PBE static calculations via atomate2

## MCP Tool Call

```python
result = fine_tune_model(
    training_data_path="training_data_filtered.json",
    epochs=10,
    learning_rate=0.001,
    batch_size=4,
    output_dir="output/",
    training_config={
        "freeze_backbone": True,
    }
)
```

## Expected Results

| Metric | Epoch 0 | Epoch 4 (best) | Epoch 9 |
|:-------|:--------|:---------------|:--------|
| Energy MAE (val) | 230.3 meV/atom | 185.9 meV/atom | 244.1 meV/atom |
| Force MAE (val) | 78.8 meV/Å | 79.8 meV/Å | 89.9 meV/Å |
| Stress MAE (val) | 4.1 meV/Å³ | 3.7 meV/Å³ | 3.5 meV/Å³ |
| Train loss | 4.89 | 3.23 | 2.99 |
| Val loss | 1.34 | 1.07 | 1.15 |


## Output Files

- `training_history.json` — Convergence metrics per epoch
- `training_history.png` — Training curve plot
