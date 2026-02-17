# UMA Si-O Fine-Tuning with Frozen Backbone

Fine-tuning UMA-S-1p1 on 93 Si-O structures with MatPES-PBE DFT labels.
Backbone parameters are frozen — only the output head is trained.
Dataset was filtered to remove 2 outlier structures with extreme forces/stresses.

## Data Preparation

The original 95-structure dataset was filtered to remove 2 outliers:
- 1 structure with max force = 81 eV/Å and max stress = 5.2 eV/Å³
- 1 structure with max force = 4.4 eV/Å

**Filter criteria**: max|F| ≤ 2.0 eV/Å, max|S| ≤ 0.5 eV/Å³

## Setup

- **Model**: `uma-s-1p1`
- **Dataset**: 93 Si-O structures (filtered)
- **Labels**: MatPES-PBE static calculations via atomate2

## MCP Tool Call

```python
result = fine_tune_model(
    training_data_path="training_data_filtered.json",
    epochs=10,
    learning_rate=4e-4,
    output_dir="output/",
    training_config={
        "freeze_backbone": True,
    }
)
```

## Expected Results

| Metric | Epoch 0 | Epoch 9 |
|:-------|:--------|:--------|
| Energy MAE (val) | 266.1 meV/atom | 259.7 meV/atom |
| Force MAE (val) | 46.5 meV/Å | 46.7 meV/Å |
| Stress MAE (val) | 9.0 meV/Å³ | 9.0 meV/Å³ |
| Val loss | 65.84 | 64.74 |

> [!NOTE]
> UMA shows very slow convergence with frozen backbone — energy MAE decreases
> only slightly (266→260 meV/atom) over 10 epochs, while forces and stress
> remain essentially flat. This is expected because FairChem uses a cosine
> annealing LR scheduler with warmup, and the frozen backbone limits the
> model's capacity to adapt.

## Output Files

- `training_history.json` — Convergence metrics per epoch
- `training_history.png` — Training curve plot
