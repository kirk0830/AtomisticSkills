# MatGL Fine-Tuning Example: WBM High-Energy States

This example demonstrates how to fine-tune the `CHGNet-MatPES-PBE-2025.2.10-2.7M-PES` foundation model on a subset of the WBM (Wolverton-Bhatia-Mills) dataset containing high-energy crystal structures. It highlights the use of the `--vasp-stress-conversion` flag, which mathematically corrects raw VASP stress labels (`kB`) into the `eV/Å³` standard natively supported by both the MatGL implementation and `general-property-units`.

## Goal
To document a functional 10-epoch fine-tuning workflow on challenging high-energy VASP data, proving the stability of the decoupled `train_matgl.py` infrastructure directly without wrapping the internal MatGL native components.

## Instructions

### 1. Prepare Training Data 
The dataset is processed using the `prepare_matgl_data.py` script. The training data must contain `structure` (pymatgen dict or ASE atoms), `energy` (or `vasp_e`), `forces` (or `vasp_f`), and `stress` (or `vasp_s`).

Because the raw `vasp_s` labels in this dataset are recorded in `kB`, we MUST pass the `--vasp-stress-conversion` flag to multiply them by `-1/1602.1766208` during extraction, converting them to `eV/Å³` and matching standard convention.

```bash
# Env: matgl-agent
conda run -n matgl-agent python .agent/skills/ml-matgl-finetune/scripts/prepare_matgl_data.py \
    --data private_data/WBM_high_energy_states.json \
    --output-dir .agent/skills/ml-matgl-finetune/examples/matgl-wbm-finetune/output \
    --val-split 0.1 \
    --vasp-stress-conversion
```

### 2. Run MatGL Trainer
The script structures data internally with `MGLDataset` and natively connects directly to PyTorch Lightning via the `train_matgl.py` protocol. You can boot the loop directly onto GPU without supplementary extraction passes:

```bash
# Env: matgl-agent
conda run -n matgl-agent python .agent/skills/ml-matgl-finetune/scripts/train_matgl.py \
    --train-data .agent/skills/ml-matgl-finetune/examples/matgl-wbm-finetune/output/train_data.json \
    --val-data .agent/skills/ml-matgl-finetune/examples/matgl-wbm-finetune/output/val_data.json \
    --model CHGNet-MatPES-PBE-2025.2.10-2.7M-PES \
    --output-dir .agent/skills/ml-matgl-finetune/examples/matgl-wbm-finetune/output \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 10 \
    --freeze-backbone \
    --patience 3
```

### 3. Verify Stress Unit Conversions

To explicitly secure that the MAE scaling mathematically binds identically against external workflows, you can natively invoke `CHGNet` via ASE isolated from lightning handlers and benchmark the resulting output stress vector against the exact identical tensor targets:

```bash
# Env: matgl-agent
cd .agent/skills/ml-matgl-finetune/examples/matgl-wbm-finetune/
conda run -n matgl-agent python test_stress_mae.py
```

## Expected Outputs
The `output` directory contains the artifacts generated from a localized run splitting 8,377 structures over a randomized 90/10 split.

- `fine_tuned_model.pth`: The natively wrapped checkpoint perfectly mapped back to standalone `MatGLWrapper.load_checkpoint()` logic.
- `training_history.json`: Parsed loss and MAE metrics dynamically tracked per-epoch independent of DGL dependencies.
- `training_history.png`: Standardized loss degradation plot identical to the original wrapper output format verifying stability.
- `finetune_record.json`: Summary configuration defining model paths and basic convergence variables.

---

**Author:** Bowen Deng  
**Contact:** [GitHub @bowen-bd](https://github.com/bowen-bd)
