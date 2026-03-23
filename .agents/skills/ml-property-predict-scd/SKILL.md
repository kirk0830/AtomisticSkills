---
name: ml-property-predict-scd
description: Use pretrained SelfConditionedDenoisingAtoms checkpoints for atomistic property prediction, frozen embeddings, linear probing, full-model finetuning, new configs, and new dataset adapters.
category: [machine-learning]
---

# SCD Property Prediction

## Goal

Use `SelfConditionedDenoisingAtoms` for four related workflows:

- apply frozen SCD embeddings as features for downstream ML
- fit a simple linear probe on top of frozen SCD embeddings
- fine-tune an entire pretrained SCD checkpoint on a new property task
- pretrain a new SCD model or add a new dataset adapter

## First Checks

1. Use the `scd-agent` environment from `conda-envs/scd-agent/`.
2. Confirm the upstream repo exists at `../SelfConditionedDenoisingAtoms` relative to `AtomisticSkills`, or create it with `conda-envs/scd-agent/install.sh`.
3. Read the upstream `README.md` and `examples.ipynb`.
4. Then read the local references in this skill:
   - `references/repo-map.md`
   - `references/transfer-recipes.md`
   - `references/config-recipes.md`
   - `references/dataset-contract.md` if a new dataset is involved

## Checkpoint Selection

- Use `ct-scd-pcq` for molecule property prediction, molecule embeddings, and molecule-side transfer learning.
- Use `ct-scd-amp` for materials property prediction, periodic materials embeddings, and materials-side transfer learning.

Do not swap these by default. The public checkpoints were pretrained on different domains.

## Preferred Workflow

### 1. Frozen embeddings

Default to `out["mol_emb"]` for graph-level property prediction.

- Use `return_atom_embs=True` only when the downstream task needs atom- or site-level features.
- Keep the model in `eval()` mode and use `torch.no_grad()`.
- Reuse `templates/extract_embeddings.py` as the starting point.

### 2. Linear probe on frozen embeddings

The cleanest linear-probe path is external to `train.py`:

1. Extract `mol_emb` features from a frozen checkpoint.
2. Fit a simple linear regressor or classifier on those cached embeddings.

Use `templates/linear_probe_from_embeddings.py` as the starting point.

Important caveat:

- the upstream `train.py` path does not expose a native "freeze full backbone and train only a linear probe on `mol_emb`" switch
- if you need a true linear probe, train it outside the Lightning trainer on cached embeddings

### 3. Full-model finetuning

Use the native training path when you want all model weights updated:

```bash
cd ../SelfConditionedDenoisingAtoms
python train.py --conf configs/my_finetune.yaml --load-hf ct-scd-pcq --job-id my_run
```

or

```bash
cd ../SelfConditionedDenoisingAtoms
python train.py --conf configs/my_finetune.yaml --load-hf ct-scd-amp --job-id my_run
```

Start from:

- `templates/finetune_config.template.yaml`
- `configs/finetune_qm9.yaml` for public molecular examples

For public materials finetuning, do not rely on `configs/finetune_matbench.yaml` unless the private `StructureCloud` dependency is available. Copy the template and configure the periodic settings yourself.

### 4. Pretraining from scratch

Use the native training path:

```bash
cd ../SelfConditionedDenoisingAtoms
python train.py --conf configs/my_pretrain.yaml --job-id my_pretrain
```

Start from:

- `templates/pretrain_config.template.yaml`
- `configs/pretrain_pcq.yaml` for molecules
- `configs/pretrain_amp20.yaml` for periodic materials

## Dataset Onboarding

When adding a new dataset class under `SelfConditionedDenoisingAtoms/data/datasets/`:

1. Start from `templates/dataset_template.py`.
2. Preserve the constructor signature `__init__(root, dataset_arg=None, transform=None, **kwargs)`.
3. Return `torch_geometric.data.Data` objects with the required fields for the training mode.
4. Export the dataset from `data/datasets/__init__.py`, or `train.py --dataset ...` will reject it.
5. Add a new config file and smoke-test the run with a short `--num-steps` override.

## Constraints

- `train.py` always creates a `WandbLogger`.
- `train.py` is configured with `accelerator="gpu"`.
- `noise_in_loader: true` is required for periodic materials and is the fallback when the optional TorchMD CUDA extension is not built.
- Public repo examples cover `PCQM4MV2`, `AlexMP20`, `QM9`, `MD17`, and `OMOL25` best.
- The public checkpoints are downloaded as `last.ckpt` files from Hugging Face.
- The upstream pretraining path freezes the scalar head, so `reset_head: true` is a reasonable downstream default when switching to a new scalar property. That is an inference from the code path, not a documented upstream requirement.
