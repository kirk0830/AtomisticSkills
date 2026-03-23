# Config Recipes

Use the nearest public config as the base case, then change only the fields required by the new dataset and task.

## Molecular SCD pretraining

Closest public baseline:
`configs/pretrain_pcq.yaml`

Expected settings:

- `pretraining: true`
- `self_cond: true`
- `dataset: <new_dataset_name>`
- `energy_weight: 0.0`
- `force_weight: 1.0`
- `denoising_weight: 1.0`
- `noise_scale: 0.04`
- `allow_periodic: false`

Graph mode choice:

- compiled extension available:
  `noise_in_loader: false`
- no compiled extension:
  `noise_in_loader: true`

## Periodic materials SCD pretraining

Closest public baseline:
`configs/pretrain_amp20.yaml`

Expected settings:

- `pretraining: true`
- `self_cond: true`
- `allow_periodic: true`
- `noise_in_loader: true`
- `graph_cutoff: 5.0`
- `max_neighbors: 32`
- optional cell augmentation:
  `p_cell_repeat`, `cell_repeat_iters`, `rep_min_atoms`

## Molecular full-model finetuning

Closest public baseline:
`configs/finetune_qm9.yaml`

Expected settings:

- `pretraining: false`
- `self_cond: false`
- `dataset: <new_dataset_name>`
- `standardize: true` for most scalar regression targets
- choose checkpoint:
  `--load-hf ct-scd-pcq`
- `allow_periodic: false`
- `noise_in_loader: false` if the TorchMD extension is built, otherwise `true`

Reasonable downstream defaults:

- `reset_head: true`
- `reset_embeddings: false`

Those reset recommendations are practical defaults inferred from the code path, not explicit upstream documentation.

## Materials full-model finetuning

Do not treat `configs/finetune_matbench.yaml` as the public baseline unless `StructureCloud` is available.

Instead:

1. start from `templates/finetune_config.template.yaml`
2. set `allow_periodic: true`
3. set `noise_in_loader: true`
4. set `graph_cutoff`, `max_neighbors`, and any cell-repeat augmentation fields
5. load the materials checkpoint with `--load-hf ct-scd-amp`

## Force-aware finetuning

Patterns exist in `models/trainer.py`, `data/datasets/md17.py`, and `data/datasets/omol25.py`.

Key switches:

- `derivative: true`
  predict energy and derive forces
- `direct_force_pred: true`
  predict forces directly from the denoising head

Only enable one intentionally.

## Batch clipping

If large systems cause OOM:

```yaml
max_nodes_per_batch: 4000
batch_clipper_cache_size: 1000
allow_test_clipping: false
```

Prefer disabling test clipping for strict evaluation comparability.

## Good first-run commands

### New molecular pretraining config

```bash
python train.py --conf configs/my_pretrain.yaml --job-id smoke_pretrain --num-steps 2000 --val-interval 1
```

### New molecular finetuning config from a public checkpoint

```bash
python train.py --conf configs/my_finetune.yaml --load-hf ct-scd-pcq --job-id smoke_ft --num-steps 2000 --val-interval 1
```

### New materials finetuning config from a public checkpoint

```bash
python train.py --conf configs/my_finetune.yaml --load-hf ct-scd-amp --job-id smoke_ft --num-steps 2000 --val-interval 1
```

Put overrides after `--conf`.
