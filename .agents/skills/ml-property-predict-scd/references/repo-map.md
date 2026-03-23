# Repo Map

## Entry points

- `train.py`
  The only training entrypoint. It parses YAML from `--conf`, loads public Hugging Face checkpoints through `--load-hf`, builds the datamodule, and launches Lightning training plus final test.
- `data/loaders.py`
  Instantiates datasets, applies molecule or periodic transforms, handles splitting, and computes normalization statistics when `standardize: true`.
- `models/trainer.py`
  Contains the actual pretraining and finetuning logic, denoising loss, force loss, and batch clipping.
- `models/model_helper.py`
  Loads model configs, resolves checkpoint loading, and applies reset options such as `reset_head`, `reset_embeddings`, and `reset_norms`.

## Public dataset patterns already available

- `data/datasets/pcqm4mv2.py`
  Unlabeled molecular pretraining dataset.
- `data/datasets/alexmp20.py`
  Periodic materials pretraining dataset.
- `data/datasets/qm9.py`
  Scalar molecular finetuning dataset with target selection through `dataset_arg`.
- `data/datasets/md17.py`
  Energy-plus-force dataset pattern exposing `y` and `dy`.
- `data/datasets/omol25.py`
  Another energy/force dataset pattern with periodic metadata.

## Config layering

There are two config layers:

1. run config passed to `train.py --conf ...`
2. model config referenced through the run config field `model_config`

`models/model_helper.py:create_model()` loads the model YAML and then lets overlapping run-config keys override model-config keys.

## Important parser behavior

`--conf` is handled by a custom argparse action.

- unknown YAML keys fail immediately
- later CLI arguments override values loaded from YAML
- overrides placed before `--conf` can be overwritten by the YAML

Use this ordering:

```bash
python train.py --conf configs/my_run.yaml --noise_in_loader True --job-id trial_1
```

## Graph path selection

- `allow_periodic: false` and `noise_in_loader: false`
  Uses the faster TorchMD extension path for non-periodic graphs when the extension is built.
- `noise_in_loader: true`
  Uses loader-side graph creation and corruption logic from `data/datasets/transforms.py`.
- `allow_periodic: true`
  Requires loader-side graph creation for periodic neighbor lists.

## Checkpoints

- public Hugging Face checkpoints:
  `ct-scd-pcq`
  `ct-scd-amp`
- local checkpoint loading:
  `--load-model <ckpt>`
- restart latest checkpoint from an existing run dir:
  `--restart True`

## Outputs

Each run writes into `log_dir/job_id`.

Typical contents:

- `input.yaml`
- `last.ckpt`
- epoch checkpoints named with step, epoch, and losses
- split indices as `splits.npz` when random splitting is used

## Current practical limitations

- `train.py` always constructs a `WandbLogger`.
- the trainer is configured with `accelerator="gpu"`.
- `configs/finetune_matbench.yaml` depends on unreleased `StructureCloud` utilities and should not be treated as a public baseline.
- new datasets are not usable until they are exported from `data/datasets/__init__.py`.
