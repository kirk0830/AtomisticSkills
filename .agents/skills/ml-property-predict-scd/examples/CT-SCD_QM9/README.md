# CT-SCD_QM9 Tutorial

This directory contains a small, practical example showing how to fine-tune the pretrained `ct-scd-pcq` checkpoint on the QM9 `homo` property using the native `SelfConditionedDenoisingAtoms/train.py` entrypoint and the upstream `configs/finetune_qm9.yaml` recipe.

The example follows the public `SelfConditionedDenoisingAtoms/README.md` finetuning instructions:

```bash
python train.py --conf configs/finetune_qm9.yaml --load-hf ct-scd-pcq --job-id scd-pcq_qm9-homo
```

## Contents

1. `run_ct_scd_qm9.py`: a wrapper script that locates the `SelfConditionedDenoisingAtoms` checkout, relaunches itself inside the `scd-agent` environment when needed, and runs a safe smoke test by default.

## Running the Example

From a general environment, let the script restart itself inside `scd-agent`:

```bash
python run_ct_scd_qm9.py --dry-run
```

Inside `scd-agent`, you can run it directly:

```bash
python run_ct_scd_qm9.py
```

Use `--dry-run` first to verify the resolved command, repo root, and conda environment without launching training:

```bash
python run_ct_scd_qm9.py --dry-run --property gap
```

By default this example performs a short smoke test with `--num-steps 100`. To launch the full upstream schedule instead:

```bash
python run_ct_scd_qm9.py --full-run
```

## Other QM9 Properties

The script exposes `--property`, which maps directly to `--dataset-arg` for the QM9 loader. For example:

```bash
python run_ct_scd_qm9.py --property lumo
python run_ct_scd_qm9.py --property gap
python run_ct_scd_qm9.py --property cv
```

Common QM9 property names from `data/datasets/qm9.py` include:

- `alpha`
- `homo`
- `lumo`
- `gap`
- `zpve`
- `cv`
- `mu`
- `R2`
- `u0`
- `u298`
- `h298`
- `g298`
- `u0_atom`
- `u298_atom`
- `h298_atom`
- `g298_atom`

For most QM9 targets, changing `--property` is the main change. For targets with different normalization or prior behavior, inspect `SelfConditionedDenoisingAtoms/data/datasets/qm9.py` and consider copying `configs/finetune_qm9.yaml` into a new task-specific config YAML before a full run.

If you create a task-specific config, pass it with `--config`:

```bash
python run_ct_scd_qm9.py --property lumo --config configs/finetune_qm9_lumo.yaml
```

## Notes

- If the TorchMD compiled graph kernel is not built, the wrapper automatically adds `--noise_in_loader True`, matching the upstream README guidance.
- The default target environment is `scd-agent`, matching the environment created in `AtomisticSkills/conda-envs/scd-agent`.
- Training outputs are written under `SelfConditionedDenoisingAtoms/experiments/<job_id>`.
- The W&B project for this example is derived by `train.py` from `dataset: QM9`, so it appears as `SCD_bench_QM9`.
