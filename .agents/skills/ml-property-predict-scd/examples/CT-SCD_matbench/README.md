# CT-SCD_matbench Tutorial

This directory contains a small example showing how to fine-tune the pretrained `ct-scd-amp` checkpoint on the Matbench `MBgap` task using the native `SelfConditionedDenoisingAtoms/train.py` entrypoint and the upstream `configs/finetune_matbench.yaml` recipe.

This example mirrors the intended upstream flow:

```bash
python train.py --conf configs/finetune_matbench.yaml --load-hf ct-scd-amp --job-id CT-SCD_matbench_fold0
```

## Important Availability Note

The public `SelfConditionedDenoisingAtoms/README.md` explicitly notes that `configs/finetune_matbench.yaml` depends on the unreleased `StructureCloud` utilities. Treat this example as the correct pattern for checkouts where the Matbench dataset wrapper is available, not as a guaranteed public quickstart.

## Contents

1. `run_ct_scd_matbench.py`: a wrapper script that locates the `SelfConditionedDenoisingAtoms` checkout, relaunches itself inside the `scd-agent` environment when needed, and runs a safe smoke test by default.

## Running the Example

From a general environment, let the script restart itself inside `scd-agent`:

```bash
python run_ct_scd_matbench.py --dry-run
```

Inside `scd-agent`, you can run it directly:

```bash
python run_ct_scd_matbench.py
```

Use `--dry-run` first to confirm the resolved config, fold, and dataset wrapper without launching training:

```bash
python run_ct_scd_matbench.py --dry-run --fold 2
```

By default this example performs a short smoke test with `--num-steps 100`. To launch the full upstream schedule instead:

```bash
python run_ct_scd_matbench.py --full-run
```

The example uses Matbench fold `0` by default. To change folds:

```bash
python run_ct_scd_matbench.py --fold 1
python run_ct_scd_matbench.py --fold 2
```

## Other Matbench Properties

In the current upstream recipe, `finetune_matbench.yaml` is specifically wired for `dataset: MBgap`, and `dataset_arg` selects the fold rather than the property.

To train on another Matbench property:

1. create a sibling dataset wrapper in `SelfConditionedDenoisingAtoms/data/datasets/matbench.py` similar to `mbench_gap`, but pointing to another task from `MBDataset_base.avail_tasks`
2. export that new wrapper from `SelfConditionedDenoisingAtoms/data/datasets/__init__.py`
3. copy `configs/finetune_matbench.yaml` into a new task-specific config YAML
4. run this example script with `--dataset-class <NewWrapper>` and `--config <new_config.yaml>`

For example, once a new wrapper is exported:

```bash
python run_ct_scd_matbench.py --dataset-class MBdielectric --config configs/finetune_matbench_dielectric.yaml
```

## Notes

- This example follows the upstream material finetuning settings: `noise_in_loader=True`, `allow_periodic=True`, and `set_head_agg: mean`.
- The default target environment is `scd-agent`, matching the environment created in `AtomisticSkills/conda-envs/scd-agent`.
- Training outputs are written under `SelfConditionedDenoisingAtoms/experiments/<job_id>`.
- The W&B project is derived by `train.py` from the dataset class name, so the default `MBgap` run appears under `SCD_bench_MBgap`.
