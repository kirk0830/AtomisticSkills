import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_CONDA_ENV = "scd-agent"


def resolve_repo_root(user_value=None):
    candidates = []
    if user_value is not None:
        candidates.append(Path(user_value).expanduser())

    cwd = Path.cwd()
    this_file = Path(__file__).resolve()
    candidates.extend(
        [
            cwd,
            cwd / "SelfConditionedDenoisingAtoms",
            this_file.parents[6] / "SelfConditionedDenoisingAtoms",
            this_file.parents[5].parent / "SelfConditionedDenoisingAtoms",
        ]
    )

    for candidate in candidates:
        if (candidate / "train.py").exists() and (candidate / "configs").exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate SelfConditionedDenoisingAtoms. Pass --repo-root explicitly."
    )


def resolve_config_path(repo_root, config_value):
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    if config_path.is_relative_to(repo_root):
        return str(config_path.relative_to(repo_root))
    return str(config_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the CT-SCD Matbench MBgap finetuning example with the ct-scd-amp checkpoint."
    )
    parser.add_argument("--repo-root", default=None, help="Path to SelfConditionedDenoisingAtoms.")
    parser.add_argument("--config", default="configs/finetune_matbench.yaml", help="Run config relative to the SCD repo.")
    parser.add_argument("--dataset-class", default="MBgap", help="Dataset class passed to --dataset.")
    parser.add_argument("--fold", type=int, default=0, help="Matbench fold index passed through --dataset-arg.")
    parser.add_argument("--checkpoint", default="ct-scd-amp", help="Checkpoint name passed to --load-hf.")
    parser.add_argument("--job-id", default=None, help="Optional explicit job id.")
    parser.add_argument(
        "--conda-env",
        default=os.environ.get("SCD_EXAMPLE_CONDA_ENV", DEFAULT_CONDA_ENV),
        help="Conda environment used to run SelfConditionedDenoisingAtoms.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training command without launching it.")
    parser.add_argument("--full-run", action="store_true", help="Run the full upstream schedule instead of a smoke test.")
    parser.add_argument("--num-steps", type=int, default=100, help="Smoke-test max steps when --full-run is not used.")
    parser.add_argument("--val-interval", type=int, default=1, help="Validation interval for smoke tests.")
    return parser


def maybe_restart_in_conda_env(target_env):
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env == target_env:
        return
    print(f"Restarting CT-SCD_matbench example in {target_env} environment...", flush=True)
    subprocess.run(["conda", "run", "-n", target_env, "python", __file__, *sys.argv[1:]], check=True)
    raise SystemExit(0)


def main():
    args = build_parser().parse_args()
    maybe_restart_in_conda_env(args.conda_env)

    repo_root = resolve_repo_root(args.repo_root)
    config_path = resolve_config_path(repo_root, args.config)

    job_id = args.job_id
    if job_id is None:
        job_id = f"CT-SCD_matbench_{args.dataset_class}_fold{args.fold}"
        if not args.full_run:
            job_id += "_smoke"

    cmd = [
        "python",
        "train.py",
        "--conf",
        config_path,
        "--load-hf",
        args.checkpoint,
        "--job-id",
        job_id,
        "--dataset",
        args.dataset_class,
        "--dataset-arg",
        str(args.fold),
    ]

    if not args.full_run:
        cmd.extend(["--num-steps", str(args.num_steps), "--val-interval", str(args.val_interval)])

    print(f"Using conda environment: {args.conda_env}")
    print("Running CT-SCD Matbench command from", repo_root)
    print(shlex.join(cmd))
    print(
        "Note: this example assumes the StructureCloud-backed Matbench dataset wrapper is available "
        "in your SelfConditionedDenoisingAtoms checkout."
    )
    if args.dry_run:
        print("Dry run only; training command was not launched.")
        return
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
