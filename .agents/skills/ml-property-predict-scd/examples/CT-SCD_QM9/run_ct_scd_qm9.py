import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


QM9_PROPERTIES = [
    "alpha",
    "homo",
    "lumo",
    "gap",
    "zpve",
    "cv",
    "mu",
    "R2",
    "u0",
    "u298",
    "h298",
    "g298",
    "u0_atom",
    "u298_atom",
    "h298_atom",
    "g298_atom",
]
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


def has_compiled_torchmd_extension(repo_root):
    extension_root = repo_root / "models" / "ET_models"
    patterns = ("*.so", "*.pyd")
    for pattern in patterns:
        if list(extension_root.rglob(pattern)):
            return True
    return False


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
        description="Run the CT-SCD QM9 finetuning example with the ct-scd-pcq checkpoint."
    )
    parser.add_argument("--repo-root", default=None, help="Path to SelfConditionedDenoisingAtoms.")
    parser.add_argument("--config", default="configs/finetune_qm9.yaml", help="Run config relative to the SCD repo.")
    parser.add_argument("--property", default="homo", choices=QM9_PROPERTIES, help="QM9 target property.")
    parser.add_argument("--checkpoint", default="ct-scd-pcq", help="Checkpoint name passed to --load-hf.")
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
    print(f"Restarting CT-SCD_QM9 example in {target_env} environment...", flush=True)
    subprocess.run(["conda", "run", "-n", target_env, "python", __file__, *sys.argv[1:]], check=True)
    raise SystemExit(0)


def main():
    args = build_parser().parse_args()
    maybe_restart_in_conda_env(args.conda_env)

    repo_root = resolve_repo_root(args.repo_root)
    config_path = resolve_config_path(repo_root, args.config)
    use_loader_graphs = not has_compiled_torchmd_extension(repo_root)

    job_id = args.job_id
    if job_id is None:
        job_id = f"CT-SCD_QM9_{args.property}"
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
        "--dataset-arg",
        args.property,
    ]

    if use_loader_graphs:
        cmd.extend(["--noise_in_loader", "True"])

    if not args.full_run:
        cmd.extend(["--num-steps", str(args.num_steps), "--val-interval", str(args.val_interval)])

    print(f"Using conda environment: {args.conda_env}")
    print(f"Running CT-SCD QM9 command from {repo_root}:")
    print(shlex.join(cmd))
    if args.dry_run:
        print("Dry run only; training command was not launched.")
        return
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
