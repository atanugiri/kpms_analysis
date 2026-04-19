#!/usr/bin/env python3
"""Convenience CLI: fit/resume model fitting via scripts/run_kpms.py.

Typical usage patterns
----------------------
1) First time training (after prepare):

    python scripts/fit_kpms.py --project-path /path/to/project --config configs/my.yml

2) Resume from a checkpoint and run additional iterations:

    python scripts/fit_kpms.py \
        --project-path /path/to/project \
        --resume-model-name 2026_04_13-15_40_25 \
        --continue-iters 200 \
        --kappa 1e4
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    run_script = repo_root / "scripts" / "run_kpms.py"

    parser = argparse.ArgumentParser(description="Run kpms fit step (AR-only + full) or resume from checkpoint.")
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--config", default=str(repo_root / "configs" / "config_example.yml"))
    parser.add_argument("--model-name", default=None)

    parser.add_argument("--resume-model-name", default=None)
    parser.add_argument("--resume-iteration", type=int, default=None)
    parser.add_argument("--continue-iters", type=int, default=None)
    parser.add_argument("--resume-ar-only", action="store_true", default=False)
    parser.add_argument("--kappa", type=float, default=None)

    parser.add_argument("--jax-platform", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(run_script),
        "--project-path",
        args.project_path,
        "--config",
        args.config,
        "--steps",
        "fit",
        "--jax-platform",
        args.jax_platform,
    ]

    if args.model_name:
        cmd.extend(["--model-name", args.model_name])

    if args.resume_model_name:
        cmd.extend(["--resume-model-name", args.resume_model_name])
    if args.resume_iteration is not None:
        cmd.extend(["--resume-iteration", str(args.resume_iteration)])
    if args.continue_iters is not None:
        cmd.extend(["--continue-iters", str(args.continue_iters)])
    if args.resume_ar_only:
        cmd.append("--resume-ar-only")
    if args.kappa is not None:
        cmd.extend(["--kappa", str(args.kappa)])

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
