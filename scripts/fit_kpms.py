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

# Ensure repo is in path before importing kpms_utils
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kpms_utils import build_cmd_list


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

    base_cmd = [
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
    
    cmd = build_cmd_list(base_cmd, args)

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
