#!/usr/bin/env python3
"""Convenience CLI: run only the 'prepare' stage of scripts/run_kpms.py.

This wrapper exists to mirror the tutorial's step-by-step workflow while
reusing the same underlying implementation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    run_script = repo_root / "scripts" / "run_kpms.py"

    parser = argparse.ArgumentParser(description="Run kpms prepare step (setup, load, format, PCA).")
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--config", default=str(repo_root / "configs" / "config_example.yml"))
    parser.add_argument("--use-filtered", action="store_true", default=False)
    parser.add_argument("--dlc-config", default=None)
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
        "prepare",
        "--jax-platform",
        args.jax_platform,
    ]
    if args.use_filtered:
        cmd.append("--use-filtered")
    if args.dlc_config:
        cmd.extend(["--dlc-config", args.dlc_config])

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
