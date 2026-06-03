#!/usr/bin/env python3
"""Create a KPMS project directory and generate config.yml.

This script mirrors the setup actions in the original pipeline but exposes them
as a standalone CLI step (step 1). It creates the `kpms_project` directory
inside `results/<project_name>/` and seeds it with settings from the workspace
config and optional DLC config.

Usage
-----
python scripts/01_setup_project.py --project-path /path/to/project --config configs/my.yml
"""

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kpms_utils.logging_utils import setup_logging
from kpms_utils.config_utils import load_yaml_config
from kpms_utils.path_utils import get_kpms_project_dir, get_log_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create KPMS project directory and generate config.yml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "config_example.yml"))
    parser.add_argument("--dlc-config", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    log_path = get_log_path(project_path)
    logger = setup_logging(log_path)

    logger.info("Setting up KPMS project for: %s", project_path)

    if not project_path.is_dir():
        logger.error("Project path does not exist: %s", project_path)
        raise SystemExit(1)

    config = load_yaml_config(config_path)

    kpms_project_dir = get_kpms_project_dir(project_path)

    logger.info("Creating KPMS project directory: %s", kpms_project_dir)

    import keypoint_moseq as kpms  # noqa: PLC0415

    # Build keyword overrides similar to run_kpms.step_prepare
    setup_kwargs = {}
    for key in (
        "bodyparts",
        "use_bodyparts",
        "skeleton",
        "anterior_bodyparts",
        "posterior_bodyparts",
        "fps",
        "latent_dim",
        "num_states",
        "kappa",
        "num_iters",
        "ar_iters",
        "save_every_n_iters",
    ):
        if key in config:
            setup_kwargs[key] = config[key]

    kpms.setup_project(
        str(kpms_project_dir),
        deeplabcut_config=args.dlc_config,
        overwrite=args.overwrite,
        **setup_kwargs,
    )

    logger.info("KPMS project setup complete. Config written to: %s", kpms_project_dir / "config.yml")


if __name__ == "__main__":
    main()
