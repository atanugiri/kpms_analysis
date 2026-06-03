#!/usr/bin/env python3
"""Train or resume the keypoint-moseq model using prepared data and PCA.

This is step 5 in the pipeline. It expects the KPMS project to contain:
- formatted_data.pkl (or expects you to provide data via --formatted)
- PCA stored via kpms.save_pca in the KPMS project
- metadata.pkl produced by 04_fit_pca.py

Usage
-----
python scripts/05_fit_model.py --project-path /path/to/project --config configs/my.yml [--model-name name] [--jax-platform gpu]
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kpms_utils.logging_utils import setup_logging
from kpms_utils.config_utils import load_yaml_config
from kpms_utils.path_utils import get_kpms_project_dir, get_log_path


def configure_jax(platform: str) -> None:
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        if platform and platform != "auto":
            jax.config.update("jax_platform_name", platform)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit or resume keypoint-moseq model.")
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "config_example.yml"))
    parser.add_argument("--formatted", default=None, help="Path to formatted_data.pkl")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--resume-model-name", default=None)
    parser.add_argument("--resume-iteration", type=int, default=None)
    parser.add_argument("--continue-iters", type=int, default=None)
    parser.add_argument("--resume-ar-only", action="store_true", default=False)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    log_path = get_log_path(project_path)
    logger = setup_logging(log_path)

    logger.info("Fit model: project=%s", project_path)

    if not project_path.is_dir():
        logger.error("Project path does not exist: %s", project_path)
        raise SystemExit(1)

    config = load_yaml_config(str(config_path))
    kpms_project_dir = get_kpms_project_dir(project_path)

    import keypoint_moseq as kpms  # noqa: PLC0415

    configure_jax(args.jax_platform)

    # Load formatted data
    data = None
    metadata = None
    if args.formatted:
        p = Path(args.formatted)
        if p.exists():
            with open(p, "rb") as f:
                obj = pickle.load(f)
                data = obj.get("data")
                metadata = obj.get("metadata")
            logger.info("Loaded formatted data from: %s", p)

    default_formatted = kpms_project_dir / "formatted_data.pkl"
    if data is None and default_formatted.exists():
        with open(default_formatted, "rb") as f:
            obj = pickle.load(f)
            data = obj.get("data")
            metadata = obj.get("metadata")
        logger.info("Loaded formatted data from: %s", default_formatted)

    if data is None:
        logger.error("Formatted data not found. Run 02_load_and_preprocess.py --format or 04_fit_pca.py first.")
        raise SystemExit(1)

    # Load PCA
    pca = kpms.load_pca(str(kpms_project_dir))
    kpms_config = kpms.load_config(str(kpms_project_dir))

    # Initialise model
    logger.info("Initialising model (num_states=%s).", kpms_config.get("num_states"))
    model = kpms.init_model(data, pca=pca, **kpms_config)

    # Prefer values from the KPMS project config if present; otherwise fall back
    # to the CLI/config file or hardcoded defaults.
    num_iters = kpms_config.get("num_iters", config.get("num_iters", 200))
    ar_iters = kpms_config.get("ar_iters", config.get("ar_iters", 50))
    save_every = kpms_config.get("save_every_n_iters", config.get("save_every_n_iters", 25))

    logger.info("Using num_iters=%s (project config preferred if present)", num_iters)
    logger.info("Using ar_iters=%s (project config preferred if present)", ar_iters)
    logger.info("Using save_every_n_iters=%s (project config preferred if present)", save_every)

    model_name = args.model_name

    # AR-only pretraining
    if ar_iters > 0:
        logger.info("AR-only pre-training for %d iterations.", ar_iters)
        model, model_name = kpms.fit_model(
            model,
            data,
            metadata,
            project_dir=str(kpms_project_dir),
            model_name=model_name,
            ar_only=True,
            num_iters=ar_iters,
            save_every_n_iters=save_every,
        )
        logger.info("AR-only phase complete. Model name: %s", model_name)

    # Full fitting
    logger.info("Fitting full keypoint-SLDS for %d iterations.", num_iters)
    model, model_name = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir=str(kpms_project_dir),
        model_name=model_name,
        ar_only=False,
        num_iters=num_iters,
        save_every_n_iters=save_every,
    )
    logger.info("Model fitting complete. Model name: %s", model_name)


if __name__ == "__main__":
    main()
