#!/usr/bin/env python3
"""Fit PCA from preprocessed/formatted data and save PCA to KPMS project dir.

This script corresponds to step 4 in the proposed pipeline. It loads either
an existing formatted dataset (formatted_data.pkl) or the preprocessed
snapshot (preprocessed_data.pkl), runs `kpms.format_data` if needed, fits PCA
via `kpms.fit_pca(...)`, and saves the PCA object to the KPMS project.

Usage
-----
python scripts/04_fit_pca.py --project-path /path/to/project [--use-filtered] [--preprocessed path]
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
from kpms_utils.path_utils import get_kpms_project_dir, get_pose_data_dir, get_log_path


def configure_jax(platform: str) -> None:
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        if platform and platform != "auto":
            jax.config.update("jax_platform_name", platform)
    except Exception:
        # Let failures surface later when running heavy jobs
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PCA and save to KPMS project.")
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--use-filtered", action="store_true", default=False)
    parser.add_argument("--preprocessed", default=None, help="Path to preprocessed_data.pkl")
    parser.add_argument("--formatted", default=None, help="Path to formatted_data.pkl (contains data, metadata)")
    parser.add_argument("--jax-platform", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--plot", action="store_true", help="Generate and save PCA diagnostic plots to the KPMS project figures/ folder")
    parser.add_argument("--explained-variance", type=float, default=0.9, help="Threshold for dims to explain variance when printing" )
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()

    log_path = get_log_path(project_path)
    logger = setup_logging(log_path)

    logger.info("Fit PCA: project=%s", project_path)

    if not project_path.is_dir():
        logger.error("Project path does not exist: %s", project_path)
        raise SystemExit(1)

    kpms_project_dir = get_kpms_project_dir(project_path)

    import keypoint_moseq as kpms  # noqa: PLC0415

    # Configure JAX for PCA if requested
    configure_jax(args.jax_platform)

    # Try to load formatted data first
    data = None
    metadata = None
    if args.formatted:
        formatted_path = Path(args.formatted)
        if formatted_path.exists():
            with open(formatted_path, "rb") as f:
                obj = pickle.load(f)
                data = obj.get("data")
                metadata = obj.get("metadata")
            logger.info("Loaded formatted data from: %s", formatted_path)

    if data is None:
        # Try default formatted path in kpms_project_dir
        default_formatted = kpms_project_dir / "formatted_data.pkl"
        if default_formatted.exists():
            with open(default_formatted, "rb") as f:
                obj = pickle.load(f)
                data = obj.get("data")
                metadata = obj.get("metadata")
            logger.info("Loaded formatted data from: %s", default_formatted)

    if data is None:
        # Load preprocessed and run format_data
        preprocessed_path = None
        if args.preprocessed:
            p = Path(args.preprocessed)
            if p.exists():
                preprocessed_path = p
        else:
            candidate = kpms_project_dir / "preprocessed_data.pkl"
            if candidate.exists():
                preprocessed_path = candidate

        if not preprocessed_path:
            logger.error("No formatted or preprocessed data found. Run 02_load_and_preprocess.py first.")
            raise SystemExit(1)

        with open(preprocessed_path, "rb") as f:
            obj = pickle.load(f)
            coordinates = obj.get("coordinates")
            confidences = obj.get("confidences")
            bodyparts = obj.get("bodyparts")

        # Load kpms project config to pass options
        kpms_config = kpms.load_config(str(kpms_project_dir))
        logger.info("Formatting data with kpms.format_data()")
        data, metadata = kpms.format_data(coordinates, confidences, **kpms_config)

        # Save formatted
        formatted_path = kpms_project_dir / "formatted_data.pkl"
        with open(formatted_path, "wb") as f:
            pickle.dump({"data": data, "metadata": metadata}, f)
        logger.info("Saved formatted data to: %s", formatted_path)

    # Load kpms_config for PCA params
    kpms_config = kpms.load_config(str(kpms_project_dir))

    logger.info("Fitting PCA (latent_dim=%s)...", kpms_config.get("latent_dim"))
    pca = kpms.fit_pca(**data, **kpms_config)
    kpms.save_pca(pca, str(kpms_project_dir))
    logger.info("PCA saved to KPMS project.")

    # Also save metadata pickle for downstream steps
    meta_path = kpms_project_dir / "metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info("Saved metadata to: %s", meta_path)

    logger.info("04_fit_pca complete.")

    # Optional plotting/visualization
    if args.plot:
        fig_dir = kpms_project_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Printing number of components to reach %.2f explained variance", args.explained_variance)
        if hasattr(kpms, "print_dims_to_explain_variance"):
            try:
                kpms.print_dims_to_explain_variance(pca, args.explained_variance)
            except Exception as exc:  # pragma: no cover - plotting optional
                logger.warning("print_dims_to_explain_variance failed: %s", exc)
        else:
            logger.warning("kpms.print_dims_to_explain_variance not available in installed keypoint_moseq")

        logger.info("Generating scree plot and component visualizations (saved under %s)", fig_dir)
        if hasattr(kpms, "plot_scree"):
            try:
                kpms.plot_scree(pca, project_dir=str(kpms_project_dir))
            except Exception as exc:  # pragma: no cover - plotting optional
                logger.warning("plot_scree failed: %s", exc)
        else:
            logger.warning("kpms.plot_scree not available in installed keypoint_moseq")

        if hasattr(kpms, "plot_pcs"):
            try:
                kpms.plot_pcs(pca, project_dir=str(kpms_project_dir))
            except Exception as exc:  # pragma: no cover - plotting optional
                logger.warning("plot_pcs failed: %s", exc)
        else:
            logger.warning("kpms.plot_pcs not available in installed keypoint_moseq; skipping")


if __name__ == "__main__":
    main()
