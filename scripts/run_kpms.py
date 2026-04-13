#!/usr/bin/env python3
"""Run a keypoint-moseq analysis on an external DLC / pose-estimation project.

This script accepts an *external* project directory (which lives outside this
repository) and writes all outputs into ``results/<project_name>/`` inside
this repository.

Usage examples
--------------
Basic run with raw pose data (default):

    python scripts/run_kpms.py \\
        --project-path /path/to/ElevatedMazeFood-Atanu-2026-04-04

Use filtered pose data instead:

    python scripts/run_kpms.py \\
        --project-path /path/to/ElevatedMazeFood-Atanu-2026-04-04 \\
        --use-filtered

Supply a custom config file:

    python scripts/run_kpms.py \\
        --project-path /path/to/project \\
        --config configs/my_project.yml

Run only the data-preparation step (no model fitting):

    python scripts/run_kpms.py \\
        --project-path /path/to/project \\
        --steps prepare

Run all steps explicitly:

    python scripts/run_kpms.py \\
        --project-path /path/to/project \\
        --steps prepare fit export

Pipeline steps
--------------
1. **prepare** – Validate external project layout, call
   ``kpms.setup_project()``, load keypoints, format data, and fit PCA.
2. **fit** – Initialise and fit the AR-HMM + keypoint-SLDS model.
3. **export** – Extract per-recording syllable sequences and save as CSV.

TODO notes
----------
* ``kpms.setup_project`` can optionally accept a DLC ``config.yml`` path via
  ``deeplabcut_config=``.  If your project has one, pass ``--dlc-config``.
* The ``ar_iters`` parameter (AR-only pre-training phase) is passed via the
  config.  If it is absent from your config the model trains the full
  keypoint-SLDS from the start.
* ``kpms.apply_model`` (not ``fit_model``) should be used when applying an
  already-trained model to new recordings.  This script uses ``fit_model``
  which trains from scratch every time.
* Verify the ``pose_estimation_format`` value in your config matches your
  tracker.  See ``keypoint_moseq.io.load_keypoints`` for supported formats.
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running this script from any working directory by adding the repo
# root to sys.path so that ``kpms_utils`` can be imported.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kpms_utils.path_utils import (
    get_project_name,
    get_video_dir,
    get_pose_data_dir,
    get_results_dir,
    get_kpms_project_dir,
    get_log_path,
)
from kpms_utils.config_utils import load_yaml_config, merge_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> logging.Logger:
    """Configure a logger that writes to both *stdout* and a log file.

    Parameters
    ----------
    log_path : Path
        Destination file for log output.

    Returns
    -------
    logging.Logger
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run keypoint-moseq on an external pose-estimation project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project-path",
        required=True,
        metavar="PATH",
        help=(
            "Absolute path to the external DLC/pose-estimation project. "
            "Must contain videos/, raw_pose_data/, and filtered_pose_data/."
        ),
    )
    parser.add_argument(
        "--use-filtered",
        action="store_true",
        default=False,
        help=(
            "Load pose data from filtered_pose_data/ instead of "
            "raw_pose_data/ (default)."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="CONFIG_YML",
        default=str(_REPO_ROOT / "configs" / "config_example.yml"),
        help=(
            "Path to a YAML config file (relative to repo root or absolute). "
            "Defaults to configs/config_example.yml."
        ),
    )
    parser.add_argument(
        "--dlc-config",
        metavar="DLC_CONFIG_YML",
        default=None,
        help=(
            "Optional path to the DLC project config.yml.  When provided, "
            "keypoint-moseq will read bodyparts and skeleton from it."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["prepare", "fit", "export"],
        default=["prepare", "fit", "export"],
        help=(
            "Pipeline steps to run.  Defaults to all three: "
            "prepare fit export."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Name for the model checkpoint directory inside the KPMS project. "
            "If omitted, keypoint-moseq generates a timestamp-based name."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate the external project and configuration but do not "
            "execute prepare/fit/export steps (useful for CI and checks)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Step 1: Prepare
# ---------------------------------------------------------------------------

def step_prepare(
    project_path: Path,
    kpms_project_dir: Path,
    pose_data_dir: Path,
    config: dict,
    dlc_config: str | None,
    logger: logging.Logger,
) -> tuple:
    """Set up the KPMS project, load keypoints, format data, and fit PCA.

    Parameters
    ----------
    project_path : Path
        External DLC project directory.
    kpms_project_dir : Path
        Output directory for the KPMS project (inside this repo).
    pose_data_dir : Path
        Directory containing pose-estimation files.
    config : dict
        Merged configuration dictionary.
    dlc_config : str or None
        Optional path to the DLC config.yml.
    logger : logging.Logger

    Returns
    -------
    tuple
        ``(data, metadata, pca, kpms_config)`` ready for fitting.
    """
    import keypoint_moseq as kpms  # noqa: PLC0415 – deferred to allow dry-runs

    fmt = config.get("pose_estimation_format", "deeplabcut")
    ext = config.get("pose_file_extension") or None
    recursive = config.get("recursive_search", True)

    # ------------------------------------------------------------------
    # 1a. Create the KPMS project directory and generate config.yml
    # ------------------------------------------------------------------
    logger.info("Setting up KPMS project at: %s", kpms_project_dir)

    # Build keyword overrides from our config to pass to setup_project.
    # TODO: Extend this dict with any additional options you want to seed
    #       into the generated config.yml.
    setup_kwargs: dict = {}
    for key in ("bodyparts", "use_bodyparts", "skeleton", "anterior_bodyparts", "posterior_bodyparts", "fps"):
        if key in config:
            setup_kwargs[key] = config[key]

    kpms.setup_project(
        str(kpms_project_dir),
        deeplabcut_config=dlc_config,
        overwrite=False,
        **setup_kwargs,
    )

    # Load the generated config (merge any additional options from our YAML).
    kpms_config = kpms.load_config(str(kpms_project_dir))

    # ------------------------------------------------------------------
    # 1b. Load keypoints
    # ------------------------------------------------------------------
    # Build a glob pattern that captures all files inside pose_data_dir.
    # keypoint_moseq.load_keypoints supports a directory path directly.
    pose_pattern = str(pose_data_dir)
    logger.info(
        "Loading keypoints from: %s (format=%s, recursive=%s)",
        pose_pattern,
        fmt,
        recursive,
    )

    coordinates, confidences, bodyparts = kpms.load_keypoints(
        pose_pattern,
        format=fmt,
        extension=ext,
        recursive=recursive,
    )
    logger.info("Loaded %d recording(s).", len(coordinates))

    if not coordinates:
        raise RuntimeError(
            f"No pose-estimation files found in {pose_data_dir}.\n"
            "Check that pose_estimation_format and pose_file_extension are "
            "correct in your config."
        )

    # ------------------------------------------------------------------
    # 1c. Format data for inference
    # ------------------------------------------------------------------
    logger.info("Formatting data for keypoint-moseq inference.")
    # TODO: format_data is in keypoint_moseq.util and re-exported from the
    #       top-level package.  Verify that this call is correct.
    data, metadata = kpms.format_data(
        coordinates,
        confidences,
        **kpms_config,
    )

    # ------------------------------------------------------------------
    # 1d. Fit PCA
    # ------------------------------------------------------------------
    logger.info("Fitting PCA (latent_dim=%s).", kpms_config.get("latent_dim"))
    pca = kpms.fit_pca(**data, **kpms_config)
    kpms.save_pca(pca, str(kpms_project_dir))
    logger.info("PCA saved.")

    return data, metadata, pca, kpms_config


# ---------------------------------------------------------------------------
# Step 2: Fit
# ---------------------------------------------------------------------------

def step_fit(
    data: dict,
    metadata: tuple,
    pca,
    kpms_project_dir: Path,
    kpms_config: dict,
    config: dict,
    model_name: str | None,
    logger: logging.Logger,
) -> tuple:
    """Initialise and fit the keypoint-SLDS model.

    Parameters
    ----------
    data : dict
        Formatted data from :func:`step_prepare`.
    metadata : tuple
        Recording keys and frame-boundary arrays.
    pca : sklearn PCA object
        Fitted PCA model.
    kpms_project_dir : Path
        KPMS project directory.
    kpms_config : dict
        Config loaded from the KPMS project.
    config : dict
        Merged configuration dictionary (workspace config.yml).
    model_name : str or None
        Optional model name override.
    logger : logging.Logger

    Returns
    -------
    tuple
        ``(model, model_name)``
    """
    import keypoint_moseq as kpms  # noqa: PLC0415

    num_iters: int = config.get("num_iters", 200)
    ar_iters: int = config.get("ar_iters", 50)
    save_every: int | None = config.get("save_every_n_iters", 25)

    # ------------------------------------------------------------------
    # 2a. Initialise model
    # ------------------------------------------------------------------
    logger.info("Initialising model (num_states=%s).", kpms_config.get("num_states"))
    model = kpms.init_model(data, pca=pca, **kpms_config)

    # ------------------------------------------------------------------
    # 2b. AR-only pre-training phase (optional but recommended)
    # ------------------------------------------------------------------
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
        logger.info("AR-only phase complete.  Model name: %s", model_name)

    # ------------------------------------------------------------------
    # 2c. Full keypoint-SLDS fitting
    # ------------------------------------------------------------------
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
    logger.info("Model fitting complete.  Model name: %s", model_name)

    return model, model_name


# ---------------------------------------------------------------------------
# Step 3: Export
# ---------------------------------------------------------------------------

def step_export(
    model: dict,
    metadata: tuple,
    kpms_project_dir: Path,
    model_name: str,
    results_dir: Path,
    logger: logging.Logger,
) -> None:
    """Extract model results and save syllable CSVs.

    Results are written to:
    - ``<kpms_project_dir>/<model_name>/results.h5``  (raw HDF5)
    - ``<results_dir>/syllables/<recording_name>.csv``  (per-recording CSVs)

    Parameters
    ----------
    model : dict
        Fitted model dictionary.
    metadata : tuple
        Recording keys and frame-boundary arrays.
    kpms_project_dir : Path
        KPMS project directory.
    model_name : str
        Name of the fitted model.
    results_dir : Path
        Top-level results directory for this project (inside the repo).
    logger : logging.Logger
    """
    import keypoint_moseq as kpms  # noqa: PLC0415

    # ------------------------------------------------------------------
    # 3a. Extract and save results.h5
    # ------------------------------------------------------------------
    logger.info("Extracting model results.")
    results = kpms.extract_results(
        model,
        metadata,
        project_dir=str(kpms_project_dir),
        model_name=model_name,
        save_results=True,
    )

    # ------------------------------------------------------------------
    # 3b. Save per-recording CSVs into results/<project_name>/syllables/
    # ------------------------------------------------------------------
    syllables_dir = results_dir / "syllables"
    syllables_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving syllable CSVs to: %s", syllables_dir)

    # TODO: kpms.save_results_as_csv writes one CSV per recording with
    #       columns: syllable, centroid x, centroid y, heading, latent_state*.
    #       The ``export_syllables.py`` script provides a trimmed version
    #       (video_name, frame, syllable) for downstream analysis.
    kpms.save_results_as_csv(
        results,
        save_dir=str(syllables_dir),
    )
    logger.info("Export complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the requested pipeline steps."""
    parser = _build_parser()
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    use_filtered: bool = args.use_filtered
    steps: list[str] = args.steps
    model_name: str | None = args.model_name

    # ------------------------------------------------------------------
    # Resolve directories
    # ------------------------------------------------------------------
    kpms_project_dir = get_kpms_project_dir(project_path)
    results_dir = get_results_dir(project_path)
    pose_data_dir = get_pose_data_dir(project_path, use_filtered=use_filtered)
    log_path = get_log_path(project_path)

    # ------------------------------------------------------------------
    # Set up logging
    # ------------------------------------------------------------------
    logger = _setup_logging(log_path)
    logger.info("=" * 60)
    logger.info("Project     : %s", project_path)
    logger.info("Pose data   : %s", pose_data_dir)
    logger.info("KPMS dir    : %s", kpms_project_dir)
    logger.info("Results dir : %s", results_dir)
    logger.info("Steps       : %s", steps)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Validate external project layout
    # ------------------------------------------------------------------
    if not project_path.is_dir():
        logger.error("Project path does not exist: %s", project_path)
        sys.exit(1)

    if not pose_data_dir.is_dir():
        logger.error("Pose data directory not found: %s", pose_data_dir)
        sys.exit(1)

    video_dir = get_video_dir(project_path)
    if not video_dir.is_dir():
        logger.warning(
            "videos/ directory not found at %s. "
            "Continuing without video validation.",
            video_dir,
        )

    # ------------------------------------------------------------------
    # Load workspace config
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    logger.info("Loading workspace config from: %s", config_path)
    config = load_yaml_config(config_path)

    # ------------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dry-run: validate configuration and paths without running heavy work
    # ------------------------------------------------------------------
    if getattr(args, "dry_run", False):
        logger.info("Dry-run enabled: validation complete. No steps will be executed.")
        logger.info("Requested steps: %s", steps)
        logger.info("KPMS project dir: %s", kpms_project_dir)
        logger.info("Pose data dir: %s", pose_data_dir)
        logger.info("Results dir: %s", results_dir)
        logger.info("Exiting due to --dry-run.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Run pipeline steps
    # ------------------------------------------------------------------
    data = metadata = pca = kpms_config = model = None

    if "prepare" in steps:
        data, metadata, pca, kpms_config = step_prepare(
            project_path=project_path,
            kpms_project_dir=kpms_project_dir,
            pose_data_dir=pose_data_dir,
            config=config,
            dlc_config=args.dlc_config,
            logger=logger,
        )

    if "fit" in steps:
        if data is None or kpms_config is None:
            # Reload from disk when skipping prepare
            import keypoint_moseq as kpms  # noqa: PLC0415
            kpms_config = kpms.load_config(str(kpms_project_dir))
            pca = kpms.load_pca(str(kpms_project_dir))
            logger.warning(
                "Step 'prepare' was skipped; loading PCA from disk. "
                "Make sure pose data was formatted in a previous run."
            )
            # TODO: data and metadata are not persisted between runs in this
            #       minimal implementation.  If you skip 'prepare', ensure you
            #       have a checkpoint to resume from and use apply_model
            #       instead of fit_model.
            logger.error(
                "Cannot run 'fit' without 'prepare' in this run "
                "(data/metadata not available).  Add 'prepare' to --steps."
            )
            sys.exit(1)

        model, model_name = step_fit(
            data=data,
            metadata=metadata,
            pca=pca,
            kpms_project_dir=kpms_project_dir,
            kpms_config=kpms_config,
            config=config,
            model_name=model_name,
            logger=logger,
        )

    if "export" in steps:
        if model is None or model_name is None:
            logger.error(
                "Cannot run 'export' without 'fit' in this run "
                "(model not available).  Add 'fit' to --steps."
            )
            sys.exit(1)

        step_export(
            model=model,
            metadata=metadata,
            kpms_project_dir=kpms_project_dir,
            model_name=model_name,
            results_dir=results_dir,
            logger=logger,
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()
