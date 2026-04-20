#!/usr/bin/env python3
"""Export a frame-by-frame syllable time series from a fitted model.

For each recording the script writes a CSV file with three columns::

    video_name, frame, syllable

The CSVs are saved under ``results/<project_name>/syllable_timeseries/``.

This script is intended as a lightweight post-processing step that can be
run independently of the main ``run_kpms.py`` pipeline once a model has been
fitted and ``results.h5`` has been written to disk.

Usage examples
--------------
Export from the most-recent (or only) model checkpoint:

    python scripts/export_syllables.py \\
        --project-path /path/to/ElevatedMazeFood-Atanu-2026-04-04

Export from a specific model:

    python scripts/export_syllables.py \\
        --project-path /path/to/project \\
        --model-name 2026_04_04-12_00_00

Load results from an explicit HDF5 path:

    python scripts/export_syllables.py \\
        --project-path /path/to/project \\
        --results-h5 results/my_project/kpms_project/my_model/results.h5

# TODO notes
----------
* ``kpms.load_results`` expects the path
  ``<kpms_project_dir>/<model_name>/results.h5``.  If model discovery fails,
  supply the path explicitly via ``--results-h5``.
* Syllable indices start at 0 and are assigned by the model.  Re-indexing
  (e.g. sorting by frequency) can be done with
  ``keypoint_moseq.reindex_syllables_in_checkpoint``.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Allow running from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kpms_utils.path_utils import (
    get_project_name,
    get_results_dir,
    get_kpms_project_dir,
    get_log_path,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> logging.Logger:
    """Configure a logger writing to stdout and a file."""
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
        description=(
            "Export per-recording syllable time series (video_name, frame, "
            "syllable) from a fitted keypoint-moseq model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project-path",
        required=True,
        metavar="PATH",
        help="Absolute path to the external DLC project.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Name of the fitted model checkpoint directory inside the KPMS "
            "project.  If omitted, all ``results.h5`` files are discovered "
            "automatically."
        ),
    )
    parser.add_argument(
        "--results-h5",
        default=None,
        metavar="PATH",
        help=(
            "Explicit path to a ``results.h5`` file.  Overrides automatic "
            "discovery from --project-path and --model-name."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for the syllable CSVs.  Defaults to "
            "``results/<project_name>/syllable_timeseries/``."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=("List discovered results files and exit without writing CSVs."),
    )
    return parser


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def find_results_h5(kpms_project_dir: Path, model_name: str | None) -> list[Path]:
    """Locate ``results.h5`` files inside the KPMS project directory.

    Parameters
    ----------
    kpms_project_dir : Path
        KPMS project directory (contains model sub-directories).
    model_name : str or None
        If given, look only inside ``<kpms_project_dir>/<model_name>/``.

    Returns
    -------
    list[Path]
        Sorted list of found ``results.h5`` paths.
    """
    if model_name:
        candidate = kpms_project_dir / model_name / "results.h5"
        return [candidate] if candidate.exists() else []

    return sorted(kpms_project_dir.rglob("results.h5"))


def load_results(results_h5_path: Path) -> dict:
    """Load model results from a ``results.h5`` file.

    Parameters
    ----------
    results_h5_path : Path
        Path to the HDF5 results file.

    Returns
    -------
    dict
        Results dictionary with one entry per recording.  Each entry
        contains at least a ``'syllable'`` array of shape ``(T,)``.

    Raises
    ------
    FileNotFoundError
        If ``results_h5_path`` does not exist.
    """
    import keypoint_moseq as kpms  # noqa: PLC0415 – deferred import

    if not results_h5_path.exists():
        raise FileNotFoundError(f"results.h5 not found: {results_h5_path}")

    return kpms.load_results(path=str(results_h5_path))


def results_to_dataframe(recording_name: str, syllable_array) -> pd.DataFrame:
    """Convert a per-recording syllable array to a tidy DataFrame.

    Parameters
    ----------
    recording_name : str
        Identifier for this recording (used as ``video_name``).
    syllable_array : array-like of shape (T,)
        Frame-by-frame syllable index sequence.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``video_name``, ``frame``, ``syllable``.
    """
    import numpy as np  # noqa: PLC0415

    syllables = np.asarray(syllable_array, dtype=int)
    n_frames = len(syllables)
    return pd.DataFrame(
        {
            "video_name": recording_name,
            "frame": range(n_frames),
            "syllable": syllables,
        }
    )


def export_syllables_from_results(
    results: dict,
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """Write one CSV per recording into *out_dir*.

    Parameters
    ----------
    results : dict
        Results dictionary loaded from ``results.h5``.
    out_dir : Path
        Destination directory (created if it does not exist).
    logger : logging.Logger
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for recording_name, rec_data in results.items():
        if "syllable" not in rec_data:
            logger.warning("No 'syllable' key for recording '%s'; skipping.", recording_name)
            continue

        df = results_to_dataframe(recording_name, rec_data["syllable"])

        # Sanitise the recording name for use as a file name.
        safe_name = recording_name.replace("/", "-").replace("\\", "-")
        csv_path = out_dir / f"{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %d frames → %s", len(df), csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and export syllable CSVs."""
    parser = _build_parser()
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    kpms_project_dir = get_kpms_project_dir(project_path)
    results_dir = get_results_dir(project_path)
    log_path = get_log_path(project_path)

    logger = _setup_logging(log_path)
    logger.info("Project path : %s", project_path)
    logger.info("KPMS dir    : %s", kpms_project_dir)

    # ------------------------------------------------------------------
    # Resolve output directory
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "syllable_timeseries"
    logger.info("Output dir  : %s", out_dir)

    # ------------------------------------------------------------------
    # Find results.h5 files
    # ------------------------------------------------------------------
    if args.results_h5:
        h5_paths = [Path(args.results_h5).resolve()]
    else:
        h5_paths = find_results_h5(kpms_project_dir, args.model_name)

    if not h5_paths:
        logger.error(
            "No results.h5 found in %s.\n"
            "Run 'scripts/run_kpms.py' first, or supply --results-h5.",
            kpms_project_dir,
        )
        sys.exit(1)

    logger.info("Found %d results file(s).", len(h5_paths))

    # If dry-run, list discovered files and exit without writing CSVs
    if getattr(args, "dry_run", False):
        for p in h5_paths:
            logger.info("[dry-run] would process: %s", p)
        logger.info("Dry-run complete: no files were written.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Export each results file
    # ------------------------------------------------------------------
    for h5_path in h5_paths:
        logger.info("Processing: %s", h5_path)
        try:
            results = load_results(h5_path)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            continue

        # Place CSVs in a sub-directory named after the model.
        model_dir_name = h5_path.parent.name
        dest = out_dir / model_dir_name
        export_syllables_from_results(results, dest, logger)

    logger.info("Export complete.")


if __name__ == "__main__":
    main()
