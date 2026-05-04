#!/usr/bin/env python3
"""Load keypoints, optionally remove outliers, format data, and save preprocessed outputs.

This script implements step 2 of the proposed pipeline:
- Load keypoints from the external project's pose-data directory
- (Optional) Run outlier removal or preprocessing hooks
- Save a preprocessed snapshot (pickle) so later steps don't need to re-run
  raw loading every time
- Optionally run `kpms.format_data` and save the formatted `data` and `metadata`

Usage
-----
python scripts/02_load_and_preprocess.py \
    --project-path /path/to/project \
    --config configs/ElevatedMazeFood.yml \
    [--use-filtered] [--format]

Outputs (written to KPMS project dir):
- preprocessed_data.pkl      (coordinates, confidences, bodyparts)
- formatted_data.pkl        (optional: kpms.format_data output: data, metadata)
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
from kpms_utils.path_utils import get_kpms_project_dir, get_pose_data_dir, get_log_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load keypoints and save preprocessed snapshot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--config", default=str(_REPO_ROOT / "configs" / "config_example.yml"))
    parser.add_argument("--use-filtered", action="store_true", default=False)
    parser.add_argument("--format", action="store_true", default=False, help="Run kpms.format_data and save formatted output")
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path

    log_path = get_log_path(project_path)
    logger = setup_logging(log_path)

    if not project_path.is_dir():
        logger.error("Project path does not exist: %s", project_path)
        raise SystemExit(1)

    config = load_yaml_config(str(config_path))

    kpms_project_dir = get_kpms_project_dir(project_path)
    pose_data_dir = get_pose_data_dir(project_path, use_filtered=args.use_filtered)

    logger.info("Loading keypoints from: %s (use_filtered=%s)", pose_data_dir, args.use_filtered)

    import keypoint_moseq as kpms  # noqa: PLC0415

    fmt = config.get("pose_estimation_format", "deeplabcut")
    ext = config.get("pose_file_extension") or None
    recursive = config.get("recursive_search", True)

    coordinates, confidences, bodyparts = kpms.load_keypoints(
        str(pose_data_dir),
        format=fmt,
        extension=ext,
        recursive=recursive,
    )

    logger.info("Loaded %d recording(s)", len(coordinates))
    logger.info("Detected %d bodyparts", len(bodyparts))

    preprocessed_path = kpms_project_dir / "preprocessed_data.pkl"
    with open(preprocessed_path, "wb") as f:
        pickle.dump({"coordinates": coordinates, "confidences": confidences, "bodyparts": bodyparts}, f)

    logger.info("Saved preprocessed snapshot to: %s", preprocessed_path)

    if args.format:
        logger.info("Formatting data via kpms.format_data()")
        # Load KPMS project config
        kpms_config = kpms.load_config(str(kpms_project_dir))
        data, metadata = kpms.format_data(coordinates, confidences, **kpms_config)

        # Save formatted objects
        formatted_path = kpms_project_dir / "formatted_data.pkl"
        with open(formatted_path, "wb") as f:
            pickle.dump({"data": data, "metadata": metadata}, f)
        logger.info("Saved formatted data to: %s", formatted_path)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
