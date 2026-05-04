#!/usr/bin/env python3
"""Extract model results and save per-recording CSVs.

This mirrors the tutorial's export step and is a convenience wrapper
for producing CSV outputs from a fitted model.

Usage
-----
python scripts/06_extract_results.py --project-path /path/to/project [--model-name timestamp]
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kpms_utils.logging_utils import setup_logging
from kpms_utils.path_utils import get_kpms_project_dir, get_results_dir, get_log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract model results and save per-recording CSVs.")
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--model-name", default=None)
    args = parser.parse_args()

    project_path = Path(args.project_path).resolve()
    kpms_project_dir = get_kpms_project_dir(project_path)
    results_dir = get_results_dir(project_path)
    log_path = get_log_path(project_path)

    logger = setup_logging(log_path)

    import keypoint_moseq as kpms  # noqa: PLC0415

    if not kpms_project_dir.is_dir():
        logger.error("KPMS project directory not found: %s", kpms_project_dir)
        raise SystemExit(1)

    model_name = args.model_name
    if model_name is None:
        subdirs = [d for d in kpms_project_dir.iterdir() if d.is_dir()]
        if not subdirs:
            logger.error("No model directories found in %s", kpms_project_dir)
            raise SystemExit(1)
        model_name = max(subdirs, key=lambda p: p.stat().st_mtime).name
        logger.info("Auto-detected model: %s", model_name)

    model_results_path = kpms_project_dir / model_name / "results.h5"
    if not model_results_path.exists():
        logger.error("Model results not found: %s", model_results_path)
        raise SystemExit(1)

    logger.info("Loading and extracting results from: %s", model_results_path)
    results = kpms.load_results_from_h5(str(model_results_path)) if hasattr(kpms, 'load_results_from_h5') else None

    # Fallback: use kpms.extract_results by loading checkpoint
    # Here we rely on kpms.extract_results to accept a model or results dict
    if results is None:
        import h5py
        with h5py.File(model_results_path, 'r') as f:
            results_dict = {k: f[k][()] for k in f.keys()}
        # Try to load metadata
        import pickle
        meta_path = kpms_project_dir / 'metadata.pkl'
        if not meta_path.exists():
            logger.error('Metadata not found: %s', meta_path)
            raise SystemExit(1)
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        extracted = kpms.extract_results(results_dict, metadata, project_dir=str(kpms_project_dir), model_name=model_name, save_results=False)
    else:
        # If kpms provides a loader
        extracted = kpms.extract_results(results, None, project_dir=str(kpms_project_dir), model_name=model_name, save_results=False)

    syllables_dir = results_dir / 'syllables'
    syllables_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving syllable CSVs to: %s", syllables_dir)
    kpms.save_results_as_csv(extracted, save_dir=str(syllables_dir))
    logger.info("Export complete.")


if __name__ == "__main__":
    main()
