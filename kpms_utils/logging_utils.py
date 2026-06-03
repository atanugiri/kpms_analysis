"""Logging utilities for the keypoint-moseq analysis workspace."""

import logging
import sys
from pathlib import Path


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure a logger that writes to both stdout and a log file.

    Parameters
    ----------
    log_path : Path
        Destination file for log output.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger(__name__)
