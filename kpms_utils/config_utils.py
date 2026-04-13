"""Configuration utilities for the keypoint-moseq analysis workspace.

These helpers make it easy to load a base YAML config and overlay
per-run overrides supplied via the command line or caller code.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a plain dictionary.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.  Returns an empty dict if the file is empty.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as fh:
        config = yaml.safe_load(fh)

    return config or {}


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge *overrides* into a shallow copy of *base*.

    Keys present in *overrides* take precedence over those in *base*.
    Nested dicts are merged one level deep; deeper nesting is replaced in
    full by the override value.

    Parameters
    ----------
    base : dict
        The base configuration dictionary (not mutated).
    overrides : dict
        Values to overlay on top of *base*.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
