#!/usr/bin/env python3
"""Shared helper utilities for KPMS scripts."""

from __future__ import annotations

import pickle
from pathlib import Path


def normalize_recording_name(name: str) -> str:
    """Strip DLC suffix so recording ids match video stems."""
    if "DLC_" in name:
        return name.split("DLC_")[0]
    return name


def infer_linear_skeleton(bodyparts: list[str]) -> list[list[str]]:
    """Create a simple chain skeleton from ordered bodyparts."""
    if len(bodyparts) < 2:
        return []
    return [[bodyparts[i], bodyparts[i + 1]] for i in range(len(bodyparts) - 1)]


def resolve_model_name(project_dir: Path, model_name: str | None) -> str:
    """Return explicit model name or read it from latest_model_name.txt."""
    if model_name:
        return model_name

    latest_model_path = project_dir / "latest_model_name.txt"
    if not latest_model_path.exists():
        raise FileNotFoundError(
            f"No --model-name provided and latest_model_name.txt not found in {project_dir}"
        )

    resolved = latest_model_path.read_text(encoding="utf-8").strip()
    if not resolved:
        raise RuntimeError(f"latest_model_name.txt is empty: {latest_model_path}")
    return resolved


def load_coordinates_from_snapshot(snapshot_path: Path) -> dict:
    """Load coordinates dict from a saved cleaned_keypoints snapshot."""
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Input snapshot not found: {snapshot_path}")

    with open(snapshot_path, "rb") as f:
        snap = pickle.load(f)

    coordinates = snap.get("coordinates")
    if coordinates is None:
        raise RuntimeError("Input snapshot is missing 'coordinates'. Expected cleaned_keypoints.pkl output.")
    return coordinates