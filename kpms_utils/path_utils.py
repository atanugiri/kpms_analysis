"""Path utilities for the keypoint-moseq analysis workspace.

All functions accept and return ``pathlib.Path`` objects so that callers never
need to manipulate raw strings for file-system operations.

External project layout assumed by these helpers:

    <project_path>/
    ├── videos/
    ├── raw_pose_data/
    └── filtered_pose_data/

Outputs are always written inside *this* repository, never back into the
external project directory.
"""

from pathlib import Path


def get_repo_root() -> Path:
    """Return the absolute path to the root of this repository.

    The root is defined as the parent of the directory containing this file
    (i.e. ``kpms_utils/``).

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parent.parent


def get_project_name(project_path: str | Path) -> str:
    """Extract the project name from an external project path.

    The project name is simply the final component of the path, which
    follows the convention that each DLC project lives in its own folder.

    Parameters
    ----------
    project_path : str or Path
        Path to the external DLC project directory.

    Returns
    -------
    str
        The directory name, e.g. ``"ElevatedMazeFood-Atanu-2026-04-04"``.
    """
    return Path(project_path).resolve().name


def get_video_dir(project_path: str | Path) -> Path:
    """Return the ``videos/`` sub-directory of an external project.

    Parameters
    ----------
    project_path : str or Path
        Path to the external DLC project directory.

    Returns
    -------
    Path
        Absolute path to ``<project_path>/videos/``.
    """
    return Path(project_path).resolve() / "videos"


def get_pose_data_dir(project_path: str | Path, use_filtered: bool = False) -> Path:
    """Return the pose-data sub-directory for an external project.

    Parameters
    ----------
    project_path : str or Path
        Path to the external DLC project directory.
    use_filtered : bool, default=False
        When ``True`` return ``filtered_pose_data/``, otherwise return
        ``raw_pose_data/``.

    Returns
    -------
    Path
        Absolute path to the chosen pose-data directory.
    """
    sub = "filtered_pose_data" if use_filtered else "raw_pose_data"
    return Path(project_path).resolve() / sub


def get_results_dir(project_path: str | Path) -> Path:
    """Return the results output directory inside this repository.

    Outputs are always written to ``<repo_root>/results/<project_name>/``
    so that external project directories are never modified.

    Parameters
    ----------
    project_path : str or Path
        Path to the external DLC project directory.

    Returns
    -------
    Path
        Absolute path to ``<repo_root>/results/<project_name>/``.
    """
    project_name = get_project_name(project_path)
    return get_repo_root() / "results" / project_name


def get_kpms_project_dir(project_path: str | Path) -> Path:
    """Return the keypoint-moseq project directory inside this repository.

    keypoint-moseq writes its ``config.yml`` and model checkpoints into a
    dedicated sub-folder.  We keep that folder under
    ``<repo_root>/results/<project_name>/kpms_project/`` so that all outputs
    remain self-contained within this repository.

    Parameters
    ----------
    project_path : str or Path
        Path to the external DLC project directory.

    Returns
    -------
    Path
        Absolute path to the KPMS project directory.
    """
    return get_results_dir(project_path) / "kpms_project"


def get_log_path(project_path: str | Path) -> Path:
    """Return the path for the run log file.

    Log files are written to ``<repo_root>/logs/<project_name>.log``.

    Parameters
    ----------
    project_path : str or Path
        Path to the external DLC project directory.

    Returns
    -------
    Path
        Absolute path to the log file.
    """
    project_name = get_project_name(project_path)
    return get_repo_root() / "logs" / f"{project_name}.log"
