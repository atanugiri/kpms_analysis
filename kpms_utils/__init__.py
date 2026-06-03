"""kpms_utils – shared utilities for the keypoint-moseq analysis workspace."""

import sys
from pathlib import Path

from .path_utils import (
    get_project_name,
    get_video_dir,
    get_pose_data_dir,
    get_repo_root,
    get_results_dir,
    get_kpms_project_dir,
    get_log_path,
    resolve_results_dir,
)
from .config_utils import (
    load_yaml_config,
    merge_config,
)
from .logging_utils import (
    setup_logging,
)
from .subprocess_utils import (
    build_cmd_list,
)


def ensure_repo_in_path() -> Path:
    """Ensure the repository root is in sys.path for imports.

    This function adds the repository root to sys.path if not already present.
    Useful at the top of scripts to allow importing kpms_utils and other
    modules regardless of the working directory.

    Returns
    -------
    Path
        The absolute path to the repository root.
    """
    repo_root = get_repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


__all__ = [
    "get_project_name",
    "get_video_dir",
    "get_pose_data_dir",
    "get_repo_root",
    "get_results_dir",
    "get_kpms_project_dir",
    "get_log_path",
    "resolve_results_dir",
    "load_yaml_config",
    "merge_config",
    "setup_logging",
    "build_cmd_list",
    "ensure_repo_in_path",
]
