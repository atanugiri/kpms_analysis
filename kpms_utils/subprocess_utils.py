"""Subprocess utilities for the keypoint-moseq analysis workspace.

Provides helpers for constructing and executing subprocess commands,
particularly useful for wrapper scripts that delegate to run_kpms.py.
"""

from typing import Any


def build_cmd_list(base_cmd: list[str], args: Any) -> list[str]:
    """Build a subprocess command list from a base command and optional arguments.

    This utility simplifies the pattern of conditionally appending arguments
    to a subprocess command.

    Parameters
    ----------
    base_cmd : list[str]
        The base command (e.g., [sys.executable, script_path, "--project-path", path]).
    args : object
        An argparse Namespace or similar object with attributes.

    Returns
    -------
    list[str]
        Extended command list with optional arguments appended.

    Examples
    --------
    >>> base = [sys.executable, "run_kpms.py", "--project-path", "/path"]
    >>> cmd = build_cmd_list(base, args)
    >>> # Add optional arguments more concisely than if-elif chains
    """
    cmd = list(base_cmd)
    
    # These are commonly used optional arguments in the wrappers
    optional_args = [
        ("model_name", "--model-name"),
        ("resume_model_name", "--resume-model-name"),
        ("resume_iteration", "--resume-iteration"),
        ("continue_iters", "--continue-iters"),
        ("dlc_config", "--dlc-config"),
        ("results_h5", "--results-h5"),
        ("out_dir", "--out-dir"),
        ("kappa", "--kappa"),
    ]

    boolean_flags = [
        ("use_filtered", "--use-filtered"),
        ("resume_ar_only", "--resume-ar-only"),
        ("dry_run", "--dry-run"),
    ]

    # Add optional arguments (if value is not None)
    for attr, flag in optional_args:
        value = getattr(args, attr, None)
        if value is not None:
            cmd.extend([flag, str(value)])

    # Add boolean flags (if True)
    for attr, flag in boolean_flags:
        if getattr(args, attr, False):
            cmd.append(flag)

    return cmd
