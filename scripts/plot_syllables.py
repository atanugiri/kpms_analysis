#!/usr/bin/env python3
"""Plot syllable time-series from exported CSVs.

Reads the tidy CSVs produced by `scripts/export_syllables.py` (columns:
`video_name`, `frame`, `syllable`) and plots a horizontal color bar per
recording where colors indicate syllable indices. The script accepts an
FPS argument to convert frames to seconds on the x-axis.

Usage example
-------------
python scripts/plot_syllables.py \
  --project-path /path/to/project \
  --model-name 2026_04_20-15_12_55 \
  --fps 15 \
  --out plot.png

The function `plot_syllable_timeseries` can be imported and used
programmatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plotting.\n"
        "Run this script inside your keypoint_moseq environment (recommended), e.g.:\n\n"
        "  conda activate keypoint_moseq\n"
        "  python scripts/plot_syllables.py --project-path results/<project> --fps 15\n\n"
        "Or install it into your current environment: pip install matplotlib"
    ) from exc


# Allow running from any working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _read_recording_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def plot_syllable_timeseries(
    sequences: Iterable[tuple[str, np.ndarray]],
    fps: float = 15.0,
    ax: plt.Axes | None = None,
    cmap: str = "tab20",
    xlabel: str = "Time (s)",
    title: str | None = None,
    show_legend: bool = False,
) -> plt.Axes:
    """Plot one or more syllable sequences as stacked color bars.

    Parameters
    ----------
    sequences : iterable of (label, syllable_array)
        Each item is a tuple where ``label`` is the recording name and
        ``syllable_array`` is a 1D integer array of per-frame syllable
        indices.
    fps : float
        Frames per second used to convert frames -> seconds on the x-axis.
    ax : matplotlib.axes.Axes, optional
        Ax to draw into. If None a new figure/ax is created.
    cmap : str
        Matplotlib colormap name to use for discrete syllable colors.
    xlabel : str
        X-axis label.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    sequences = list(sequences)
    n_rows = len(sequences)
    if n_rows == 0:
        raise ValueError("No sequences to plot")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(1.5, 0.5 * n_rows)))
    else:
        fig = ax.figure

    # Find global min/max syllable to set colormap levels
    all_vals = np.concatenate([np.asarray(s[1], dtype=int).ravel() for s in sequences])
    min_val = int(np.nanmin(all_vals))
    max_val = int(np.nanmax(all_vals))
    n_colors = max_val - min_val + 1

    cmap_obj = plt.get_cmap(cmap, max(1, n_colors))

    # Build an image matrix of shape (n_rows, max_len)
    max_len = max(len(s[1]) for s in sequences)
    img = np.full((n_rows, max_len), fill_value=min_val - 1, dtype=int)
    yticks = []
    yticklabels = []
    for i, (label, arr) in enumerate(sequences):
        arr = np.asarray(arr, dtype=int)
        img[i, : len(arr)] = arr
        yticks.append(i)
        yticklabels.append(label)

    # Map invalid fill value (min_val-1) to a distinct color (e.g., white)
    # We'll display the image with a ListedColormap that includes a background
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = ["#ffffff"] + [cmap_obj(j) for j in range(n_colors)]
    listed = ListedColormap(colors)
    bounds = [min_val - 1 + i for i in range(n_colors + 2)]
    norm = BoundaryNorm(bounds, listed.N)

    extent = [0, max_len / float(fps), n_rows, 0]
    ax.imshow(img, aspect="auto", cmap=listed, norm=norm, extent=extent)

    ax.set_yticks([r + 0.5 for r in range(n_rows)])
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)

    if show_legend:
        # Build a legend mapping syllable -> color (can be huge for many states)
        import matplotlib.patches as mpatches

        patches = []
        for s in range(min_val, max_val + 1):
            color = listed((s - (min_val - 1)) / (n_colors + 1))
            patches.append(mpatches.Patch(color=color, label=str(s)))

        ax.legend(
            handles=patches,
            title="syllable",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
    ax.set_xlim(0, max_len / float(fps))

    return ax


def resolve_results_dir(project_path: Path) -> Path:
    """Resolve the repo `results/<project_name>/` directory.

    `--project-path` can be either:
    1) The external DLC project directory (contains raw_pose_data/), or
    2) The repo results directory itself: results/<project_name>/
    """
    p = project_path.resolve()

    # If the user pointed at a subfolder inside the results dir, normalize.
    if p.name in {"kpms_project", "syllables", "syllable_timeseries"}:
        return p.parent

    # If the user already pointed at the repo results folder, use it.
    if (p / "kpms_project").exists() or (p / "syllables").exists() or (p / "syllable_timeseries").exists():
        return p

    # Otherwise, treat it as an external project path and map to repo results.
    try:
        from kpms_utils.path_utils import get_results_dir

        return get_results_dir(p)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not resolve results directory from --project-path. "
            "Pass either the external project directory or the repo results/<project_name>/ folder."
        ) from exc


def find_syllable_csvs(
    results_dir: Path,
    model_name: str | None = None,
    source: str = "auto",
) -> tuple[str, list[Path]]:
    """Locate syllable CSVs inside a project's results directory.

    Parameters
    ----------
    results_dir : Path
        Repo results directory (results/<project_name>/).
    model_name : str or None
        Model folder name under syllable_timeseries/ (only used when source is
        'timeseries' or 'auto').
    source : {'auto', 'timeseries', 'syllables'}
        - 'timeseries' reads tidy CSVs from `syllable_timeseries/<model>/*.csv`
          produced by scripts/export_syllables.py.
        - 'syllables' reads per-recording CSVs from `syllables/*.csv` produced
          by kpms.save_results_as_csv.
        - 'auto' prefers timeseries if present, else falls back to syllables.

    Returns
    -------
    (resolved_source, csv_paths)
    """
    results_dir = results_dir.resolve()
    source = source.lower()
    if source not in {"auto", "timeseries", "syllables"}:
        raise ValueError("source must be one of: auto, timeseries, syllables")

    timeseries_root = results_dir / "syllable_timeseries"
    syllables_root = results_dir / "syllables"

    def _timeseries_paths() -> list[Path]:
        if model_name:
            candidate = timeseries_root / model_name
            return sorted(candidate.glob("*.csv")) if candidate.exists() else []
        return sorted(timeseries_root.glob("*/*.csv")) if timeseries_root.exists() else []

    def _syllables_paths() -> list[Path]:
        return sorted(syllables_root.glob("*.csv")) if syllables_root.exists() else []

    if source == "timeseries":
        paths = _timeseries_paths()
        if not paths:
            raise FileNotFoundError(f"No timeseries CSVs found under: {timeseries_root}")
        return "timeseries", paths

    if source == "syllables":
        paths = _syllables_paths()
        if not paths:
            raise FileNotFoundError(f"No syllables CSVs found under: {syllables_root}")
        return "syllables", paths

    # auto
    paths = _timeseries_paths()
    if paths:
        return "timeseries", paths
    paths = _syllables_paths()
    if paths:
        return "syllables", paths
    raise FileNotFoundError(
        f"No CSVs found under {results_dir}. Expected syllable_timeseries/ or syllables/."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot syllable time series CSVs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--project-path",
        help=(
            "External project path OR repo results/<project_name>/ directory. "
            "If provided, the script will search for syllable CSVs under that project."
        ),
    )
    group.add_argument(
        "--csv",
        help=(
            "Path to a single syllables CSV file to plot. "
            "Must contain a 'syllable' column (and optionally a 'frame' column)."
        ),
    )
    parser.add_argument("--model-name", default=None, help="Model directory name under syllable_timeseries/")
    parser.add_argument("--fps", type=float, default=15.0, help="Frames per second (default: 15)")
    parser.add_argument("--out", default=None, help="Optional output image path")
    parser.add_argument("--max-recordings", type=int, default=None, help="Limit number of recordings to plot")
    parser.add_argument(
        "--source",
        choices=["auto", "timeseries", "syllables"],
        default="auto",
        help=(
            "Which CSVs to read. 'timeseries' uses results/<project>/syllable_timeseries/ (from export_syllables.py). "
            "'syllables' uses results/<project>/syllables/ (from kpms.save_results_as_csv). "
            "'auto' prefers timeseries if present."
        ),
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        default=False,
        help="Show a legend mapping syllable index -> color (can be large).",
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = _read_recording_csv(csv_path)
        if "syllable" not in df.columns:
            raise ValueError(f"CSV does not contain a 'syllable' column: {csv_path}")

        if "frame" in df.columns:
            seq = df.sort_values("frame")["syllable"].to_numpy()
        else:
            seq = df["syllable"].to_numpy()

        sequences = [(csv_path.stem, seq)]
        ax = plot_syllable_timeseries(
            sequences,
            fps=args.fps,
            title=f"Syllables: {csv_path.stem}",
            show_legend=args.legend,
        )
        plt.tight_layout()
        if args.out:
            plt.savefig(args.out, dpi=150, bbox_inches="tight")
            print("Saved plot to", args.out)
        else:
            plt.show()
        return

    project_path = Path(args.project_path)
    results_dir = resolve_results_dir(project_path)
    source, csvs = find_syllable_csvs(results_dir, args.model_name, source=args.source)
    if not csvs:
        print("No syllable CSVs found.")
        return

    sequences = []
    for csv in csvs[: args.max_recordings if args.max_recordings else None]:
        df = _read_recording_csv(csv)
        if "syllable" not in df.columns:
            continue

        name = str(csv.stem)
        if source == "timeseries" and "frame" in df.columns:
            seq = df.sort_values("frame")["syllable"].to_numpy()
        else:
            # kpms.save_results_as_csv format doesn't always include a frame column.
            seq = df["syllable"].to_numpy()
        sequences.append((name, seq))

    ax = plot_syllable_timeseries(
        sequences,
        fps=args.fps,
        title=f"Syllables: {results_dir.name}",
        show_legend=args.legend,
    )
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print("Saved plot to", args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
