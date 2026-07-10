import argparse
from pathlib import Path

import h5py


def summarize_h5(path: Path) -> None:
    print(f"Checkpoint: {path}")
    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        print("\nDatasets/groups (up to depth 2):")

        def visit(name: str, obj) -> None:
            depth = name.count("/")
            if depth > 2:
                return
            indent = "  " * depth
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}- {name} [dataset] shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"{indent}- {name} [group]")

        f.visititems(visit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a keypoint-MoSeq checkpoint HDF5 file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/ElevatedMazeFood/kappa_scan_20260708-1000/checkpoint.h5"),
        help="Path to checkpoint.h5",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent.parent / checkpoint_path

    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint.h5 not found: {checkpoint_path}")

    summarize_h5(checkpoint_path)