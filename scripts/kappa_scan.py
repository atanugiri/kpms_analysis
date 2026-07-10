#!/usr/bin/env python3

import os
import pickle
from pathlib import Path

import jax
import keypoint_moseq as kpms
from jax_moseq.utils.debugging import convert_data_precision


def parse_kappas(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


# Read run configuration from environment variables (set in Slurm wrapper).
project_dir = Path(os.environ.get("PROJECT_DIR", "results/ElevatedMazeFood"))
formatted_snapshot = Path(os.environ.get("FORMATTED_SNAPSHOT", str(project_dir / "formatted_data.pkl")))
kappas = parse_kappas(os.environ.get("KAPPAS", "1e3,1e4,1e5,1e6,1e7"))
decrease_kappa_factor = float(os.environ.get("DECREASE_KAPPA_FACTOR", "10"))
num_ar_iters = int(os.environ.get("NUM_AR_ITERS", "50"))
num_full_iters = int(os.environ.get("NUM_FULL_ITERS", "200"))
prefix = os.environ.get("KAPPA_SCAN_PREFIX", "my_kappa_scan")

if not formatted_snapshot.exists():
    raise FileNotFoundError(f"formatted_data.pkl not found: {formatted_snapshot}")

jax.config.update("jax_enable_x64", True)

config = kpms.load_config(str(project_dir))
pca = kpms.load_pca(str(project_dir))

with open(formatted_snapshot, "rb") as f:
    snap = pickle.load(f)

data = convert_data_precision(snap["data"], x64=True)
metadata = snap["metadata"]

for kappa in kappas:
    print(f"Fitting model with kappa={kappa}")
    # Match docs naming so plot_kappa_scan can discover checkpoints by kappa.
    model_name = f"{prefix}-{kappa}"
    model = kpms.init_model(data, pca=pca, **config)

    # Stage 1: fit AR-HMM with initial kappa.
    model = kpms.update_hypparams(model, kappa=kappa)
    model = kpms.fit_model(
        model,
        data,
        metadata,
        str(project_dir),
        model_name,
        ar_only=True,
        num_iters=num_ar_iters,
        save_every_n_iters=25,
    )[0]

    # Stage 2: fit full model with reduced kappa.
    model = kpms.update_hypparams(model, kappa=kappa / decrease_kappa_factor)
    kpms.fit_model(
        model,
        data,
        metadata,
        str(project_dir),
        model_name,
        ar_only=False,
        start_iter=num_ar_iters,
        num_iters=num_ar_iters + num_full_iters,
        save_every_n_iters=25,
    )

fig, final_medians = kpms.plot_kappa_scan(kappas, str(project_dir), prefix)
fig.savefig(project_dir / f"{prefix}.pdf", dpi=200, bbox_inches="tight")