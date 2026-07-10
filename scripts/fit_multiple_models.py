#!/usr/bin/env python3

import os
import pickle
from pathlib import Path

import jax
import keypoint_moseq as kpms
from jax_moseq.utils.debugging import convert_data_precision


# Read run configuration from environment variables (set in Slurm wrapper).
project_dir = Path(os.environ.get("PROJECT_DIR", "results/ElevatedMazeFood"))
formatted_snapshot = Path(os.environ.get("FORMATTED_SNAPSHOT", str(project_dir / "formatted_data.pkl")))
prefix = os.environ.get("MULTI_MODEL_PREFIX", "my_models")
num_model_fits = int(os.environ.get("NUM_MODEL_FITS", "20"))
ar_only_kappa = float(os.environ.get("AR_ONLY_KAPPA", "1e3"))
full_model_kappa = float(os.environ.get("FULL_MODEL_KAPPA", "1e2"))
num_ar_iters = int(os.environ.get("NUM_AR_ITERS", "50"))
num_full_iters = int(os.environ.get("NUM_FULL_ITERS", "500"))
save_every_n_iters = int(os.environ.get("SAVE_EVERY_N_ITERS", "25"))

if not formatted_snapshot.exists():
    raise FileNotFoundError(f"formatted_data.pkl not found: {formatted_snapshot}")

jax.config.update("jax_enable_x64", True)

config = kpms.load_config(str(project_dir))
pca = kpms.load_pca(str(project_dir))

with open(formatted_snapshot, "rb") as f:
    snap = pickle.load(f)

data = convert_data_precision(snap["data"], x64=True)
metadata = snap["metadata"]

for restart in range(num_model_fits):
    seed = restart
    model_name = f"{prefix}-{restart}"
    print(f"Fitting model {restart + 1}/{num_model_fits}: {model_name} (seed={seed})")

    model = kpms.init_model(
        data,
        pca=pca,
        seed=jax.random.PRNGKey(seed),
        **config,
    )

    # Stage 1: fit AR-HMM with fixed kappa.
    model = kpms.update_hypparams(model, kappa=ar_only_kappa)
    model = kpms.fit_model(
        model,
        data,
        metadata,
        str(project_dir),
        model_name,
        ar_only=True,
        num_iters=num_ar_iters,
        save_every_n_iters=save_every_n_iters,
    )[0]

    # Stage 2: fit full model with fixed kappa.
    model = kpms.update_hypparams(model, kappa=full_model_kappa)
    kpms.fit_model(
        model,
        data,
        metadata,
        str(project_dir),
        model_name,
        ar_only=False,
        start_iter=num_ar_iters,
        num_iters=num_ar_iters + num_full_iters,
        save_every_n_iters=save_every_n_iters,
    )

    kpms.reindex_syllables_in_checkpoint(str(project_dir), model_name)
    model_ckpt, _, metadata_ckpt, _ = kpms.load_checkpoint(str(project_dir), model_name)
    kpms.extract_results(model_ckpt, metadata_ckpt, str(project_dir), model_name)

print("Finished fitting all model restarts.")
