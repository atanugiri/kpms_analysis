# kpms_analysis

A reusable workspace for running [keypoint-MoSeq](https://github.com/dattalab/keypoint-moseq) on external DLC (DeepLabCut) pose-estimation projects.

---

## Design philosophy

* External pose-estimation projects live **outside** this repository (e.g. on a shared drive or in a DLC project folder).
* This repository contains only **code, configs, and results** — never raw video or raw pose data.
* All outputs are written to `results/<project_name>/` inside this repo.
* Scripts are parameterised with `--project-path` so the same code works for many projects.

---

## Repository structure

```
kpms_analysis/
├── configs/
│   └── config_example.yml      ← copy and edit for each project
├── kpms_utils/
│   ├── __init__.py
│   ├── path_utils.py           ← shared path helpers (uses pathlib)
│   └── config_utils.py         ← YAML loading and merging helpers
├── notebooks/                  ← Jupyter notebooks (add as needed)
├── results/                    ← outputs written here (one folder per project)
│   └── <project_name>/
│       ├── kpms_project/       ← keypoint-moseq project dir (config.yml, checkpoints)
│       ├── syllables/          ← per-recording CSVs from kpms.save_results_as_csv
│       └── syllable_timeseries/← tidy (video_name, frame, syllable) CSVs
├── logs/                       ← run logs (one file per project)
├── scripts/
│   ├── run_kpms.py             ← main pipeline script
│   └── export_syllables.py     ← post-processing: tidy syllable CSVs
└── README.md
```

---

## Requirements

Install keypoint-moseq and its dependencies:

```bash
pip install keypoint-moseq
# or, for GPU support:
pip install "keypoint-moseq[gpu]"
```

This workspace also uses standard scientific Python packages (NumPy, pandas, PyYAML) which are pulled in transitively by keypoint-moseq.

---

## External project layout

The scripts expect the following directory structure for every external project:

```
<project_path>/
├── videos/               ← raw video files (read-only, never written to)
├── raw_pose_data/        ← DLC/SLEAP output files (CSV or HDF5)
└── filtered_pose_data/   ← filtered pose files (optional)
```

For example:

```
/Users/atanugiri/Downloads/dlc-pose-estimation/ElevatedMazeFood-Atanu-2026-04-04/
├── videos/
│   ├── session1.mp4
│   └── session2.mp4
├── raw_pose_data/
│   ├── session1DLC_resnet50_...csv
│   └── session2DLC_resnet50_...csv
└── filtered_pose_data/
    ├── session1DLC_resnet50_...filtered.csv
    └── session2DLC_resnet50_...filtered.csv
```

---

## Quick start

### 1 — Copy and edit the config

```bash
cp configs/config_example.yml configs/ElevatedMazeFood.yml
# then open configs/ElevatedMazeFood.yml and update:
#   - bodyparts / use_bodyparts / skeleton
#   - pose_estimation_format (deeplabcut | sleap | ...)
#   - num_states, kappa, latent_dim, num_iters, ...
```

### 2 — Run the full pipeline

```bash
python scripts/run_kpms.py \
    --project-path /Users/atanugiri/Downloads/dlc-pose-estimation/ElevatedMazeFood-Atanu-2026-04-04 \
    --config configs/ElevatedMazeFood.yml
```

This runs all three steps in sequence: **prepare → fit → export**.

### 3 — Run individual steps

```bash
# Data preparation only (load keypoints, fit PCA):
python scripts/run_kpms.py \
    --project-path /path/to/project \
    --config configs/ElevatedMazeFood.yml \
    --steps prepare

# Fit model only (requires a prior prepare run):
python scripts/run_kpms.py \
    --project-path /path/to/project \
    --steps prepare fit

# Export syllable CSVs only (requires a fitted model):
python scripts/export_syllables.py \
    --project-path /path/to/project
```

### 4 — Use filtered pose data

```bash
python scripts/run_kpms.py \
    --project-path /path/to/project \
    --use-filtered
```

### 5 — Point to a DLC config for automatic bodypart discovery

```bash
python scripts/run_kpms.py \
    --project-path /path/to/project \
    --dlc-config /path/to/project/config.yaml
```

---

## Pipeline details

### Step 1: prepare

1. Calls `kpms.setup_project()` to create a `config.yml` inside
   `results/<project_name>/kpms_project/`.
2. Loads pose-estimation files with `kpms.load_keypoints()`.
3. Formats data for inference with `kpms.format_data()`.
4. Fits PCA with `kpms.fit_pca()` and saves it to disk.

> Note: The keypoint-MoSeq **noise calibration** step (`kpms.noise_calibration`) is
> an interactive JupyterLab widget and is intentionally **not** part of this
> headless CLI pipeline. If you want to run it, use:
>
> ```bash
> python scripts/run_kpms.py --project-path /path/to/project --launch-noise-calibration
> ```

### Step 2: fit

1. Initialises the model with `kpms.init_model()`.
2. Runs an optional AR-only pre-training phase (`ar_only=True`).
3. Fits the full keypoint-SLDS model with `kpms.fit_model()`.
4. Saves checkpoints every `save_every_n_iters` iterations.

### Step 3: export

1. Calls `kpms.extract_results()` to produce a `results.h5` file.
2. Calls `kpms.save_results_as_csv()` to write one CSV per recording
   (columns: `syllable`, `centroid x`, `centroid y`, `heading`, `latent_state*`).

The standalone `export_syllables.py` script reads `results.h5` and produces a
simpler tidy format with only `video_name`, `frame`, and `syllable` columns.

---

## Output files

| Path | Description |
|------|-------------|
| `results/<project>/kpms_project/config.yml` | keypoint-moseq configuration |
| `results/<project>/kpms_project/pca.p` | Fitted PCA model |
| `results/<project>/kpms_project/<model_name>/checkpoint.h5` | Model checkpoint |
| `results/<project>/kpms_project/<model_name>/results.h5` | Raw model outputs |
| `results/<project>/syllables/<recording>.csv` | Full results CSV per recording |
| `results/<project>/syllable_timeseries/<model>/<recording>.csv` | Tidy syllable CSV |
| `logs/<project_name>.log` | Run log |

---

## Configuration reference

See `configs/config_example.yml` for all available settings with inline
documentation.  The most important parameters are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pose_estimation_format` | `"deeplabcut"` | Format of pose files |
| `bodyparts` | — | Full list of keypoints from the tracker |
| `use_bodyparts` | — | Subset used for modelling |
| `skeleton` | — | Pairs of bodypart names for visualisation |
| `latent_dim` | `4` | PCA dimensionality of the latent trajectory |
| `num_states` | `100` | Number of discrete syllables |
| `kappa` | `1e5` | Syllable stickiness (higher = longer syllables) |
| `num_iters` | `200` | Total Gibbs-sampling iterations |
| `ar_iters` | `50` | AR-only pre-training iterations |
| `fps` | `30` | Video frame rate (used for duration analysis) |

---

## Assumptions and TODOs

> These items should be verified against the current keypoint-moseq
> documentation before running in production.

1. **`kpms.load_keypoints` signature** – The third return value (`bodyparts`)
   was verified against the source at
   `dattalab/keypoint-moseq @ fe6599b`.  Double-check if you are on a
   different version.

2. **`kpms.format_data` location** – This function lives in
   `keypoint_moseq.util` but is re-exported from the top-level package.
   If `import keypoint_moseq as kpms; kpms.format_data(...)` fails,
   import it directly: `from keypoint_moseq.util import format_data`.

3. **`kpms.fit_pca` location** – Exported from
   `jax_moseq.models.keypoint_slds` and re-exported at the top level.

4. **AR-only pre-training** – Calling `fit_model(..., ar_only=True)` then
   `fit_model(..., ar_only=False)` with the same `model_name` is the
   recommended approach; verify this is still the case for your version.

5. **`kpms.setup_project` overwrite** – The script passes `overwrite=False`
   so a second run will skip re-creating the project directory.  Set
   `overwrite=True` in `step_prepare` if you want to regenerate config.yml.

6. **Multi-animal projects** – These are supported by keypoint-moseq but not
   specifically tested in this workspace.  Per-individual keys are
   generated automatically by `load_keypoints`.

7. **GPU acceleration** – Install `jax[cuda]` and the CUDA toolkit for GPU
   support.  The model automatically uses the available backend.

---

## References

* [keypoint-MoSeq GitHub](https://github.com/dattalab/keypoint-moseq)
* [keypoint-MoSeq documentation](https://keypoint-moseq.readthedocs.io/en/latest/)
* [jax-moseq](https://github.com/dattalab/jax-moseq) (the JAX backend)
