# Keypoint-MoSeq Stepwise Workflow (Scripted)

This repository exposes a step-by-step, CLI-first workflow matching the tutorial. Each step is a small script that is easy to run, debug, and re-run independently.

Scripts (ordered)

1. `scripts/01_setup_project.py` — create KPMS project directory and write `config.yml`
2. `scripts/02_load_and_preprocess.py` — load keypoints, optional outlier removal, save `preprocessed_data.pkl` (and optionally `formatted_data.pkl`)
3. `scripts/03_noise_calibration.ipynb` — interactive Jupyter widget for noise calibration (open in JupyterLab)
4. `scripts/04_fit_pca.py` — load preprocessed/formatted data, run `kpms.format_data` if needed, fit PCA, save PCA and `metadata.pkl`
5. `scripts/05_fit_model.py` — initialise and train (or resume) the keypoint-SLDS model; supports AR-only pretraining + full fit
6. `scripts/06_extract_results.py` — extract model results and save per-recording CSVs

Quick workflow example

```bash
export PROJECT_PATH="/path/to/external_project"
export CONFIG="configs/ElevatedMazeFood.yml"

# Step 1: create KPMS project and seed config
python scripts/01_setup_project.py --project-path "$PROJECT_PATH" --config "$CONFIG"

# Step 2: load and save preprocessed snapshot
python scripts/02_load_and_preprocess.py --project-path "$PROJECT_PATH" --config "$CONFIG"

# Step 3: (interactive) open calibration notebook
# Launch JupyterLab manually or use the launcher script
python scripts/03_noise_calibration.ipynb

# Step 4: fit PCA (uses preprocessed/formatted files and the calibrated project config)
python scripts/04_fit_pca.py --project-path "$PROJECT_PATH" --jax-platform gpu

# Step 5: fit model (train)
python scripts/05_fit_model.py --project-path "$PROJECT_PATH" --config "$CONFIG" --jax-platform gpu

# Step 6: export results
python scripts/06_extract_results.py --project-path "$PROJECT_PATH"
```

Notes
- For Steps 2–5 you should activate your `keypoint_moseq` conda environment so dependencies (keypoint-moseq, jax, pyyaml) are available.
- `scripts/03_noise_calibration.ipynb` is interactive and expects `ipympl` / widget support in JupyterLab.
- Preprocessed snapshots are saved to the KPMS project directory (`results/<project>/kpms_project/`). If you re-run earlier steps, the existing snapshots will be overwritten unless you move or rename them.
- If you prefer a single convenience script, `scripts/run_kpms.py` still exists and runs the prepare→fit→export pipeline as before.
