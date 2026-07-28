# Detailed Instructions for `README.md`

This document expands the short project summary in `README.md` into a practical, step-by-step guide.

## 1) Project Overview

The repository follows a notebook-first workflow for Keypoint-MoSeq (KPMS) analysis.

Core idea:
- Use Jupyter notebooks for setup, inspection, analysis, and reporting.
- Use HPC-oriented scripts to run computationally heavy model fitting and scanning.
- Bring generated model outputs back into notebooks for comparison, visualization, and statistical interpretation.

The `README.md` describes this as:
- setup -> HPC fitting -> model comparison -> visualization -> statistics.

## 2) What "Notebook-First" Means Here

In this project, notebooks are the main orchestration layer.

You should expect notebooks to:
- Define project configuration and paths.
- Load and inspect model outputs.
- Produce figures and analytical summaries.
- Drive interpretation after model fitting is complete.

HPC scripts are supporting execution tools, not the final reporting interface.

## 3) Expanded Primary Workflow

The `README.md` lists five steps. Below is a detailed interpretation of each step.

### Step 1: Run `notebooks/01_project_setup.ipynb`

Purpose:
- Initialize the analysis environment.
- Confirm project structure and expected files are available.
- Set up configuration needed before fitting/comparison.

***Why cell 6 and cell 8 are necessary:***
- Cell 6 is where keypoints are loaded and `orig_coordinates`/`orig_confidences` backups are created. Those backups are required later if you want to restore selected bodyparts after automated cleaning.
- Cell 8 is where outlier removal is applied before downstream formatting and model fitting. This cleaning step improves data quality and helps prevent noisy keypoints from propagating into PCA and later analyses.

***Why cells 10-14 are necessary (snapshot checkpoints):***
- Cell 10 saves `cleaned_snapshot` as a preprocessing checkpoint (`coordinates`, `confidences`, `bodyparts`). This lets you resume from cleaned keypoints without rerunning load/outlier steps.
- Cell 12 optionally reloads `cleaned_snapshot` after a kernel restart, so downstream formatting can continue from a known-clean state.
- Cell 13 runs `kpms.format_data(...)` to build model-ready arrays (`data`, `metadata`) from cleaned keypoints.
- Cell 14 saves `formatted_snapshot` as a modeling checkpoint. This lets you restart from formatted tensors without re-running preprocessing or formatting.
- Keeping both snapshots preserves two restart points: one for cleaning-level iteration and one for model-fitting-level iteration.

Typical outcomes:
- Validated paths to data and result directories.
- Ready-to-run parameters for downstream fitting and analysis.

### Step 2: Train/Fit Models on HPC using `hpc/`

Purpose:
- Execute compute-intensive fitting outside the notebook runtime.
- Use Slurm job scripts for reproducible batch execution.

Relevant files in `hpc/` include:
- `fit_multiple_models_slurm.sh`
- `kappa_scan_slurm.sh`
- `README.md` (HPC-specific usage notes)

Before submitting jobs, only these variables usually need editing:
- In `kappa_scan_slurm.sh`: `PROJECT_DIR`
- In `fit_multiple_models_slurm.sh`: `PROJECT_DIR`, `AR_ONLY_KAPPA`, `FULL_MODEL_KAPPA`

Typical outcomes:
- New model result folders under `results/`.
- Checkpoints and fit outputs (for example, `checkpoint.h5`, `results.h5`).

### Step 3: Compare Model Results in `notebooks/02_model_fitting.ipynb`

Purpose:
- Evaluate and compare outputs from different model runs.
- Select or justify a preferred fit configuration.

Typical comparisons can include:
- Different kappa values from scan runs.
- Multiple fit replicates.
- Quality and completeness of generated output artifacts.

### Step 4: Run `notebooks/03_visualization.ipynb`

Purpose:
- Turn model outputs into interpretable visual summaries.
- Inspect behavior-level structure and trajectory patterns.

Likely outputs include:
- Figures.
- Trajectory or behavior visualizations.
- Visual diagnostics useful for reporting and QA.

### Step 5: Run `notebooks/04_statistical_analysis.ipynb`

Purpose:
- Perform quantitative/statistical analyses on processed KPMS outputs.
- Move from qualitative inspection to statistical conclusions.

Typical outcomes:
- Statistical summaries and tests.
- Analysis tables and final interpretation-ready results.

## 4) Directory Roles (As Implied by `README.md` and Layout)

- `notebooks/`: Main user-facing analysis pipeline.
- `hpc/`: Batch execution scripts for fitting/training workloads.
- `results/`: Generated model artifacts and run outputs.
- `scripts/`: Utility Python scripts related to fitting and scanning.
- `data/`: Source input data (for example, videos).
- `literature/`: Notes and reference material.

## 5) Practical Run Order

Follow this exact order for reproducibility:
1. Complete setup notebook first.
2. Launch HPC fitting/scans.
3. Wait for results to finish writing to `results/`.
4. Run model comparison notebook.
5. Run visualization notebook.
6. Run statistical analysis notebook.

This preserves dependency order and reduces avoidable errors from missing intermediate outputs.

## 6) Minimal Checklist Before You Start

- Confirm notebook environment and dependencies are available.
- Confirm HPC access and Slurm submission workflow are ready.
- Confirm write access to `results/`.

## 8) Summary

`README.md` defines a concise end-to-end KPMS pipeline:
- initialize project in notebooks,
- run compute-heavy fitting on HPC,
- then compare, visualize, and statistically analyze results in notebooks.

This file provides the detailed execution intent behind that concise workflow.