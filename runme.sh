#!/usr/bin/env bash
set -euo pipefail

# Notebook-first workflow runner for KPMS.
# Use notebooks for setup/preprocessing/model fitting:
# - notebooks/01_project_setup.ipynb
# - notebooks/02_model_fitting.ipynb
#
# This script keeps only post-fit utilities and sanity checks.

WORKDIR="/Users/atanugiri/Downloads/kpms_analysis_2"
PROJECT_DIR="$WORKDIR/results/toystick_kpms_project"
MODEL_NAME="${MODEL_NAME:-$(cat "$PROJECT_DIR/latest_model_name.txt" 2>/dev/null || true)}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate keypoint_moseq
cd "$WORKDIR"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "MODEL_NAME is empty and latest_model_name.txt was not found."
  echo "Set MODEL_NAME explicitly, e.g.: MODEL_NAME=2026_06_03-15_22_13 bash runme.sh"
  exit 1
fi

echo "Using project: $PROJECT_DIR"
echo "Using model:   $MODEL_NAME"

# Step A: sort syllables by frequency in checkpoint
python scripts/sort_kpms_syllables.py \
  --project-dir "$PROJECT_DIR" \
  --model-name "$MODEL_NAME"

# Step B: extract model results
python scripts/extract_kpms_results.py \
  --project-dir "$PROJECT_DIR" \
  --model-name "$MODEL_NAME"

# Optional: also export per-recording CSV files
python scripts/extract_kpms_results.py \
  --project-dir "$PROJECT_DIR" \
  --model-name "$MODEL_NAME" \
  --save-csv

# Optional: visualization - trajectory plots
python scripts/visualize_kpms_trajectory.py \
  --project-dir "$PROJECT_DIR" \
  --input-snapshot "$PROJECT_DIR/cleaned_keypoints.pkl" \
  --model-name "$MODEL_NAME" \
  --fps 10 \
  --pre 0.3 \
  --post 0.7 \
  --min-frequency 0.005

# Optional: visualization - grid movies (2D only)
python scripts/visualize_kpms_grid_movies.py \
  --project-dir "$PROJECT_DIR" \
  --input-snapshot "$PROJECT_DIR/cleaned_keypoints.pkl" \
  --model-name "$MODEL_NAME" \
  --fps 10 \
  --pre 0.3 \
  --post 0.7 \
  --min-frequency 0.005 \
  --min-duration 3

# Optional: visualization - syllable dendrogram
python scripts/visualize_kpms_dendrogram.py \
  --project-dir "$PROJECT_DIR" \
  --input-snapshot "$PROJECT_DIR/cleaned_keypoints.pkl"

# Sanity checks for notebook-generated artifacts
ls -lh "$PROJECT_DIR/cleaned_keypoints.pkl"
ls -lh "$PROJECT_DIR/formatted_data.pkl"
ls -lh "$PROJECT_DIR/pca.p"
ls -lh "$PROJECT_DIR/latest_model_name.txt"
