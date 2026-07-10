#!/bin/bash
#SBATCH --job-name=kpms_multi_fit
#SBATCH --output=/work/%u/logs/%x-%j.out
#SBATCH --error=/work/%u/logs/%x-%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --mem=48G

set -euo pipefail

REPO_DIR="$WORK/kpms_analysis_2"
CONDA_ENV_NAME=keypoint_moseq

# Multi-model parameters exported for scripts/fit_multiple_models.py.
export PROJECT_DIR="$REPO_DIR/results/ElevatedMazeFood"
export FORMATTED_SNAPSHOT="$PROJECT_DIR/formatted_data.pkl"
export MULTI_MODEL_PREFIX="multi_fit_$(date +%Y%m%d)"
export NUM_MODEL_FITS="20"
export AR_ONLY_KAPPA="1e3"
export FULL_MODEL_KAPPA="1e2"
export NUM_AR_ITERS="50"
export NUM_FULL_ITERS="500"
export SAVE_EVERY_N_ITERS="25"

mkdir -p "$WORK/logs" "$WORK/.cache" "$WORK/pip-cache" "$WORK/mplconfig" "$WORK/tmp"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PIP_CACHE_DIR="$WORK/pip-cache"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/mplconfig"
export TMPDIR="$WORK/tmp"
export TMP="$WORK/tmp"
export TEMP="$WORK/tmp"
export MPLBACKEND=Agg

export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$REPO_DIR"

echo "Running scripts/fit_multiple_models.py on $(hostname) at $(date)"
python scripts/fit_multiple_models.py

echo "Done at $(date)"
