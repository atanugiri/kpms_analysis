#!/bin/bash
#SBATCH --job-name=kpms_kappa_scan
#SBATCH --output=/work/agiri/logs/%x-%j.out
#SBATCH --error=/work/agiri/logs/%x-%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2:00:00
#SBATCH --mem=48G

set -euo pipefail

REPO_DIR="$WORK/kpms_analysis_2"
CONDA_ENV_NAME=keypoint_moseq

# Scan parameters exported for scripts/kappa_scan.py.
export PROJECT_DIR="$REPO_DIR/results/ElevatedMazeFood"
export FORMATTED_SNAPSHOT="$PROJECT_DIR/formatted_data.pkl"
export KAPPAS="1e3,1e4,1e5,1e6,1e7"
export DECREASE_KAPPA_FACTOR="10"
export NUM_AR_ITERS="50"
export NUM_FULL_ITERS="200"
export KAPPA_SCAN_PREFIX="kappa_scan_$(date +%Y%m%d)"

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

echo "Running scripts/kappa_scan.py on $(hostname) at $(date)"
python scripts/kappa_scan.py

echo "Done at $(date)"
