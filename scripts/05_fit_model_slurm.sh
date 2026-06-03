#!/bin/bash
#SBATCH --job-name=kpms_fit_model
#SBATCH --output=/work/agiri/logs/%x-%j.out
#SBATCH --error=/work/agiri/logs/%x-%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --mem=32G

set -euo pipefail

# <- EDIT THIS to your workspace root on the cluster
WORK=/work/agiri
PROJECT_DIR="$WORK/kpms_analysis/results/Black-ToyStick"
CONDA_ENV_NAME=keypoint_moseq

mkdir -p "$WORK/logs" "$WORK/.cache" "$WORK/pip-cache" "$WORK/mplconfig" "$WORK/tmp" "$WORK/home_fake"

echo "Job started on $(hostname) at $(date)"

# Enable conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Optional environment tuning for headless and JAX
export PIP_CACHE_DIR="$WORK/pip-cache"
export XDG_CACHE_HOME="$WORK/.cache"
export MPLCONFIGDIR="$WORK/mplconfig"
export TMPDIR="$WORK/tmp"
export TMP="$WORK/tmp"
export TEMP="$WORK/tmp"
export HOME="$WORK/home_fake"
export MPLBACKEND="Agg"

# Limit threads for numpy/mkl
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Prevent JAX from preallocating all GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run from the repository root so outputs/logs are written under the repo
cd "$WORK/kpms_analysis"

# Step 5: Fit model (GPU)
# Uses the project config in results/<project>/kpms_project/config.yml for
# model hyperparameters; pipeline config controls run-time settings.
python scripts/05_fit_model.py \
  --project-path "$PROJECT_DIR" \
  --config configs/BlackToyStick.yml \
  --jax-platform gpu

echo "Job completed at $(date)"
