#!/bin/bash
#SBATCH --job-name=kpms_analysis
#SBATCH --output=/work/agiri/logs/%x-%j.out
#SBATCH --error=/work/agiri/logs/%x-%j.err
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2:00:00
#SBATCH --mem=16G

set -euo pipefail

# <- EDIT THIS to your workspace root on the cluster
WORK=/work/agiri
PROJECT_DIR="$WORK/ElevatedMazeFood-Atanu-2026-04-04"
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

# Prevent JAX from preallocating all GPU memory (optional)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Run from the repository root so all outputs/logs are written under the
# repo directory structure (results/, logs/, etc.). Keep PROJECT_DIR set to
# the external DLC project location and pass it as --project-path.
cd "$WORK/kpms_analysis"

# Run the pipeline (example)
python scripts/run_kpms.py \
  --project-path "$PROJECT_DIR" \
  --config "$WORK/kpms_analysis/configs/BlackToyStick.yml" \
  --steps prepare fit export \
  --jax-platform gpu