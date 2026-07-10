# HPC Scripts

This folder contains cluster-oriented scripts for heavy keypoint-MoSeq jobs.

## Automatic kappa scan

Use `kappa_scan_slurm.sh` to run a full kappa scan on Slurm.

1. The Slurm wrapper configures environment variables for the run:
   - `PROJECT_DIR`, `FORMATTED_SNAPSHOT`
   - `KAPPAS`, `DECREASE_KAPPA_FACTOR`
   - `NUM_AR_ITERS`, `NUM_FULL_ITERS`
   - `KAPPA_SCAN_PREFIX`
2. It activates the conda environment.
3. It runs `python scripts/kappa_scan.py`.
4. The Python script performs AR + full-stage fitting for each kappa and saves `<project_dir>/<prefix>_kappa_scan.pdf`.

### Files

- `kappa_scan_slurm.sh`: Thin Slurm wrapper for environment setup and job launch.
- `scripts/kappa_scan.py`: Main kappa scan logic.

### How to use

1. Edit paths and scan settings at the top of `kappa_scan_slurm.sh`.
2. Submit:

```bash
sbatch hpc/kappa_scan_slurm.sh
```
