#!/bin/bash
#SBATCH --job-name=snail_collect
# See snail_sweep.slurm: directives cannot expand $OUTDIR, and the log dir must
# pre-exist. Logs go to the submit dir; override with sbatch -o/-e if desired.
#SBATCH --output=snail_collect_%j.out
#SBATCH --error=snail_collect_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu        # match the sweep job's partition/account
# #SBATCH --account=YOUR_ALLOC

# Gather all per-point .npz into summary.csv + combined.npz. Chain after the
# sweep so it runs only once every point has succeeded:
#   JID=$(RUNNER=run_sweep_zhou.py OUTDIR=results_zhou/ \
#         sbatch --parsable --array=0-119 snail_sweep.slurm)
#   RUNNER=run_sweep_zhou.py OUTDIR=results_zhou/ \
#         sbatch --dependency=afterok:${JID} snail_collect.slurm
set -euo pipefail
OUTDIR="${OUTDIR:-results}"
RUNNER="${RUNNER:-run_sweep_zhou.py}"
LAUNCHER="${LAUNCHER:-uv run python}"

# module load python/3.12   # or: conda activate qsim / source venv/bin/activate (then LAUNCHER=python)
cd "${SLURM_SUBMIT_DIR:-.}"
srun ${LAUNCHER} "${RUNNER}" collect --outdir "${OUTDIR}"
