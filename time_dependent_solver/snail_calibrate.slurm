#!/bin/bash
#SBATCH --job-name=snail_calib
# Logs go to the submit dir (always exists); see snail_sweep.slurm for why
# #SBATCH paths cannot use $OUTDIR. Override with sbatch -o/-e if you want.
#SBATCH --output=snail_calib_%j.out
#SBATCH --error=snail_calib_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=06:00:00          # ~15-25 QuTiP solves; raise for very long t_g (small eta)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcalphon@syr.edu
# --- REQUIRED ON MOST CLUSTERS (a rejected submit shows nothing in squeue) ---
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
# ----------------------------------------------------------------------------

# One-shot iSWAP pump calibration for a device, run BEFORE the spectator sweep.
# Finds the amplitude scale (and optional frequency offset) that maximizes the
# spectator-free gate, writing both a calibration JSON and a calibrated device
# JSON to feed straight into run_sweep_zhou.py:
#
#   DEVICE=dev.json TARGET_ETA=0.1 sbatch snail_calibrate.slurm
#   # -> writes calibration.json and device_calibrated.json, then:
#   uv run run_sweep_zhou.py prepare --outdir results_zhou/ --device device_calibrated.json
#
# Needs QuTiP (it integrates the exact Hamiltonian) -- run on a compute node.
set -euo pipefail

DEVICE="${DEVICE:-device.json}"
CALIB="${CALIB:-calibration.json}"
CALIBRATED_DEVICE="${CALIBRATED_DEVICE:-device_calibrated.json}"
TARGET_ETA="${TARGET_ETA:-1.2}"          # perturbative target; sets t_g if device has none
TUNE_FREQ="${TUNE_FREQ:-0}"              # set to 1 to also search a small pump-freq offset
LAUNCHER="${LAUNCHER:-uv run python}"    # e.g. "python" after activating a venv
GPU_FLAG=""; [ -n "${GPU:-}" ] && GPU_FLAG="--gpu"   # GPU=1 -> qutip-jax/diffrax; also add "#SBATCH --gres=gpu:1"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-.}"
echo "host=$(hostname) job=${SLURM_JOB_ID} device=${DEVICE} target_eta=${TARGET_ETA} gpu=${GPU:-0} start=$(date -Is)"

FREQ_FLAG=""
[ "${TUNE_FREQ}" = "1" ] && FREQ_FLAG="--tune-freq"

srun ${LAUNCHER} calibrate_iswap.py \
    --device "${DEVICE}" \
    --out "${CALIB}" \
    --target-eta "${TARGET_ETA}" \
    --update-device "${CALIBRATED_DEVICE}" \
    ${FREQ_FLAG} ${GPU_FLAG}

echo "done end=$(date -Is)"
