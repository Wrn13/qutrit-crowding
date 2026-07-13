#!/bin/bash
#SBATCH --job-name=snail_calgate
# Logs go to the submit dir (always exists); see snail_sweep.slurm for why
# #SBATCH paths cannot use $OUTDIR. Override with sbatch -o/-e if you want.
#SBATCH --output=snail_calgate_%A_%a.out
#SBATCH --error=snail_calgate_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # chevron scans parallelize over pump offsets
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00          # ITERS * (amp scan + chevron); raise for many iters / long t_g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcalphon@syr.edu
# --- REQUIRED ON MOST CLUSTERS (a rejected submit shows nothing in squeue) ---
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
# ----------------------------------------------------------------------------
# For GPU (qutip-jax/diffrax): GPU=1, request a device, load CUDA (forces --jobs 1):
# #SBATCH --gres=gpu:1
# module load cuda   # and: uv pip install qutip-jax diffrax jax[cuda]

# Full experimental iSWAP tune-up (amplitude + Stark-frequency, iterated) with a
# final fidelity test, run as an ARRAY over many device configurations. Provide a
# manifest: one device JSON path per line. Submit with an array matching its length:
#
#   printf '%s\n' devices/*.json > configs.txt
#   MANIFEST=configs.txt TARGET_ETA=1.5 ITERS=2 OUTDIR=cal_out \
#       sbatch --array=0-$(( $(wc -l < configs.txt) - 1 )) snail_calibrate_gate.slurm
#
# Each task writes <OUTDIR>/cal_<device-stem>.json (+ .npz) and a calibrated device
# JSON <OUTDIR>/<device-stem>_calibrated.json to feed run_sweep_zhou.py. Plot the
# Stark shift afterwards with calibration_plots.py.
#
# Needs QuTiP (it integrates the exact Hamiltonian) -- run on compute nodes.
set -euo pipefail

MANIFEST="${MANIFEST:-configs.txt}"
OUTDIR="${OUTDIR:-cal_out}"
TARGET_ETA="${TARGET_ETA:-1.5}"          # operating |eta| (sets t_g); or set T_G
T_G="${T_G:-}"                           # explicit gate time (ns), overrides via --t-g
ITERS="${ITERS:-2}"                      # amplitude/frequency rounds
SPAN_MHZ="${SPAN_MHZ:-60}"
CHEVRON_POINTS="${CHEVRON_POINTS:-21}"
LAUNCHER="${LAUNCHER:-uv run python}"    # e.g. "python" after activating a venv
GPU_FLAG=""; [ -n "${GPU:-}" ] && GPU_FLAG="--gpu"   # GPU=1 -> qutip-jax/diffrax (forces --jobs 1)

JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p "${OUTDIR}"

# select this task's device from the manifest (0-indexed array -> line N+1)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
DEVICE="$(sed -n "$(( IDX + 1 ))p" "${MANIFEST}")"
if [ -z "${DEVICE}" ]; then
    echo "ERROR: no device at manifest line $(( IDX + 1 )) of ${MANIFEST}" >&2
    exit 1
fi
STEM="$(basename "${DEVICE}" .json)"
OUT="${OUTDIR}/cal_${STEM}.json"
CAL_DEV="${OUTDIR}/${STEM}_calibrated.json"

echo "host=$(hostname) job=${SLURM_JOB_ID} task=${IDX} device=${DEVICE} jobs=${JOBS} gpu=${GPU:-0} start=$(date -Is)"

ARGS=(--device "${DEVICE}" --iters "${ITERS}"
      --span-MHz "${SPAN_MHZ}" --chevron-points "${CHEVRON_POINTS}"
      --jobs "${JOBS}" --out "${OUT}" --update-device "${CAL_DEV}")
if [ -n "${T_G}" ]; then ARGS+=(--t-g "${T_G}"); else ARGS+=(--target-eta "${TARGET_ETA}"); fi
[ -n "${GPU_FLAG}" ] && ARGS+=(${GPU_FLAG})

srun ${LAUNCHER} calibrate_gate.py "${ARGS[@]}"

echo "done task=${IDX} device=${DEVICE} end=$(date -Is)"
