#!/bin/bash
#SBATCH --job-name=snail_fig8
#SBATCH --output=snail_fig8_%j.out
#SBATCH --error=snail_fig8_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00          # full eta x time map + spectator scan on the exact coupler
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcalphon@syr.edu
# --- REQUIRED ON MOST CLUSTERS (a rejected submit shows nothing in squeue) ---
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
# ----------------------------------------------------------------------------

# Generate the Fig.8 data (McKinney arXiv:2409.18262) with the EXACT Zhou coupler.
#   DEVICE=dev.json sbatch snail_figure8.slurm
# Defaults to the built-in FIG8_DEFAULTS if DEVICE is unset. Uses the QuTiP backend
# (compiled sparse sesolve); the dense backend is for small in-sandbox checks only.
set -euo pipefail

DEVICE="${DEVICE:-}"                     # optional device JSON
MODE="${MODE:-both}"                     # map | spectator | both
OUT="${OUT:-fig8.npz}"
PLOT="${PLOT:-fig8.png}"
ETA_MIN="${ETA_MIN:-0.1}"; ETA_MAX="${ETA_MAX:-2.4}"; ETA_POINTS="${ETA_POINTS:-32}"
T_POINTS="${T_POINTS:-160}"
DELTAS_MHZ="${DELTAS_MHZ:-100,200,400}"
LAM_SPEC="${LAM_SPEC:-0.20}"
LAUNCHER="${LAUNCHER:-uv run python}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-.}"
echo "host=$(hostname) job=${SLURM_JOB_ID} mode=${MODE} start=$(date -Is)"

DEVICE_FLAG=""
[ -n "${DEVICE}" ] && DEVICE_FLAG="--device ${DEVICE}"

srun ${LAUNCHER} figure8_data.py \
    ${DEVICE_FLAG} \
    --mode "${MODE}" --backend qutip \
    --out "${OUT}" --plot "${PLOT}" \
    --eta-min "${ETA_MIN}" --eta-max "${ETA_MAX}" --eta-points "${ETA_POINTS}" \
    --t-points "${T_POINTS}" \
    --deltas-MHz "${DELTAS_MHZ}" --lam-spec "${LAM_SPEC}"

echo "done end=$(date -Is)"
