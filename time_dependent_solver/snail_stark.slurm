#!/bin/bash
#SBATCH --job-name=snail_stark
# Logs go to the submit dir (always exists); see snail_sweep.slurm for why
# #SBATCH paths cannot use $OUTDIR. Override with sbatch -o/-e if you want.
#SBATCH --output=snail_stark_%j.out
#SBATCH --error=snail_stark_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # chevron is embarrassingly parallel over pump offsets
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00          # ~POINTS trajectories over WINDOW; raise for long t_g / many points
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcalphon@syr.edu
# --- REQUIRED ON MOST CLUSTERS (a rejected submit shows nothing in squeue) ---
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
# ----------------------------------------------------------------------------
# For GPU (qutip-jax/diffrax): set GPU=1 below, request a device, load CUDA, and
# note GPU forces --jobs 1 (one big solve, not a CPU offset-pool):
# #SBATCH --gres=gpu:1
# module load cuda   # and: uv pip install qutip-jax diffrax jax[cuda]

# Locate the AC-Stark-shifted iSWAP resonance for a device, so run_sweep_zhou.py
# can drive the pump at the frequency that actually closes the swap rather than the
# bare |w_b - w_a|. Writes a chevron .npz (+ optional PNG) and, with UPDATE_DEVICE,
# a device JSON carrying wp_offset_GHz. Typical use, before the allocation sweep:
#
#   DEVICE=dev.json TARGET_ETA=0.3 PLOT=stark.png UPDATE_DEVICE=dev_stark.json \
#       sbatch snail_stark.slurm
#   # -> reports the offset and writes dev_stark.json, then:
#   uv run run_sweep_zhou.py prepare --outdir results_zhou/ --device dev_stark.json
#
# Alternatively, let the sweep find the (per-point) resonance itself:
#   uv run run_sweep_zhou.py prepare --sweep target --stark --outdir results_zhou/ --device dev.json
#
# Needs QuTiP (it integrates the exact Hamiltonian) -- run on a compute node.
set -euo pipefail

DEVICE="${DEVICE:-device.json}"
OUT="${OUT:-stark.npz}"
PLOT="${PLOT:-}"                         # set to a .png to also render the chevron
UPDATE_DEVICE="${UPDATE_DEVICE:-}"       # set to a path to write wp_offset_GHz into a device JSON
CALIB="${CALIB:-}"                       # optional calibration.json to read amp_scale / t_g from
TARGET_ETA="${TARGET_ETA:-}"            # operating |eta| (sets t_g); or use T_G / AMP_SCALE
T_G="${T_G:-}"                           # explicit operating gate time (ns)
AMP_SCALE="${AMP_SCALE:-1.0}"            # amplitude-scale from a prior amplitude calibration
SPAN_MHZ="${SPAN_MHZ:-60}"               # scan is +/- SPAN/2 about |w_b - w_a|
POINTS="${POINTS:-41}"                   # offset grid points
WINDOW_FACTOR="${WINDOW_FACTOR:-2.0}"    # time window = factor * t_g
TIME_POINTS="${TIME_POINTS:-200}"
LAUNCHER="${LAUNCHER:-uv run python}"    # e.g. "python" after activating a venv
GPU_FLAG=""; [ -n "${GPU:-}" ] && GPU_FLAG="--gpu"   # GPU=1 -> qutip-jax/diffrax (forces --jobs 1)

# One worker per allocated CPU (GPU path is single-process; --gpu overrides anyway).
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:-.}"
echo "host=$(hostname) job=${SLURM_JOB_ID} device=${DEVICE} jobs=${JOBS} gpu=${GPU:-0} start=$(date -Is)"

# Assemble the amplitude/eta selection (precedence mirrors the script: --calibration,
# then --target-eta, then --t-g, then device t_g_ns).
ARGS=(--device "${DEVICE}" --out "${OUT}"
      --amp-scale "${AMP_SCALE}"
      --span-MHz "${SPAN_MHZ}" --points "${POINTS}"
      --window-factor "${WINDOW_FACTOR}" --time-points "${TIME_POINTS}"
      --jobs "${JOBS}")
[ -n "${CALIB}" ]         && ARGS+=(--calibration "${CALIB}")
[ -n "${TARGET_ETA}" ]    && ARGS+=(--target-eta "${TARGET_ETA}")
[ -n "${T_G}" ]           && ARGS+=(--t-g "${T_G}")
[ -n "${PLOT}" ]          && ARGS+=(--plot "${PLOT}")
[ -n "${UPDATE_DEVICE}" ] && ARGS+=(--update-device "${UPDATE_DEVICE}")
[ -n "${GPU_FLAG}" ]      && ARGS+=(${GPU_FLAG})

srun ${LAUNCHER} find_stark_resonance.py "${ARGS[@]}"

echo "done end=$(date -Is)"
