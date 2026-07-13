#!/bin/bash
#SBATCH --job-name=snail_stark_det
#SBATCH --output=snail_stark_det_%j.out
#SBATCH --error=snail_stark_det_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # each chevron is parallel over pump offsets
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00          # ~N_DETUNINGS chevrons run in series; raise as needed
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcalphon@syr.edu
# --- REQUIRED ON MOST CLUSTERS (a rejected submit shows nothing in squeue) ---
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
# ----------------------------------------------------------------------------
# For GPU (qutip-jax/diffrax): set GPU=1, request a device, load CUDA; GPU forces
# --jobs 1 (one solve at a time), so it mainly helps large Hilbert spaces.
# #SBATCH --gres=gpu:1
# module load cuda   # and: uv pip install qutip-jax diffrax jax[cuda]

# Stark resonance offset vs spectator detuning: the frequency-domain companion to
# the DRAG spectator sweep. For each Delta = w_b - w_spec it locates the AC-Stark-
# shifted iSWAP resonance with the spectator present, so you can plot how the
# resonance moves as the spectator walks the band (feature at Delta = w_p). Pair the
# PNG with plot_results.py's DRAG-vs-frequency panels from the SAME device/detunings.
#
#   DEVICE=devices/warren_device.json TARGET_ETA=1.5 OUTDIR=results/stark_det \
#       sbatch slurm/snail_stark_detuning.slurm
#   # match a spectator sweep's axis exactly:
#   DEVICE=devices/warren_device.json SPECFREQS=0.20,0.25,...,0.90 \
#       OUTDIR=results/stark_det sbatch slurm/snail_stark_detuning.slurm
#
# The detuning loop is SERIAL (each chevron uses all CPUs for its offset scan). For
# many detunings, prefer a fat node or split SPECFREQS across two submissions.
#
# Needs QuTiP (it integrates the exact Hamiltonian) -- run on a compute node.
set -euo pipefail

DEVICE="${DEVICE:-device.json}"
OUTDIR="${OUTDIR:-results/stark_det}"
SPECFREQS="${SPECFREQS:-}"               # comma detunings; empty -> spectator-sweep default axis
TARGET_ETA="${TARGET_ETA:-}"            # operating |eta| (sets t_g); or use T_G / AMP_SCALE
T_G="${T_G:-}"                           # explicit operating gate time (ns)
AMP_SCALE="${AMP_SCALE:-}"              # amplitude-scale (default: device amp_scale or 1.0)
SPAN_MHZ="${SPAN_MHZ:-60}"               # scan is +/- SPAN/2 about |w_b - w_a|
POINTS="${POINTS:-41}"                   # offset grid points per chevron
WINDOW_FACTOR="${WINDOW_FACTOR:-2.0}"    # time window = factor * t_g
TIME_POINTS="${TIME_POINTS:-200}"
NO_BASELINE="${NO_BASELINE:-}"           # set to 1 to skip the spectator-free reference
LAUNCHER="${LAUNCHER:-uv run python}"    # e.g. "python" after activating a venv
GPU_FLAG=""; [ -n "${GPU:-}" ] && GPU_FLAG="--gpu"

JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Submit from the code root:  sbatch slurm/snail_stark_detuning.slurm  (or set CODE_DIR).
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"
echo "host=$(hostname) job=${SLURM_JOB_ID} device=${DEVICE} jobs=${JOBS} gpu=${GPU:-0} start=$(date -Is)"

ARGS=(--device "${DEVICE}" --outdir "${OUTDIR}"
      --span-MHz "${SPAN_MHZ}" --points "${POINTS}"
      --window-factor "${WINDOW_FACTOR}" --time-points "${TIME_POINTS}"
      --jobs "${JOBS}")
[ -n "${SPECFREQS}" ]  && ARGS+=(--specfreqs "${SPECFREQS}")
[ -n "${TARGET_ETA}" ] && ARGS+=(--target-eta "${TARGET_ETA}")
[ -n "${T_G}" ]        && ARGS+=(--t-g "${T_G}")
[ -n "${AMP_SCALE}" ]  && ARGS+=(--amp-scale "${AMP_SCALE}")
[ -n "${NO_BASELINE}" ]&& ARGS+=(--no-baseline)
[ -n "${GPU_FLAG}" ]   && ARGS+=(${GPU_FLAG})

srun ${LAUNCHER} stark_vs_detuning.py "${ARGS[@]}"

echo "done end=$(date -Is)"
