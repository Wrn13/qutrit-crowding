#!/bin/bash
#SBATCH --job-name=snail_spec
# See snail_sweep.slurm: #SBATCH directives are parsed before any shell runs, so
# they cannot expand $OUTDIR/$DEVICE, and any log directory must pre-exist.
#SBATCH --output=snail_spec_%A_%a.out
#SBATCH --error=snail_spec_%A_%a.err
#SBATCH --array=0-15             # one shard per task; NSHARDS must match this count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # --nproc workers inside each task (columns)
#SBATCH --mem-per-cpu=4G   # QuTiP sesolve on the full Hilbert space
#SBATCH --time=12:00:00          # exact sesolve per (frequency, amplitude) point; cost
                                 # ~ f_points * amp_points * probe_ns, and REDUCE=max
                                 # adds n_times outputs -- time one shard first
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC

# Power-vs-frequency spectroscopy of a transmon transition, parallelised two ways:
#   * ACROSS tasks: the frequency axis is sharded (strided, so the per-column cost --
#     which grows with drive frequency -- balances across tasks).
#   * WITHIN a task: --nproc workers over that shard's columns.
# Total parallelism is (array size) x (cpus-per-task).
#
# Scan tasks integrate the EXACT Hamiltonian with QuTiP sesolve via the same
# ZhouCoupler the sweeps use, so QuTiP IS required here (the MERGE=1 step is not).
#
#   DEVICE=warren_device.json PREFIX=results/ge_map NSHARDS=16 \
#       F_LO=1.5 F_HI=3.8 F_POINTS=192 AMP_HI=3.0 AMP_POINTS=61 PROBE=200 \
#       sbatch --array=0-15 slurm/snail_spectroscopy.slurm
#
# Map the iSWAP gate itself (an exchange, so it needs a starting excitation), with
# PROBE set to the gate length so a complete swap shows up as full transfer:
#   DEVICE=warren_clean.json PREFIX=results/iswap_map NSHARDS=16 \
#       F_LO=1.04 F_HI=1.16 F_POINTS=61 PROBE=92.6 \
#       AMP_LO=0 AMP_HI=2.5 AMP_POINTS=51 INIT=0,1,0 TARGET_MODE=0 REDUCE=max \
#       sbatch --array=0-15 slurm/snail_spectroscopy.slurm
#
# Then merge + plot (chain it so it waits for the array):
#   JID=$(... sbatch --parsable ...)
#   PREFIX=results/ge_map sbatch --dependency=afterok:${JID} slurm/snail_spectroscopy.slurm
#     ...with MERGE=1 set, i.e.:
#   MERGE=1 PREFIX=results/ge_map sbatch --dependency=afterok:${JID} \
#       slurm/snail_spectroscopy.slurm
#
# Each shard writes ${PREFIX}_shard<NNN>.npz. The merge step overlays them, WARNS
# about any column no shard filled (a failed task shows up as a missing column
# rather than a silently striped figure), and writes the merged grid + figure.
#
# IMPORTANT: F_POINTS/AMP_POINTS/PROBE and the physics flags must be IDENTICAL across
# every shard, or the axes will not match and the merge refuses. Set them once in the
# environment for both the array and the merge job.
set -euo pipefail

PREFIX="${PREFIX:?set PREFIX=<path prefix for shard files, e.g. results/ge_map>}"
LAUNCHER="${LAUNCHER:-uv run python}"

# scan definition (must match across shards)
F_LO="${F_LO:-1.5}"
F_HI="${F_HI:-3.8}"
F_POINTS="${F_POINTS:-121}"
AMP_LO="${AMP_LO:-0.0}"
AMP_HI="${AMP_HI:-3.0}"
AMP_POINTS="${AMP_POINTS:-41}"
PROBE="${PROBE:-200}"
N_TIMES="${N_TIMES:-41}"                # output times for REDUCE=max/mean
RTOL="${RTOL:-1e-8}"
ATOL="${ATOL:-1e-10}"
ENVELOPE="${ENVELOPE:-constant}"      # constant (sharp lines) | raised_cosine
REDUCE="${REDUCE:-max}"               # max avoids Rabi-phase aliasing; final = fixed duration
LEVEL="${LEVEL:-1}"                   # 1 = |e>, 2 = |f>
TARGET_MODE="${TARGET_MODE:-0}"

# BLAS stays single-threaded: parallelism comes from --nproc processes, and letting
# each worker also spawn BLAS threads oversubscribes the allocation.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# module load python/3.12   # or activate a venv/conda env, then LAUNCHER=python
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"
mkdir -p "$(dirname "${PREFIX}")"

COMMON=(--f-lo "${F_LO}" --f-hi "${F_HI}" --f-points "${F_POINTS}"
        --amp-lo "${AMP_LO}" --amp-hi "${AMP_HI}" --amp-points "${AMP_POINTS}"
        --probe-ns "${PROBE}" --n-times "${N_TIMES}" --rtol "${RTOL}" --atol "${ATOL}"
        --envelope "${ENVELOPE}" --reduce "${REDUCE}" --level "${LEVEL}"
        --target-mode "${TARGET_MODE}")
[ -n "${SPEC_ABS:-}" ] && COMMON+=(--spec-abs-GHz "${SPEC_ABS}")
# INIT: initial Fock occupations, one per mode (e.g. INIT=0,1,0). Default is |g>.
# An EXCHANGE resonance such as the iSWAP conserves excitation number and so is
# invisible from the ground state at any drive frequency -- to map the a<->b gate at
# |w_b - w_a| set INIT=0,1,0 with TARGET_MODE=0, which records the transfer
# probability. Mode count must match the build (3 without a spectator, 4 with).
[ -n "${INIT:-}" ] && COMMON+=(--init "${INIT}")

# ----------------------------------------------------------------- merge mode
if [ "${MERGE:-0}" = "1" ]; then
    echo "host=$(hostname) MERGE ${PREFIX} start=$(date -Is)"
    srun ${LAUNCHER} spectroscopy.py --merge "${PREFIX}" \
        --save-data "${PREFIX}_full.npz" \
        ${SAVE_CSV:+--save-csv "${PREFIX}_full.csv"} \
        --out "${OUT:-${PREFIX}.png}" ${TITLE:+--title "${TITLE}"}
    echo "done merge end=$(date -Is)"
    exit 0
fi

# ----------------------------------------------------------------- scan shard
DEVICE="${DEVICE:?set DEVICE=<device>.json}"
NSHARDS="${NSHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
SHARD="${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}"
NPROC="${NPROC:-${SLURM_CPUS_PER_TASK:-1}}"

echo "host=$(hostname) shard=${SHARD}/${NSHARDS} nproc=${NPROC} device=${DEVICE} start=$(date -Is)"
srun ${LAUNCHER} spectroscopy.py --device "${DEVICE}" "${COMMON[@]}" \
    --shard "${SHARD}" --nshards "${NSHARDS}" --nproc "${NPROC}" \
    --save-data "${PREFIX}.npz"

echo "done shard=${SHARD} end=$(date -Is)"
echo "merge when all tasks finish:  MERGE=1 PREFIX=${PREFIX} sbatch slurm/snail_spectroscopy.slurm"
