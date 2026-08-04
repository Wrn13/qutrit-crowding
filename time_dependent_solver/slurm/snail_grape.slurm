#!/bin/bash
#SBATCH --job-name=snail_grape
# See snail_sweep.slurm for why directives cannot reference runtime variables.
#SBATCH --output=snail_grape_%A_%a.out
#SBATCH --error=snail_grape_%A_%a.err
#SBATCH --array=0-2              # one task per gate length in T_G_LIST (override to match)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # single process; cores help only via BLAS on small matrices
#SBATCH --mem=8G
#SBATCH --time=06:00:00          # L-BFGS-B with a NUMERICAL gradient: each iteration costs
                                 # ~(2*N_CTRL+1) propagations, so cost ~ MAXITER*N_CTRL.
                                 # Keep N_CTRL small (12-24) or raise this.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC

# GRAPE optimal control vs the DRAG raised cosine, one array task per gate length.
# Answers "can a shaped pulse recover what first-order DRAG structurally cannot at
# short t_g" -- the follow-on to the dF-vs-eta question.
#
# Needs NO QuTiP (scipy expm in the rotating frame), so a cheap CPU partition is fine.
#
#   DEVICE=warren_device.json sbatch --array=0-2 slurm/snail_grape.slurm
#   DEVICE=warren_clean.json T_G_LIST="30 40 60 92.6" DRAG_BEAT=-0.04 \
#       sbatch --array=0-3 slurm/snail_grape.slurm
#
# Results are appended per task to ${OUTDIR}/grape_<t_g>.txt; the fidelities quoted
# are from the REDUCED model (carrier cutoff), so validate the winning pulse in the
# full QuTiP iswap_fidelity before quoting it.
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
T_G_LIST="${T_G_LIST:-40 60 92.6}"
LAUNCHER="${LAUNCHER:-uv run python}"
OUTDIR="${OUTDIR:-results/grape}"
N_CTRL="${N_CTRL:-16}"
CUTOFF="${CUTOFF:-0.5}"
MAXITER="${MAXITER:-60}"

# pick this task's gate length
read -r -a TGS <<< "${T_G_LIST}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#TGS[@]}" ]; then
    echo "task ${IDX} beyond T_G_LIST (${#TGS[@]} entries) -- nothing to do"; exit 0
fi
T_G="${TGS[${IDX}]}"

EXTRA=""
[ -n "${DRAG_BEAT:-}" ] && EXTRA="${EXTRA} --drag-beat-GHz ${DRAG_BEAT}"
[ -n "${SPEC_ABS:-}"  ] && EXTRA="${EXTRA} --spec-abs-GHz ${SPEC_ABS}"
[ -n "${AMP_SCALE:-}" ] && EXTRA="${EXTRA} --amp-scale ${AMP_SCALE}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg

# module load python/3.12   # or activate a venv/conda env, then LAUNCHER=python
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/grape_tg${T_G}.txt"
echo "host=$(hostname) task=${IDX} t_g=${T_G} device=${DEVICE} start=$(date -Is)"

srun ${LAUNCHER} grape.py \
    --device "${DEVICE}" --t-g-ns "${T_G}" ${EXTRA} \
    --n-ctrl "${N_CTRL}" --cutoff-GHz "${CUTOFF}" --maxiter "${MAXITER}" \
    | tee "${LOG}"

echo "done task=${IDX} t_g=${T_G} -> ${LOG} end=$(date -Is)"
