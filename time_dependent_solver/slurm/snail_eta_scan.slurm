#!/bin/bash
#SBATCH --job-name=snail_eta
# See snail_sweep.slurm for why directives cannot reference runtime variables.
#SBATCH --output=snail_eta_%A_%a.out
#SBATCH --error=snail_eta_%A_%a.err
#SBATCH --array=0-5              # one task per eta in ETA_LIST (override to match)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00          # a couple of integrated points per task; raise for
                                 # long t_g (small eta) since t_g = 138.889/eta ns
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC

# DRAG gain dF = F_drag - F_nodrag versus pump strength eta, at a FIXED spectator
# detuning. Each eta is a separate gate length (t_g = 138.889/eta ns for the
# baseline g3/lambda), hence a separate grid -- so each array task does its own
# prepare + run rather than sharing one grid.json.
#
# This is the figure that answers "does driving harder (eta ~ 1.8) hurt DRAG?":
# expect dF to rise, peak near eta ~ 1.0-1.1, then fall as the eta^2 Stark
# residual and leakage outgrow what first-order DRAG removes.
#
# NEEDS QuTiP (integrated dynamics).
#
#   DEVICE=warren_clean.json DELTA=2.17 sbatch --array=0-5 slurm/snail_eta_scan.slurm
#   ETA_LIST="0.6 0.8 1.0 1.2 1.5 1.8"   # default
#
# DELTA is the spectator-sweep detuning of the point being probed; use an
# OFF-resonant shoulder (DRAG is skipped exactly on collision, so a resonant
# point yields dF = 0 by construction and tells you nothing).
#
# Afterwards, aggregate + plot:
#   uv run python - <<'EOF'
#   import csv, glob, json
#   rows=[]
#   for f in sorted(glob.glob("results/eta_*/summary.csv")):
#       off=on=None
#       for r in csv.DictReader(open(f)):
#           try: F=float(r["F_avg"])
#           except (TypeError, ValueError): continue
#           (on:=F) if r["drag"].strip().lower()=="true" else (off:=F)
#       print(f, off, on, None if None in (off,on) else on-off)
#   EOF
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
DELTA="${DELTA:?set DELTA=<spectator detuning w_b - w_spec, GHz>}"
ETA_LIST="${ETA_LIST:-0.6 0.8 1.0 1.2 1.5 1.8}"
LAUNCHER="${LAUNCHER:-uv run python}"
OUTROOT="${OUTROOT:-eta}"

read -r -a ETAS <<< "${ETA_LIST}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#ETAS[@]}" ]; then
    echo "task ${IDX} beyond ETA_LIST (${#ETAS[@]} entries) -- nothing to do"; exit 0
fi
ETA="${ETAS[${IDX}]}"
OUTDIR="${OUTROOT}_$(echo "${ETA}" | tr '.' 'p')"

# array tasks provide the parallelism; keep the libraries single-threaded
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# module load python/3.12   # or activate a venv/conda env, then LAUNCHER=python
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"
echo "host=$(hostname) task=${IDX} eta=${ETA} outdir=${OUTDIR} start=$(date -Is)"

# --target-eta sets t_g from the gate area, so do NOT also pass --t-g-ns.
# An operating point calibrated at a different t_g would be flagged, so we let the
# per-eta gate stand on the device defaults unless POINT is given explicitly.
srun ${LAUNCHER} run_sweep_zhou.py prepare \
    --sweep spectator --device "${DEVICE}" --outdir "${OUTDIR}" \
    --drags false,true --specfreqs "${DELTA}" --target-eta "${ETA}" \
    ${POINT:+--operating-point ${POINT}}

srun ${LAUNCHER} run_sweep_zhou.py local \
    --outdir "${OUTDIR}" --nproc "${SLURM_CPUS_PER_TASK:-1}"

echo "done task=${IDX} eta=${ETA} -> results/${OUTDIR}/summary.csv end=$(date -Is)"
