#!/bin/bash
#SBATCH --job-name=snail_calmap
# See snail_sweep.slurm: #SBATCH directives are parsed before any shell runs, so
# they cannot expand $OUTDIR/$DEVICE, and any log directory must pre-exist.
#SBATCH --output=snail_calmap_%j.out
#SBATCH --error=snail_calmap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # single process; the cores only help via BLAS, and
                                 # these matrices are small (45-135), so gains are modest
#SBATCH --mem=8G
#SBATCH --time=04:00:00          # 41x41 = 1681 expm propagations: hours at full
                                 # resolution/dim. Shrink with WP_POINTS/AMP_POINTS
                                 # or a smaller CUTOFF before raising this.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu        # match your sweep jobs
# #SBATCH --account=YOUR_ALLOC

# 2D (pump frequency offset x pump strength) landscape -> named operating point.
# This replaces the alternating 1D amplitude/frequency tune-up, which rails when
# the AC-Stark shift and the Rabi amplitude feed back on each other.
#
# NOTE: this program does NOT need QuTiP (pure NumPy/SciPy via grape.py), so it
# can run on a cheap CPU partition -- and small scans are fine on a login node.
#
#   DEVICE=warren_clean.json T_G=92.6 POINT=nominal \
#       sbatch slurm/snail_calibration_map.slurm
#
# CONTEXT (which gate the point is calibrated for) -- set at most one spectator form:
#   WB=5.5            override w_b (the target sweep's swept frequency)
#   SPEC_ABS=4.0      absolute spectator frequency
#   DELTA=1.2         spectator-sweep detuning  Delta = w_b - w_spec
#   DRAG_BEAT=-0.04   map the DRAG-on gate instead of the bare one
#
# RACE WARNING: --save-point does a read-modify-write on the device JSON. Do NOT
# run several of these concurrently against the SAME device file -- points will be
# lost. Calibrate contexts serially, or give each context its own device file.
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
T_G="${T_G:?set T_G=<gate length in ns>}"
POINT="${POINT:-nominal}"          # operating-point name to write ("" = do not save)
LAUNCHER="${LAUNCHER:-uv run python}"
WP_SPAN="${WP_SPAN:-40}"           # full pump-offset width (MHz)
WP_POINTS="${WP_POINTS:-41}"
AMP_LO="${AMP_LO:-0.6}"
AMP_HI="${AMP_HI:-1.4}"
AMP_POINTS="${AMP_POINTS:-41}"
CUTOFF="${CUTOFF:-1.0}"            # rotating-frame carrier cutoff (GHz)
METRIC="${METRIC:-fidelity}"       # fidelity | transfer
FIG="${FIG:-figs/calibration_map_${POINT}.png}"

# optional context flags
CTX=""
[ -n "${WA:-}"        ] && CTX="${CTX} --wa-GHz ${WA}"
[ -n "${WB:-}"        ] && CTX="${CTX} --wb-GHz ${WB}"
[ -n "${SPEC_ABS:-}"  ] && CTX="${CTX} --spec-abs-GHz ${SPEC_ABS}"
[ -n "${DELTA:-}"     ] && CTX="${CTX} --delta-GHz ${DELTA}"
[ -n "${DRAG_BEAT:-}" ] && CTX="${CTX} --drag-beat-GHz ${DRAG_BEAT}"
SAVE=""
[ -n "${POINT}" ] && SAVE="--save-point ${POINT} ${OVERWRITE:+--overwrite}"

# single process: let BLAS use the allocated cores
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg

# module load python/3.12   # or: conda activate qsim / source venv/bin/activate (then LAUNCHER=python)
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"
mkdir -p "$(dirname "${FIG}")"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} device=${DEVICE} t_g=${T_G} point=${POINT:-<none>} start=$(date -Is)"

srun ${LAUNCHER} calibration_map.py \
    --device "${DEVICE}" --t-g-ns "${T_G}" ${CTX} \
    --wp-span-MHz "${WP_SPAN}" --wp-points "${WP_POINTS}" \
    --amp-lo "${AMP_LO}" --amp-hi "${AMP_HI}" --amp-points "${AMP_POINTS}" \
    --cutoff-GHz "${CUTOFF}" --metric "${METRIC}" \
    --out "${FIG}" ${SAVE}

echo "done end=$(date -Is)"
echo "CHECK THE FIGURE: one smooth maximum, not against a grid edge. If the optimum"
echo "sits on an amplitude bound, widen AMP_LO/AMP_HI and rerun before using the point."
