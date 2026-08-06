#!/bin/bash
#SBATCH --job-name=snail_sweep
# NOTE: #SBATCH directives are parsed by sbatch BEFORE any shell runs, so they
# CANNOT reference $OUTDIR/$DEVICE, and any directory in --output must already
# exist. These land in the submit directory; move them afterwards if you prefer.
#SBATCH --output=snail_sweep_%A_%a.out
#SBATCH --error=snail_sweep_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4        # one point per task. Cores are used by (a) the
                                 # per-point --stark chevron when STARK=1
                                 # (stark_jobs = cpus-per-task), else (b) BLAS on
                                 # small matrices (dim 45-135), where gains are modest.
#SBATCH --mem=8G
#SBATCH --time=02:00:00          # PER TASK, and with CHUNK>1 one task runs CHUNK
                                 # points IN SERIES -- scale this up accordingly.
                                 # Rough per-point cost (integrate=true):
                                 #   plain iSWAP fidelity ...... seconds to ~1 min
                                 #   + --drag-compare .......... x2
                                 #   + --stark ................. + chevron scan
                                 #                               (stark_points x t)
                                 #   + --calibrate ............. x(3-10), the tune-up
                                 #   + GRAPE reduced ........... + ~1-5 min
                                 #   + GRAPE qutip (GOAT/JOPT) . + ~10 min to hours
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu        # set to your partition
# #SBATCH --account=YOUR_ALLOC
#
# =============================================================================
# ARRAY sweep driver for run_sweep_zhou.py  (mode=point, one point per task)
# =============================================================================
# The grid is written ONCE by `prepare` into <OUTDIR>/grid.json; each array task
# then evaluates its own slice and writes <OUTDIR>/points/point_XXXXX.npz.
# Requires QuTiP on the compute nodes (integrate=true drives qt.sesolve);
# `prepare`/`collect`/`missing` need only numpy+scipy and run fine on a login node.
#
# -----------------------------------------------------------------------------
# WORKFLOW  (copy-paste; run steps 1/3/5 on the login node)
# -----------------------------------------------------------------------------
# 1. Build the grid. ALL physics/grid choices go here (see PREPARE-TIME below):
#      uv run python run_sweep_zhou.py prepare --sweep target \
#          --device evan_device.json --target-eta 0.9 \
#          --drag-compare --outdir results/allocation
#    It prints the exact submit line, and switches to CHUNK automatically if the
#    point count exceeds a typical SLURM MaxArraySize.
#
# 2. Submit exactly what step 1 printed, e.g.:
#      OUTDIR=results/allocation sbatch --array=0-269 slurm/snail_sweep.slurm
#    or, chunked (one task runs CHUNK points in series -- raise --time):
#      OUTDIR=results/allocation CHUNK=4 sbatch --array=0-67 slurm/snail_sweep.slurm
#
# 3. Find gaps (a point file is written only on success, so a missing file = a
#    failed/killed task):
#      uv run python run_sweep_zhou.py missing --outdir results/allocation
#    -> writes results/allocation/missing.txt and prints a RESUME= line.
#
# 4. Resubmit only the gaps (array is 0..M-1 over the LIST, so scattered indices
#    are fine):
#      RESUME=results/allocation/missing.txt OUTDIR=results/allocation \
#          sbatch --array=0-11 slurm/snail_sweep.slurm
#
# 5. Collect to summary.csv (+ prints the best placement / mean DRAG gain):
#      uv run python run_sweep_zhou.py collect --outdir results/allocation
#      uv run python run_sweep_zhou.py chevrons --outdir results/allocation  # if --stark
#
# Small grids don't need the array at all -- pool them on one fat node:
#      MODE=local NPROC=16 OUTDIR=results/allocation sbatch slurm/snail_sweep.slurm
#
# -----------------------------------------------------------------------------
# WHICH FLAGS GO WHERE  (this trips people up -- read once)
# -----------------------------------------------------------------------------
# `mode=point` LOADS THE CONFIG FROM grid.json, which overwrites anything passed
# on the command line. run_sweep_zhou re-applies only a specific set afterwards.
#
#   RUN-TIME (settable here, per task -- safe to add on a rerun of an existing grid):
#       STARK, STARK_MATCH, STARK_JOBS, DRAG_SUBHARM, SUBHARM_MODES,
#       and every GRAPE knob (GRAPE, GRAPE_BACKEND, GRAPE_ALG, GRAPE_NBASIS,
#       GRAPE_NCTRL, GRAPE_CUTOFF, GRAPE_MAXITER, GRAPE_WARMSTART)
#
#   PREPARE-TIME ONLY (silently ignored if passed at task time -- they must be on
#   the `prepare` command, because they are baked into grid.json):
#       --device, --sweep, --t-g-ns / --target-eta, --calibrate,
#       --calibrate-iters, --drag, --drag-compare, --drag-compare-below-MHz,
#       --no-spectator, --no-integrate, --operating-point, --stark-span-MHz,
#       --stark-points, and all grid axes (--wb-GHz, --spec-GHz, --specfreqs,
#       --beats, --beat-span-MHz, --clip-band, --drags)
#
#   So: `--stark` can be turned on here, but its SPAN/POINTS come from the grid.
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#   OUTDIR    (required) sweep directory containing grid.json
#   RUNNER    driver script                              [run_sweep_zhou.py]
#   MODE      point (array) | local (pool on one node)   [point]
#   CHUNK     points per array task, in series           [1]
#   RESUME    resume-list file from `missing`            [unset]
#   NPROC     worker processes for MODE=local            [cpus-per-task]
#   LAUNCHER  how to invoke python                       [uv run python]
#   CODE_DIR  where the .py files live                   [submit dir]
#   GRID      explicit path to grid.json                 [OUTDIR/grid.json]
#
#   STARK=1           per-point AC-Stark chevron (drive at the shifted resonance)
#   STARK_MATCH=1     chevron uses the ACTUAL pulse (so DRAG-on points calibrate
#                     onto the DRAG-on resonance); needs STARK=1
#   STARK_JOBS=n      processes for the chevron's offset scan [cpus-per-task]
#   DRAG_SUBHARM=1    include w_i/2 = w_p subharmonic channels in the DRAG target
#   SUBHARM_MODES=a,b restrict which modes' subharmonics are considered
#
#   GRAPE=1           per-point optimal control; adds grape_baseline_F / F_grape /
#                     dF_grape / leak_grape to summary.csv and stores the optimized
#                     envelope in the point .npz. Needs the integrated run.
#   GRAPE_BACKEND     qutip (qutip-qoc GOAT/JOPT, exact propagation, SLOW) |
#                     reduced (in-house scipy over the rotating-frame model, fast)
#                                                        [qutip]
#   GRAPE_ALG         GOAT | JOPT | CRAB   (qutip backend)                     [GOAT]
#                     GOAT/JOPT are qutip-qoc GRADIENT methods (JOPT needs
#                       qutip-jax + jax); both need `pip install qutip-qoc`.
#                     CRAB is GRADIENT-FREE over a randomized Fourier basis and
#                       needs only qutip -- no qutip-qoc. Use it when a gradient
#                       method stalls, when qutip-qoc is unavailable, or when you
#                       want the objective to be the exact QuTiP propagator.
#   CRAB_RESTARTS     [CRAB] DCRAB super-iterations; each appends a fresh random
#                     basis and warm-starts from the previous optimum, so the
#                     fidelity is monotone. Params grow by 4*GRAPE_NBASIS each
#                     time and Nelder-Mead degrades in high dimension -- a few
#                     harmonics and 1-3 restarts is the sweet spot.            [1]
#   CRAB_SCORE        [CRAB] qutip (exact propagator_columns per evaluation) |
#                     reduced (fast rotating-frame model, for exploration) [qutip]
#   CRAB_SEED         [CRAB] RNG seed for the random basis (reproducible)  [random]
#   CRAB_METHOD       [CRAB] gradient-free scipy method                [Nelder-Mead]
#   GRAPE_NBASIS      sin() basis functions per quadrature (qutip backend)      [6]
#   GRAPE_NCTRL       control points (reduced) / tlist resolution (qutip)       [24]
#   GRAPE_CUTOFF      rotating-frame carrier cutoff in GHz                      [1.0]
#   GRAPE_MAXITER     optimizer iteration cap                                   [200]
#   GRAPE_WARMSTART=1 on DRAG-off points, seed from a DRAG pulse at the nearest
#                     beat (better start near a collision; auto-skipped inside the
#                     drag-skip window, where the 1/beat quadrature blows up)
#
# RECOMMENDED GRAPE STRATEGY: every qutip-backend optimizer is minutes-to-hours
# PER POINT (GOAT integrates the exact gradient system each iteration; CRAB is
# gradient-free and so needs many exact propagations). So sweep broadly with the
# cheap optimizer and escalate only on the interesting rows:
#   # 1. broad pass, fast reduced optimizer
#   GRAPE=1 GRAPE_BACKEND=reduced OUTDIR=... sbatch --array=0-269 slurm/snail_sweep.slurm
#   # 2. exact cross-check on a hand-picked subset (gradient method)
#   GRAPE=1 GRAPE_BACKEND=qutip GRAPE_ALG=GOAT RESUME=picked.txt OUTDIR=... \
#       sbatch --array=0-9 slurm/snail_sweep.slurm
#   # 3. where GOAT stalls (dF ~ 0 but you expect headroom), retry gradient-free
#   GRAPE=1 GRAPE_BACKEND=qutip GRAPE_ALG=CRAB CRAB_RESTARTS=3 \
#       RESUME=stalled.txt OUTDIR=... sbatch --array=0-3 slurm/snail_sweep.slurm
# (`picked.txt`/`stalled.txt` are hand-written lists of point indices, one per line
#  -- the RESUME mechanism does not care that they came from `missing`.)
# (`picked.txt` is just a hand-written list of point indices, one per line -- the
#  RESUME mechanism does not care that it came from `missing`.)
# =============================================================================
set -euo pipefail

OUTDIR="${OUTDIR:?set OUTDIR=<sweep dir containing grid.json>}"
RUNNER="${RUNNER:-run_sweep_zhou.py}"
MODE="${MODE:-point}"
CHUNK="${CHUNK:-1}"
LAUNCHER="${LAUNCHER:-uv run python}"
NPROC="${NPROC:-${SLURM_CPUS_PER_TASK:-4}}"

cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"

GRID_PATH="${GRID:-${OUTDIR}/grid.json}"
if [ ! -f "${GRID_PATH}" ]; then
    echo "ERROR: no grid at ${GRID_PATH} -- run \`${RUNNER} prepare\` first." >&2
    exit 2
fi

if [ "${MODE}" != "local" ] && [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: MODE=point needs an ARRAY job. Submit the line that \`prepare\` printed," >&2
    echo "       e.g.  OUTDIR=${OUTDIR} sbatch --array=0-<N-1> slurm/snail_sweep.slurm" >&2
    echo "       (or pool on a single node with  MODE=local NPROC=16 ...)" >&2
    exit 2
fi

# ---- assemble the run-time flags that SURVIVE load_grid (see notes above) ----
OPTS=""
[ -n "${STARK:-}"         ] && OPTS="${OPTS} --stark"
[ -n "${STARK_MATCH:-}"   ] && OPTS="${OPTS} --stark-match-pulse"
[ -n "${STARK_JOBS:-}"    ] && OPTS="${OPTS} --stark-jobs ${STARK_JOBS}"
[ -n "${DRAG_SUBHARM:-}"  ] && OPTS="${OPTS} --drag-subharmonic"
[ -n "${SUBHARM_MODES:-}" ] && OPTS="${OPTS} --subharmonic-modes ${SUBHARM_MODES}"
[ -n "${GRAPE:-}"           ] && OPTS="${OPTS} --grape"
[ -n "${GRAPE_WARMSTART:-}" ] && OPTS="${OPTS} --grape-warmstart-drag"
[ -n "${GRAPE_BACKEND:-}"   ] && OPTS="${OPTS} --grape-backend ${GRAPE_BACKEND}"
[ -n "${GRAPE_ALG:-}"       ] && OPTS="${OPTS} --grape-alg ${GRAPE_ALG}"
[ -n "${GRAPE_NBASIS:-}"    ] && OPTS="${OPTS} --grape-nbasis ${GRAPE_NBASIS}"
[ -n "${GRAPE_NCTRL:-}"     ] && OPTS="${OPTS} --grape-nctrl ${GRAPE_NCTRL}"
[ -n "${GRAPE_CUTOFF:-}"    ] && OPTS="${OPTS} --grape-cutoff-GHz ${GRAPE_CUTOFF}"
[ -n "${GRAPE_MAXITER:-}"   ] && OPTS="${OPTS} --grape-maxiter ${GRAPE_MAXITER}"
[ -n "${CRAB_RESTARTS:-}"   ] && OPTS="${OPTS} --grape-crab-restarts ${CRAB_RESTARTS}"
[ -n "${CRAB_SEED:-}"       ] && OPTS="${OPTS} --grape-crab-seed ${CRAB_SEED}"
[ -n "${CRAB_SCORE:-}"      ] && OPTS="${OPTS} --grape-crab-score ${CRAB_SCORE}"
[ -n "${CRAB_METHOD:-}"     ] && OPTS="${OPTS} --grape-crab-method ${CRAB_METHOD}"
[ -n "${GRID:-}"            ] && OPTS="${OPTS} --grid ${GRID}"

# ---- threads: avoid oversubscription when the job forks worker processes ------
# The chevron (STARK) and MODE=local both fork; pin BLAS to 1 thread there.
if [ "${MODE}" = "local" ] || [ -n "${STARK:-}" ]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg

# module load python/3.12      # or: conda activate qsim
# source .venv/bin/activate    # (then set LAUNCHER=python)
# GRAPE_BACKEND=qutip with GRAPE_ALG=GOAT/JOPT needs: pip install qutip-qoc
#   (+ qutip-jax jax for JOPT).  GRAPE_ALG=CRAB needs only qutip itself.

mkdir -p "${OUTDIR}/points"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}:${SLURM_ARRAY_TASK_ID:-none}"
echo "mode=${MODE} outdir=${OUTDIR} grid=${GRID_PATH} chunk=${CHUNK} opts=${OPTS:-<none>}"
echo "start=$(date -Is)"

if [ "${MODE}" = "local" ]; then
    # One node pools all points; no array needed.
    srun ${LAUNCHER} "${RUNNER}" local --outdir "${OUTDIR}" --nproc "${NPROC}" ${OPTS}
else
    RESUME_OPT=""
    if [ -n "${RESUME:-}" ]; then
        [ -f "${RESUME}" ] || { echo "ERROR: RESUME file ${RESUME} not found" >&2; exit 2; }
        RESUME_OPT="--resume-list ${RESUME}"
        echo "resume-list=${RESUME} ($(wc -l < "${RESUME}") indices); --index is the CHUNK"
        echo "  position into that list, not a grid index."
    fi
    srun ${LAUNCHER} "${RUNNER}" point \
        --outdir "${OUTDIR}" \
        --index "${SLURM_ARRAY_TASK_ID}" \
        --chunk "${CHUNK}" \
        ${RESUME_OPT} ${OPTS}
fi

echo "done end=$(date -Is)"
echo
echo "NEXT:  ${RUNNER} missing  --outdir ${OUTDIR}   # gaps + resubmit line"
echo "       ${RUNNER} collect  --outdir ${OUTDIR}   # -> summary.csv"
