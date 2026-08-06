#!/bin/bash
#SBATCH --job-name=snail_calmap
# NOTE: #SBATCH directives are parsed by sbatch BEFORE any shell runs, so they
# CANNOT reference $OUTDIR/$DEVICE, and any directory in --output must pre-exist.
#SBATCH --output=snail_calmap_%j.out
#SBATCH --error=snail_calmap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # ENGINE=qutip parallelizes over the grid with joblib
                                 # (JOBS below). For ENGINE=reduced drop to ~4: it is a
                                 # single process and the matrices (dim 45-135) are too
                                 # small for BLAS threading to help much.
#SBATCH --mem=16G
#SBATCH --time=04:00:00          # 41x41 = 1681 grid points.
                                 #   ENGINE=qutip   : 4 sesolve trajectories per point;
                                 #                    ~10-60 min at JOBS=16.
                                 #   ENGINE=reduced : hours single-core at full resolution.
                                 # Shrink WP_POINTS/AMP_POINTS before raising this.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu        # match your sweep jobs
# #SBATCH --account=YOUR_ALLOC
#
# =============================================================================
# 2D calibration landscape: pump frequency offset x pump strength
# =============================================================================
# Scans the (offset, amplitude) plane in ONE pass and writes the optimum into the
# device JSON as a named operating point. This replaces the alternating 1D
# amplitude/frequency tune-up, which rails when the AC-Stark shift and the Rabi
# amplitude feed back on each other (the optimum is a diagonal ridge, not a cross).
#
# TWO ENGINES:
#   ENGINE=qutip   exact, compiled qt.sesolve on the FULL Hamiltonian via
#                  ZhouCoupler.propagator_columns, parallel over the grid (JOBS).
#                  This is the trustworthy map. Needs QuTiP.
#   ENGINE=reduced fast NumPy/SciPy rotating-frame model. No QuTiP, cheap enough
#                  for a login node. It DROPS the fast leakage carriers and so
#                  OVER-READS the swap at strong drive -- use it for quick looks
#                  and confirm the optimum with ENGINE=qutip.
#
# TWO METRICS:
#   METRIC=fidelity  leakage-aware iSWAP fidelity F (default; what you calibrate on)
#   METRIC=transfer  swap population P(|01>->|10>) = |U[2,1]|^2 -- the chevron-style
#                    map. Use this to SEE whether the gate completes at all.
#
# -----------------------------------------------------------------------------
# QUICK START
# -----------------------------------------------------------------------------
#   # trustworthy map + save the optimum as operating point "nominal"
#   DEVICE=evan_device.json T_G=154.3 POINT=nominal \
#       sbatch slurm/snail_calibration_map.slurm
#
#   # swap-population map (does the gate close?), no point written
#   DEVICE=evan_device.json T_G=154.3 METRIC=transfer POINT= \
#       sbatch slurm/snail_calibration_map.slurm
#
#   # cheap look first, on a login node (no QuTiP needed):
#   DEVICE=evan_device.json T_G=154.3 ENGINE=reduced POINT= \
#       LAUNCHER="uv run python" bash slurm/snail_calibration_map.slurm
#
# Then USE the point in a sweep (it supplies the calibrated amp_scale + wp_offset):
#   uv run python run_sweep_zhou.py prepare --device evan_device.json \
#       --operating-point nominal --outdir results/sweep
#
# -----------------------------------------------------------------------------
# CHOOSING T_G  (peak |eta| <-> gate time, for THIS device family)
# -----------------------------------------------------------------------------
# The raised-cosine full-iSWAP normalization gives  t_g = 1 / (12 g3 la lb eta),
# so with g3 = 60 MHz and la = lb = 0.1:   peak |eta| = 138.9 / t_g[ns]
#     eta 0.8 -> 173.6 ns    eta 1.0 -> 138.9 ns    eta 1.4 ->  99.2 ns
#     eta 1.8 ->  77.2 ns    eta 2.0 ->  69.4 ns
# At these parameters the coupling is weak (iSWAP rate 6 g3 la lb |eta| is only
# ~6.5 MHz at eta=1.8), and exact integration shows the swap does NOT complete at
# eta~1.8: the amplitude needed to close it in 77 ns dumps population into the
# coupler and the qubit |2> levels. Start near eta ~ 0.8-1.0 (T_G ~ 140-175) where
# a clean vertex exists, and treat eta>~1.4 as leakage-limited unless g3*lambda^2
# is larger. Also note the chevron is only a few MHz wide here, so keep WP_SPAN
# tight (~20-40 MHz) or the maximum washes out between samples.
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#   DEVICE  (required) device JSON (bare name resolves under devices/)
#   T_G     (required) gate duration in ns
#   POINT   operating-point name to write; POINT= (empty) means DO NOT save [nominal]
#   OVERWRITE=1        allow POINT to replace an existing entry
#   ENGINE  qutip | reduced                                     [qutip]
#   METRIC  fidelity | transfer                                 [fidelity]
#   JOBS    joblib workers over the grid (ENGINE=qutip)         [cpus-per-task]
#   WP_SPAN full pump-offset width in MHz (centred on 0)        [40]
#   WP_POINTS / AMP_POINTS   grid resolution                    [41 / 41]
#   AMP_LO / AMP_HI          amplitude-scale range (x eta_nom)  [0.6 / 1.4]
#   CUTOFF  carrier cutoff in GHz (ENGINE=reduced only)         [1.0]
#   COUPLER_LEVELS  override the device coupler truncation      [device]
#   ATOL / RTOL / NSTEPS     solver controls (ENGINE=qutip)     [1e-10/1e-8/500000]
#   FIG / NPZ                output paths          [figs/calibration_map_<POINT>.*]
#   LAUNCHER / CODE_DIR      as in snail_sweep.slurm
#
# CONTEXT -- which gate the point is calibrated FOR. Set at most one spectator form:
#   WA=3.5            override w_a
#   WB=5.5            override w_b (the target sweep's swept frequency)
#   SPEC_ABS=4.0      absolute spectator frequency
#   DELTA=1.2         spectator-sweep detuning  Delta = w_b - w_spec
#   DRAG_BEAT=-0.04   map the DRAG-on gate instead of the bare one
# A point is only valid for the context it was calibrated in; run_sweep_zhou warns
# on mismatch (--operating-point-strict makes that an error).
#
# CONVERGENCE AT STRONG DRIVE: raise COUPLER_LEVELS (5 -> 6 -> 7) and confirm the
# bright region stops moving. A map that is not truncation-converged will show a
# fictitiously high optimum.
#
# RACE WARNING: --save-point does a read-modify-write on the device JSON. Do NOT
# run several of these concurrently against the SAME device file -- points will be
# lost. Calibrate contexts serially, or give each context its own device file.
# =============================================================================
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
T_G="${T_G:?set T_G=<gate length in ns>}"
POINT="${POINT-nominal}"           # note: ${VAR-default}, so POINT= means "empty"
LAUNCHER="${LAUNCHER:-uv run python}"
ENGINE="${ENGINE:-qutip}"
METRIC="${METRIC:-fidelity}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-1}}"
WP_SPAN="${WP_SPAN:-40}"
WP_POINTS="${WP_POINTS:-41}"
AMP_LO="${AMP_LO:-0.6}"
AMP_HI="${AMP_HI:-1.4}"
AMP_POINTS="${AMP_POINTS:-41}"
CUTOFF="${CUTOFF:-1.0}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-8}"
NSTEPS="${NSTEPS:-500000}"
if [ -n "${POINT}" ]; then TAG="_${POINT}"; else TAG=""; fi
FIG="${FIG:-figs/calibration_map${TAG}.png}"
NPZ="${NPZ:-figs/calibration_map${TAG}.npz}"

# optional context / truncation flags
CTX=""
[ -n "${WA:-}"             ] && CTX="${CTX} --wa-GHz ${WA}"
[ -n "${WB:-}"             ] && CTX="${CTX} --wb-GHz ${WB}"
[ -n "${SPEC_ABS:-}"       ] && CTX="${CTX} --spec-abs-GHz ${SPEC_ABS}"
[ -n "${DELTA:-}"          ] && CTX="${CTX} --delta-GHz ${DELTA}"
[ -n "${DRAG_BEAT:-}"      ] && CTX="${CTX} --drag-beat-GHz ${DRAG_BEAT}"
[ -n "${COUPLER_LEVELS:-}" ] && CTX="${CTX} --coupler-levels ${COUPLER_LEVELS}"

SAVE=""
if [ -n "${POINT}" ]; then
    SAVE="--save-point ${POINT}"
    [ -n "${OVERWRITE:-}" ] && SAVE="${SAVE} --overwrite"
fi

# joblib workers get the cores; keep per-worker BLAS single-threaded so that
# (workers x BLAS threads) does not exceed the allocation and thrash.
if [ "${ENGINE}" = "qutip" ] && [ "${JOBS}" -gt 1 ]; then
    export OMP_NUM_THREADS=1
else
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg

# module load python/3.12      # or: conda activate qsim
# source .venv/bin/activate    # (then set LAUNCHER=python)
# ENGINE=qutip needs QuTiP importable here; ENGINE=reduced does not.
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"
mkdir -p "$(dirname "${FIG}")" "$(dirname "${NPZ}")"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} device=${DEVICE} t_g=${T_G}"
echo "engine=${ENGINE} metric=${METRIC} jobs=${JOBS} point=${POINT:-<none, not saving>}"
echo "grid=${WP_POINTS}x${AMP_POINTS} span=${WP_SPAN}MHz amp=[${AMP_LO},${AMP_HI}]"
echo "start=$(date -Is)"

srun ${LAUNCHER} calibration_map.py \
    --device "${DEVICE}" --t-g-ns "${T_G}" ${CTX} \
    --engine "${ENGINE}" --jobs "${JOBS}" \
    --wp-span-MHz "${WP_SPAN}" --wp-points "${WP_POINTS}" \
    --amp-lo "${AMP_LO}" --amp-hi "${AMP_HI}" --amp-points "${AMP_POINTS}" \
    --cutoff-GHz "${CUTOFF}" --metric "${METRIC}" \
    --atol "${ATOL}" --rtol "${RTOL}" --nsteps "${NSTEPS}" \
    --save-npz "${NPZ}" --out "${FIG}" ${SAVE}

echo "done end=$(date -Is)"
echo
echo "CHECK THE FIGURE (${FIG}): you want ONE smooth interior maximum."
echo "  * optimum against an amplitude bound -> widen AMP_LO/AMP_HI and rerun"
echo "    BEFORE using the point (a railed optimum is not a calibration)."
echo "  * no bright vertex at all -> the gate does not close at this T_G; try a"
echo "    longer T_G (lower eta) and METRIC=transfer to see the swap directly."
echo "  * ENGINE=reduced -> re-run the winner with ENGINE=qutip before trusting it."
echo "Raw arrays are in ${NPZ} (offsets_MHz, amps, Z, leakage, best_*, con_*/ope_*)."
