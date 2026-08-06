#!/bin/bash
#SBATCH --job-name=snail_grape
# NOTE: #SBATCH directives are parsed before any shell runs, so they cannot
# reference $DEVICE, and any directory in --output must pre-exist.
#SBATCH --output=snail_grape_%j.out
#SBATCH --error=snail_grape_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # single optimization process; the cores help via
                                 # BLAS on the propagator/gradient matrix products.
#SBATCH --mem=16G
#SBATCH --time=06:00:00          # BACKEND=reduced : minutes.
                                 # BACKEND=qutip   : GOAT integrates the augmented
                                 #   gradient system through the FULL Hamiltonian
                                 #   every iteration -> ~10 min to hours depending on
                                 #   dim, N_BASIS and MAXITER. Start with MAXITER=30.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC
#
# =============================================================================
# Single-point optimal control for the SNAIL iSWAP  (grape.py)
# =============================================================================
# Optimizes the complex pump envelope eta(t) against the leakage-aware iSWAP
# fidelity and reports it next to the DRAG raised-cosine baseline, so the number
# you get is the OPTIMAL-CONTROL HEADROOM over the gate you are actually running.
#
# WHY NOT PLAIN GRAPE: this gate is control-NONLINEAR -- the pump enters as eta,
# eta^2 and eta^3 through g3*X^3, and on this device the nonlinear terms carry
# 31% of the Hamiltonian magnitude at eta=0.5, 51% at eta=1.0 and 68% at eta=1.8
# (where the eta^2 piece is LARGER than the linear one). Textbook control-linear
# GRAPE, H = H_d + sum_k u_k H_k, cannot represent that, and linearizing in eta
# throws away the very terms (coupler heating, |2> leakage) you are fighting.
# Both backends here handle the nonlinearity exactly, via the chain rule through
# d(eta^p conj(eta)^q)/d(eta):
#
#   BACKEND=qutip    QuTiP's optimal-control package (qutip-qoc) with an ANALYTIC
#                    control ansatz: ALG=GOAT differentiates the exact propagator
#                    by integrating the gradient equations of motion; ALG=JOPT does
#                    it by JAX auto-differentiation. Propagation is QuTiP's.
#                    Needs:  pip install qutip-qoc      (+ qutip-jax jax for JOPT)
#   BACKEND=reduced  in-house scipy L-BFGS-B over the rotating-frame reduced model
#                    (no QuTiP). Fast; validate its winner with BACKEND=qutip.
#
# -----------------------------------------------------------------------------
# QUICK START
# -----------------------------------------------------------------------------
#   # exact optimal control at a chosen operating point
#   DEVICE=evan_device.json T_G=154.3 sbatch slurm/snail_grape.slurm
#
#   # against a specific spectator, with the DRAG-on gate as the baseline
#   DEVICE=evan_device.json T_G=154.3 SPEC_ABS=4.6 DRAG_BEAT=-0.04 \
#       sbatch slurm/snail_grape.slurm
#
#   # hard near-resonant collision: seed the optimizer from the DRAG pulse
#   DEVICE=evan_device.json T_G=154.3 SPEC_ABS=4.6 WARMSTART_BEAT=-0.04 \
#       sbatch slurm/snail_grape.slurm
#
#   # cheap first pass (no QuTiP; fine on a login node)
#   DEVICE=evan_device.json T_G=154.3 BACKEND=reduced \
#       bash slurm/snail_grape.slurm
#
# Use the CALIBRATED operating point, not the raw normalization: read amp_scale and
# wp_offset off `operating_points.py --device <dev>` (written by
# snail_calibration_map.slurm) and pass them as AMP_SCALE / WP_OFFSET, so GRAPE
# starts from the gate you actually run rather than the analytic pi/2 guess.
#
# For a GRAPE column across a whole sweep instead of one point, use
# snail_sweep.slurm with GRAPE=1 (see its GRAPE_* variables) -- same optimizer,
# invoked per grid point.
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#   DEVICE   (required) device JSON (bare name resolves under devices/)
#   T_G      (required) gate duration in ns
#   BACKEND  qutip | reduced                                      [qutip]
#   ALG      GOAT | JOPT   (BACKEND=qutip; JOPT needs JAX)        [GOAT]
#   N_BASIS  sin() basis functions per quadrature (qutip)         [6]
#   N_CTRL   control points (reduced) / tlist resolution (qutip)  [24]
#   CUTOFF   rotating-frame carrier cutoff in GHz                 [1.0]
#   MAXITER  optimizer iteration cap                              [200]
#   AMP_SCALE   calibrated pump-amplitude correction              [1.0]
#   WP_OFFSET   calibrated pump-frequency offset in GHz           [0.0]
#   SPEC_ABS    absolute spectator frequency (loads a spectator)  [none]
#   DRAG_BEAT   beat for the DRAG BASELINE quadrature             [none]
#   WARMSTART_BEAT  seed the optimizer from a DRAG pulse at this beat, leaving the
#                   baseline (and therefore the reported dF) unchanged   [none]
#   LAUNCHER / CODE_DIR   as in snail_sweep.slurm
#
# READING THE OUTPUT: F_baseline is the DRAG/raised-cosine gate, F_grape the
# optimized one, and both are scored with the SAME leakage-aware metric the sweeps
# use -- so F_baseline should track the sweep's F_avg at the same point, and a large
# gap between them means the reduced model and the full sim disagree there, not
# that GRAPE won. A big dF means the error is optimal-control-recoverable (dynamic
# Stark shift, off-resonant spectator exchange); a small dF on a near-resonant
# collision means it is NOT -- that needs frequency allocation, not pulse shaping.
# Either way, re-score the optimized envelope in the full QuTiP iswap_fidelity
# before quoting an absolute number.
# =============================================================================
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
T_G="${T_G:?set T_G=<gate length in ns>}"
LAUNCHER="${LAUNCHER:-uv run python}"
BACKEND="${BACKEND:-qutip}"
ALG="${ALG:-GOAT}"
N_BASIS="${N_BASIS:-6}"
N_CTRL="${N_CTRL:-24}"
CUTOFF="${CUTOFF:-1.0}"
MAXITER="${MAXITER:-200}"
AMP_SCALE="${AMP_SCALE:-1.0}"
WP_OFFSET="${WP_OFFSET:-0.0}"

OPTS=""
[ -n "${SPEC_ABS:-}"       ] && OPTS="${OPTS} --spec-abs-GHz ${SPEC_ABS}"
[ -n "${DRAG_BEAT:-}"      ] && OPTS="${OPTS} --drag-beat-GHz ${DRAG_BEAT}"
[ -n "${WARMSTART_BEAT:-}" ] && OPTS="${OPTS} --warmstart-drag-beat-GHz ${WARMSTART_BEAT}"

# single optimization process -> let BLAS use the allocation
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg

# module load python/3.12      # or: conda activate qsim
# source .venv/bin/activate    # (then set LAUNCHER=python)
# BACKEND=qutip needs: pip install qutip-qoc   (ALG=JOPT also needs qutip-jax jax)
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} device=${DEVICE} t_g=${T_G}"
echo "backend=${BACKEND} alg=${ALG} n_basis=${N_BASIS} n_ctrl=${N_CTRL} maxiter=${MAXITER}"
echo "amp_scale=${AMP_SCALE} wp_offset=${WP_OFFSET} GHz opts=${OPTS:-<bare gate>}"
echo "start=$(date -Is)"

srun ${LAUNCHER} grape.py \
    --device "${DEVICE}" --t-g-ns "${T_G}" \
    --amp-scale "${AMP_SCALE}" --wp-offset-GHz "${WP_OFFSET}" \
    --backend "${BACKEND}" --alg "${ALG}" --n-basis "${N_BASIS}" \
    --n-ctrl "${N_CTRL}" --cutoff-GHz "${CUTOFF}" --maxiter "${MAXITER}" \
    ${OPTS}

echo "done end=$(date -Is)"
