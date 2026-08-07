#!/bin/bash
#SBATCH --job-name=snail_gpu_scan
# NOTE: #SBATCH directives are parsed before any shell runs, so they cannot
# reference $DEVICE, and any directory in --output must pre-exist.
#SBATCH --output=snail_gpu_scan_%j.out
#SBATCH --error=snail_gpu_scan_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             # ONE process drives the device; the batching is
                                 # internal (vmap), so do NOT also raise --ntasks.
#SBATCH --cpus-per-task=4        # host-side term expansion + plotting only
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_ALLOC
#
# =============================================================================
# Batched calibration / chirp scan on a GPU  (jax_engine.py via calibration_map.py)
# =============================================================================
# The whole (amp x pump-offset) grid runs as ONE vmapped program: the device is
# fixed, so every grid point shares one sparse operator stack and varies only the
# frequency vector and the pulse parameters. That is why this asks for a single
# GPU and a single task -- unlike MODE=local in snail_sweep.slurm, there is no
# process-level fan-out here.
#
# BEFORE TRUSTING A NUMBER FROM THIS JOB, run the validation once per device:
#
#     python validate_engines.py --device <dev>.json --cutoff-scan
#
# It reports fidelity and max|U_engine - U_qutip| against the exact QuTiP solver
# across cutoffs, so ENGINE_CUTOFF below is chosen from data. The engine's own
# integrator is 4th-order and converges to ~1e-10; the CUTOFF is the physics
# approximation, and it is the one that can bite.
#
# ENGINE_CUTOFF DEFAULTS TO inf (exact) DELIBERATELY. A finite cutoff is a
# rotating-wave reduction that only holds at weak drive, and this gate frequently
# is not: measured on a 15 ns gate (|eta| = 9.3), a 3 GHz cutoff was off by
# max|dU| ~ 0.9 -- worthless -- while inf agreed with QuTiP to 5e-4 AND still ran
# 174x faster (0.56 s vs 97 s). The speed here comes from the CF4 integrator, the
# sparse operator stack and the batching, not from throwing terms away. Only lower
# ENGINE_CUTOFF if a cutoff-scan on YOUR operating point justifies it.
#
# -----------------------------------------------------------------------------
# QUICK START
# -----------------------------------------------------------------------------
#   # calibration map on GPU
#   DEVICE=2Gate4.9SNAIL.json T_G=77.2 sbatch slurm/snail_gpu_scan.slurm
#
#   # same, but scanning a linear chirp instead of a constant offset
#   DEVICE=2Gate4.9SNAIL.json T_G=77.2 CHIRP=0,0.05 sbatch slurm/snail_gpu_scan.slurm
#
#   # exact (no pruning) -- much slower, use to spot-check the cutoff choice
#   DEVICE=2Gate4.9SNAIL.json T_G=77.2 ENGINE_CUTOFF=inf CARRIER_RES=0.05 \
#       sbatch slurm/snail_gpu_scan.slurm
#
#   # CPU fallback: identical code path, just no device. Drop --gres when
#   # submitting, or run directly:
#   DEVICE=2Gate4.9SNAIL.json T_G=77.2 bash slurm/snail_gpu_scan.slurm
#
# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#   DEVICE          (required) device JSON (bare name resolves under devices/)
#   T_G             (required) gate duration in ns
#   ENGINE_CUTOFF   carrier cutoff in GHz; 'inf' is exact             [3.0]
#   CARRIER_RES     max |Omega|*dt for the CF4 propagator; 4th order,
#                   so halving it cuts the error ~16x                 [0.1]
#   BATCH           grid points per vmapped call; raise until the GPU
#                   is saturated or memory complains                  [64]
#   PRECISION       f64 | f32   (f32 is MIXED: f64 phases, c64 state)  [f64]
#   WP_SPAN_MHZ / WP_POINTS / AMP_LO / AMP_HI / AMP_POINTS  grid
#   CHIRP           comma list of Legendre chirp coefficients (GHz)   [none]
#   SPEC_ABS        absolute spectator frequency                      [none]
#   METRIC          fidelity | transfer                          [fidelity]
#   LAUNCHER / CODE_DIR   as in snail_sweep.slurm
#
# ON PRECISION: PRECISION=f32 is unlikely to help much here. At dim ~ 135 each
# step is a small sparse scatter-add, so the kernel is launch/bandwidth bound
# rather than FLOP bound, and the tensor-core speedups quoted for narrow floats
# do not apply. Measure before switching:
#     python validate_engines.py --device <dev>.json --precision-scan
# float16 is not offered at all: near t = 77 ns its spacing is 0.0625 ns while
# the required step is ~0.0024 ns, so t + dt == t and the integrator cannot move.
# =============================================================================
set -euo pipefail

DEVICE="${DEVICE:?set DEVICE=<device>.json}"
T_G="${T_G:?set T_G=<gate length in ns>}"
LAUNCHER="${LAUNCHER:-uv run python}"
ENGINE_CUTOFF="${ENGINE_CUTOFF:-inf}"
CARRIER_RES="${CARRIER_RES:-0.1}"
BATCH="${BATCH:-64}"
PRECISION="${PRECISION:-f64}"
WP_SPAN_MHZ="${WP_SPAN_MHZ:-40}"
WP_POINTS="${WP_POINTS:-41}"
AMP_LO="${AMP_LO:-0.6}"
AMP_HI="${AMP_HI:-1.4}"
AMP_POINTS="${AMP_POINTS:-41}"
METRIC="${METRIC:-fidelity}"

OPTS=""
[ -n "${SPEC_ABS:-}" ] && OPTS="${OPTS} --spec-abs-GHz ${SPEC_ABS}"
[ -n "${CHIRP:-}" ]    && OPTS="${OPTS} --chirp-GHz ${CHIRP}"

cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"

# module load cuda            # then: uv pip install -e '.[gpu]'

# Let JAX grow its allocation instead of grabbing the whole device up front --
# otherwise a second job (or the plotting step) on a shared GPU fails to allocate.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The engine batches internally; host BLAS threads only fight it for cores.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} device=${DEVICE} t_g=${T_G}"
echo "engine=jax cutoff=${ENGINE_CUTOFF} carrier_res=${CARRIER_RES} batch=${BATCH} precision=${PRECISION}"
echo "grid=${AMP_POINTS}x${WP_POINTS} span=${WP_SPAN_MHZ}MHz amp=[${AMP_LO},${AMP_HI}] metric=${METRIC}"
echo "opts=${OPTS:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible (CPU fallback)"
echo "start=$(date -Is)"

srun ${LAUNCHER} calibration_map.py \
    --device "${DEVICE}" --t-g-ns "${T_G}" \
    --engine jax \
    --engine-cutoff-GHz "${ENGINE_CUTOFF}" \
    --engine-carrier-resolution "${CARRIER_RES}" \
    --engine-batch "${BATCH}" \
    --engine-precision "${PRECISION}" \
    --wp-span-MHz "${WP_SPAN_MHZ}" --wp-points "${WP_POINTS}" \
    --amp-lo "${AMP_LO}" --amp-hi "${AMP_HI}" --amp-points "${AMP_POINTS}" \
    --metric "${METRIC}" \
    ${OPTS}

echo "done end=$(date -Is)"
