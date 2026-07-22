#!/bin/bash
#SBATCH --job-name=snail_sweep
# IMPORTANT: #SBATCH directives are parsed by sbatch BEFORE any shell runs, so
# they CANNOT see $OUTDIR or any runtime variable, and the directory named here
# must already exist or the task fails at launch with no log. We therefore write
# logs to the submit directory (which always exists). To route logs into OUTDIR
# instead, create it first and override on the CLI, e.g.:
#   mkdir -p results_zhou/logs
#   sbatch -o results_zhou/logs/point_%A_%a.out -e results_zhou/logs/point_%A_%a.err \
#          --array=0-119 snail_sweep.slurm
#SBATCH --output=snail_sweep_%A_%a.out
#SBATCH --error=snail_sweep_%A_%a.err
#SBATCH --array=0-59%32          # OVERRIDE on submit: sbatch --array=0-<M-1> snail_sweep.slurm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12        # 1 for plain sweeps; for --stark raise this (e.g. 8-16)
                                 # so the per-point chevron parallelizes over its offsets
                                 # (mode=point auto-uses SLURM_CPUS_PER_TASK). Override:
                                 #   sbatch --cpus-per-task=12 --array=... slurm/snail_sweep.slurm
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:30:00          # per-point wall limit; raise for large N / long t_g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wcalphon@syr.edu   # NOTE: on an array, ALL e-mails per task (120!);
                                       # use --mail-type=END,FAIL to cut the volume
# --- REQUIRED ON MOST CLUSTERS: set these or sbatch will REJECT the job ------
# (a rejected submit also shows nothing in squeue and writes no logs)
# #SBATCH --partition=cpu        # uncomment / edit for your cluster
# #SBATCH --account=YOUR_ALLOC
# ----------------------------------------------------------------------------

# One array task == one (channel, spec_freq, lam_spec, drag) grid point.
#   1) uv run run_sweep_zhou.py prepare --outdir results_zhou/   # prints the range
#   2) RUNNER=run_sweep_zhou.py OUTDIR=results_zhou/ \
#          sbatch --array=0-<M-1> snail_sweep.slurm              # use the printed range
#   3) collect (dependent job, see snail_collect.slurm) or:
#          uv run run_sweep_zhou.py collect --outdir results_zhou/
# The `%32` in --array caps concurrently running tasks at 32.
set -euo pipefail

OUTDIR="${OUTDIR:-results/results_zhou}"
RUNNER="${RUNNER:-run_sweep_zhou.py}"   # the Zhou-coupler sweep driver
LAUNCHER="${LAUNCHER:-uv run python}"   # how to invoke python; e.g. "python" after activating a venv
CHUNK="${CHUNK:-1}"                      # points per array task; set >1 when N > MaxArraySize
                                        #   (array size = ceil(N/CHUNK)); e.g. CHUNK=25 for 11551 pts
RESUME="${RESUME:-}"                     # path to a resume list (e.g. ${OUTDIR}/missing.txt) from
                                        #   `missing`; --index then selects the chunk-th slice of it.
RESUME_FLAG=""; [ -n "${RESUME}" ] && RESUME_FLAG="--resume-list ${RESUME}"
GPU_FLAG=""; [ -n "${GPU:-}" ] && GPU_FLAG="--gpu"   # GPU=1 -> qutip-jax/diffrax; also add "#SBATCH --gres=gpu:1"
mkdir -p "${OUTDIR}/points"

# Keep numerical libraries single-threaded: we parallelize over array tasks.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# --- environment: pick ONE for your cluster (uv works from the project dir) --
# module load python/3.12               # (Lmod site module), or:
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate qsim   # then LAUNCHER=python, or:
# source /path/to/venv/bin/activate                                   # then LAUNCHER=python
# GPU (optional): request a device and install a CUDA-enabled JAX, e.g.
#   #SBATCH --gres=gpu:1
#   module load cuda cudnn ; uv pip install "qutip-jax" "jax[cuda12]"
#   GPU only helps for large Hilbert spaces (many modes/levels); small couplers are faster on CPU.
# ----------------------------------------------------------------------------

# Run from the code root (the time_dependent_solver/ folder, parent of slurm/).
# Submit from there:  sbatch slurm/snail_sweep.slurm   (or set CODE_DIR to that folder).
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} runner=${RUNNER} outdir=${OUTDIR} resume=${RESUME:-none} gpu=${GPU:-0} start=$(date -Is)"

srun ${LAUNCHER} "${RUNNER}" point ${GPU_FLAG} ${RESUME_FLAG} \
    --outdir "${OUTDIR}" \
    --grid   "${OUTDIR}/grid.json" \
    --chunk  "${CHUNK}" \
    --index  "${SLURM_ARRAY_TASK_ID}"

echo "done task=${SLURM_ARRAY_TASK_ID} end=$(date -Is)"
