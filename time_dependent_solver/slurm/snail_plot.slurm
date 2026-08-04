#!/bin/bash
#SBATCH --job-name=snail_plot
# See snail_sweep.slurm: directives cannot expand $OUTDIR; logs go to the submit dir.
#SBATCH --output=snail_plot_%j.out
#SBATCH --error=snail_plot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wcalphon@syr.edu
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ALLOC

# Render figures for a finished sweep. Chain after collect:
#   JID=$(RUNNER=run_sweep_zhou.py OUTDIR=clean_v1 sbatch --parsable --array=0-49 slurm/snail_sweep.slurm)
#   CID=$(OUTDIR=clean_v1 sbatch --parsable --dependency=afterok:${JID} slurm/snail_collect.slurm)
#   OUTDIR=clean_v1 KIND=spectator sbatch --dependency=afterok:${CID} slurm/snail_plot.slurm
#
# KIND selects the plotter, matching how the sweep was prepared:
#   spectator  -> plot_results.py       (infidelity/leakage/heatmap vs Delta)
#   bare       -> plot_bare_sweep.py    (1D no-spectator sweep: infidelity + n_coupler)
#   target     -> plot_fidelity_map.py  (2D allocation map, plus the --diff panel)
#
# NOTE: the plotters take a LITERAL path, unlike run_sweep_zhou.py which maps a bare
# --outdir through paths.in_results. This script resolves it for you.
set -euo pipefail

OUTDIR="${OUTDIR:?set OUTDIR=<sweep outdir>}"
KIND="${KIND:?set KIND=spectator|bare|target}"
LAUNCHER="${LAUNCHER:-uv run python}"

export MPLBACKEND=Agg
# module load python/3.12   # or activate a venv/conda env, then LAUNCHER=python
cd "${CODE_DIR:-${SLURM_SUBMIT_DIR:-.}}"

# bare name -> results/<name>; an explicit path is passed through unchanged
RESULTS="${OUTDIR}"
[ -d "${RESULTS}" ] || RESULTS="results/${OUTDIR}"
[ -f "${RESULTS}/summary.csv" ] || {
    echo "no ${RESULTS}/summary.csv -- run collect first"; exit 1; }
echo "host=$(hostname) results=${RESULTS} kind=${KIND} start=$(date -Is)"

case "${KIND}" in
  spectator)
    srun ${LAUNCHER} plot_results.py --outdir "${RESULTS}"
    ;;
  bare)
    srun ${LAUNCHER} plot_bare_sweep.py --outdir "${RESULTS}" ${XAXIS:+--xaxis ${XAXIS}}
    ;;
  target)
    srun ${LAUNCHER} plot_fidelity_map.py --outdir "${RESULTS}" --metric infidelity --log
    srun ${LAUNCHER} plot_fidelity_map.py --outdir "${RESULTS}" --diff
    ;;
  *)
    echo "unknown KIND=${KIND} (expected spectator|bare|target)"; exit 2
    ;;
esac

echo "done end=$(date -Is)"
ls -1 "${RESULTS}/figs/" 2>/dev/null || true
