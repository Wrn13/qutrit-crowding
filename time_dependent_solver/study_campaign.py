"""Figure-campaign driver for the SNAIL iSWAP / DRAG study.

Runs the study end to end in dependency order and emits the figure set, with a
manifest recording every command, timing, and output so any figure is traceable.

The ordering is not cosmetic: everything downstream of calibration is
uninterpretable if the calibration landscape is not clean (a railed amplitude
scan produces fidelity scatter that looks like physics). So tier 0 calibrates
once, saves a named operating point, and every later tier reuses it via
``--operating-point`` instead of re-deriving a per-point tune-up.

Tiers
-----
0 ``calib``          (w_p offset x pump strength) landscape; saves the operating point.
1 ``mechanism``      DRAG I/Q time domain + spectral null (no simulation).
2 ``clean``          Isolated one-pump spectator collision -- the headline DRAG result.
3 ``allocation``     2D frequency-allocation map + DRAG-helps/hurts difference.
  ``subharm_spec``   Spectator on the pump 2nd harmonic (w_spec = 2 w_p).
  ``subharm_snail``  Bare 3-mode gate through the SNAIL subharmonic (w_c = 2 w_p).
4 ``eta_scan``       DRAG improvement dF vs pump strength eta (where DRAG stops paying).
  ``grape``          GRAPE vs DRAG at several gate lengths.
5 ``robustness``     Tier-2 sweep at t_g +/-5%: separates real collisions from
                     fixed-t_g fringe/revival accidents.

Usage
-----
    python campaign.py --list
    python campaign.py --dry-run                     # print the plan, run nothing
    python campaign.py --analytic --tier 0 --tier 3  # fast, no QuTiP dynamics
    python campaign.py --execute local --nproc 8     # full local run
    python campaign.py --execute slurm               # emit sbatch commands

``--analytic`` adds ``--no-integrate`` so the construction/plumbing path runs
without QuTiP; fidelity columns stay empty, so it validates the pipeline rather
than producing physics. Full figures need ``--execute local`` or ``slurm``.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

#: shared figure palette (matches the standalone mechanism figures)
BLUE, RED = "#2980B9", "#C0392B"

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


# --------------------------------------------------------------------------- #
# campaign context
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    """Resolved campaign settings shared by every stage.

    Parameters
    ----------
    device : str
        Base device JSON (bare name resolves under ``devices/``).
    outroot : str
        Prefix for per-stage results directories.
    execute : {'local', 'slurm'}
        Run sweeps locally, or emit sbatch commands without running them.
    analytic : bool
        Append ``--no-integrate`` (fast, QuTiP-free plumbing validation).
    nproc : int
        Workers for ``local`` mode.
    dry_run : bool
        Print commands without running them.
    quick : bool
        Coarse grids and fewer points: smoke-tests the whole campaign
        cheaply before committing cluster time. NOT for figures.
    op_point : str
        Operating-point name written by tier 0 and consumed by later tiers.
    figs_dir : str
        Where the mechanism-figure scripts live (tier 1).
    """
    device: str = "warren_device.json"
    outroot: str = "campaign"
    execute: str = "local"
    analytic: bool = False
    nproc: int = 4
    dry_run: bool = False
    quick: bool = False
    op_point: str = "campaign"
    figs_dir: str = os.path.dirname(HERE)
    log: List[Dict[str, Any]] = field(default_factory=list)

    def out(self, stage: str) -> str:
        """Results directory name for a stage (bare name -> ``results/``)."""
        return f"{self.outroot}_{stage}"

    def npts(self, n: int) -> int:
        """Sweep point count, thinned in --quick mode."""
        return max(5, n // 4) if self.quick else n

    def cal_pts(self) -> str:
        """Calibration-grid resolution per axis, coarse in --quick mode."""
        return "11" if self.quick else "41"


# --------------------------------------------------------------------------- #
# command helpers
# --------------------------------------------------------------------------- #
def _run(ctx: Context, cmd: Sequence[str], *, label: str = "",
         capture: bool = False, allow_fail: bool = False) -> Optional[str]:
    """Run one command, logging it into the manifest.

    Returns captured stdout when ``capture`` is set, else None.
    """
    pretty = " ".join(cmd)
    print(f"  $ {pretty}")
    entry: Dict[str, Any] = dict(stage=label, cmd=list(cmd))
    if ctx.dry_run:
        entry["skipped"] = "dry-run"
        ctx.log.append(entry)
        return None
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)
    entry["seconds"] = round(time.time() - t0, 2)
    entry["returncode"] = proc.returncode
    ctx.log.append(entry)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"stage {label!r} failed ({pretty})")
    return proc.stdout if capture else None


def _has_op_point(ctx: Context, device: str) -> bool:
    """True when ``ctx.op_point`` exists in the given device JSON.

    Lets a tier run standalone: without a tier-0 calibration the sweep falls back
    to the device's own amp_scale/wp_offset instead of failing.
    """
    try:
        from paths import resolve_device
        with open(resolve_device(device)) as fh:
            return ctx.op_point in (json.load(fh).get("operating_points") or {})
    except Exception:
        return False


def _sweep(ctx: Context, stage: str, prepare_args: Sequence[str], *,
           plot: Optional[Sequence[str]] = None,
           use_op_point: bool = True) -> str:
    """Prepare + run + plot one sweep; returns the results directory name.

    Parameters
    ----------
    stage : str
        Stage id (names the results directory).
    prepare_args : sequence of str
        Arguments after ``prepare`` (sweep kind, frequencies, flags).
    plot : sequence of str, optional
        Plot command; ``{outdir}`` is substituted.
    use_op_point : bool
        Pass ``--operating-point`` so the sweep uses the tier-0 calibration
        (silently skipped when that point has not been created yet).
    """
    outdir = ctx.out(stage)
    prep = [PY, "run_sweep_zhou.py", "prepare", *prepare_args,
            "--device", ctx.device, "--outdir", outdir]
    if use_op_point:
        if _has_op_point(ctx, ctx.device):
            prep += ["--operating-point", ctx.op_point]
        else:
            print(f"    note: no operating point {ctx.op_point!r} in {ctx.device}; "
                  f"using the device's own amp_scale/wp_offset (run tier 0 first "
                  f"for a calibrated campaign)")
    if ctx.analytic:
        prep += ["--no-integrate"]
    _run(ctx, prep, label=stage)

    if ctx.execute == "slurm":
        print(f"    [slurm] RUNNER=run_sweep_zhou.py OUTDIR={outdir} "
              f"sbatch slurm/snail_sweep.slurm    # then re-run with --execute local "
              f"or plot manually")
        return outdir
    _run(ctx, [PY, "run_sweep_zhou.py", "local", "--outdir", outdir,
               "--nproc", str(ctx.nproc)], label=stage)
    if plot:
        # plotters take a literal path, so substitute the RESOLVED results dir
        # (run_sweep_zhou maps a bare --outdir through in_results, plotters do not)
        try:
            from paths import in_results
            resolved = in_results(outdir)
        except Exception:
            resolved = os.path.join("results", outdir)
        _run(ctx, [c.format(outdir=resolved) for c in plot], label=stage,
             allow_fail=True)
    return outdir


def _derive_device(ctx: Context, name: str, overrides: Dict[str, Any]) -> str:
    """Write a device variant with ``overrides`` applied; return its bare name.

    Tiers that need a different pair or coupler frequency (the isolated-collision
    geometry, the reachable SNAIL subharmonic) get their own device file rather
    than mutating the base one.
    """
    from paths import resolve_device
    base_path = resolve_device(ctx.device)
    with open(base_path) as fh:
        raw = json.load(fh)
    raw.update(overrides)
    raw.pop("operating_points", None)          # a variant must be recalibrated
    out_name = f"_campaign_{name}.json"
    out_path = os.path.join(os.path.dirname(base_path), out_name)
    if not ctx.dry_run:
        with open(out_path, "w") as fh:
            json.dump(raw, fh, indent=2)
    print(f"    derived device {out_name}: {overrides}")
    return out_name


def _summary_path(outdir: str) -> str:
    """Absolute path to a stage's collected summary.csv."""
    sys.path.insert(0, HERE)
    from paths import in_results
    return os.path.join(in_results(outdir), "summary.csv")


def _read_summary(outdir: str) -> List[Dict[str, str]]:
    """Load a stage summary.csv (empty list when absent)."""
    path = _summary_path(outdir)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _fnum(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _spec_list(center: float, span_GHz: float, n: int) -> str:
    """Comma list of spectator-sweep Delta values centered on ``center``."""
    return ",".join(f"{v:.6f}" for v in
                    np.linspace(center - span_GHz / 2, center + span_GHz / 2, n))


def _wb_list(lo: float, hi: float, step: float) -> str:
    """Comma list of w_b values from lo to hi, clamped to hi.

    A step that does not divide (hi - lo) must not overshoot the band edge --
    an out-of-band w_b is silently unphysical rather than an obvious error.
    """
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    vals = [lo + step * k for k in range(n)]
    if vals[-1] < hi - 1e-9:
        vals.append(hi)
    return ",".join(f"{v:.4f}" for v in vals)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_calib(ctx: Context) -> None:
    """Tier 0: calibration landscape; saves the operating point every tier reuses."""
    from device_utils import load_device
    from paths import resolve_device
    cfg = load_device(resolve_device(ctx.device))
    t_g = float(cfg["t_g_ns"])
    _run(ctx, [PY, "calibration_map.py", "--device", ctx.device,
               "--t-g-ns", f"{t_g:.4f}", "--wp-span-MHz", "40", "--wp-points", ctx.cal_pts(),
               "--amp-lo", "0.6", "--amp-hi", "1.4", "--amp-points", ctx.cal_pts(),
               "--metric", "fidelity",
               "--out", f"figs/{ctx.outroot}_calibration_map.png",
               "--save-point", ctx.op_point, "--overwrite"], label="calib")
    print("    ACCEPTANCE: one smooth maximum, not against a grid edge. "
          "If the optimum sits on a bound, widen --amp-lo/--amp-hi and rerun.")


def stage_mechanism(ctx: Context) -> None:
    """Tier 1: DRAG I/Q and spectral-null figures (no simulation)."""
    for script in ("drag_iq_figure.py", "drag_fourier_figure.py"):
        path = os.path.join(ctx.figs_dir, script)
        if not os.path.exists(path):
            print(f"    skip {script} (not found under {ctx.figs_dir}; "
                  f"pass --figs-dir)")
            continue
        _run(ctx, [PY, path], label="mechanism", allow_fail=True)


def stage_clean(ctx: Context) -> None:
    """Tier 2: isolated one-pump spectator collision -- the headline DRAG figure.

    Geometry: w_a=4.6, w_b=5.7 (w_p=1.1), coupler 4.2. The a-side one-pump
    collision sits at w_spec = w_a - w_p = 3.5 GHz, i.e. Delta = w_b - w_spec = 2.2,
    with every other channel >= ~0.7 GHz away -- so DRAG's effect is unambiguous
    (unlike the inner root, which lands exactly on the partner qubit).
    """
    dev = _derive_device(ctx, "clean", dict(qubit_freqs_GHz=[4.6, 5.7],
                                            coupler_freq_GHz=4.2))
    saved, ctx.device = ctx.device, dev
    try:
        # the variant has no operating point of its own -> calibrate it first
        _run(ctx, [PY, "calibration_map.py", "--device", dev, "--t-g-ns", "92.6",
                   "--wp-span-MHz", "40", "--amp-lo", "0.6", "--amp-hi", "1.4",
                   "--wp-points", ctx.cal_pts(), "--amp-points", ctx.cal_pts(),
                   "--out", f"figs/{ctx.outroot}_clean_calibration.png",
                   "--save-point", ctx.op_point, "--overwrite"], label="clean")
        _sweep(ctx, "clean",
               ["--sweep", "spectator", "--drags", "false,true",
                "--specfreqs", _spec_list(2.2, 0.20, ctx.npts(25))],
               plot=[PY, "plot_results.py", "--outdir", "{outdir}"])
    finally:
        ctx.device = saved
    print("    ACCEPTANCE: both curves smooth; DRAG lifts BOTH shoulders. "
          "Any spike where amp_scale_used railed is a calibration artifact, not physics.")


def stage_allocation(ctx: Context) -> None:
    """Tier 3a: 2D allocation map + the DRAG-helps/hurts difference panel."""
    outdir = _sweep(ctx, "allocation",
                    ["--sweep", "target", "--drags", "false,true",
                     "--wb-GHz", _wb_list(3.8, 5.7, 0.4 if ctx.quick else 0.10),
                     "--spec-min-GHz", "3.1", "--spec-max-GHz", "6.1",
                     "--spec-step-GHz", "0.4" if ctx.quick else "0.10"],
                    plot=[PY, "plot_fidelity_map.py", "--outdir", "{outdir}",
                          "--metric", "infidelity", "--log"])
    try:
        from paths import in_results
        resolved = in_results(outdir)
    except Exception:
        resolved = os.path.join("results", outdir)
    _run(ctx, [PY, "plot_fidelity_map.py", "--outdir", resolved, "--diff"],
         label="allocation", allow_fail=True)


def stage_subharm_spec(ctx: Context) -> None:
    """Tier 3b: spectator parked on the pump's 2nd harmonic, w_spec = 2 w_p.

    On the base device (w_a=3.5, w_b=5.7) that is w_spec = 4.4 GHz, i.e.
    Delta = w_b - 2 w_p = 1.3. DRAG here is the eta^2 (two-pump) correction, and
    it is intrinsically one-sided -- the spectral tilt nulls one shoulder and
    amplifies the other -- so expect asymmetry rather than a clean dip-lift.
    """
    _sweep(ctx, "subharm_spec",
           ["--sweep", "spectator", "--drags", "false,true",
            "--drag-subharmonic", "--subharmonic-modes", "spec",
            "--specfreqs", _spec_list(1.3, 0.24, ctx.npts(25))],
           plot=[PY, "plot_results.py", "--outdir", "{outdir}"])


def stage_subharm_snail(ctx: Context) -> None:
    """Tier 3c: bare 3-mode gate swept through the SNAIL subharmonic w_c = 2 w_p.

    Needs w_p = w_c/2 in band: with w_a=3.5 the pump caps at 2.2 GHz, so a 4.7 GHz
    coupler is unreachable (it would need w_b=5.85). The variant lowers the coupler
    to 4.2, putting the resonance at w_b = 5.60.
    """
    dev = _derive_device(ctx, "wc42", dict(coupler_freq_GHz=4.2))
    saved, ctx.device = ctx.device, dev
    try:
        _run(ctx, [PY, "calibration_map.py", "--device", dev, "--t-g-ns", "92.6",
                   "--wp-span-MHz", "40", "--amp-lo", "0.6", "--amp-hi", "1.4",
                   "--wp-points", ctx.cal_pts(), "--amp-points", ctx.cal_pts(),
                   "--out", f"figs/{ctx.outroot}_wc42_calibration.png",
                   "--save-point", ctx.op_point, "--overwrite"], label="subharm_snail")
        _sweep(ctx, "subharm_snail",
               ["--sweep", "target", "--no-spectator", "--drags", "false,true",
                "--drag-subharmonic", "--subharmonic-modes", "s",
                "--wb-GHz", _wb_list(5.45, 5.75, 0.05 if ctx.quick else 0.0125)],
               plot=[PY, "plot_bare_sweep.py", "--outdir", "{outdir}"])
    finally:
        ctx.device = saved
    print("    ACCEPTANCE: n_coupler peaks at detuning 0 while F dips there -- that "
          "coincidence is what identifies the dip as the SNAIL subharmonic.")


def stage_eta_scan(ctx: Context) -> None:
    """Tier 4a: DRAG improvement dF = F_drag - F_nodrag vs pump strength eta.

    Runs the isolated-collision shoulder at several ``--target-eta`` and plots the
    improvement, which should rise, peak near eta ~ 1, then fall as the eta^2 Stark
    residual and leakage outgrow what first-order DRAG removes.
    """
    dev = _derive_device(ctx, "clean", dict(qubit_freqs_GHz=[4.6, 5.7],
                                            coupler_freq_GHz=4.2))
    etas = [0.8, 1.2, 1.8] if ctx.quick else [0.6, 0.8, 1.0, 1.2, 1.5, 1.8]
    shoulder = _spec_list(2.2 - 0.03, 0.0, 1)          # one off-resonant shoulder point
    rows: List[Dict[str, float]] = []
    for eta in etas:
        stage = f"eta{eta:.1f}".replace(".", "p")
        outdir = ctx.out(stage)
        prep = [PY, "run_sweep_zhou.py", "prepare", "--sweep", "spectator",
                "--device", dev, "--outdir", outdir, "--drags", "false,true",
                "--specfreqs", shoulder, "--target-eta", f"{eta}"]
        if ctx.analytic:
            prep += ["--no-integrate"]
        _run(ctx, prep, label="eta_scan")
        if ctx.execute == "slurm":
            print(f"    [slurm] OUTDIR={outdir} sbatch slurm/snail_sweep.slurm")
            continue
        _run(ctx, [PY, "run_sweep_zhou.py", "local", "--outdir", outdir,
                   "--nproc", str(ctx.nproc)], label="eta_scan")
        f_off = f_on = None
        for r in _read_summary(outdir):
            F = _fnum(r, "F_avg")
            if F is None:
                continue
            if str(r.get("drag", "")).strip().lower() == "true":
                f_on = F
            else:
                f_off = F
        if f_off is not None and f_on is not None:
            rows.append(dict(eta=eta, F_nodrag=f_off, F_drag=f_on,
                             dF=f_on - f_off))
    if rows:
        _plot_eta_scan(ctx, rows)
    elif not ctx.dry_run:
        print("    no integrated fidelities collected (analytic mode?) -- "
              "skipping the dF-vs-eta figure")


def _plot_eta_scan(ctx: Context, rows: List[Dict[str, float]]) -> None:
    """Plot DRAG improvement vs eta from collected eta-scan rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["eta"])
    eta = [r["eta"] for r in rows]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True, dpi=200,
                                   gridspec_kw=dict(height_ratios=[2, 1.4]))
    ax0.plot(eta, [1 - r["F_nodrag"] for r in rows], color=BLUE, lw=2.2, marker="o",
             mfc="white", mec=BLUE, label="no DRAG")
    ax0.plot(eta, [1 - r["F_drag"] for r in rows], color=RED, lw=2.2, marker="o",
             label="with DRAG")
    ax0.set_yscale("log")
    ax0.set_ylabel(r"infidelity  $1-F$")
    ax0.legend(frameon=False, fontsize=11)
    ax1.axhline(0.0, color="0.6", lw=1.0, ls=":")
    ax1.plot(eta, [r["dF"] for r in rows], color=RED, lw=2.2, marker="s")
    ax1.set_ylabel(r"$\Delta F$ (DRAG gain)")
    ax1.set_xlabel(r"pump strength  $|\eta|_{\rm peak}$")
    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Where DRAG stops paying off", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "figs", f"{ctx.outroot}_eta_scan.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"    wrote {out}")


def stage_grape(ctx: Context) -> None:
    """Tier 4b: GRAPE vs the DRAG raised cosine at several gate lengths."""
    for t_g in ((40.0, 92.6) if ctx.quick else (40.0, 60.0, 92.6)):
        out = _run(ctx, [PY, "grape.py", "--device", ctx.device,
                         "--t-g-ns", f"{t_g}", "--n-ctrl", "16",
                         "--cutoff-GHz", "0.5", "--maxiter", "60"],
                   label="grape", capture=True, allow_fail=True)
        if out:
            print("    " + " | ".join(l.strip() for l in out.splitlines()
                                      if "F =" in l or "dF" in l))


def stage_robustness(ctx: Context) -> None:
    """Tier 5: repeat the tier-2 sweep at t_g +/-5%.

    A feature that shifts by more than its own width under a small t_g change is a
    fixed-t_g fringe/revival artifact, not a collision -- this has bitten this
    study twice, so any quoted point should survive the check.
    """
    dev = _derive_device(ctx, "clean", dict(qubit_freqs_GHz=[4.6, 5.7],
                                            coupler_freq_GHz=4.2))
    saved, ctx.device = ctx.device, dev
    try:
        for tag, scale in (("tg095", 0.95), ("tg105", 1.05)):
            t_g = 92.6 * scale
            outdir = ctx.out(f"robust_{tag}")
            prep = [PY, "run_sweep_zhou.py", "prepare", "--sweep", "spectator",
                    "--device", dev, "--outdir", outdir, "--drags", "false,true",
                    "--specfreqs", _spec_list(2.2, 0.20, ctx.npts(21)),
                    "--t-g-ns", f"{t_g:.4f}"]
            if ctx.analytic:
                prep += ["--no-integrate"]
            _run(ctx, prep, label="robustness")
            if ctx.execute == "slurm":
                print(f"    [slurm] OUTDIR={outdir} sbatch slurm/snail_sweep.slurm")
                continue
            _run(ctx, [PY, "run_sweep_zhou.py", "local", "--outdir", outdir,
                       "--nproc", str(ctx.nproc)], label="robustness")
            _run(ctx, [PY, "plot_results.py", "--outdir", outdir],
                 label="robustness", allow_fail=True)
    finally:
        ctx.device = saved


STAGES: List[Dict[str, Any]] = [
    dict(id="calib", tier=0, fn=stage_calib,
         title="calibration landscape -> operating point"),
    dict(id="mechanism", tier=1, fn=stage_mechanism,
         title="DRAG I/Q + spectral null (no sim)"),
    dict(id="clean", tier=2, fn=stage_clean,
         title="isolated one-pump collision (headline DRAG result)"),
    dict(id="allocation", tier=3, fn=stage_allocation,
         title="2D allocation map + DRAG difference"),
    dict(id="subharm_spec", tier=3, fn=stage_subharm_spec,
         title="spectator on the pump 2nd harmonic"),
    dict(id="subharm_snail", tier=3, fn=stage_subharm_snail,
         title="bare gate through the SNAIL subharmonic"),
    dict(id="eta_scan", tier=4, fn=stage_eta_scan,
         title="DRAG gain vs pump strength eta"),
    dict(id="grape", tier=4, fn=stage_grape,
         title="GRAPE vs DRAG at several t_g"),
    dict(id="robustness", tier=5, fn=stage_robustness,
         title="tier-2 sweep at t_g +/-5% (fringe check)"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="warren_device.json")
    ap.add_argument("--outroot", default="campaign")
    ap.add_argument("--execute", choices=["local", "slurm"], default="local")
    ap.add_argument("--analytic", action="store_true",
                    help="add --no-integrate: validates the pipeline without QuTiP "
                         "(no fidelities, so no physics figures)")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--quick", action="store_true",
                    help="coarse grids / fewer points: smoke-test the campaign "
                         "cheaply. Validates the pipeline, NOT publication figures.")
    ap.add_argument("--tier", type=int, action="append", default=None,
                    help="run only these tiers (repeatable)")
    ap.add_argument("--only", action="append", default=None,
                    help="run only these stage ids (repeatable)")
    ap.add_argument("--op-point", default="campaign",
                    help="operating-point name written by tier 0 and reused after")
    ap.add_argument("--figs-dir", default=os.path.dirname(HERE),
                    help="directory holding the tier-1 mechanism figure scripts")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    args = ap.parse_args()

    if args.list:
        for s in STAGES:
            print(f"  tier {s['tier']}  {s['id']:16s} {s['title']}")
        return

    stages = STAGES
    if args.tier:
        stages = [s for s in stages if s["tier"] in set(args.tier)]
    if args.only:
        stages = [s for s in stages if s["id"] in set(args.only)]
    if not stages:
        raise SystemExit("no stages selected")

    ctx = Context(device=args.device, outroot=args.outroot, execute=args.execute,
                  analytic=args.analytic, nproc=args.nproc, dry_run=args.dry_run, quick=args.quick,
                  op_point=args.op_point, figs_dir=args.figs_dir)
    sys.path.insert(0, HERE)

    if args.quick:
        print("NOTE: --quick uses coarse grids -- smoke test only, "
              "not publication figures.")
    if args.analytic:
        print("NOTE: --analytic runs the construction/plumbing path only; fidelity "
              "columns stay empty, so figures validate the pipeline, not the physics.\n")
    started = datetime.datetime.now().isoformat(timespec="seconds")
    for s in stages:
        print(f"\n=== tier {s['tier']}  {s['id']}: {s['title']} ===")
        s["fn"](ctx)

    manifest = dict(started=started,
                    finished=datetime.datetime.now().isoformat(timespec="seconds"),
                    device=args.device, outroot=args.outroot, execute=args.execute,
                    analytic=args.analytic, op_point=args.op_point,
                    stages=[s["id"] for s in stages], commands=ctx.log)
    if not ctx.dry_run:
        path = os.path.join(HERE, f"{args.outroot}_manifest.json")
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nwrote manifest {path}")


if __name__ == "__main__":
    main()