"""
stark_vs_detuning.py
====================

Companion to the spectator sweep (run_sweep_zhou.py --sweep spectator). That sweep
shows how DRAG suppresses a spectator as its detuning Delta = w_b - w_spec varies.
This tool answers the paired question: how does the AC-Stark-shifted iSWAP
resonance itself move as the SAME spectator is walked across the band?

For each detuning it builds the full (a, b, coupler, spectator) system with the
spectator at w_spec = w_b - Delta, drives a CONSTANT pump at the operating |eta|,
and runs a pump-frequency chevron (find_stark_resonance.scan with spec_abs_GHz) to
locate the resonance. The spectator's dispersive pull on the a<->b resonance makes
the located offset vary with Delta, with a feature at the one-pump collision
Delta = w_p = |w_b - w_a|. The bare (spectator-free) offset is drawn as a baseline
so the spectator contribution is the gap between the two.

Physics note: this is a constant-pulse Stark probe (see find_stark_resonance), so
the amplitude is the constant-pump full-iSWAP |eta| for t_g -- half the raised-
cosine peak. The probe amplitude is set by the (a, b) pair and is the same at every
detuning; only the resonance LOCATION moves with the spectator.

Needs QuTiP (it integrates the exact Hamiltonian) -- run on a compute node. Pair
the resulting figure with plot_results.py's DRAG-vs-frequency panels from the same
device and --specfreqs for a like-for-like comparison.

Usage
-----
    python stark_vs_detuning.py --device devices/warren_device.json \
        --target-eta 1.5 --outdir results/stark_det

    # match a spectator sweep's detuning axis exactly:
    python stark_vs_detuning.py --device devices/warren_device.json \
        --specfreqs 0.20,0.25,0.30,...,0.90 --outdir results/stark_det
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np

import find_stark_resonance as FS
from device_utils import load_device, target_eta_area
from run_sweep_zhou import DEFAULT_SPECFREQS_GHz


def sweep_detunings(config: Dict[str, Any], detunings_GHz: List[float], t_g: float,
                    amp_scale: float, span_MHz: float, points: int,
                    window_factor: float, time_points: int,
                    solver: Dict[str, Any], n_jobs: Optional[int],
                    baseline: bool = True) -> Dict[str, Any]:
    """Locate the Stark-shifted iSWAP resonance at each spectator detuning.

    Parameters
    ----------
    config : dict
        Merged device configuration (qubit_freqs_GHz = [w_a, w_b]).
    detunings_GHz : list of float
        Spectator detunings Delta = w_b - w_spec (GHz) to scan.
    t_g : float
        Operating gate time (ns); sets the constant-pump probe |eta|.
    amp_scale : float
        Amplitude-scale correction (1.0 = raw analytic normalization).
    span_MHz : float
        Full chevron pump-offset width (MHz), centered on the bare |w_b - w_a|.
    points : int
        Number of pump-offset samples per chevron.
    window_factor : float
        Chevron time window = window_factor * t_g.
    time_points : int
        Time samples across the window.
    solver : dict
        QuTiP tolerances (atol, rtol, nsteps).
    n_jobs : int, optional
        Worker processes for the per-chevron offset scan.
    baseline : bool, default True
        Also compute the spectator-free (bare-pair) offset for reference.

    Returns
    -------
    dict
        detunings_GHz, w_spec_GHz, stark_offset_MHz [n_det], contrast [n_det],
        baseline_offset_MHz (float or nan), eta_op, w_p_bare_GHz, t_g_ns.
    """
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    w_p_bare = abs(wb - wa)
    span = span_MHz / 1000.0
    offsets = np.linspace(-span / 2, span / 2, points)
    window = window_factor * t_g

    det = np.asarray(detunings_GHz, dtype=float)
    w_spec = wb - det                                   # absolute spectator freq
    offs_MHz = np.full(det.size, np.nan)
    contrast = np.full(det.size, np.nan)
    eta_op = np.nan
    for k, ws_abs in enumerate(w_spec):
        res = FS.scan(config, t_g, amp_scale, offsets, window, time_points,
                      solver, n_jobs=n_jobs, spec_abs_GHz=float(ws_abs))
        offs_MHz[k] = res["resonance_offset_GHz"] * 1e3
        contrast[k] = float(res["max_transfer"].max())
        eta_op = res["eta_op"]
        print(f"  Delta={det[k]:.3f} GHz  w_spec={ws_abs:.4f}  "
              f"Stark={offs_MHz[k]:+.2f} MHz  contrast={contrast[k]:.4f}")

    base_MHz = np.nan
    if baseline:
        rb = FS.scan(config, t_g, amp_scale, offsets, window, time_points,
                     solver, n_jobs=n_jobs, spec_abs_GHz=None)
        base_MHz = rb["resonance_offset_GHz"] * 1e3
        eta_op = rb["eta_op"] if np.isnan(eta_op) else eta_op
        print(f"  baseline (no spectator)  Stark={base_MHz:+.2f} MHz")

    return {"detunings_GHz": det, "w_spec_GHz": w_spec,
            "stark_offset_MHz": offs_MHz, "contrast": contrast,
            "baseline_offset_MHz": float(base_MHz), "eta_op": float(eta_op),
            "w_p_bare_GHz": float(w_p_bare), "t_g_ns": float(t_g)}


def plot_dispersion(npz_path: str, png_path: str) -> None:
    """Render Stark resonance offset (MHz) vs spectator detuning Delta, with the
    bare-pair baseline and the one-pump collision Delta = w_p marked.

    Parameters
    ----------
    npz_path : str
        .npz produced by main().
    png_path : str
        Output PNG.

    Returns
    -------
    None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from plot_results import set_literature_style
        set_literature_style()
    except Exception:
        pass

    d = np.load(npz_path)
    det = d["detunings_GHz"]
    off = d["stark_offset_MHz"]
    base = float(d["baseline_offset_MHz"])
    w_p = float(d["w_p_bare_GHz"])

    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    ax.plot(det, off, "-o", ms=3.5, lw=1.4, color="#0072B2",
            label="with spectator")
    if np.isfinite(base):
        ax.axhline(base, ls="--", lw=1.1, color="#555555",
                   label=f"bare pair ({base:+.2f} MHz)")
    if det.min() <= w_p <= det.max():
        ax.axvline(w_p, ls=":", lw=1.1, color="#D55E00",
                   label=rf"$\Delta=w_p={w_p:.3f}$ GHz")
    ax.set_xlabel(r"spectator detuning $\Delta = w_b - w_\mathrm{spec}$ (GHz)")
    ax.set_ylabel(r"Stark resonance offset (MHz)")
    ax.set_title(rf"Stark shift vs spectator detuning  ($t_g={float(d['t_g_ns']):.0f}$ ns, "
                 rf"$|\eta|={float(d['eta_op']):.3f}$)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """CLI entry point: scan the Stark resonance vs spectator detuning and plot."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON (bare name -> devices/)")
    ap.add_argument("--outdir", default="results/stark_vs_detuning",
                    help="output dir (bare name -> results/)")
    ap.add_argument("--specfreqs", help="comma list of detunings Delta (GHz); "
                                        "default matches the spectator sweep axis")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="operating |eta| (sets t_g via the constant-pulse relation)")
    ap.add_argument("--t-g", type=float, default=None, help="explicit gate time (ns)")
    ap.add_argument("--amp-scale", type=float, default=None,
                    help="amplitude-scale correction (default: device amp_scale or 1.0)")
    ap.add_argument("--span-MHz", type=float, default=None,
                    help="chevron pump-offset width (default: device stark_span_MHz or 60)")
    ap.add_argument("--points", type=int, default=None,
                    help="chevron offset samples (default: device stark_points or 21)")
    ap.add_argument("--window-factor", type=float, default=None,
                    help="chevron window = factor * t_g (default: device or 2.0)")
    ap.add_argument("--time-points", type=int, default=None,
                    help="chevron time samples (default: device stark_time_points or 120)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the spectator-free reference offset")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (offset scan)")
    ap.add_argument("--gpu", action="store_true", help="run the solver on GPU (forces jobs=1)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    args = ap.parse_args()

    if args.gpu:
        import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from paths import resolve_device, in_results
    import os
    outdir = in_results(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    config = load_device(resolve_device(args.device))

    amp_scale = float(args.amp_scale if args.amp_scale is not None
                      else config.get("amp_scale", 1.0))
    t_g = args.t_g
    if args.target_eta is not None:
        t_g = target_eta_area(float(config["g3_GHz"]), float(config["lam_a"]),
                              float(config["lam_b"])) / float(args.target_eta)
    if t_g is None:
        t_g = float(config.get("t_g_ns", 200.0))
    t_g = float(t_g)

    span_MHz = float(args.span_MHz if args.span_MHz is not None
                     else config.get("stark_span_MHz", 60.0))
    points = int(args.points if args.points is not None
                 else config.get("stark_points", 21))
    window_factor = float(args.window_factor if args.window_factor is not None
                          else config.get("stark_window_factor", 2.0))
    time_points = int(args.time_points if args.time_points is not None
                      else config.get("stark_time_points", 120))
    detunings = ([float(x) for x in args.specfreqs.split(",")]
                 if args.specfreqs else list(DEFAULT_SPECFREQS_GHz))
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}

    print(f"device={args.device}  t_g={t_g:.1f} ns  amp_scale={amp_scale}  "
          f"detunings={len(detunings)}  jobs={FS._resolve_jobs(args.jobs)}"
          f"{' GPU' if args.gpu else ''}")
    result = sweep_detunings(config, detunings, t_g, amp_scale, span_MHz, points,
                             window_factor, time_points, solver, args.jobs,
                             baseline=not args.no_baseline)

    npz = os.path.join(outdir, "stark_vs_detuning.npz")
    png = os.path.join(outdir, "stark_vs_detuning.png")
    np.savez(npz, amp_scale=amp_scale, **result)
    print(f"written {npz}")
    plot_dispersion(npz, png)
    print(f"plotted {png}")


if __name__ == "__main__":
    main()