#!/usr/bin/env python3
"""
find_stark_resonance.py
=======================

Locate the AC-Stark-shifted iSWAP resonance so the pump can be driven at the
frequency that actually closes the swap, rather than the bare guess
w_p = |w_b - w_a|.

Why this is needed
------------------
The pump does not only activate the a<->b exchange; through the SNAIL non-linearity
it also AC-Stark shifts the two qubits (and dresses the coupler) by an amount that
grows with the pump photon number |eta|^2. The resonance therefore moves to

    w_p^res = |w_b - w_a| + Delta_Stark(eta),   Delta_Stark = differential shift,

so a pump placed at the bare |w_b - w_a| sits at a residual detuning delta, which
caps the achievable transfer at g_eff^2 / (g_eff^2 + delta^2) and injects coherent
error. This tool reproduces the hardware chevron calibration: it holds a CONSTANT
pump at the operating |eta| and sweeps the pump-frequency offset (and time),
records the |01>->|10> exchange, and returns the offset that maximises the swap
contrast -- i.e. the Stark-shifted resonance. Feed the reported offset back into
run_sweep_zhou / calibrate as ``wp_offset_GHz``.

The scan uses a constant (not raised-cosine) pump on purpose: the classic chevron
is an amplitude-independent frequency measurement -- on resonance the exchange
reaches full contrast regardless of whether the pi/2 amplitude is perfectly
calibrated, so the vertex locates the frequency cleanly. Anharmonicity and the
configured qutrit levels are included (they shift the resonance too).

Usage
-----
    python find_stark_resonance.py --device dev.json --target-eta 0.3 \
        --span-MHz 60 --points 41 --out stark.npz --plot stark.png
    python find_stark_resonance.py --device dev.json --t-g 200 --amp-scale 0.9 --jobs 16

`dev.json` is the run_sweep_zhou schema (merged over its DEFAULT_CONFIG). QuTiP is
required (the scan integrates the exact Hamiltonian); the resonance-location logic
(`locate_resonance`) is pure numpy and unit-testable without QuTiP.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from device_utils import load_device, target_eta_area

TWO_PI: float = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Coupler at a fixed (constant) pump strength held over the whole window
# ---------------------------------------------------------------------------
def operating_eta(config: Dict[str, Any], t_g: float, amp_scale: float) -> float:
    """Constant-pump |eta| that performs a full iSWAP in t_g (then scaled by
    amp_scale). The Stark chevron is a CONSTANT pulse, so the probe amplitude is
    the constant-pulse operating point -- not the raised-cosine peak:

        eta = (pi/2) / (6 (2pi g3) la lb t_g) .

    For a raised-cosine gate of the same t_g this is exactly half the RC peak, and
    is a better single-amplitude proxy for that gate's pulse-averaged Stark shift
    (Hann <eta^2> = 0.375 eta_peak^2, vs eta_peak^2 at the peak) than the peak is.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Operating gate time (ns).
    amp_scale : float
        Amplitude-scale correction from a prior amplitude calibration.

    Returns
    -------
    float
        Constant-pump operating |eta|.
    """
    from device_utils import build_coupler
    sub = dict(config)
    sub["envelope"] = "constant"                 # Stark calibration is constant-pulse
    _cpl, _w_p, eta_peak = build_coupler(sub, t_g, amp_scale, 0.0)
    return float(eta_peak)


def build_chevron_coupler(config: Dict[str, Any], eta_op: float,
                          wp_offset_GHz: float, window_ns: float,
                          spec_abs_GHz: Optional[float] = None,
                          shape: str = "constant", t_g_ns: Optional[float] = None,
                          drag_beat_GHz: Optional[float] = None,
                          amp_scale: float = 1.0):
    """(a, b, coupler[, spectator]) system driven by a probe pump, with the pump
    frequency offset from |w_b - w_a| by wp_offset_GHz. Anharmonicity / qutrit
    levels are included.

    Two probe shapes:
      * ``shape="constant"`` (default) -- a CONSTANT pump of peak |eta| = eta_op
        held over [0, window_ns]. Amplitude-robust; it has d eta/dt = 0, so it
        cannot see the DRAG-quadrature Stark shift.
      * ``shape="raised_cosine"`` -- the ACTUAL gate pulse: a Hann full iSWAP over
        [0, t_g_ns] normalized on (a, b) and scaled by amp_scale, with the DRAG
        quadrature applied when ``drag_beat_GHz`` is given (tuned to that beat).
        This DOES carry the DRAG-quadrature shift, so the located resonance is the
        DRAG-ON resonance.

    With ``spec_abs_GHz`` given, a 4th spectator mode is added at that ABSOLUTE
    frequency (participation lam_b, ``spec_levels`` levels, ``anharm_spec_GHz``).

    Parameters
    ----------
    config : dict
        Merged device configuration.
    eta_op : float
        Constant pump strength |eta| (used only for ``shape="constant"``).
    wp_offset_GHz : float
        Offset added to the bare pump frequency w_b - w_a (GHz).
    window_ns : float
        Evolution window (ns); for the constant shape the pump is held over it.
    spec_abs_GHz : float, optional
        Spectator ABSOLUTE frequency (GHz). None -> no spectator (bare pair).
    shape : str, default "constant"
        "constant" or "raised_cosine".
    t_g_ns : float, optional
        Gate time (ns) for the raised-cosine pulse; required if shape="raised_cosine".
    drag_beat_GHz : float, optional
        DRAG beat detuning delta = Delta - w_p (GHz). If given (raised-cosine only),
        apply the first-order DRAG quadrature tuned to it.
    amp_scale : float, default 1.0
        Amplitude-scale correction applied after the full-iSWAP normalization
        (raised-cosine only).

    Returns
    -------
    (ZhouCoupler, float)
        The coupler and its pump frequency w_p (GHz).
    """
    from zhou_coupler import ZhouCoupler, PumpTone, ConstantPulse, RaisedCosine

    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    ws = float(config["coupler_freq_GHz"])
    w_p_GHz = abs(wb - wa) + wp_offset_GHz
    aq = float(config.get("anharm_qubit_GHz", 0.0))
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    freqs = [wa, wb, ws]
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"])]
    participations = {0: float(config["lam_a"]), 1: float(config["lam_b"])}
    anharm = {0: aq, 1: aq}
    if spec_abs_GHz is not None:                    # add the spectator as a 4th mode
        freqs.append(float(spec_abs_GHz))
        levels.append(int(config.get("spec_levels", 3)))
        participations[3] = float(config["lam_b"])           # spectator participation = lam_b
        anharm[3] = float(config.get("anharm_spec_GHz", 0.0))

    cpl = ZhouCoupler(mode_freqs_GHz=freqs, coupler_index=2,
                      participations=participations,
                      nonlinearities=nonlin, levels=levels,
                      anharmonicities_GHz=anharm)

    if shape == "raised_cosine":
        # The actual gate pulse: Hann full iSWAP over t_g, DRAG tuned to the beat.
        # Built exactly as run_sweep_zhou.build_point does (normalize then scale).
        t_g = float(t_g_ns if t_g_ns is not None else window_ns)
        env = RaisedCosine(amp=1.0, t_g=t_g)
        cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=env, is_eta=True,
                              drag=(drag_beat_GHz is not None),
                              delta_drag_GHz=drag_beat_GHz),
                     normalize_iswap=(0, 1))
        cpl.scale_pump_amplitude(float(amp_scale))
    else:
        # constant pump at fixed |eta| (is_eta=True, no normalization): peak_eta == eta_op
        cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=ConstantPulse(amp=eta_op, t_g=window_ns),
                              is_eta=True), normalize_iswap=None)
    return cpl, w_p_GHz


# ---------------------------------------------------------------------------
# Parallel chevron scan
# ---------------------------------------------------------------------------
def _resolve_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs and n_jobs > 0:
        return int(n_jobs)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def _chevron_worker(args: Tuple) -> np.ndarray:
    """One pump-offset column: P(|01>->|10>) over the time grid. Works for the
    3-mode bare pair or the 4-mode pair+spectator (state/index built from n_modes),
    and for constant or shaped(+DRAG) probe pulses (``build_kw``)."""
    config, eta_op, wp_offset, times, solver, spec_abs_GHz, build_kw = args
    cpl, _w_p = build_chevron_coupler(config, eta_op, wp_offset, float(times[-1]),
                                      spec_abs_GHz=spec_abs_GHz, **build_kw)
    init = [0] * cpl.n_modes; init[1] = 1          # |01...> : qubit b excited
    tgt = [0] * cpl.n_modes; tgt[0] = 1            # |10...> : qubit a excited
    states = cpl.evolve_trajectory(init, times, **solver)
    return np.abs(states[:, cpl.fock_index(tgt)]) ** 2


def _parabolic_vertex(x: Sequence[float], y: Sequence[float]) -> float:
    """Sub-grid maximum via a 3-point parabola around the discrete argmax; falls
    back to the grid point at the boundary.

    Parameters
    ----------
    x, y : sequence of float
        Sampled abscissae (sorted) and values.

    Returns
    -------
    float
        Interpolated x of the maximum.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    k = int(np.argmax(y))
    if k == 0 or k == len(x) - 1:
        return float(x[k])
    x0, x1, x2 = x[k - 1], x[k], x[k + 1]
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-15:
        return float(x1)
    # vertex of the parabola through the three points (uniform spacing not required)
    return float(x1 + 0.25 * (x2 - x0) * (y0 - y2) / denom)


def locate_resonance(offsets_GHz: np.ndarray, max_transfer: np.ndarray) -> float:
    """Stark-shifted resonance offset (GHz) = the offset that maximises the swap
    contrast, parabolically refined. Pure numpy (no QuTiP).

    Parameters
    ----------
    offsets_GHz : ndarray
        Scanned pump-frequency offsets (GHz).
    max_transfer : ndarray
        Max over time of P(|10>) at each offset (the chevron envelope).

    Returns
    -------
    float
        Interpolated resonance offset (GHz).
    """
    return _parabolic_vertex(offsets_GHz, max_transfer)


def scan(config: Dict[str, Any], t_g: float, amp_scale: float,
         offsets_GHz: np.ndarray, window_ns: float, n_time: int,
         solver: Optional[Dict[str, Any]] = None,
         n_jobs: Optional[int] = None,
         spec_abs_GHz: Optional[float] = None,
         shape: str = "constant",
         drag_beat_GHz: Optional[float] = None) -> Dict[str, Any]:
    """Run the pump-frequency chevron and locate the Stark-shifted resonance.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Operating gate time (ns); sets the probe amplitude via operating_eta.
    amp_scale : float
        Amplitude-scale correction from a prior amplitude calibration.
    offsets_GHz : ndarray
        Pump-frequency offsets to scan, relative to |w_b - w_a| (GHz).
    window_ns : float
        Chevron time window (ns); ~2 t_g captures a full on-resonance exchange.
    n_time : int
        Number of output times across the window.
    solver : dict, optional
        QuTiP tolerances (atol, rtol, nsteps); defaults provided.
    n_jobs : int, optional
        Worker processes (0/None -> SLURM_CPUS_PER_TASK or CPU count). Forced to 1
        on GPU.
    spec_abs_GHz : float, optional
        If given, include a spectator mode at this ABSOLUTE frequency so the
        located resonance includes its dispersive pull; None -> bare pair.

    Returns
    -------
    dict
        offsets_GHz, times_ns, P10 [n_off, n_time], max_transfer [n_off],
        eta_op, w_p_bare_GHz, resonance_offset_GHz, resonance_w_p_GHz, and the
        probe metadata shape, drag_beat_GHz, spec_abs_GHz.
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    eta_op = operating_eta(config, t_g, amp_scale)
    times = np.linspace(0.0, window_ns, n_time)
    build_kw = {"shape": shape, "t_g_ns": float(t_g),
                "drag_beat_GHz": (float(drag_beat_GHz) if drag_beat_GHz is not None else None),
                "amp_scale": float(amp_scale)}
    args = [(config, eta_op, float(off), times, solver, spec_abs_GHz, build_kw)
            for off in offsets_GHz]

    jobs = _resolve_jobs(n_jobs)
    if jobs <= 1 or len(args) <= 1:
        cols = [_chevron_worker(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            cols = list(pool.map(_chevron_worker, args))

    P10 = np.array(cols)                       # [n_off, n_time]
    max_transfer = P10.max(axis=1)
    # Resonance criterion: for the CONSTANT probe, max-over-time = max Rabi contrast.
    # For the SHAPED full-iSWAP the gate is evaluated at t_g (the pump is off after),
    # so locate on P(|10>) AT t_g -- max-over-time would credit off-resonant offsets
    # with their best mid-pulse value, broadening the peak and pulling the resonance
    # off the frequency that actually completes the swap at t_g.
    if shape == "raised_cosine":
        k_tg = int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(t_g))))
        metric = P10[:, k_tg]
        metric_label = f"P(|10>) at t_g={float(t_g):.0f} ns"
    else:
        metric = max_transfer
        metric_label = "max-over-time P(|10>)"
    res_off = locate_resonance(np.asarray(offsets_GHz, dtype=float), metric)
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    w_p_bare = abs(wb - wa)
    return {"offsets_GHz": np.asarray(offsets_GHz, dtype=float), "times_ns": times,
            "P10": P10, "max_transfer": max_transfer,
            "resonance_metric": metric, "metric_label": metric_label,
            "eta_op": float(eta_op),
            "w_p_bare_GHz": float(w_p_bare),
            "resonance_offset_GHz": float(res_off),
            "resonance_w_p_GHz": float(w_p_bare + res_off),
            "shape": shape,
            "drag_beat_GHz": (float(drag_beat_GHz) if drag_beat_GHz is not None else np.nan),
            "spec_abs_GHz": (float(spec_abs_GHz) if spec_abs_GHz is not None else np.nan)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def render_chevron(chev: Dict[str, Any], png_path: str, title_suffix: str = "") -> None:
    """Render one chevron (heatmap + max-transfer envelope) from a dict of arrays.

    Parameters
    ----------
    chev : dict
        Must contain offsets_GHz, times_ns, P10 [n_off, n_time], max_transfer,
        resonance_offset_GHz, eta_op; optional shape / drag_beat_GHz / spec_abs_GHz
        annotate the title.
    png_path : str
        Output PNG.
    title_suffix : str, default ""
        Extra text appended to the figure title (e.g. the point index / detuning).

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

    off_MHz = np.asarray(chev["offsets_GHz"]) * 1e3
    res_MHz = float(chev["resonance_offset_GHz"]) * 1e3
    shape = str(chev.get("shape", "constant"))
    drag_beat = float(chev.get("drag_beat_GHz", np.nan))
    probe = ("raised-cosine" if shape == "raised_cosine" else "constant")
    if shape == "raised_cosine" and np.isfinite(drag_beat):
        probe += rf" + DRAG($\delta$={drag_beat*1e3:+.0f} MHz)"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")
    mesh = ax0.pcolormesh(off_MHz, np.asarray(chev["times_ns"]), np.asarray(chev["P10"]).T,
                          shading="auto", cmap="viridis", vmin=0, vmax=1)
    ax0.axvline(res_MHz, color="r", ls="--", lw=1.6, label=f"resonance {res_MHz:+.1f} MHz")
    ax0.set_xlabel(r"pump offset from $|w_b-w_a|$ (MHz)"); ax0.set_ylabel("time (ns)")
    ax0.set_title(rf"chevron ({probe}): $P(|01\rangle\to|10\rangle)$")
    ax0.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(mesh, ax=ax0, label=r"$P(|10\rangle)$")

    ax1.plot(off_MHz, np.asarray(chev.get("resonance_metric", chev["max_transfer"])),
             "o-", ms=3)
    ax1.axvline(res_MHz, color="r", ls="--", lw=1.6)
    ax1.set_xlabel(r"pump offset from $|w_b-w_a|$ (MHz)")
    ax1.set_ylabel(str(chev.get("metric_label", "max-over-time $P(|10\\rangle)$")))
    ax1.set_title(rf"offset = {res_MHz:+.1f} MHz  ($|\eta|$={float(chev['eta_op']):.3f})  {title_suffix}")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_chevron(npz_path: str, png_path: str) -> None:
    """Load a chevron .npz (from main()) and render it via render_chevron."""
    d = np.load(npz_path)
    chev = {"offsets_GHz": d["offsets_GHz"], "times_ns": d["times_ns"],
            "P10": d["P10"], "max_transfer": d["max_transfer"],
            "resonance_offset_GHz": float(d["resonance_offset_GHz"]),
            "eta_op": float(d["eta_op"]),
            "shape": str(d["shape"]) if "shape" in d.files else "constant",
            "drag_beat_GHz": float(d["drag_beat_GHz"]) if "drag_beat_GHz" in d.files else np.nan}
    if "resonance_metric" in d.files:
        chev["resonance_metric"] = d["resonance_metric"]
        chev["metric_label"] = str(d["metric_label"]) if "metric_label" in d.files else ""
    render_chevron(chev, png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Find the Stark-shifted iSWAP resonance and report the pump frequency to drive."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON (run_sweep schema)")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="operating |eta| (sets t_g); overrides --t-g / device t_g_ns")
    ap.add_argument("--t-g", type=float, default=None, help="operating gate time (ns)")
    ap.add_argument("--amp-scale", type=float, default=1.0,
                    help="amplitude-scale correction from a prior amplitude calibration")
    ap.add_argument("--calibration", default=None,
                    help="calibration JSON to read amp_scale (and t_g) from")
    ap.add_argument("--span-MHz", type=float, default=60.0,
                    help="offset scan is +/- span/2 about |w_b-w_a|")
    ap.add_argument("--points", type=int, default=41, help="offset grid points")
    ap.add_argument("--window-factor", type=float, default=2.0,
                    help="time window = window_factor * t_g")
    ap.add_argument("--time-points", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = SLURM_CPUS_PER_TASK/CPU)")
    ap.add_argument("--gpu", action="store_true", help="run via qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--out", default="stark.npz", help="output .npz")
    ap.add_argument("--plot", default=None, help="optional output PNG")
    ap.add_argument("--update-device", default=None,
                    help="write device JSON with wp_offset_GHz set to the resonance")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    args = ap.parse_args()

    if args.gpu:
        import zhou_coupler
        zhou_coupler.use_gpu(True)
        args.jobs = 1

    from paths import resolve_device, in_results
    args.out = in_results(args.out)
    if args.update_device:
        args.update_device = in_results(args.update_device)
    if args.plot:
        args.plot = in_results(args.plot)
    config = load_device(resolve_device(args.device))
    amp_scale = float(args.amp_scale)
    t_g = args.t_g
    if args.calibration:
        with open(args.calibration) as f:
            cal = json.load(f)
        amp_scale = float(cal.get("amp_scale", amp_scale))
        t_g = cal.get("t_g_ns", t_g)
    if args.target_eta is not None:
        # constant-pulse full iSWAP: area = eta * t_g = target_eta_area  ->  t_g = area/eta
        t_g = target_eta_area(float(config["g3_GHz"]), float(config["lam_a"]),
                              float(config["lam_b"])) / float(args.target_eta)
    if t_g is None:
        t_g = float(config.get("t_g_ns", 200.0))
    t_g = float(t_g)

    span = args.span_MHz / 1000.0
    offsets = np.linspace(-span / 2, span / 2, args.points)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}
    window = args.window_factor * t_g

    print(f"device={args.device}  t_g={t_g:.1f} ns  amp_scale={amp_scale}  "
          f"jobs={_resolve_jobs(args.jobs)}{' GPU' if args.gpu else ''}")
    result = scan(config, t_g, amp_scale, offsets, window, args.time_points,
                  solver, n_jobs=args.jobs)

    np.savez(args.out, t_g_ns=t_g, amp_scale=amp_scale, **{
        k: v for k, v in result.items()})
    print(f"operating |eta|        = {result['eta_op']:.4f}")
    print(f"bare w_p = |w_b-w_a|   = {result['w_p_bare_GHz']:.6f} GHz")
    print(f"Stark resonance offset = {result['resonance_offset_GHz']*1e3:+.2f} MHz")
    print(f"=> drive the pump at w_p = {result['resonance_w_p_GHz']:.6f} GHz "
          f"(set wp_offset_GHz = {result['resonance_offset_GHz']:.6f})")
    print(f"peak contrast at resonance ~ {result['max_transfer'].max():.4f}")
    print(f"written {args.out}")

    if args.update_device:
        dev = dict(config)
        dev["wp_offset_GHz"] = float(result["resonance_offset_GHz"])
        with open(args.update_device, "w") as f:
            json.dump(dev, f, indent=2)
        print(f"wrote {args.update_device} with wp_offset_GHz={dev['wp_offset_GHz']:.6f}")

    if args.plot:
        plot_chevron(args.out, args.plot)
        print(f"plotted {args.plot}")


if __name__ == "__main__":
    main()