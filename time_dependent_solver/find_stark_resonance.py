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

from calibrate_iswap import load_device, auto_t_g

TWO_PI: float = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Coupler at a fixed (constant) pump strength held over the whole window
# ---------------------------------------------------------------------------
def operating_eta(config: Dict[str, Any], t_g: float, amp_scale: float) -> float:
    """Peak |eta| of the calibrated operating gate (normalized full iSWAP over
    t_g, then scaled by amp_scale). This is the amplitude at which we probe the
    Stark shift.

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
        Peak pump strength |eta| of the operating point.
    """
    from calibrate_iswap import build_coupler
    _cpl, _w_p, eta_peak = build_coupler(config, t_g, amp_scale, 0.0)
    return float(eta_peak)


def build_chevron_coupler(config: Dict[str, Any], eta_op: float,
                          wp_offset_GHz: float, window_ns: float):
    """Spectator-free (a, b, coupler) system driven by a CONSTANT pump of peak
    |eta| = eta_op held over [0, window_ns], with the pump frequency offset from
    |w_b - w_a| by wp_offset_GHz. Anharmonicity / qutrit levels are included.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    eta_op : float
        Constant pump strength |eta| to hold (the operating amplitude).
    wp_offset_GHz : float
        Offset added to the bare pump frequency w_b - w_a (GHz).
    window_ns : float
        Envelope duration (ns); the pump stays on for the whole scan window.

    Returns
    -------
    (ZhouCoupler, float)
        The coupler and its pump frequency w_p (GHz).
    """
    from zhou_coupler import ZhouCoupler, PumpTone, ConstantPulse

    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    ws = float(config["coupler_freq_GHz"])
    w_p_GHz = abs(wb - wa) + wp_offset_GHz
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"])]
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])
    aq = float(config.get("anharm_qubit_GHz", 0.0))

    cpl = ZhouCoupler(mode_freqs_GHz=[wa, wb, ws], coupler_index=2,
                      participations={0: float(config["lam_a"]), 1: float(config["lam_b"])},
                      nonlinearities=nonlin, levels=levels,
                      anharmonicities_GHz={0: aq, 1: aq})
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
    """One pump-offset column: P(|01>->|10>) over the time grid."""
    config, eta_op, wp_offset, times, solver = args
    cpl, _w_p = build_chevron_coupler(config, eta_op, wp_offset, float(times[-1]))
    states = cpl.evolve_trajectory([0, 1, 0], times, **solver)
    i10 = cpl.fock_index([1, 0, 0])
    return np.abs(states[:, i10]) ** 2


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
         n_jobs: Optional[int] = None) -> Dict[str, Any]:
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

    Returns
    -------
    dict
        offsets_GHz, times_ns, P10 [n_off, n_time], max_transfer [n_off],
        eta_op, w_p_bare_GHz, resonance_offset_GHz, resonance_w_p_GHz.
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    eta_op = operating_eta(config, t_g, amp_scale)
    times = np.linspace(0.0, window_ns, n_time)
    args = [(config, eta_op, float(off), times, solver) for off in offsets_GHz]

    jobs = _resolve_jobs(n_jobs)
    if jobs <= 1 or len(args) <= 1:
        cols = [_chevron_worker(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            cols = list(pool.map(_chevron_worker, args))

    P10 = np.array(cols)                       # [n_off, n_time]
    max_transfer = P10.max(axis=1)
    res_off = locate_resonance(np.asarray(offsets_GHz, dtype=float), max_transfer)
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    w_p_bare = abs(wb - wa)
    return {"offsets_GHz": np.asarray(offsets_GHz, dtype=float), "times_ns": times,
            "P10": P10, "max_transfer": max_transfer, "eta_op": float(eta_op),
            "w_p_bare_GHz": float(w_p_bare),
            "resonance_offset_GHz": float(res_off),
            "resonance_w_p_GHz": float(w_p_bare + res_off)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_chevron(npz_path: str, png_path: str) -> None:
    """Render the chevron heatmap P(|10>)(offset, time) and the max-transfer
    envelope with the located resonance marked.

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
    off_MHz = d["offsets_GHz"] * 1e3
    res_MHz = float(d["resonance_offset_GHz"]) * 1e3
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")

    mesh = ax0.pcolormesh(off_MHz, d["times_ns"], d["P10"].T, shading="auto",
                          cmap="viridis", vmin=0, vmax=1)
    ax0.axvline(res_MHz, color="r", ls="--", lw=1.6, label=f"resonance {res_MHz:+.1f} MHz")
    ax0.set_xlabel(r"pump offset from $|w_b-w_a|$ (MHz)"); ax0.set_ylabel("time (ns)")
    ax0.set_title("chevron: $P(|01\\rangle\\to|10\\rangle)$"); ax0.legend(loc="upper right", framealpha=0.9)
    fig.colorbar(mesh, ax=ax0, label=r"$P(|10\rangle)$")

    ax1.plot(off_MHz, d["max_transfer"], "o-", ms=3)
    ax1.axvline(res_MHz, color="r", ls="--", lw=1.6)
    ax1.set_xlabel(r"pump offset from $|w_b-w_a|$ (MHz)")
    ax1.set_ylabel(r"max-over-time $P(|10\rangle)$")
    ax1.set_title(f"resonance offset = {res_MHz:+.1f} MHz  ($|\\eta|$={float(d['eta_op']):.3f})")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    if not args.plot:
        args.plot = args.device.rsplit(".",1)[0] + "_Frequency_Response_Plot.png"

    config = load_device(args.device)
    amp_scale = float(args.amp_scale)
    t_g = args.t_g
    if args.calibration:
        with open(args.calibration) as f:
            cal = json.load(f)
        amp_scale = float(cal.get("amp_scale", amp_scale))
        t_g = cal.get("t_g_ns", t_g)
    if args.target_eta is not None:
        t_g = auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                       float(config["lam_b"]), float(args.target_eta))
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
