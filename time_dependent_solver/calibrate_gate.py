#!/usr/bin/env python3
"""
calibrate_gate.py
=================

End-to-end iSWAP tune-up that mirrors the experimental sequence, then validates
the calibrated gate.

At the pump strengths a real device runs (|eta| ~ 1-1.5), the open-loop pi/2
normalization is no longer accurate: the pulse over-rotates (amplitude error) and
the AC-Stark shift moves the resonance (frequency error), and the two are coupled
because the shift grows with |eta|^2 while the amplitude calibration assumes the
pump is on resonance. Experimentalists resolve this by iterating two 1-D
calibrations, FREQUENCY FIRST:

  1. Frequency (Stark) chevron -- sweep the pump frequency, take the offset that
     maximises swap contrast (find_stark_resonance). Contrast-based, so the vertex
     is (to leading order) independent of the drive amplitude -- robust even before
     the amplitude is calibrated.
  2. Amplitude (Rabi) scan -- ON the located resonance, vary the pump-amplitude
     scale to maximise the |01>->|10> transfer. This MUST follow the frequency
     scan: off-resonance the transfer at t_g is capped at g^2/(g^2+delta^2) with a
     shifted optimum, so a Rabi calibration at the wrong frequency is corrupted.

Repeating (1)-(2) a couple of times converges (wp_offset, amp_scale): the second
chevron refines the (~eta^2) Stark magnitude at the calibrated amplitude. A final
leakage-aware fidelity test on the calibrated gate then reports F_avg, leakage,
transfer, and the residual conditional (ZZ) phase.

The full per-iteration record (amplitude curves + chevrons) is written for
calibration_plots.py to visualise the Stark shift.

Usage
-----
    python calibrate_gate.py --device dev.json --target-eta 1.5 \
        --iters 2 --jobs 16 --out cal_dev.json --update-device dev_calibrated.json
    # dev_calibrated.json then carries amp_scale + wp_offset_GHz for run_sweep_zhou.py

Requires QuTiP (it integrates the exact Hamiltonian); run on a compute node. The
search/bookkeeping logic (amplitude_scan bookkeeping, iteration, phase extraction)
is exercised without QuTiP in the module self-test.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from device_utils import (load_device, auto_t_g, build_coupler,
                             transfer_probability, maximize_1d)
import find_stark_resonance as FS

TWO_PI: float = 2.0 * np.pi
_DEFAULT_SOLVER: Dict[str, Any] = {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}


# ---------------------------------------------------------------------------
# Individual calibration scans (experiment analogues)
# ---------------------------------------------------------------------------
def amplitude_scan(config: Dict[str, Any], t_g: float, wp_offset_GHz: float,
                   amp_bounds: Tuple[float, float], n_points: int,
                   solver: Dict[str, Any]) -> Dict[str, Any]:
    """Amplitude (Rabi) calibration: sweep the pump-amplitude scale at a fixed
    pump frequency and maximise the |01>->|10> transfer.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    wp_offset_GHz : float
        Current pump-frequency offset from |w_b - w_a| (GHz).
    amp_bounds : (float, float)
        Search interval for the amplitude scale.
    n_points : int
        Grid points per refinement round.
    solver : dict
        QuTiP integrator options.

    Returns
    -------
    dict
        amps, transfer (the recorded curve over the initial grid), amp_best,
        transfer_best.
    """
    grid = np.linspace(amp_bounds[0], amp_bounds[1], n_points)
    curve = np.array([transfer_probability(config, t_g, float(a), wp_offset_GHz, solver)
                      for a in grid])
    amp_best, transfer_best, _n = maximize_1d(
        lambda s: transfer_probability(config, t_g, s, wp_offset_GHz, solver),
        amp_bounds[0], amp_bounds[1], n_points=n_points)
    return {"amps": grid, "transfer": curve,
            "amp_best": float(amp_best), "transfer_best": float(transfer_best)}


def frequency_chevron(config: Dict[str, Any], t_g: float, amp_scale: float,
                      span_MHz: float, n_points: int, window_factor: float,
                      time_points: int, solver: Dict[str, Any],
                      n_jobs: Optional[int]) -> Dict[str, Any]:
    """Frequency (Stark) calibration: constant-pump chevron at the calibrated
    amplitude; the resonance offset maximises swap contrast. Thin wrapper over
    find_stark_resonance.scan that keeps the arrays for plotting.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns); sets the probe |eta|.
    amp_scale : float
        Calibrated pump-amplitude scale.
    span_MHz : float
        Chevron scan is +/- span/2 about |w_b - w_a|.
    n_points : int
        Offset grid points.
    window_factor : float
        Time window = window_factor * t_g.
    time_points : int
        Time samples.
    solver : dict
        QuTiP integrator options.
    n_jobs : int, optional
        Worker processes for the offset scan.

    Returns
    -------
    dict
        The find_stark_resonance.scan result (offsets_GHz, times_ns, P10,
        max_transfer, eta_op, resonance_offset_GHz, ...).
    """
    span = span_MHz / 1000.0
    offsets = np.linspace(-span / 2.0, span / 2.0, n_points)
    return FS.scan(config, t_g, amp_scale, offsets, window_factor * t_g,
                   time_points, solver, n_jobs=n_jobs)


# ---------------------------------------------------------------------------
# Final validation of the calibrated gate
# ---------------------------------------------------------------------------
def conditional_phase(U: np.ndarray) -> float:
    """Gauge-invariant conditional (ZZ) phase of a 2-qubit gate in the
    {|00>,|01>,|10>,|11>} basis:

        phi = arg U00 + arg U11 - arg det(M),

    where M is the single-excitation block on {|01>,|10>}. Using det(M) makes this
    correct for both diagonal (CZ-like) and antidiagonal (iSWAP-like) gates -- for
    an ideal iSWAP det(M) = -U(01->10) U(10->01) = 1, so phi = 0 -- and invariant
    under single-qubit Z (each Z phase cancels between U11/U00 and det M), so it
    isolates the two-qubit phase that virtual-Z cannot remove. Wrapped to (-pi, pi].

    Parameters
    ----------
    U : ndarray, shape (4, 4)
        Projected propagator on the computational subspace.

    Returns
    -------
    float
        Conditional phase (rad).
    """
    det_M = U[1, 1] * U[2, 2] - U[1, 2] * U[2, 1]      # 1-excitation block {01,10}
    phi = np.angle(U[0, 0]) + np.angle(U[3, 3]) - np.angle(det_M)
    return float((phi + np.pi) % (2 * np.pi) - np.pi)


def final_test(config: Dict[str, Any], t_g: float, amp_scale: float,
               wp_offset_GHz: float, solver: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the calibrated gate: leakage-aware F_avg, transfer, leakage, the
    residual conditional phase, and the operating |eta|.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    amp_scale : float
        Calibrated pump-amplitude scale.
    wp_offset_GHz : float
        Calibrated pump-frequency offset (GHz).
    solver : dict
        QuTiP integrator options.

    Returns
    -------
    dict
        amp_scale, wp_offset_GHz, w_p_GHz, eta_peak, F_avg, leakage, transfer,
        conditional_phase_rad.
    """
    cpl, w_p_GHz, eta_peak = build_coupler(config, t_g, amp_scale, wp_offset_GHz)
    F_avg, leakage, U = cpl.iswap_fidelity(0, 1, t_g, fit_virtual_z=True, **solver)
    transfer = transfer_probability(config, t_g, amp_scale, wp_offset_GHz, solver)
    return {"amp_scale": round(float(amp_scale), 5),
            "wp_offset_GHz": round(float(wp_offset_GHz), 6),
            "w_p_GHz": round(float(w_p_GHz), 6),
            "eta_peak": round(float(eta_peak), 5),
            "F_avg": round(float(F_avg), 5),
            "leakage": round(float(leakage), 5),
            "transfer": round(float(transfer), 5),
            "conditional_phase_rad": round(conditional_phase(U), 5)}


# ---------------------------------------------------------------------------
# Iterated tune-up
# ---------------------------------------------------------------------------
def run_calibration(config: Dict[str, Any], t_g: float, *, iters: int = 2,
                    amp_bounds: Tuple[float, float] = (0.6, 1.4), amp_points: int = 9,
                    span_MHz: float = 60.0, chevron_points: int = 21,
                    window_factor: float = 2.0, time_points: int = 160,
                    solver: Optional[Dict[str, Any]] = None,
                    n_jobs: Optional[int] = None) -> Dict[str, Any]:
    """Iterate Stark-frequency and amplitude calibration to self-consistency, then
    run the final gate test.

    Each round locates the Stark resonance first (measured from the bare
    |w_b - w_a|, so wp_offset is SET, not accumulated), then calibrates the Rabi
    amplitude ON that resonance. Frequency-first because the chevron vertex is
    amplitude-robust while the amplitude scan is corrupted by detuning; a second
    round refines the amplitude-dependent (~eta^2) Stark magnitude. Two rounds
    usually suffice.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    iters : int, default 2
        Number of amplitude/frequency rounds.
    amp_bounds : (float, float), default (0.6, 1.4)
        Amplitude-scale search interval (wider than the low-eta default since
        high-|eta| gates over-rotate more).
    amp_points : int, default 9
        Amplitude grid points per round.
    span_MHz, chevron_points, window_factor, time_points
        Chevron scan settings (see frequency_chevron).
    solver : dict, optional
        QuTiP integrator options.
    n_jobs : int, optional
        Worker processes for the chevron scans.

    Returns
    -------
    dict
        {"t_g_ns", "iterations": [...per-round records...], "final": {...}}.
        Each iteration record carries the amplitude curve and the chevron arrays.
    """
    solver = solver or dict(_DEFAULT_SOLVER)
    amp_scale, wp_offset = 1.0, 0.0
    history: List[Dict[str, Any]] = []

    for it in range(iters):
        # Frequency FIRST: the chevron's contrast-based vertex is (to leading order)
        # independent of the drive amplitude, whereas the amplitude scan maximises
        # transfer AT t_g, which off-resonance is capped at g^2/(g^2+delta^2) with a
        # shifted optimum. So we locate the resonance, then calibrate the Rabi
        # amplitude ON it. The chevron probe amplitude still tracks amp_scale, so a
        # second round refines the (~eta^2) Stark magnitude at the calibrated amp.
        chev = frequency_chevron(config, t_g, amp_scale, span_MHz, chevron_points,
                                 window_factor, time_points, solver, n_jobs)
        wp_offset = float(chev["resonance_offset_GHz"])
        amp = amplitude_scan(config, t_g, wp_offset, amp_bounds, amp_points, solver)
        amp_scale = amp["amp_best"]
        history.append({"iter": it, "amp_scale": amp_scale, "wp_offset_GHz": wp_offset,
                        "eta_op": float(chev["eta_op"]),
                        "amp_curve": amp, "chevron": chev})
        print(f"  iter {it}: wp_offset={wp_offset*1e3:+.2f} MHz  "
              f"amp_scale={amp_scale:.4f}  |eta|={chev['eta_op']:.3f}  "
              f"peak_contrast={chev['max_transfer'].max():.4f}")

    final = final_test(config, t_g, amp_scale, wp_offset, solver)
    print(f"  final: F_avg={final['F_avg']:.4f}  leakage={final['leakage']:.4f}  "
          f"transfer={final['transfer']:.4f}  "
          f"phi_cond={final['conditional_phase_rad']:+.3f} rad")
    return {"t_g_ns": float(t_g), "iterations": history, "final": final}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_record(record: Dict[str, Any], json_path: str) -> str:
    """Write scalar calibration results to JSON and the per-iteration arrays
    (amplitude curves + chevrons) to a sibling .npz for calibration_plots.py.

    Parameters
    ----------
    record : dict
        Output of run_calibration.
    json_path : str
        Destination JSON path; the .npz is the same stem with .npz.

    Returns
    -------
    str
        The .npz path written.
    """
    scalars = {"t_g_ns": record["t_g_ns"], "final": record["final"],
               "iterations": [{k: v for k, v in it.items()
                               if k not in ("amp_curve", "chevron")}
                              for it in record["iterations"]]}
    with open(json_path, "w") as f:
        json.dump(scalars, f, indent=2)

    npz_path = json_path.rsplit(".", 1)[0] + ".npz"
    arrays: Dict[str, Any] = {}
    for it in record["iterations"]:
        i = it["iter"]
        arrays[f"amp{i}_amps"] = it["amp_curve"]["amps"]
        arrays[f"amp{i}_transfer"] = it["amp_curve"]["transfer"]
        arrays[f"chev{i}_offsets_GHz"] = it["chevron"]["offsets_GHz"]
        arrays[f"chev{i}_times_ns"] = it["chevron"]["times_ns"]
        arrays[f"chev{i}_P10"] = it["chevron"]["P10"]
        arrays[f"chev{i}_resonance_GHz"] = np.array(it["chevron"]["resonance_offset_GHz"])
        arrays[f"chev{i}_eta_op"] = np.array(it["chevron"]["eta_op"])
    arrays["n_iters"] = np.array(len(record["iterations"]))
    np.savez_compressed(npz_path, **arrays)
    return npz_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Calibrate one device end-to-end and validate the calibrated iSWAP gate."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON (run_sweep schema)")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="operating |eta| (sets t_g); overrides --t-g / device t_g_ns")
    ap.add_argument("--t-g", type=float, default=None, help="operating gate time (ns)")
    ap.add_argument("--iters", type=int, default=2, help="amplitude/frequency rounds")
    ap.add_argument("--amp-lo", type=float, default=0.6)
    ap.add_argument("--amp-hi", type=float, default=1.4)
    ap.add_argument("--amp-points", type=int, default=9)
    ap.add_argument("--span-MHz", type=float, default=60.0)
    ap.add_argument("--chevron-points", type=int, default=21)
    ap.add_argument("--window-factor", type=float, default=2.0)
    ap.add_argument("--time-points", type=int, default=160)
    ap.add_argument("--jobs", type=int, default=0, help="workers for chevron scans")
    ap.add_argument("--gpu", action="store_true", help="qutip-jax/diffrax (forces --jobs 1)")
    ap.add_argument("--out", default="calibration.json", help="output JSON (+ .npz)")
    ap.add_argument("--update-device", default=None,
                    help="write device JSON with amp_scale + wp_offset_GHz set")
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
    config = load_device(resolve_device(args.device))
    t_g = args.t_g
    if args.target_eta is not None:
        t_g = auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                       float(config["lam_b"]), float(args.target_eta))
    if t_g is None:
        t_g = float(config.get("t_g_ns", 200.0))
    t_g = float(t_g)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}

    print(f"device={args.device}  t_g={t_g:.1f} ns  iters={args.iters}"
          f"{' GPU' if args.gpu else ''}")
    record = run_calibration(config, t_g, iters=args.iters,
                             amp_bounds=(args.amp_lo, args.amp_hi),
                             amp_points=args.amp_points, span_MHz=args.span_MHz,
                             chevron_points=args.chevron_points,
                             window_factor=args.window_factor,
                             time_points=args.time_points, solver=solver,
                             n_jobs=args.jobs)
    npz = save_record(record, args.out)
    print(f"written {args.out} and {npz}")

    if args.update_device:
        dev = dict(config)
        dev["amp_scale"] = record["final"]["amp_scale"]
        dev["wp_offset_GHz"] = record["final"]["wp_offset_GHz"]
        dev["t_g_ns"] = record["t_g_ns"]
        with open(args.update_device, "w") as f:
            json.dump(dev, f, indent=2)
        print(f"wrote {args.update_device} "
              f"(amp_scale={dev['amp_scale']}, wp_offset_GHz={dev['wp_offset_GHz']})")


# ---------------------------------------------------------------------------
# Self-test (no QuTiP): iteration bookkeeping + conditional-phase extraction
# ---------------------------------------------------------------------------
if __name__ == "__main__" and __import__("sys").argv[1:2] == ["selftest"]:
    # conditional_phase on a synthetic iSWAP with a known phi on |11>
    for phi_true in (0.0, 0.44 * np.pi, np.pi):
        U = np.zeros((4, 4), dtype=complex)
        U[0, 0] = 1.0
        U[1, 2] = 1j; U[2, 1] = 1j                 # iSWAP swap block
        U[3, 3] = np.exp(-1j * phi_true)
        got = conditional_phase(U)
        print(f"phi_true={phi_true:+.4f}  extracted={got:+.4f}  "
              f"ok={np.isclose(abs(got), abs(phi_true), atol=1e-9)}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()