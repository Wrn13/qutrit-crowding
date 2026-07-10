"""
calibrate_iswap.py
==================

Per-device calibration of the SNAIL iSWAP pump, to be run ONCE before a spectator
sweep (see run_sweep_zhou.py). Standalone and SLURM-friendly: one device in, one
calibration JSON out.

Why this is needed
------------------
The analytic pump normalization sets the pulse area from the leading-order rate
6 g3 lambda_a lambda_b |eta| (Zhou Eqs. 55/73). At finite participation/eta the
true effective rate is dressed upward (virtual-coupler / Bloch-Siegert
corrections), so the open-loop "pi/2" pulse OVER-rotates and the |eg> -> |ge>
swap overshoots -- the dominant cause of the ~0.85 transfer / ~0.6 fidelity floor
seen off-resonance. This script mimics a hardware Rabi calibration: it integrates
the SPECTATOR-FREE gate with QuTiP and finds the pump-amplitude scale (and,
optionally, a small pump-frequency offset) that maximizes the bare iSWAP. The
result is written as `amp_scale` / `wp_offset_GHz`, which run_sweep_zhou.py
applies on top of the analytic normalization.

It also helps choose a perturbative gate time: `--target-eta` sets t_g so that
the normalized |eta| peak equals the requested value (the counter-rotating
coupler error shrinks with eta, so eta <~ 0.1 is a good target).

Usage
-----
    python calibrate_iswap.py --device dev.json --out calib.json \
        [--target-eta 0.1] [--t-g 200] [--tune-freq] \
        [--update-device dev_calibrated.json]

`dev.json` uses the same schema as run_sweep_zhou.py's --device file (it is merged
over run_sweep_zhou.DEFAULT_CONFIG). QuTiP is required (the calibration integrates
the exact Hamiltonian), so run it on a compute node; see snail_calibrate.slurm.
The pure search logic (`maximize_1d`) needs only numpy and is unit-testable
without QuTiP.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

TWO_PI: float = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Device / pulse helpers
# ---------------------------------------------------------------------------
def load_device(path: str) -> Dict[str, Any]:
    """Load a device JSON merged over run_sweep_zhou.DEFAULT_CONFIG.

    Parameters
    ----------
    path : str
        Path to the device JSON (same schema as run_sweep_zhou's --device file).

    Returns
    -------
    dict
        The merged configuration (device values override the defaults).
    """
    from run_sweep_zhou import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    with open(path) as f:
        config.update(json.load(f))
    return config


def target_eta_area(g3_GHz: float, lam_a: float, lam_b: float) -> float:
    """Pulse area integral|eta|dt (ns) that the analytic normalization targets for
    a full iSWAP: (pi/2) / (6 g3 lambda_a lambda_b), with g3 in rad/ns.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.

    Returns
    -------
    float
        The required integral of |eta| over the gate, in ns.
    """
    return (np.pi / 2) / (6 * (g3_GHz * TWO_PI) * lam_a * lam_b)


def auto_t_g(g3_GHz: float, lam_a: float, lam_b: float, target_eta: float) -> float:
    """Gate time (ns) for which a raised-cosine pump normalized to a full iSWAP has
    peak |eta| = `target_eta`. For the Hann window integral|eta|dt = eta_peak * t_g/2,
    so t_g = 2 * area / target_eta.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.
    target_eta : float
        Desired peak |eta| (perturbative: <~ 0.1).

    Returns
    -------
    float
        The gate duration in ns.
    """
    if target_eta <= 0.0:
        raise ValueError("target_eta must be positive.")
    return 2.0 * target_eta_area(g3_GHz, lam_a, lam_b) / target_eta


def build_coupler(config: Dict[str, Any], t_g: float, amp_scale: float,
                  wp_offset_GHz: float):
    """Build the SPECTATOR-FREE (3-mode: qubit a, qubit b, coupler) coupler with
    the pump applied, normalized to a full iSWAP and scaled by `amp_scale`.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    amp_scale : float
        Multiplicative correction on the normalized pump amplitude.
    wp_offset_GHz : float
        Offset added to the pump frequency w_b - w_a (GHz).

    Returns
    -------
    (ZhouCoupler, float, float)
        The coupler, the pump frequency w_p (GHz), and the resulting peak |eta|.
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

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
    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=EnvCls(amp=1.0, t_g=t_g), is_eta=True),
                 normalize_iswap=(0, 1))
    cpl.scale_pump_amplitude(amp_scale)
    return cpl, w_p_GHz, cpl.peak_eta()


def transfer_probability(config: Dict[str, Any], t_g: float, amp_scale: float,
                         wp_offset_GHz: float, solver: Dict[str, Any]) -> float:
    """Single-shot swap probability P(|eg> -> |ge>) at t_g (QuTiP sesolve). This is
    a fast (one-trajectory) proxy for the rotation angle, used as the search
    objective; the full leakage-aware fidelity is evaluated once at the optimum.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    amp_scale : float
        Pump-amplitude correction to test.
    wp_offset_GHz : float
        Pump-frequency offset to test (GHz).
    solver : dict
        QuTiP integrator options (atol, rtol, nsteps).

    Returns
    -------
    float
        P(|ge>) starting from |eg>.
    """
    cpl, _w_p, _eta = build_coupler(config, t_g, amp_scale, wp_offset_GHz)
    state = cpl.evolve_state([1, 0, 0], t_g, **solver)
    return float(np.abs(state[cpl.fock_index([0, 1, 0])]) ** 2)


# ---------------------------------------------------------------------------
# Pure 1-D maximizer (no QuTiP; unit-testable)
# ---------------------------------------------------------------------------
def maximize_1d(func: Callable[[float], float], lo: float, hi: float,
                n_points: int = 7, n_refine: int = 2) -> Tuple[float, float, int]:
    """Maximize a unimodal `func` on [lo, hi] by a coarse grid plus successive
    zoom-ins around the best point. Deterministic and cache-backed.

    Parameters
    ----------
    func : callable(float) -> float
        Objective to MAXIMIZE.
    lo, hi : float
        Search bounds.
    n_points : int, default 7
        Grid points per refinement round.
    n_refine : int, default 2
        Number of zoom-in rounds after the initial grid.

    Returns
    -------
    (float, float, int)
        Best x, best func(x), and the number of distinct evaluations.
    """
    cache: Dict[float, float] = {}

    def evaluate(x: float) -> float:
        key = round(x, 10)
        if key not in cache:
            cache[key] = func(x)
        return cache[key]

    best_x, best_f = lo, -np.inf
    for _ in range(n_refine + 1):
        grid = np.linspace(lo, hi, n_points)
        for x in grid:
            value = evaluate(float(x))
            if value > best_f:
                best_f, best_x = value, float(x)
        step = (hi - lo) / (n_points - 1)
        lo, hi = best_x - step, best_x + step          # zoom to +/- one step
    return best_x, best_f, len(cache)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(config: Dict[str, Any], t_g: float, *,
              amp_bounds: Tuple[float, float] = (0.6, 1.15), amp_points: int = 7,
              tune_freq: bool = False, freq_span_MHz: float = 12.0, freq_points: int = 5,
              solver: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Find the pump-amplitude scale (and optional frequency offset) that maximizes
    the spectator-free iSWAP transfer, then report the full fidelity at the optimum.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns) to calibrate at.
    amp_bounds : (float, float), default (0.6, 1.15)
        Search interval for the amplitude scale.
    amp_points : int, default 7
        Grid points per amplitude refinement round.
    tune_freq : bool, default False
        Also search a small pump-frequency offset (one alternating amp/freq pass).
    freq_span_MHz : float, default 12.0
        Half-width of the frequency-offset search (MHz), used when `tune_freq`.
    freq_points : int, default 5
        Grid points for the frequency search.
    solver : dict, optional
        QuTiP integrator options; defaults to atol=1e-10, rtol=1e-8, nsteps=5e5.

    Returns
    -------
    dict
        Calibration record: amp_scale, wp_offset_GHz, w_p_GHz, t_g_ns, eta_peak,
        transfer, F_avg, leakage, and search metadata.
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}

    wp_offset = 0.0
    amp_scale, transfer, n_eval = maximize_1d(
        lambda s: transfer_probability(config, t_g, s, wp_offset, solver),
        amp_bounds[0], amp_bounds[1], n_points=amp_points)

    if tune_freq:
        span = freq_span_MHz / 1000.0
        wp_offset, _t, n2 = maximize_1d(
            lambda off: transfer_probability(config, t_g, amp_scale, off, solver),
            -span, span, n_points=freq_points, n_refine=1)
        amp_scale, transfer, n3 = maximize_1d(
            lambda s: transfer_probability(config, t_g, s, wp_offset, solver),
            amp_bounds[0], amp_bounds[1], n_points=amp_points)
        n_eval += n2 + n3

    # full leakage-aware fidelity at the optimum
    cpl, w_p_GHz, eta_peak = build_coupler(config, t_g, amp_scale, wp_offset)
    F_avg, leakage, _U = cpl.iswap_fidelity(0, 1, t_g, **solver)

    return {
        "amp_scale": round(float(amp_scale), 5),
        "wp_offset_GHz": round(float(wp_offset), 6),
        "w_p_GHz": round(float(w_p_GHz), 6),
        "t_g_ns": float(t_g),
        "eta_peak": round(float(eta_peak), 5),
        "transfer": round(float(transfer), 5),
        "F_avg": round(float(F_avg), 5),
        "leakage": round(float(leakage), 5),
        "n_evaluations": int(n_eval),
        "tuned_frequency": bool(tune_freq),
    }


def main() -> None:
    """Command-line entry point. Calibrates one device and writes the result JSON;
    with --update-device, also writes a device file with the calibration merged in
    (t_g_ns, amp_scale, wp_offset_GHz) ready for run_sweep_zhou.py."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON (run_sweep schema)")
    ap.add_argument("--out", default="calibration.json", help="output calibration JSON")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="pick t_g so the normalized |eta| peak equals this (<~0.1)")
    ap.add_argument("--t-g", type=float, default=None,
                    help="explicit gate time (ns); overrides --target-eta and the device t_g_ns")
    ap.add_argument("--tune-freq", action="store_true",
                    help="also search a small pump-frequency offset")
    ap.add_argument("--gpu", action="store_true",
                    help="run the solver on GPU via qutip-jax/diffrax (see zhou_coupler.use_gpu)")
    ap.add_argument("--amp-lo", type=float, default=0.6, help="amplitude-scale lower bound")
    ap.add_argument("--amp-hi", type=float, default=1.15, help="amplitude-scale upper bound")
    ap.add_argument("--amp-points", type=int, default=7, help="grid points per amplitude round")
    ap.add_argument("--update-device", default=None,
                    help="write a device JSON with the calibration merged in")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    args = ap.parse_args()

    if args.gpu:
        import zhou_coupler
        zhou_coupler.use_gpu(True)

    config = load_device(args.device)

    if args.t_g is not None:
        t_g = args.t_g
    elif args.target_eta is not None:
        t_g = auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                       float(config["lam_b"]), args.target_eta)
    else:
        t_g = float(config["t_g_ns"])

    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}
    result = calibrate(config, t_g, amp_bounds=(args.amp_lo, args.amp_hi),
                       amp_points=args.amp_points, tune_freq=args.tune_freq,
                       solver=solver)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("calibration:")
    for k, v in result.items():
        print(f"  {k:16s}= {v}")
    print(f"written to {args.out}")

    if args.update_device:
        calibrated = dict(config)
        calibrated.update({"t_g_ns": result["t_g_ns"],
                           "amp_scale": result["amp_scale"],
                           "wp_offset_GHz": result["wp_offset_GHz"]})
        with open(args.update_device, "w") as f:
            json.dump(calibrated, f, indent=2)
        print(f"calibrated device written to {args.update_device}")


if __name__ == "__main__":
    main()