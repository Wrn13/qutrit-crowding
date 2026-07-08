#!/usr/bin/env python3
"""
calibrate_iswap.py
=================

Closed-loop iSWAP calibration for a ``zhou_coupler.ZhouCoupler`` device.

Why this exists
---------------
The analytic pump normalization sets the pulse area from the leading-order rate
g_eff = 6 g3 lambda_a lambda_b |eta| (Zhou Eqs. 55/73). That is *open loop*: at
finite participation/eta the true effective rate is dressed upward (virtual
coupler / Bloch-Siegert processes), so the nominal pi/2 pulse over-rotates and the
realized iSWAP fidelity floors well below 1 even with no spectator. A real device
fixes this with a tune-up -- measure the Rabi rate and set the amplitude to land
exactly on pi/2. This tool does the numerical equivalent: holding the
spectator-free two-qubit + coupler system, it scans the pump amplitude scale (and
optionally w_p) at a chosen operating eta and reports the calibration that
maximizes the leakage-aware average gate fidelity.

Feed the result (t_g, w_p, amplitude scale) into run_sweep_zhou before the
spectator sweep so every point uses the calibrated target gate.

Workflow (same SLURM array pattern as run_sweep_zhou)
-----------------------------------------------------
    calibrate_zhou.py prepare --outdir cal/ --device dev.json   # writes grid.json, prints M
    sbatch --array=0-<M-1> snail_calibrate.slurm                # one task per (amp, w_p)
    calibrate_zhou.py collect --outdir cal/ --out cal/calibration.json
Single node / quick:
    calibrate_zhou.py local --outdir cal/ --nproc 8 && calibrate_zhou.py collect --outdir cal/
    calibrate_zhou.py run   --outdir cal/                       # prepare + local + collect

Solver
------
Exact QobjEvo via ZhouCoupler.iswap_fidelity (QuTiP compiled sesolve). Runs on CPU
by default; pass --gpu to route through the qutip-jax / diffrax backend
(zhou_coupler.use_gpu). GPU only helps for large Hilbert spaces.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI: float = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Default configuration (device matches run_sweep_zhou's DEFAULT_CONFIG so the
# calibration applies to the same target gate the sweep simulates).
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # device
    "qubit_freqs_GHz": [5.00, 4.60],
    "qubit_levels":    3,
    "lam_a":           0.20,
    "lam_b":           0.20,
    "coupler_freq_GHz": 7.00,
    "coupler_levels":  5,
    "g3_GHz":          0.10,
    "g4_GHz":          0.0,
    "envelope":        "raised_cosine",   # "raised_cosine" or "constant"
    # operating point: set t_g_ns directly, or leave 0 to derive it from target_eta
    "target_eta":      0.10,              # perturbative pump -> small counter-rotating error
    "t_g_ns":          0.0,               # >0 overrides target_eta
    # calibration grid
    "amp_min":         0.60,
    "amp_max":         1.10,
    "amp_points":      11,
    "freq_offsets_MHz": [0.0],            # add e.g. [-10,-5,0,5,10] to also tune w_p
    # solver
    "rtol": 1e-8, "atol": 1e-10, "nsteps": 500000,
}

A_INDEX, B_INDEX = 0, 1                    # qubit mode indices (coupler is index 2)


# ---------------------------------------------------------------------------
# Operating point and coupler construction
# ---------------------------------------------------------------------------
def t_g_for_target_eta(g3_GHz: float, lam_a: float, lam_b: float,
                       eta_target: float) -> float:
    """Gate time (ns) whose normalized raised-cosine iSWAP has peak |eta| = eta_target.

    From the area condition g_eff_per_eta * (eta_peak * t_g / 2) = pi/2 with
    g_eff_per_eta = 6 (2 pi g3) lambda_a lambda_b, so t_g = pi / (g_eff_per_eta eta).

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity (GHz).
    lam_a, lam_b : float
        Qubit participations lambda_is.
    eta_target : float
        Desired peak displaced pump amplitude |eta|.

    Returns
    -------
    float
        Gate duration in ns.
    """
    g_eff_per_eta = 6.0 * (TWO_PI * g3_GHz) * lam_a * lam_b   # rad/ns per unit |eta|
    if g_eff_per_eta <= 0 or eta_target <= 0:
        raise ValueError("g3, participations and target_eta must be positive.")
    return float(np.pi / (g_eff_per_eta * eta_target))


def operating_t_g(config: Dict[str, Any]) -> float:
    """Resolve the gate time: explicit ``t_g_ns`` if > 0, else from ``target_eta``."""
    t_g = float(config.get("t_g_ns", 0.0))
    if t_g > 0:
        return t_g
    return t_g_for_target_eta(float(config["g3_GHz"]), float(config["lam_a"]),
                              float(config["lam_b"]), float(config["target_eta"]))


def build_bare_coupler(config: Dict[str, Any], t_g: float, amp_scale: float,
                       w_p_GHz: float) -> "Any":
    """Build the spectator-free 2-qubit + coupler system with a normalized pump
    whose amplitude is then scaled by ``amp_scale``.

    Parameters
    ----------
    config : dict
        Device configuration (see DEFAULT_CONFIG).
    t_g : float
        Gate duration (ns).
    amp_scale : float
        Multiplier applied to the analytically normalized pump amplitude (1.0 =
        the open-loop pi/2 value).
    w_p_GHz : float
        Pump frequency (GHz).

    Returns
    -------
    ZhouCoupler
        The configured coupler (qubits at indices 0, 1; coupler at index 2).
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

    qubit_freqs = config["qubit_freqs_GHz"]
    freqs = [float(qubit_freqs[0]), float(qubit_freqs[1]), float(config["coupler_freq_GHz"])]
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"])]
    nonlin: Dict[int, float] = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    cpl = ZhouCoupler(freqs, 2, {A_INDEX: float(config["lam_a"]), B_INDEX: float(config["lam_b"])},
                      nonlin, levels=levels)
    envelope = (RaisedCosine(1.0, t_g) if config["envelope"] == "raised_cosine"
                else ConstantPulse(1.0, t_g))
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=envelope, is_eta=True),
                 normalize_iswap=(A_INDEX, B_INDEX))
    cpl._pump_tones[0].envelope.amp *= float(amp_scale)       # apply the calibration scale
    return cpl


# ---------------------------------------------------------------------------
# Fidelity evaluation
# ---------------------------------------------------------------------------
def evaluate(config: Dict[str, Any], amp_scale: float,
             w_p_offset_MHz: float) -> Dict[str, Any]:
    """Evaluate the bare-gate fidelity for one (amplitude scale, pump offset).

    Parameters
    ----------
    config : dict
        Device + solver configuration.
    amp_scale : float
        Pump amplitude multiplier relative to the analytic pi/2 normalization.
    w_p_offset_MHz : float
        Offset added to the bare pump frequency |w_b - w_a| (MHz).

    Returns
    -------
    dict
        amp_scale, w_p_offset_MHz, t_g_ns, w_p_GHz, eta_peak, F, leakage,
        p_transfer ( = |<ge|U|eg>|^2 ).
    """
    qubit_freqs = config["qubit_freqs_GHz"]
    t_g = operating_t_g(config)
    w_p = abs(float(qubit_freqs[1]) - float(qubit_freqs[0])) + float(w_p_offset_MHz) / 1000.0
    cpl = build_bare_coupler(config, t_g, amp_scale, w_p)

    F, leakage, U = cpl.iswap_fidelity(A_INDEX, B_INDEX, t_g, fit_virtual_z=True,
                                       atol=float(config["atol"]),
                                       rtol=float(config["rtol"]),
                                       nsteps=int(config["nsteps"]))

    # columns are |00>,|01>,|10>,|11>; |eg>=|10> is col 2, |ge>=|01> is row 1.
    p_transfer = float(abs(U[1, 2]) ** 2)
    return dict(amp_scale=float(amp_scale), w_p_offset_MHz=float(w_p_offset_MHz),
                t_g_ns=float(t_g), w_p_GHz=float(w_p), eta_peak=float(cpl.peak_eta()),
                F=float(F), leakage=float(leakage), p_transfer=p_transfer)


# ---------------------------------------------------------------------------
# Calibration grid
# ---------------------------------------------------------------------------
@dataclass
class CalPoint:
    """One calibration grid point.

    Attributes
    ----------
    index : int
        Position in the grid / output filename suffix.
    amp_scale : float
        Pump amplitude multiplier to evaluate.
    w_p_offset_MHz : float
        Pump-frequency offset (MHz) from the bare |w_b - w_a|.
    """

    index: int
    amp_scale: float
    w_p_offset_MHz: float


def build_grid(config: Dict[str, Any]) -> List[CalPoint]:
    """Build the (amplitude scale x frequency offset) grid from ``config``.

    Parameters
    ----------
    config : dict
        Uses amp_min, amp_max, amp_points and freq_offsets_MHz.

    Returns
    -------
    list of CalPoint
        Ordered amp-major, then frequency offset.
    """
    amps = np.linspace(float(config["amp_min"]), float(config["amp_max"]),
                       int(config["amp_points"]))
    offsets = [float(x) for x in config["freq_offsets_MHz"]]
    points: List[CalPoint] = []
    index = 0
    for amp in amps:
        for offset in offsets:
            points.append(CalPoint(index=index, amp_scale=float(amp), w_p_offset_MHz=offset))
            index += 1
    return points


def write_grid(outdir: str, config: Dict[str, Any], points: List[CalPoint]) -> str:
    """Persist config + grid to ``<outdir>/grid.json`` (creates ``points/``)."""
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    path = os.path.join(outdir, "grid.json")
    with open(path, "w") as f:
        json.dump({"config": config, "points": [asdict(p) for p in points]}, f, indent=2)
    return path


def load_grid(outdir: str) -> Tuple[Dict[str, Any], List[CalPoint]]:
    """Load config + grid from ``<outdir>/grid.json``."""
    with open(os.path.join(outdir, "grid.json")) as f:
        blob = json.load(f)
    return blob["config"], [CalPoint(**p) for p in blob["points"]]


def run_point(point: CalPoint, config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one grid point and tag it with its index.

    Parameters
    ----------
    point : CalPoint
        The (amp_scale, w_p_offset) to evaluate.
    config : dict
        Device + solver configuration.

    Returns
    -------
    dict
        The `evaluate` result with an added ``index`` field.
    """
    result = evaluate(config, point.amp_scale, point.w_p_offset_MHz)
    result["index"] = point.index
    return result


def save_point(result: Dict[str, Any], outdir: str) -> str:
    """Write one result dict to ``<outdir>/points/point_XXXXX.json``."""
    path = os.path.join(outdir, "points", f"point_{result['index']:05d}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Collection -> calibration.json (best point + parabolic refinement)
# ---------------------------------------------------------------------------
def _parabolic_vertex(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Vertex abscissa of the parabola through three (x, y) points, or None if the
    points are collinear / the extremum is a minimum."""
    (x0, x1, x2), (y0, y1, y2) = x, y
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a >= 0:                                   # not a maximum
        return None
    return float(-b / (2 * a))


def collect(outdir: str, out_path: Optional[str] = None) -> Dict[str, Any]:
    """Gather point results, pick the highest-fidelity calibration, and write it.

    Parameters
    ----------
    outdir : str
        Calibration directory containing ``points/point_*.json``.
    out_path : str, optional
        Where to write the calibration JSON (default ``<outdir>/calibration.json``).

    Returns
    -------
    dict
        The calibration: device echo, t_g_ns, w_p_GHz, amp_scale (grid best),
        amp_scale_refined (parabolic), eta_peak, and the achieved F / leakage /
        p_transfer at the best grid point.
    """
    files = sorted(glob.glob(os.path.join(outdir, "points", "point_*.json")))
    if not files:
        raise FileNotFoundError(f"no point files in {outdir}/points/")
    results: List[Dict[str, Any]] = []
    for path in files:
        with open(path) as f:
            results.append(json.load(f))

    best = max(results, key=lambda r: r["F"])

    # Parabolic refinement in amplitude, restricted to the best frequency offset.
    same_offset = sorted((r for r in results
                          if abs(r["w_p_offset_MHz"] - best["w_p_offset_MHz"]) < 1e-9),
                         key=lambda r: r["amp_scale"])
    amp_refined: Optional[float] = None
    amps = [r["amp_scale"] for r in same_offset]
    if best["amp_scale"] in amps:
        i = amps.index(best["amp_scale"])
        if 0 < i < len(amps) - 1:                # need a neighbour on each side
            trip = same_offset[i - 1:i + 2]
            amp_refined = _parabolic_vertex([t["amp_scale"] for t in trip],
                                            [t["F"] for t in trip])

    calibration = {
        "device": {k: best_config_echo(outdir)[k] for k in
                   ("qubit_freqs_GHz", "coupler_freq_GHz", "lam_a", "lam_b",
                    "g3_GHz", "g4_GHz", "qubit_levels", "coupler_levels")},
        "t_g_ns": best["t_g_ns"],
        "w_p_GHz": best["w_p_GHz"],
        "w_p_offset_MHz": best["w_p_offset_MHz"],
        "amp_scale": best["amp_scale"],
        "amp_scale_refined": amp_refined,
        "eta_peak": best["eta_peak"],
        "F": best["F"],
        "leakage": best["leakage"],
        "p_transfer": best["p_transfer"],
        "n_points": len(results),
    }
    out_path = out_path or os.path.join(outdir, "calibration.json")
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)
    return calibration


def best_config_echo(outdir: str) -> Dict[str, Any]:
    """Return the device config persisted in ``<outdir>/grid.json``."""
    config, _ = load_grid(outdir)
    return config


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_local(outdir: str, config: Dict[str, Any], points: List[CalPoint],
               nproc: int) -> None:
    """Evaluate every grid point (process pool if nproc > 1) and save results."""
    if nproc <= 1:
        for p in points:
            save_point(run_point(p, config), outdir)
            print(f"[{p.index + 1}/{len(points)}] amp={p.amp_scale:.3f} "
                  f"off={p.w_p_offset_MHz:+.0f}MHz")
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        futures = {pool.submit(run_point, p, config): p for p in points}
        done = 0
        for fut in futures:
            pass  # submitted; gather below
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            result = fut.result()
            save_point(result, outdir)
            done += 1
            print(f"[{done}/{len(points)}] amp={result['amp_scale']:.3f} "
                  f"F={result['F']:.4f}")


def _merge_config(base: Dict[str, Any], device_path: Optional[str],
                  overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Layer a device JSON and CLI overrides on top of DEFAULT_CONFIG."""
    config = dict(base)
    if device_path:
        with open(device_path) as f:
            config.update(json.load(f))
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def main() -> None:
    """CLI entry point: prepare / point / local / collect / run.

    Subcommands mirror run_sweep_zhou. ``prepare`` writes the grid (and prints the
    array range), ``point --index k`` evaluates one grid point, ``collect`` writes
    calibration.json, ``local`` runs the whole grid in a pool, and ``run`` does
    prepare + local + collect in one process.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "point", "local", "collect", "run"])
    ap.add_argument("--outdir", default="cal")
    ap.add_argument("--device", default=None, help="device/config JSON to overlay on defaults")
    ap.add_argument("--out", default=None, help="calibration.json output path (collect/run)")
    ap.add_argument("--index", type=int, default=None, help="grid index (point mode)")
    ap.add_argument("--grid", default=None, help="explicit grid.json path (point mode)")
    ap.add_argument("--nproc", type=int, default=1)
    # common overrides
    ap.add_argument("--gpu", action="store_true",
                    help="run the solver on GPU via qutip-jax/diffrax (see zhou_coupler.use_gpu)")
    ap.add_argument("--target-eta", type=float, default=None)
    ap.add_argument("--t_g", type=float, default=None, help="explicit gate time (ns)")
    ap.add_argument("--amp-min", type=float, default=None)
    ap.add_argument("--amp-max", type=float, default=None)
    ap.add_argument("--amp-points", type=int, default=None)
    ap.add_argument("--freq-offsets", default=None,
                    help="comma-separated MHz offsets to also scan w_p, e.g. -10,0,10")
    args = ap.parse_args()

    if args.gpu:
        import zhou_coupler
        zhou_coupler.use_gpu(True)

    overrides: Dict[str, Any] = {
        "target_eta": args.target_eta,
        "t_g_ns": args.t_g, "amp_min": args.amp_min, "amp_max": args.amp_max,
        "amp_points": args.amp_points,
        "freq_offsets_MHz": ([float(x) for x in args.freq_offsets.split(",")]
                             if args.freq_offsets else None),
    }

    if args.cmd == "prepare":
        config = _merge_config(DEFAULT_CONFIG, args.device, overrides)
        points = build_grid(config)
        write_grid(args.outdir, config, points)
        t_g = operating_t_g(config)
        print(f"Prepared {len(points)} calibration points in {args.outdir}/ "
              f"(t_g={t_g:.1f} ns).")
        print(f"Submit with:\n  sbatch --array=0-{len(points) - 1} snail_calibrate.slurm")

    elif args.cmd == "point":
        grid_dir = args.outdir if args.grid is None else os.path.dirname(args.grid) or "."
        config, points = load_grid(grid_dir)
        if args.index is None:
            raise SystemExit("point mode needs --index")
        save_point(run_point(points[args.index], config), args.outdir)

    elif args.cmd == "local":
        config, points = load_grid(args.outdir)
        _run_local(args.outdir, config, points, args.nproc)

    elif args.cmd == "collect":
        cal = collect(args.outdir, args.out)
        _print_calibration(cal)

    elif args.cmd == "run":
        config = _merge_config(DEFAULT_CONFIG, args.device, overrides)
        points = build_grid(config)
        write_grid(args.outdir, config, points)
        _run_local(args.outdir, config, points, args.nproc)
        cal = collect(args.outdir, args.out)
        _print_calibration(cal)


def _print_calibration(cal: Dict[str, Any]) -> None:
    """Pretty-print the chosen calibration."""
    refined = cal["amp_scale_refined"]
    print("\n=== calibration ===")
    print(f"  t_g            = {cal['t_g_ns']:.1f} ns   (eta_peak = {cal['eta_peak']:.3f})")
    print(f"  w_p            = {cal['w_p_GHz']:.5f} GHz  (offset {cal['w_p_offset_MHz']:+.1f} MHz)")
    print(f"  amp_scale      = {cal['amp_scale']:.4f}   (grid best of {cal['n_points']})")
    if refined is not None:
        print(f"  amp_scale*     = {refined:.4f}   (parabolic refinement)")
    print(f"  -> F = {cal['F']:.4f}, leakage = {cal['leakage']:.4f}, "
          f"transfer = {cal['p_transfer']:.4f}")


if __name__ == "__main__":
    main()