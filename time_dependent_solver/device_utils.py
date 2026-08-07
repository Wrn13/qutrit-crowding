"""
device_utils.py
===============

Shared device I/O, spectator-free gate construction, and a pure 1-D maximizer,
used by the calibration and Stark-resonance tools. These were previously housed in
calibrate_iswap.py (now removed); the physics and signatures are unchanged.

`load_device` merges a device JSON over run_sweep_zhou.DEFAULT_CONFIG. `build_coupler`
constructs the 3-mode (qubit a, qubit b, coupler) gate with the pump normalized to a
full iSWAP and scaled by amp_scale (anharmonicity included). `transfer_probability`
is the one-trajectory swap proxy used as a fast search objective. `maximize_1d` is a
deterministic grid+zoom optimizer (numpy only, unit-testable without QuTiP).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

TWO_PI: float = 2.0 * np.pi


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
    """Pulse area integral|eta|dt (ns) the analytic normalization targets for a full
    iSWAP: (pi/2) / (6 g3 lambda_a lambda_b), g3 in rad/ns.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.

    Returns
    -------
    float
        Required integral of |eta| over the gate (ns).
    """
    return (np.pi / 2) / (6 * (g3_GHz * TWO_PI) * lam_a * lam_b)


def auto_t_g(g3_GHz: float, lam_a: float, lam_b: float, target_eta: float) -> float:
    """Gate time (ns) for which a raised-cosine full-iSWAP pump has peak
    |eta| = target_eta. Hann window: integral|eta|dt = eta_peak * t_g/2, so
    t_g = 2 * area / target_eta.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.
    target_eta : float
        Desired peak |eta|.

    Returns
    -------
    float
        Gate duration (ns).
    """
    if target_eta <= 0.0:
        raise ValueError("target_eta must be positive.")
    return 2.0 * target_eta_area(g3_GHz, lam_a, lam_b) / target_eta


def build_coupler(config: Dict[str, Any], t_g: float, amp_scale: float,
                  wp_offset_GHz: float, spec_abs_GHz: Optional[float] = None,
                  drag_beat_GHz: Optional[float] = None,
                  chirp_coeffs_GHz: Optional[Sequence[float]] = None):
    """Build the (qubit a, qubit b, coupler[, spectator]) gate with the pump
    normalized to a full iSWAP and scaled by amp_scale (anharmonicity included).

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
    spec_abs_GHz : float, optional
        If given, add a 4th spectator mode at this ABSOLUTE frequency (participation
        lam_b, ``spec_levels`` levels, ``anharm_spec_GHz``) so the tune-up sees the
        spectator, i.e. a hardware-style per-point calibration. None -> bare (a, b) pair.
    drag_beat_GHz : float, optional
        If given, apply a DRAG quadrature tuned to this beat (GHz) on the pump, so the
        calibration matches a DRAG-on gate. None -> no DRAG.
    chirp_coeffs_GHz : sequence of float, optional
        Legendre coefficients of a time-dependent pump-frequency offset delta(t)
        (GHz), applied ON TOP of the constant `wp_offset_GHz`. None or all-zero
        leaves the tone un-chirped and the solver path unchanged. Defaults to
        ``config["chirp_coeffs_GHz"]``. See :class:`envelope.Chirp`.

    Returns
    -------
    (ZhouCoupler, float, float)
        The coupler, its pump frequency w_p (GHz), and the resulting peak |eta|.
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse, make_chirp

    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float))
    ws = float(config["coupler_freq_GHz"])
    w_p_GHz = abs(wb - wa) + wp_offset_GHz
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"])]
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    aq = float(config.get("anharm_qubit_GHz", 0.0))
    freqs = [wa, wb, ws]
    participations = {0: float(config["lam_a"]), 1: float(config["lam_b"])}
    anharm = {0: aq, 1: aq}
    if spec_abs_GHz is not None:                    # add the spectator as a 4th mode
        freqs.append(float(spec_abs_GHz))
        levels.append(int(config.get("spec_levels", 3)))
        participations[3] = float(config["lam_b"])              # spectator participation = lam_b
        anharm[3] = float(config.get("anharm_spec_GHz", 0.0))
    cpl = ZhouCoupler(mode_freqs_GHz=freqs, coupler_index=2,
                      participations=participations, nonlinearities=nonlin, levels=levels,
                      anharmonicities_GHz=anharm)
    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse
    if chirp_coeffs_GHz is None:
        chirp_coeffs_GHz = config.get("chirp_coeffs_GHz") or None
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=EnvCls(amp=1.0, t_g=t_g), is_eta=True,
                          drag=(drag_beat_GHz is not None),
                          delta_drag_GHz=(drag_beat_GHz if drag_beat_GHz is not None else 0.0),
                          chirp=make_chirp(chirp_coeffs_GHz, t_g)),
                 normalize_iswap=(0, 1))
    cpl.scale_pump_amplitude(amp_scale)
    return cpl, w_p_GHz, cpl.peak_eta()


def transfer_probability(config: Dict[str, Any], t_g: float, amp_scale: float,
                         wp_offset_GHz: float, solver: Dict[str, Any],
                         spec_abs_GHz: Optional[float] = None,
                         drag_beat_GHz: Optional[float] = None) -> float:
    """Single-shot swap probability P(|01> -> |10>) at t_g (QuTiP sesolve): a fast
    one-trajectory proxy for the rotation angle, used as a search objective.

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
    spec_abs_GHz : float, optional
        Spectator absolute frequency (GHz); adds the spectator mode (ground) to the
        Hilbert space so the probe sees it. None -> bare (a, b) pair.
    drag_beat_GHz : float, optional
        DRAG beat (GHz) for the probe pump. None -> no DRAG.

    Returns
    -------
    float
        P(|10>) starting from |01>, with any spectator left in its ground state.
    """
    cpl, _w_p, _eta = build_coupler(config, t_g, amp_scale, wp_offset_GHz,
                                    spec_abs_GHz, drag_beat_GHz)
    tail = [0] if spec_abs_GHz is not None else []       # spectator stays in |0>
    state = cpl.evolve_state([1, 0, 0] + tail, t_g, **solver)
    return float(np.abs(state[cpl.fock_index([0, 1, 0] + tail)]) ** 2)


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
        Zoom-in rounds after the initial grid.

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