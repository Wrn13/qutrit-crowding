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
from typing import Any, Callable, Dict, Tuple

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
                  wp_offset_GHz: float):
    """Build the spectator-free (qubit a, qubit b, coupler) gate with the pump
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

    Returns
    -------
    (ZhouCoupler, float, float)
        The coupler, its pump frequency w_p (GHz), and the resulting peak |eta|.
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

    Returns
    -------
    float
        P(|10>) starting from |01>.
    """
    cpl, _w_p, _eta = build_coupler(config, t_g, amp_scale, wp_offset_GHz)
    state = cpl.evolve_state([1, 0, 0], t_g, **solver)
    return float(np.abs(state[cpl.fock_index([0, 1, 0])]) ** 2)


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