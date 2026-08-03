"""GRAPE optimal-control comparison for the Zhou SNAIL iSWAP.

Optimizes the complex pump envelope eta(t) -- piecewise-constant I/Q control
points -- to maximize the leakage-aware iSWAP fidelity, and compares it against
the DRAG-shaped raised-cosine gate the sweeps use.

Propagation is done with scipy matrix exponentials in the coupler's rotating
frame: the interaction-picture operator basis from ``ZhouCoupler.expand_terms``
(which already folds in g_n, participations, and the pump-order structure) plus
the static transmon anharmonicity ``_anharm_op``. The pump enters those terms
NONLINEARLY (eta, eta^2, eta^3 from g3 X^3), so this is a genuine nonlinear-
control GRAPE, not a linear-control one. The resulting 4x4 projected propagator
is scored with the SAME ``ZhouCoupler._iswap_fidelity_from_U`` used by the QuTiP
sweep, so F is directly comparable to the sweep numbers.

QuTiP is NOT required here. A finite ``cutoff_GHz`` keeps the near-resonant band
(beam-splitter + Stark + leakage + the collision under study) and drops only the
fast carriers that average away; the optimized pulse should still be validated in
the full ``iswap_fidelity`` sim on the cluster.

CLI
---
    python grape.py --device warren_device.json --t-g-ns 92.6 [--drag-beat-GHz ...]
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

TWO_PI = 2.0 * np.pi


def _prepare(cpl, a: int, b: int, cutoff_GHz: float):
    """Precompute the rotating-frame terms, anharmonicity, subspace, max carrier.

    Returns
    -------
    terms : list of (Omega, n_pos, n_neg, O)
        Omega (rad/ns), pump exponents (eta^n_pos * conj(eta)^n_neg), operator.
    H_anh : ndarray
        Static transmon-anharmonicity operator (diagonal).
    idx : list of int
        The 4 computational-subspace fock indices (|00>,|01>,|10>,|11>).
    max_Omega : float
        Largest |Omega| kept (sets the fine-propagation step).
    """
    terms: List[Tuple[float, int, int, np.ndarray]] = []
    for Omega, pump_sig, O in cpl.expand_terms(cutoff_GHz=cutoff_GHz):
        n_pos = sum(1 for (_ti, conj) in pump_sig if not conj)
        n_neg = sum(1 for (_ti, conj) in pump_sig if conj)
        terms.append((float(Omega), n_pos, n_neg, np.asarray(O, dtype=complex)))
    H_anh = np.asarray(cpl._anharm_op, dtype=complex)
    idx = list(cpl._subspace_indices(a, b))
    max_Omega = max((abs(t[0]) for t in terms), default=0.0)
    return terms, H_anh, idx, max_Omega


def _H(t: float, eta: complex, terms, H_anh: np.ndarray,
       offset_rad: float = 0.0) -> np.ndarray:
    """Interaction-picture Hamiltonian at time t for pump amplitude eta.

    A pump-frequency offset shifts each term's carrier by (n_pos - n_neg) * offset
    (the net number of pump quanta), so the same precomputed operator basis serves
    any offset -- no re-expansion needed for a calibration scan.
    """
    H = H_anh.copy()
    for Omega, n_pos, n_neg, O in terms:
        Om = Omega + (n_pos - n_neg) * offset_rad
        f = (eta ** n_pos) * (np.conj(eta) ** n_neg)
        if Om != 0.0:
            f = f * np.exp(-1j * Om * t)
        H = H + f * O
    return H


def _propagate(eta_ctrl: np.ndarray, t_g: float, terms, H_anh, idx,
               n_sub: int, offset_rad: float = 0.0) -> np.ndarray:
    """Propagate the 4 computational states; return the 4x4 projected propagator.

    Parameters
    ----------
    eta_ctrl : ndarray (complex)
        N piecewise-constant control amplitudes.
    t_g : float
        Gate duration (ns).
    terms, H_anh, idx : see _prepare
    n_sub : int
        Fine sub-steps per control slice (resolves the kept carriers).
    offset_rad : float
        Pump-frequency offset (rad/ns) applied to the carriers.
    """
    N = len(eta_ctrl)
    dt_ctrl = t_g / N
    dt_fine = dt_ctrl / n_sub
    dim = H_anh.shape[0]
    Psi = np.zeros((dim, 4), dtype=complex)
    for col, s in enumerate(idx):
        Psi[s, col] = 1.0
    for j in range(N):
        eta = eta_ctrl[j]
        t0 = j * dt_ctrl
        for m in range(n_sub):
            t = t0 + (m + 0.5) * dt_fine
            Ustep = expm(-1j * _H(t, eta, terms, H_anh, offset_rad) * dt_fine)
            Psi = Ustep @ Psi
    return Psi[np.ix_(idx, range(4))]


def _score(U: np.ndarray, cpl) -> Tuple[float, float]:
    """Leakage-aware iSWAP (F, leakage) via the coupler's own scorer."""
    from zhou_coupler import ZhouCoupler
    return ZhouCoupler._iswap_fidelity_from_U(U, True)


def _raised_cosine_eta(t_g: float, peak: float, n_ctrl: int,
                       drag_beat_GHz: Optional[float]) -> np.ndarray:
    """DRAG-shaped raised-cosine envelope sampled at control-slice midpoints."""
    ts = (np.arange(n_ctrl) + 0.5) * (t_g / n_ctrl)
    rc = 0.5 * (1.0 - np.cos(2.0 * np.pi * ts / t_g))
    eta = peak * rc.astype(complex)
    if drag_beat_GHz:                       # add the first-order DRAG quadrature
        drc = (np.pi / t_g) * np.sin(2.0 * np.pi * ts / t_g)
        eta = eta - 1j * peak * drc / (2.0 * np.pi * drag_beat_GHz)
    return eta


def optimize_pulse(cpl, a: int, b: int, t_g: float, *, n_ctrl: int = 24,
                   cutoff_GHz: float = 1.0, drag_beat_GHz: Optional[float] = None,
                   maxiter: int = 200, carrier_resolution: float = 0.3,
                   verbose: bool = False) -> Dict[str, Any]:
    """Optimize the pump envelope and compare to the DRAG raised-cosine baseline.

    Parameters
    ----------
    cpl : ZhouCoupler
        Coupler with a pump already set (its peak_eta sets the amplitude scale).
    a, b : int
        Target-qubit mode indices.
    t_g : float
        Gate duration (ns).
    n_ctrl : int
        Number of piecewise-constant control points.
    cutoff_GHz : float
        Rotating-frame carrier cutoff for the reduced propagation model.
    drag_beat_GHz : float or None
        Beat for the DRAG baseline quadrature (None -> plain raised cosine).
    maxiter : int
        L-BFGS-B iteration cap.
    carrier_resolution : float
        Max Omega*dt_fine (rad) -- sets fine sub-steps per control slice.

    Returns
    -------
    dict
        eta_baseline, F_baseline, leak_baseline, eta_opt, F_grape, leak_grape,
        n_ctrl, n_sub, cutoff_GHz, nfev.
    """
    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    dt_ctrl = t_g / n_ctrl
    n_sub = max(1, int(np.ceil(max_Omega * dt_ctrl / carrier_resolution)))

    peak = float(cpl.peak_eta())
    eta0 = _raised_cosine_eta(t_g, peak, n_ctrl, drag_beat_GHz)
    U0 = _propagate(eta0, t_g, terms, H_anh, idx, n_sub)
    F0, leak0 = _score(U0, cpl)

    def infid(x: np.ndarray) -> float:
        eta = x[:n_ctrl] + 1j * x[n_ctrl:]
        U = _propagate(eta, t_g, terms, H_anh, idx, n_sub)
        F, _ = _score(U, cpl)
        return 1.0 - F

    x0 = np.concatenate([eta0.real, eta0.imag])
    # keep the optimizer from running away to non-physical amplitudes
    bound = 2.0 * (abs(peak) + 1e-6)
    res = minimize(infid, x0, method="L-BFGS-B",
                   bounds=[(-bound, bound)] * (2 * n_ctrl),
                   options=dict(maxiter=maxiter, ftol=1e-9, disp=verbose))
    eta_opt = res.x[:n_ctrl] + 1j * res.x[n_ctrl:]
    Uo = _propagate(eta_opt, t_g, terms, H_anh, idx, n_sub)
    Fg, leakg = _score(Uo, cpl)

    return dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
                eta_opt=eta_opt, F_grape=Fg, leak_grape=leakg,
                n_ctrl=n_ctrl, n_sub=n_sub, cutoff_GHz=cutoff_GHz, nfev=res.nfev)


def compare(config: Dict[str, Any], t_g: float, *, amp_scale: float = 1.0,
            wp_offset_GHz: float = 0.0, spec_abs_GHz: Optional[float] = None,
            **kw) -> Dict[str, Any]:
    """Build the coupler from a config and run optimize_pulse (a vs b = 0, 1)."""
    from device_utils import build_coupler
    cpl, w_p, eta_pk = build_coupler(config, t_g=t_g, amp_scale=amp_scale,
                                     wp_offset_GHz=wp_offset_GHz,
                                     spec_abs_GHz=spec_abs_GHz)
    out = optimize_pulse(cpl, 0, 1, t_g, **kw)
    out["w_p_GHz"] = w_p
    out["peak_eta"] = eta_pk
    return out


def main() -> None:
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--t-g-ns", type=float, required=True)
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--wp-offset-GHz", type=float, default=0.0)
    ap.add_argument("--spec-abs-GHz", type=float, default=None)
    ap.add_argument("--drag-beat-GHz", type=float, default=None)
    ap.add_argument("--n-ctrl", type=int, default=24)
    ap.add_argument("--cutoff-GHz", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args()

    from paths import resolve_device
    from device_utils import load_device
    cfg = load_device(resolve_device(args.device))
    out = compare(cfg, args.t_g_ns, amp_scale=args.amp_scale,
                  wp_offset_GHz=args.wp_offset_GHz, spec_abs_GHz=args.spec_abs_GHz,
                  drag_beat_GHz=args.drag_beat_GHz, n_ctrl=args.n_ctrl,
                  cutoff_GHz=args.cutoff_GHz, maxiter=args.maxiter)
    print(f"reduced model (cutoff={out['cutoff_GHz']} GHz, {out['n_ctrl']} ctrl pts, "
          f"{out['n_sub']} sub-steps, {out['nfev']} evals):")
    print(f"  DRAG raised-cosine : F = {out['F_baseline']:.5f}  leak = {out['leak_baseline']:.4f}")
    print(f"  GRAPE optimized    : F = {out['F_grape']:.5f}  leak = {out['leak_grape']:.4f}")
    print(f"  improvement dF = {out['F_grape'] - out['F_baseline']:+.5f}")
    print("  (validate the GRAPE pulse in the full QuTiP iswap_fidelity on the cluster)")


if __name__ == "__main__":
    main()