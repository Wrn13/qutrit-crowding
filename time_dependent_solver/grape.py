"""Optimal-control comparison for the Zhou SNAIL iSWAP.

Optimizes the complex pump envelope eta(t) to maximize the leakage-aware iSWAP
fidelity and compares it against the DRAG-shaped raised-cosine gate the sweeps
use. The pump enters the Hamiltonian NONLINEARLY (eta, eta^2, eta^3 from g3 X^3),
so control-linear GRAPE does not apply; two backends are provided:

* backend='qutip' (default): QuTiP's optimal-control package ``qutip-qoc`` with an
  analytic-control gradient method -- GOAT (propagator-gradient equations of
  motion) or JOPT (JAX auto-differentiation). Both differentiate the EXACT
  propagator w.r.t. a shared analytic-ansatz parameter vector through the
  nonlinearity, and QuTiP does the propagation. Needs ``qutip`` + ``qutip-qoc``
  (+ ``qutip-jax``/``jax`` for JOPT).
* backend='reduced': in-house scipy L-BFGS-B over the rotating-frame reduced model
  (scipy expm, no QuTiP). Fast; keeps the near-resonant band via ``cutoff_GHz``.

Both score the resulting 4x4 projected propagator with the SAME
``ZhouCoupler._iswap_fidelity_from_U`` the sweeps use, so F_baseline/F_grape are
directly comparable to the sweep's F_avg. The optimized pulse should still be
validated in the full ``iswap_fidelity`` sim before use.

CLI
---
    python grape.py --device warren_device.json --t-g-ns 92.6 \
        --backend qutip --alg GOAT [--warmstart-drag-beat-GHz ...]
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


def _iq_ansatz(t_g: float, peak: float, n_basis: int):
    """Analytic complex pump ansatz eta(t; p) for GOAT/JOPT.

    eta(t) = peak * env(t) * [ (1 + sum_k pI_k s_k(t)) + i * sum_k pQ_k s_k(t) ],
    env = raised cosine, s_k(t) = sin(k*pi*t/t_g) (zero-ended, so the pulse still
    turns on/off smoothly). p = [pI_1..K, pQ_1..K]; p = 0 -> plain raised cosine.
    Returned as a scalar-in-time function so it slots straight into a QobjEvo/qoc
    coefficient. Kept in plain numpy; JOPT wraps it through JAX automatically.
    """
    ks = np.arange(1, n_basis + 1)

    def eta(t: float, p) -> complex:
        p = np.asarray(p, dtype=float)
        env = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / t_g))
        s = np.sin(ks * np.pi * t / t_g)
        amp_I = 1.0 + float(np.dot(p[:n_basis], s))
        amp_Q = float(np.dot(p[n_basis:], s))
        return peak * env * (amp_I + 1j * amp_Q)

    return eta


def _drag_seed_params(t_g: float, peak: float, n_basis: int,
                      warmstart_beat_GHz: Optional[float]) -> np.ndarray:
    """Initial ansatz parameters. Zeros -> plain raised cosine; a DRAG warm start
    loads the k=2 quadrature (sin(2 pi t/t_g) ~ the raised-cosine derivative) with
    the first-order Motzoi weight for the given beat."""
    p0 = np.zeros(2 * n_basis)
    if warmstart_beat_GHz:                       # DRAG: Q ~ -eta'(t) / (2 pi beat)
        # env'(t) = (pi/t_g) sin(2 pi t/t_g) = (pi/t_g) s_2(t); Q amplitude coeff on s_2
        p0[n_basis + 1] = -(np.pi / t_g) / (2.0 * np.pi * float(warmstart_beat_GHz))
    return p0


def _optimize_qoc(cpl, a: int, b: int, t_g: float, *, n_basis: int, cutoff_GHz: float,
                  drag_beat_GHz: Optional[float], warmstart_beat_GHz: Optional[float],
                  maxiter: int, alg: str, n_time: int, verbose: bool) -> Dict[str, Any]:
    """Optimize the pump envelope with QuTiP's optimal-control package (qutip-qoc).

    The SNAIL gate is control-NONLINEAR (the pump enters as eta, eta^2, eta^3 via
    g3 X^3), so classic control-linear GRAPE does not apply. We use qutip-qoc's
    analytic-control gradient methods -- GOAT (propagator-gradient EOM) or JOPT
    (JAX autodiff) -- which differentiate the exact propagator w.r.t. the shared
    ansatz parameters through that nonlinearity.

    Construction:
      * operator basis: ``ZhouCoupler.expand_terms(cutoff_GHz)`` gives the rotating-
        frame terms O_j at carrier Omega_j with pump powers (n_pos, n_neg). Larger
        cutoff -> closer to the exact QobjEvo (cutoff=inf reproduces sesolve).
      * H = [drift, [O_j, c_j(t,p)] ...] where every pump-bearing term shares the
        SAME parameter vector p and c_j(t,p) = eta(t;p)^{n_pos} conj(eta)^{n_neg}
        e^{-i Omega_j t}; pump-free carrier terms + the static anharmonicity form
        the (time-dependent) drift.
      * target: iSWAP embedded on the (a,b) computational subspace, identity
        elsewhere (so leakage lowers the overlap).
      * qoc.optimize_pulses drives it to the target; we then reconstruct eta(t;p*)
        and RE-SCORE with the sweep's own reduced propagator + _iswap_fidelity_from_U
        so F_baseline/F_grape stay directly comparable to the rest of the pipeline,
        while the optimization itself ran on QuTiP.

    Requires ``qutip`` and ``qutip-qoc`` (and ``qutip-jax``/``jax`` for alg='JOPT').
    """
    import qutip as qt
    import qutip_qoc as qoc

    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    peak = float(cpl.peak_eta())
    dims = [list(cpl.dims), list(cpl.dims)]
    eta = _iq_ansatz(t_g, peak, n_basis)

    # split into (time-dependent) drift and pump-bearing controls
    drift_list: list = [qt.Qobj(np.asarray(H_anh), dims=dims)]
    control: list = []
    for Omega, n_pos, n_neg, O in terms:
        Oq = qt.Qobj(np.asarray(O), dims=dims)
        if n_pos + n_neg == 0:                   # pump-free -> drift (fixed carrier)
            if Omega == 0.0:
                drift_list.append(Oq)
            else:
                drift_list.append([Oq, (lambda Om: (lambda t, _a=None: np.exp(-1j * Om * t)))(Omega)])
            continue

        def make_coeff(Om=Omega, npv=n_pos, nnv=n_neg):
            def c(t, p):
                e = eta(t, p)
                val = (e ** npv) * (np.conj(e) ** nnv)
                return val * np.exp(-1j * Om * t) if Om != 0.0 else val
            return c
        control.append([Oq, make_coeff()])

    drift = qt.QobjEvo(drift_list) if len(drift_list) > 1 else drift_list[0]
    H = [drift] + control

    # embedded-iSWAP target on the (a,b) pair
    U = np.eye(cpl.dim, dtype=complex)
    i00, i01, i10, i11 = idx
    U[i01, i01] = U[i10, i10] = 0.0
    U[i10, i01] = U[i01, i10] = 1j
    target = qt.Qobj(U, dims=dims)
    initial = qt.qeye(cpl.dims)

    p_guess = _drag_seed_params(t_g, peak, n_basis, warmstart_beat_GHz)
    bound = 2.0                                  # |Fourier coeff| bound (dimensionless)
    result = qoc.optimize_pulses(
        objectives=[qoc.Objective(initial, H, target)],
        control_parameters={"p": {"guess": list(p_guess),
                                  "bounds": [(-bound, bound)] * (2 * n_basis)}},
        tlist=np.linspace(0.0, t_g, n_time),
        algorithm_kwargs={"alg": alg, "fid_err_targ": 1e-4, "max_iter": int(maxiter),
                          "disp": verbose})

    # extract optimized parameters (attribute name varies by qutip-qoc version)
    p_star = None
    for attr in ("optimized_params", "optimized_control_parameters", "final_params",
                 "optimized_controls"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            p_star = np.ravel(val[0] if isinstance(val, (list, tuple)) else val)
            break
    if p_star is None or len(p_star) != 2 * n_basis:
        raise RuntimeError("could not read optimized params off the qutip-qoc result; "
                           "check the result attribute name for your qutip-qoc version")
    qoc_fid_err = float(getattr(result, "fid_err", getattr(result, "infidelity", np.nan)))

    # RE-SCORE optimized and baseline pulses on the sweep's reduced propagator so the
    # reported F's line up with the rest of the pipeline (the OPTIMIZER used QuTiP).
    dt_ctrl = t_g / max(int(n_time), 1)
    n_sub = max(1, int(np.ceil(max_Omega * dt_ctrl / 0.3)))
    ts = (np.arange(n_time) + 0.5) * (t_g / n_time)
    eta_opt = np.array([eta(t, p_star) for t in ts], dtype=complex)
    eta0 = _raised_cosine_eta(t_g, peak, n_time, drag_beat_GHz)
    Fg, leakg = _score(_propagate(eta_opt, t_g, terms, H_anh, idx, n_sub), cpl)
    F0, leak0 = _score(_propagate(eta0, t_g, terms, H_anh, idx, n_sub), cpl)

    return dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
                eta_opt=eta_opt, F_grape=Fg, leak_grape=leakg,
                n_ctrl=n_time, n_sub=n_sub, cutoff_GHz=cutoff_GHz,
                nfev=int(getattr(result, "iters", getattr(result, "n_iters", -1))),
                warmstart_beat_GHz=(np.nan if warmstart_beat_GHz is None
                                    else float(warmstart_beat_GHz)),
                backend="qutip", alg=alg, n_basis=n_basis, qoc_fid_err=qoc_fid_err)


def optimize_pulse(cpl, a: int, b: int, t_g: float, *, n_ctrl: int = 24,
                   cutoff_GHz: float = 1.0, drag_beat_GHz: Optional[float] = None,
                   warmstart_beat_GHz: Optional[float] = None,
                   backend: str = "qutip", alg: str = "GOAT", n_basis: int = 6,
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
    backend : {'qutip', 'reduced'}
        'qutip' (default) optimizes with QuTiP's optimal-control package
        (``qutip-qoc``) using an analytic-control gradient method (``alg``), which
        handles the eta/eta^2/eta^3 control-nonlinearity of the SNAIL gate; the
        exact propagator is QuTiP's. 'reduced' is the in-house scipy L-BFGS-B
        optimizer over the rotating-frame reduced model (fast, no QuTiP).
    alg : {'GOAT', 'JOPT'}
        qutip-qoc analytic-control algorithm ('qutip' backend). JOPT needs JAX.
    n_basis : int
        Number of sin() basis functions per quadrature for the analytic ansatz
        ('qutip' backend). n_ctrl is the piecewise-constant control count for the
        'reduced' backend and the re-scoring/tlist resolution for 'qutip'.
    drag_beat_GHz : float or None
        Beat for the DRAG BASELINE quadrature (None -> plain raised cosine). Sets
        F_baseline / the gate the improvement is measured against.
    warmstart_beat_GHz : float or None
        Seed the optimizer from a DRAG raised cosine at THIS beat (both backends);
        baseline/dF unchanged, only the start moves. None -> start from baseline.
    maxiter : int
        Optimizer iteration cap.
    carrier_resolution : float
        Max Omega*dt_fine (rad) -- sets fine sub-steps per control slice.

    Returns
    -------
    dict
        eta_baseline, F_baseline, leak_baseline, eta_opt, F_grape, leak_grape,
        n_ctrl, n_sub, cutoff_GHz, nfev, warmstart_beat_GHz (+ backend/alg and,
        for 'qutip', qoc_fid_err). F_baseline/F_grape use the reduced scorer for
        both backends so they are directly comparable across the sweep.
    """
    if backend == "qutip":
        return _optimize_qoc(cpl, a, b, t_g, n_basis=n_basis, cutoff_GHz=cutoff_GHz,
                             drag_beat_GHz=drag_beat_GHz,
                             warmstart_beat_GHz=warmstart_beat_GHz,
                             maxiter=maxiter, alg=alg, n_time=n_ctrl, verbose=verbose)
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

    # optimizer seed: the baseline pulse, or a DRAG raised cosine at a supplied beat
    eta_seed = (eta0 if warmstart_beat_GHz is None
                else _raised_cosine_eta(t_g, peak, n_ctrl, warmstart_beat_GHz))
    x0 = np.concatenate([eta_seed.real, eta_seed.imag])
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
                n_ctrl=n_ctrl, n_sub=n_sub, cutoff_GHz=cutoff_GHz, nfev=res.nfev,
                warmstart_beat_GHz=(np.nan if warmstart_beat_GHz is None
                                    else float(warmstart_beat_GHz)),
                backend="reduced")


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
    ap.add_argument("--warmstart-drag-beat-GHz", type=float, default=None,
                    help="seed the optimizer from a DRAG raised cosine at this beat "
                         "(baseline/dF unchanged); good start for a hard collision")
    ap.add_argument("--n-ctrl", type=int, default=24)
    ap.add_argument("--cutoff-GHz", type=float, default=1.0)
    ap.add_argument("--backend", choices=["qutip", "reduced"], default="qutip",
                    help="qutip = qutip-qoc optimal control (handles the control "
                         "nonlinearity; needs qutip-qoc); reduced = in-house scipy")
    ap.add_argument("--alg", choices=["GOAT", "JOPT"], default="GOAT",
                    help="qutip-qoc analytic-control algorithm (JOPT needs JAX)")
    ap.add_argument("--n-basis", type=int, default=6,
                    help="sin() basis functions per quadrature (qutip backend)")
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args()

    from paths import resolve_device
    from device_utils import load_device
    cfg = load_device(resolve_device(args.device))
    out = compare(cfg, args.t_g_ns, amp_scale=args.amp_scale,
                  wp_offset_GHz=args.wp_offset_GHz, spec_abs_GHz=args.spec_abs_GHz,
                  drag_beat_GHz=args.drag_beat_GHz, n_ctrl=args.n_ctrl,
                  cutoff_GHz=args.cutoff_GHz, maxiter=args.maxiter,
                  warmstart_beat_GHz=args.warmstart_drag_beat_GHz,
                  backend=args.backend, alg=args.alg, n_basis=args.n_basis)
    tag = (f"qutip-qoc/{out.get('alg','GOAT')} (fid_err={out.get('qoc_fid_err', float('nan')):.2e})"
           if out.get("backend") == "qutip"
           else f"reduced (cutoff={out['cutoff_GHz']} GHz)")
    print(f"{tag}, {out['n_ctrl']} pts, {out['nfev']} iters:")
    print(f"  DRAG raised-cosine : F = {out['F_baseline']:.5f}  leak = {out['leak_baseline']:.4f}")
    print(f"  GRAPE optimized    : F = {out['F_grape']:.5f}  leak = {out['leak_grape']:.4f}")
    print(f"  improvement dF = {out['F_grape'] - out['F_baseline']:+.5f}")
    print("  (validate the GRAPE pulse in the full QuTiP iswap_fidelity on the cluster)")


if __name__ == "__main__":
    main()