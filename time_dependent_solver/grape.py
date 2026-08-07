"""Optimal-control comparison for the Zhou SNAIL iSWAP.

Optimizes the complex pump envelope eta(t) to maximize the leakage-aware iSWAP
fidelity and compares it against the DRAG-shaped raised-cosine gate the sweeps
use. The pump enters the Hamiltonian NONLINEARLY (eta, eta^2, eta^3 from g3 X^3),
so control-linear GRAPE does not apply; two backends are provided:

* backend='qutip', alg='JOPT': JAX-autodiff GRADIENT method over the analytic I/Q
  ansatz (``_iq_ansatz``), differentiating the reduced-model propagator exactly
  with ``jax.grad`` and driving scipy L-BFGS-B. Needs only ``jax``.
  NOT qutip-qoc's JOPT: that package scores the FULL-dimension propagator with no
  virtual-Z freedom (its TRACEDIFF/PSU/SU objectives), whereas this pipeline
  reports the 4x4 projection with single-qubit Z fitted out -- and on this gate the
  Z fit is worth +0.76 of a fidelity of 0.95, so a phase-rigid objective spends
  real fidelity chasing free phases and walks DOWNHILL from its own start. See
  ``_jax_pipeline_infidelity``. Correct, but currently much slower per point than
  CRAB (a ~600-step expm chain per objective), so it is opt-in, not the default.
  Because the gradient comes from tracing, the ansatz and every pump-bearing
  coefficient must be built with ``jax.numpy`` and stay free of
  ``float()``/``np.asarray(..., dtype=...)`` on parameter-dependent values --
  either one forces a tracer to a concrete value and raises. ``_iq_ansatz`` takes
  the array module as ``xp`` for exactly this reason.
* backend='qutip', alg='CRAB': Chopped RAndom Basis -- expand the pulse in a
  truncated RANDOMIZED Fourier basis over a fixed shape function and minimize the
  infidelity with a gradient-FREE optimizer (Nelder-Mead, as in qutip-qtrl).
  CRAB treats the evolution as a black box (pulse in, infidelity out), so it needs
  no propagator gradients -- and therefore no control-linear H = H_d + sum u_k H_k
  structure, which this gate does not have. The black box here is
  ``ZhouCoupler.propagator_columns`` (exact compiled ``qt.sesolve``) evaluated on
  an ``envelope.IQFourierEnvelope``, so only ``qutip`` is required, NOT
  ``qutip-qoc``. Robust where a gradient method stalls, at the cost of many exact
  propagations. ``crab_restarts>1`` gives DCRAB-style monotone super-iterations.
* backend='reduced': in-house scipy L-BFGS-B over the rotating-frame reduced model
  (scipy expm, no QuTiP). Fast; keeps the near-resonant band via ``cutoff_GHz``.

Both score the resulting 4x4 projected propagator with the SAME
``ZhouCoupler._iswap_fidelity_from_U`` the sweeps use, so F_baseline/F_grape are
directly comparable to the sweep's F_avg. The optimized pulse should still be
validated in the full ``iswap_fidelity`` sim before use.

CLI
---
    python grape.py --device warren_device.json --t-g-ns 92.6 \
        --backend qutip --alg JOPT [--warmstart-drag-beat-GHz ...]
    python grape.py --device warren_device.json --t-g-ns 92.6 \
        --backend qutip --alg CRAB [--warmstart-drag-beat-GHz ...]
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


def _H(t: float, eta: complex,

 terms, H_anh: np.ndarray,
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


def _iq_ansatz(t_g: float, peak: float, n_basis: int, xp=np):
    """Analytic complex pump ansatz eta(t; p) for the qutip-qoc gradient method.

    eta(t) = peak * env(t) * [ (1 + sum_k pI_k s_k(t)) + i * sum_k pQ_k s_k(t) ],
    env = raised cosine, s_k(t) = sin(k*pi*t/t_g) (zero-ended, so the pulse still
    turns on/off smoothly). p = [pI_1..K, pQ_1..K]; p = 0 -> plain raised cosine.
    Returned as a scalar-in-time function so it slots straight into a QobjEvo/qoc
    coefficient.

    ``xp`` is the array module the body is built from: ``numpy`` for a plain
    evaluation, ``jax.numpy`` under alg='JOPT'. JOPT obtains its gradient by
    TRACING this function in ``p``, so under JAX the body has to stay trace-clean:
    no ``asarray(p, dtype=float)`` and no ``float(...)`` of a p-dependent value,
    since either forces a tracer to a concrete value and raises
    (TracerArrayConversionError / ConcretizationTypeError).
    """
    ks = xp.arange(1, n_basis + 1)

    def eta(t, p):
        p = xp.asarray(p)                     # no dtype= : must not concretize a tracer
        env = 0.5 * (1.0 - xp.cos(2.0 * xp.pi * t / t_g))
        s = xp.sin(ks * xp.pi * t / t_g)
        amp_I = 1.0 + xp.dot(p[:n_basis], s)
        amp_Q = xp.dot(p[n_basis:], s)
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
    JOPT analytic-control gradient method, which differentiates the exact
    propagator w.r.t. the shared ansatz parameters through that nonlinearity by
    JAX auto-differentiation.

    Everything the gradient flows through -- the ansatz and every pump-bearing
    coefficient -- is therefore built from ``jax.numpy`` (``xp`` below) rather than
    numpy: JOPT traces these callables in ``p``, and a numpy ufunc applied to a
    tracer raises instead of differentiating. x64 is enabled because the default
    JAX float32/complex64 precision is far coarser than the 1e-4 fidelity target.

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

    Requires ``qutip``, ``qutip-qoc``, ``qutip-jax`` and ``jax``.
    """
    import qutip as qt

    alg = str(alg).upper()
    if alg != "JOPT":
        raise ValueError(f"_optimize_qoc: unsupported alg {alg!r}; the qutip-qoc "
                         f"gradient path is 'JOPT' (use alg='CRAB' for the "
                         f"gradient-free optimizer, which needs only qutip)")
    try:
        import qutip_qoc as qoc
        import jax
        import jax.numpy as xp
        import qutip_jax                      # noqa: F401  registers the jax data layer
    except ImportError as exc:
        raise ImportError(
            f"alg='JOPT' needs qutip-qoc, qutip-jax and jax, but importing them "
            f"failed ({exc}). Install them, or use alg='CRAB' -- the gradient-free "
            f"optimizer, which requires only qutip.") from exc
    # the fidelity target is 1e-4; JAX's default single precision cannot resolve it
    jax.config.update("jax_enable_x64", True)

    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    peak = float(cpl.peak_eta())
    dims = [list(cpl.dims), list(cpl.dims)]
    eta = _iq_ansatz(t_g, peak, n_basis, xp=xp)

    # split into (time-dependent) drift and pump-bearing controls
    drift_list: list = [qt.Qobj(np.asarray(H_anh), dims=dims)]
    control: list = []
    for Omega, n_pos, n_neg, O in terms:
        Oq = qt.Qobj(np.asarray(O), dims=dims)
        if n_pos + n_neg == 0:                   # pump-free -> drift (fixed carrier)
            if Omega == 0.0:
                drift_list.append(Oq)
            else:
                drift_list.append([Oq, (lambda Om: (lambda t, _a=None: xp.exp(-1j * Om * t)))(Omega)])
            continue

        # carries the gradient: JOPT traces c() in p, so it must be built from xp
        # (jax.numpy) end to end -- a numpy conj/exp applied to a tracer raises.
        def make_coeff(Om=Omega, npv=n_pos, nnv=n_neg):
            def c(t, p):
                e = eta(t, p)
                val = (e ** npv) * (xp.conj(e) ** nnv)
                return val * xp.exp(-1j * Om * t) if Om != 0.0 else val
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
    # Per-coefficient bound, chosen so the PULSE stays physical rather than the
    # coefficient merely looking small. amp_I = 1 + sum_k p_k s_k with |s_k| <= 1,
    # so |eta| <= peak * (1 + n_basis * bound): a fixed bound of 2.0 admitted
    # (1 + 2*n_basis)x the calibrated peak, and the optimizer duly pinned every
    # coefficient to it and returned a wildly over-driven, all-leakage pulse.
    # bound = 1/n_basis caps the worst case at 2x peak, matching the 'reduced'
    # backend's 2.0*peak bound in physical eta units.
    bound = 1.0 / float(n_basis)
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
    # Rebuild the ansatz in numpy: `eta` above is JAX-backed, and the reduced
    # propagator/scoring path downstream is plain numpy/scipy.
    eta_np = _iq_ansatz(t_g, peak, n_basis, xp=np)
    dt_ctrl = t_g / max(int(n_time), 1)
    n_sub = max(1, int(np.ceil(max_Omega * dt_ctrl / 0.3)))
    ts = (np.arange(n_time) + 0.5) * (t_g / n_time)
    eta_opt = np.array([eta_np(t, p_star) for t in ts], dtype=complex)
    eta0 = _raised_cosine_eta(t_g, peak, n_time, drag_beat_GHz)
    Fg, leakg = _score(_propagate(eta_opt, t_g, terms, H_anh, idx, n_sub), cpl)
    F0, leak0 = _score(_propagate(eta0, t_g, terms, H_anh, idx, n_sub), cpl)

    return dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
                eta_opt=eta_opt, F_grape=Fg, leak_grape=leakg,
                n_ctrl=n_time, n_sub=n_sub, cutoff_GHz=cutoff_GHz,
                nfev=int(getattr(result, "num_iter",
                                 getattr(result, "iters",
                                         getattr(result, "n_iters", -1)))),
                warmstart_beat_GHz=(np.nan if warmstart_beat_GHz is None
                                    else float(warmstart_beat_GHz)),
                backend="qutip", alg=alg, n_basis=n_basis, qoc_fid_err=qoc_fid_err)


def _jax_pipeline_infidelity(terms, H_anh, idx, t_g: float, n_sub: int,
                             n_ctrl: int, eta_max: float, z_grid: int = 1024):
    """Build a JAX-traceable ``eta_ctrl -> 1 - F`` on the SWEEP's own metric.

    This exists because qutip-qoc cannot express the metric the rest of the
    pipeline reports. Its objectives (TRACEDIFF / PSU / SU) all score the
    FULL-dimension propagator rigidly in phase, whereas the sweep scores the 4x4
    projection with single-qubit virtual-Z phases fitted out -- and those phases
    are not a detail here: on the raised-cosine baseline the Z fit is worth +0.76
    of a fidelity of 0.95. Optimizing a phase-rigid objective therefore spends
    real fidelity chasing phases that are free in software, which is exactly why
    the qutip-qoc run came back BELOW its own starting point.

    So the objective is rebuilt here in JAX, term for term the same computation as
    ``_propagate`` + ``ZhouCoupler._iswap_fidelity_from_U(U, True)``:

      * propagate the 4 computational columns through the same piecewise-constant
        slices with ``jax.scipy.linalg.expm`` (differentiable),
      * project onto the 4x4 computational block,
      * fit the virtual Z, by the same 1-D reduction ``_fit_virtual_z`` uses (see
        that docstring for the derivation): with c_k = (U U_ideal^dag)_kk the
        overlap is A(pa) + e^{i pb} B(pa), so the maximum over pb is |A| + |B|
        ANALYTICALLY and only pa is swept over ``z_grid`` points. Vectorized, and
        differentiable through ``jnp.max`` (the gradient flows through the winning
        grid point, as in max-pooling). The numpy version additionally ternary-
        searches within the winning cell, so this stays marginally pessimistic --
        the safe side, and now by ~(2 pi / z_grid)^2 in the merit rather than by a
        two-axis grid error.
      * Pedersen leakage-aware fidelity (overlap + Tr[U^dag U]) / 20.

    Because the optimizer and the reporter now evaluate the SAME function, the
    optimized F is directly comparable to F_baseline and to the sweep's F_avg --
    no re-scoring disagreement is possible by construction.

    Parameters
    ----------
    eta_max : float
        Largest |eta| the caller can present. Only used to bound ||H|| when
        choosing the propagator's squaring count (see below); pass the optimizer's
        own amplitude ceiling, not the nominal peak.

    Returns
    -------
    callable
        ``infid(eta_ctrl)`` with ``eta_ctrl`` a length-``n_ctrl`` complex vector.
    """
    import jax
    import jax.numpy as jnp

    dt_ctrl = t_g / n_ctrl
    dt_fine = dt_ctrl / n_sub
    dim = H_anh.shape[0]

    # Scaling-and-squaring with the squaring count fixed at BUILD time rather than
    # read off ||A|| at runtime. jax.scipy.linalg.expm does the latter, i.e. with
    # data-dependent control flow, which cannot compile once inside lax.scan and
    # made the reverse-mode compile of a ~600-step chain never finish.
    #
    # The count must still come from a real bound. It is NOT enough to lean on
    # carrier_resolution: that caps max_Omega*dt_fine at ~0.3, but ||H|| also has
    # H_anh and the operator norms in it, which it does not bound at all -- with a
    # small n_sub (large dt_fine) a hardcoded 3 squarings silently returns a
    # diverged Taylor series (observed: 0.798 vs a true 0.222). So bound
    # ||H|| <= ||H_anh|| + sum_j ||O_j|| * eta_max^(n_pos+n_neg) in numpy here,
    # where eta_max caps the pump the optimizer may request, and pick the squarings
    # so the scaled argument is <= 1/2. Order 12 at 1/2 truncates at ~2e-14.
    _op_norm = float(np.linalg.norm(np.asarray(H_anh), 2))
    for _Om, _np_, _nn, _O in terms:
        _op_norm += float(np.linalg.norm(np.asarray(_O), 2)) * eta_max ** (_np_ + _nn)
    _SQ = int(max(0, np.ceil(np.log2(max(_op_norm * dt_fine, 1e-12) / 0.5))))
    _ORDER = 12

    def expm(A):
        As = A / (2 ** _SQ)
        eye = jnp.eye(As.shape[-1], dtype=As.dtype)
        term, out = eye, eye
        for k in range(1, _ORDER + 1):
            term = term @ As / k
            out = out + term
        for _ in range(_SQ):
            out = out @ out
        return out

    # static (traced-constant) pieces, promoted to JAX arrays once
    H_anh_j = jnp.asarray(np.asarray(H_anh), dtype=jnp.complex128)
    Om_j = jnp.asarray([Om for Om, _, _, _ in terms], dtype=jnp.float64)
    npos = jnp.asarray([p for _, p, _, _ in terms], dtype=jnp.float64)
    nneg = jnp.asarray([n for _, _, n, _ in terms], dtype=jnp.float64)
    Ops = jnp.asarray(np.array([np.asarray(O) for _, _, _, O in terms]),
                      dtype=jnp.complex128)                      # (n_terms, dim, dim)

    Psi0 = jnp.asarray(np.eye(dim, dtype=complex)[:, list(idx)])  # (dim, 4)
    U_ideal = jnp.asarray(_ideal_iswap_np(), dtype=jnp.complex128)
    # 1-D phase sweep: phi_b is handled analytically (see below), so only phi_a is
    # gridded. Same z_grid budget buys a far finer sweep than the old 2-D version.
    phases = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, z_grid, endpoint=False))

    def H_at(t, eta):
        f = (eta ** npos) * (jnp.conj(eta) ** nneg) * jnp.exp(-1j * Om_j * t)
        return H_anh_j + jnp.tensordot(f.astype(jnp.complex128), Ops, axes=(0, 0))

    # Fine-step schedule, flattened: step k sits at time t_all[k] and uses the
    # control slice eta_ctrl[slice_of[k]]. Driving this with lax.scan rather than a
    # Python loop matters a lot -- there are n_ctrl*n_sub (here ~600) expm calls per
    # objective, and unrolling them builds a trace so large that compiling the
    # reverse-mode gradient effectively never finishes. scan compiles ONE step.
    n_steps = n_ctrl * n_sub
    t_all = jnp.asarray(((np.arange(n_steps) % n_sub) + 0.5) * dt_fine
                        + (np.arange(n_steps) // n_sub) * dt_ctrl)
    slice_of = jnp.asarray(np.repeat(np.arange(n_ctrl), n_sub))

    def infid(eta_ctrl):
        eta_steps = eta_ctrl[slice_of]                            # (n_steps,)

        def step(Psi, xs):
            t, eta = xs
            return expm(-1j * H_at(t, eta) * dt_fine) @ Psi, None

        Psi, _ = jax.lax.scan(step, Psi0, (t_all, eta_steps))
        U = Psi[jnp.asarray(list(idx)), :]                        # (4, 4)

        # virtual-Z fit, matching zhou_coupler._fit_virtual_z: with
        # c_k = (U U_ideal^dag)_kk the overlap is
        #   sum_k z_k c_k = (c0 + e^{i pa} c2) + e^{i pb} (c1 + e^{i pa} c3) = A + e^{i pb} B,
        # so max over pb is |A| + |B| ANALYTICALLY (rotate B onto A). Only pa is
        # swept, which removes the pb grid error entirely and leaves the max over a
        # 1-D grid -- still a subgradient through argmax, but on one axis, not two.
        c = jnp.diag(U @ jnp.conj(U_ideal).T)                     # c_k, length 4
        e = jnp.exp(1j * phases)
        merit = jnp.abs(c[0] + e * c[2]) + jnp.abs(c[1] + e * c[3])
        overlap = jnp.max(merit) ** 2                             # virtual-Z fitted
        trace_UU = jnp.real(jnp.trace(jnp.conj(U).T @ U))
        return 1.0 - (overlap + trace_UU) / 20.0                  # d*(d+1), d = 4

    return infid


def _ideal_iswap_np() -> np.ndarray:
    """Local copy of the 4x4 target (avoids importing zhou_coupler at module import)."""
    U = np.eye(4, dtype=complex)
    U[1, 1] = U[2, 2] = 0.0
    U[1, 2] = U[2, 1] = 1j
    return U


def _optimize_jax(cpl, a: int, b: int, t_g: float, *, n_basis: int, cutoff_GHz: float,
                  drag_beat_GHz: Optional[float], warmstart_beat_GHz: Optional[float],
                  maxiter: int, n_time: int, verbose: bool) -> Dict[str, Any]:
    """JAX-autodiff gradient optimizer over the pipeline's OWN fidelity metric.

    Same analytic I/Q ansatz as before (``_iq_ansatz``), but the objective is
    ``_jax_pipeline_infidelity`` -- the leakage-aware, virtual-Z-fitted iSWAP
    fidelity the sweeps report -- differentiated exactly by ``jax.grad`` and
    handed to scipy L-BFGS-B with ``jac=True``.

    This replaces the qutip-qoc route for the gradient method. qutip-qoc scores
    the full-dimension propagator with no virtual-Z freedom, which on this gate
    disagrees with the pipeline by ~0.13 in fidelity and sends the optimizer
    downhill; see ``_jax_pipeline_infidelity`` for the measurement. Here the
    optimizer and the reporter are the same function, so ``F_grape`` is exact
    rather than re-scored, and ``dF_grape`` is a true improvement over
    ``F_baseline``.

    Requires ``jax``. Does NOT require qutip-qoc or qutip-jax.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            f"alg='JOPT' needs jax ({exc}). Install it, or use alg='CRAB' -- the "
            f"gradient-free optimizer, which requires only qutip.") from exc
    jax.config.update("jax_enable_x64", True)     # 1e-4 fidelities need float64

    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    peak = float(cpl.peak_eta())
    dt_ctrl = t_g / max(int(n_time), 1)
    n_sub = max(1, int(np.ceil(max_Omega * dt_ctrl / 0.3)))

    eta_fn = _iq_ansatz(t_g, peak, n_basis, xp=jnp)
    ts = (np.arange(n_time) + 0.5) * (t_g / n_time)
    ts_j = jnp.asarray(ts)
    # the L-BFGS-B box below caps amp_I/amp_Q at 1 + n_basis*bound = 2, so the
    # optimizer can never present more than 2*peak; that is the bound the
    # propagator sizes its scaling-and-squaring against.
    infid_eta = _jax_pipeline_infidelity(terms, H_anh, idx, t_g, n_sub, n_time,
                                         eta_max=2.0 * peak)

    def objective(p):
        eta_ctrl = jnp.stack([eta_fn(t, p) for t in ts_j])
        return infid_eta(eta_ctrl)

    obj_and_grad = jax.jit(jax.value_and_grad(objective))

    def scipy_obj(x):
        v, g = obj_and_grad(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=float)

    p0 = _drag_seed_params(t_g, peak, n_basis, warmstart_beat_GHz)
    # keep |eta| <= 2 * peak: amp_I = 1 + sum_k p_k s_k with |s_k| <= 1
    bound = 1.0 / float(n_basis)
    res = minimize(scipy_obj, p0, method="L-BFGS-B", jac=True,
                   bounds=[(-bound, bound)] * (2 * n_basis),
                   options=dict(maxiter=maxiter, ftol=1e-12, disp=verbose))
    p_star = np.asarray(res.x, dtype=float)

    # reconstruct + score in numpy on the identical model (agreement is a check,
    # not a re-score: the optimizer minimized exactly this quantity)
    eta_np = _iq_ansatz(t_g, peak, n_basis, xp=np)
    eta_opt = np.array([eta_np(t, p_star) for t in ts], dtype=complex)
    eta0 = _raised_cosine_eta(t_g, peak, n_time, drag_beat_GHz)
    Fg, leakg = _score(_propagate(eta_opt, t_g, terms, H_anh, idx, n_sub), cpl)
    F0, leak0 = _score(_propagate(eta0, t_g, terms, H_anh, idx, n_sub), cpl)

    return dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
                eta_opt=eta_opt, F_grape=Fg, leak_grape=leakg,
                n_ctrl=n_time, n_sub=n_sub, cutoff_GHz=cutoff_GHz,
                nfev=int(res.nfev),
                warmstart_beat_GHz=(np.nan if warmstart_beat_GHz is None
                                    else float(warmstart_beat_GHz)),
                backend="qutip", alg="JOPT", n_basis=n_basis,
                qoc_fid_err=float(res.fun))


def _crab_frequencies(n_basis: int, t_g: float, rng, jitter: float = 0.5) -> np.ndarray:
    """Randomized CRAB basis frequencies omega_k = 2 pi k (1 + r_k) / t_g, rad/ns.

    The random offsets r_k ~ U(-jitter, jitter) are what make CRAB work: on the
    exact harmonics every sin(2 pi k t / t_g) vanishes at t = t_g/2 (and the basis
    has other structural blind spots), so a randomized basis both removes those
    degeneracies and lets independent draws explore different subspaces.
    """
    k = np.arange(1, int(n_basis) + 1, dtype=float)
    return TWO_PI * k * (1.0 + rng.uniform(-jitter, jitter, size=k.size)) / t_g


def _drag_seed_crab(t_g: float, freqs: np.ndarray,
                    warmstart_beat_GHz: Optional[float]) -> np.ndarray:
    """Flat CRAB coefficients approximating the first-order DRAG pulse.

    DRAG is eta -> eta - i eta'/(2 pi beta). With the Hann shape S(t), the
    derivative is S'(t) = (pi/t_g) sin(2 pi t/t_g), so the quadrature is a pure
    sin at the FIRST harmonic with weight -1/(2 beta t_g). Loaded into sin_Q[0]
    (approximate, since the drawn frequency is jittered off 2 pi/t_g).
    """
    n = int(np.asarray(freqs).size)
    p = np.zeros(4 * n)
    if warmstart_beat_GHz and n > 0:
        p[n] = -1.0 / (2.0 * float(warmstart_beat_GHz) * t_g)      # sin_Q[0]
    return p


def _optimize_crab(cpl, a: int, b: int, t_g: float, *, n_basis: int,
                   cutoff_GHz: float, drag_beat_GHz: Optional[float],
                   warmstart_beat_GHz: Optional[float], maxiter: int,
                   restarts: int, seed: Optional[int], score: str, method: str,
                   atol: float, rtol: float, nsteps: int,
                   verbose: bool) -> Dict[str, Any]:
    """Optimize the pump envelope with CRAB (Chopped RAndom Basis).

    CRAB expands the pulse in a truncated, RANDOMIZED Fourier basis on top of a
    fixed shape function and minimizes the infidelity over the basis coefficients
    with a gradient-free optimizer (Nelder-Mead, as in qutip-qtrl). Because it
    treats the time evolution as a BLACK BOX -- pulse in, infidelity out -- it
    needs no propagator gradients and therefore no control-linear
    H = H_d + sum_k u_k H_k structure. That is exactly why it suits this gate,
    whose pump enters nonlinearly as eta, eta^2, eta^3 through g3 X^3.

    Why not call ``qutip_qtrl.pulseoptim.opt_pulse_crab_unitary`` directly: that
    implementation builds its dynamics from a drift plus a LIST OF CONTROL
    OPERATORS, i.e. it assumes control-linearity, and would require linearizing
    away the eta^2/eta^3 terms (31-68% of the Hamiltonian magnitude here). So we
    run the CRAB algorithm over the model as it is, and hand the pulse to QuTiP's
    compiled solver as the black box: the objective is
    ``ZhouCoupler.propagator_columns`` (exact ``qt.sesolve``, no pruned terms) on
    an :class:`envelope.IQFourierEnvelope`, scored with the same leakage-aware
    ``_iswap_fidelity_from_U`` as the sweeps. Nothing is approximated.

    ``score='reduced'`` swaps the objective for the fast rotating-frame model
    (no QuTiP) for cheap exploration; the returned pulse is then worth re-scoring
    with ``score='qutip'``.

    ``restarts > 1`` runs DCRAB-style super-iterations: each one APPENDS a fresh
    randomized basis and warm-starts from the previous optimum (padded with
    zeros), so the previous best pulse is inside the new search space and the
    fidelity is monotone across super-iterations. Note the parameter count grows
    by ``4 * n_basis`` each time, and Nelder-Mead degrades in high dimension --
    prefer a few harmonics and a couple of restarts.

    Returns
    -------
    dict
        Same keys as :func:`optimize_pulse` plus ``crab_freqs`` / ``crab_params``
        (the basis and coefficients, so the pulse is exactly reproducible).
    """
    from zhou_coupler import ZhouCoupler
    from envelope import IQFourierEnvelope

    rng = np.random.default_rng(seed)
    tone = cpl._pump_tones[0]
    env_saved, drag_saved = tone.envelope, tone.drag
    amp = float(env_saved.amp)
    n_samples = max(int(4 * n_basis * max(restarts, 1)), 32)   # for stored eta_opt

    # reduced-model scaffolding (used by score='reduced', and cheap to build)
    terms, H_anh, idx, max_Omega = _prepare(cpl, a, b, cutoff_GHz)
    n_sub_red = max(1, int(np.ceil(max_Omega * (t_g / n_samples) / 0.3)))

    def score_current() -> Tuple[float, float]:
        """(F, leak) of whatever pulse is currently installed on the tone."""
        if score == "qutip":
            U = cpl.propagator_columns(a, b, t_g, atol=atol, rtol=rtol, nsteps=nsteps)
            return ZhouCoupler._iswap_fidelity_from_U(U, True)
        # NOTE: read the pump through cpl._eta, NOT envelope.value -- _eta is what
        # applies the DRAG quadrature, the is_eta prefactor and the pump phase, so
        # sampling the raw envelope would silently drop the DRAG baseline.
        eta = np.array([complex(cpl._eta(tone, float(t)))
                        for t in (np.arange(n_samples) + 0.5) * (t_g / n_samples)])
        return _score(_propagate(eta, t_g, terms, H_anh, idx, n_sub_red), cpl)

    try:
        # (1) baseline: the gate actually applied at this point (DRAG or raised cosine)
        tone.envelope, tone.drag = env_saved, drag_saved
        if drag_beat_GHz is not None:
            tone.drag, tone.delta_drag_GHz = True, float(drag_beat_GHz)
        F0, leak0 = score_current()
        eta0 = np.array([complex(cpl._eta(tone, float(t)))
                         for t in (np.arange(n_samples) + 0.5) * (t_g / n_samples)])

        # (2) install the CRAB ansatz. Zero coefficients reproduce the raised cosine
        #     EXACTLY, so the optimizer starts from the sweep's own baseline shape.
        freqs = _crab_frequencies(n_basis, t_g, rng)
        env = IQFourierEnvelope(amp, t_g, freqs=freqs)
        tone.envelope, tone.drag = env, False   # ansatz carries its own quadrature

        # Candidate starts, all scored before optimizing. Zero coefficients ARE the
        # plain raised cosine, and the DRAG seed reproduces the DRAG baseline to
        # first order, so taking the best of these guarantees the optimizer starts
        # at or above the applied gate -- dF_grape can then never come out negative
        # just because a warm start happened to be a bad pulse.
        seeds = [("raised-cosine", np.zeros(4 * int(n_basis)))]
        if drag_beat_GHz:
            seeds.append(("drag-baseline", _drag_seed_crab(t_g, freqs, drag_beat_GHz)))
        if warmstart_beat_GHz:
            seeds.append(("warmstart", _drag_seed_crab(t_g, freqs, warmstart_beat_GHz)))
        F_best, leak_best, p_best, seed_used = -1.0, 1.0, seeds[0][1], seeds[0][0]
        nfev_total = 0
        for name, p in seeds:
            env.set_params(p)
            F_s, leak_s = score_current()
            nfev_total += 1
            if F_s > F_best:
                F_best, leak_best, p_best, seed_used = F_s, leak_s, p, name
        if verbose:
            print(f"  CRAB start: {seed_used} (F = {F_best:.6f}) "
                  f"of {len(seeds)} candidate seed(s)")

        bound = 2.0                            # |coefficient| bound (dimensionless)

        def infid(p: np.ndarray) -> float:
            q = np.clip(np.asarray(p, dtype=float), -bound, bound)
            env.set_params(q)
            F, _ = score_current()
            return 1.0 - F

        for sup in range(max(int(restarts), 1)):
            if sup > 0:
                # DCRAB super-iteration: append a fresh random basis, keep the
                # previous optimum (zeros on the new coefficients) as the start.
                new = _crab_frequencies(n_basis, t_g, rng)
                freqs = np.concatenate([freqs, new])
                k = p_best.size // 4
                sI, sQ, cI, cQ = (p_best[:k], p_best[k:2 * k],
                                  p_best[2 * k:3 * k], p_best[3 * k:])
                z = np.zeros(new.size)
                p_best = np.concatenate([sI, z, sQ, z, cI, z, cQ, z])
                env = IQFourierEnvelope(amp, t_g, freqs=freqs)
                env.set_params(p_best)
                tone.envelope = env
            res = minimize(infid, p_best, method=method,
                           options=dict(maxiter=int(maxiter), xatol=1e-6,
                                        fatol=1e-9, disp=verbose)
                           if method == "Nelder-Mead"
                           else dict(maxiter=int(maxiter), disp=verbose))
            nfev_total += int(res.nfev)
            p_try = np.clip(np.asarray(res.x, dtype=float), -bound, bound)
            env.set_params(p_try)
            F_try, leak_try = score_current()
            if F_try >= F_best:                # monotone: keep only improvements
                F_best, leak_best, p_best = F_try, leak_try, p_try
            else:
                env.set_params(p_best)
            if verbose:
                print(f"  CRAB super-iteration {sup + 1}/{restarts}: "
                      f"{freqs.size} harmonics, {p_best.size} params, F = {F_best:.6f}")

        env.set_params(p_best)
        eta_opt = env.samples(n_samples)
        return dict(eta_baseline=eta0, F_baseline=F0, leak_baseline=leak0,
                    eta_opt=eta_opt, F_grape=F_best, leak_grape=leak_best,
                    n_ctrl=n_samples, n_sub=n_sub_red, cutoff_GHz=cutoff_GHz,
                    nfev=nfev_total,
                    warmstart_beat_GHz=(np.nan if warmstart_beat_GHz is None
                                        else float(warmstart_beat_GHz)),
                    backend="qutip", alg="CRAB", n_basis=int(n_basis),
                    crab_freqs=np.asarray(freqs, dtype=float),
                    crab_params=np.asarray(p_best, dtype=float),
                    crab_score=score, crab_restarts=int(restarts),
                    crab_seed_used=seed_used,
                    qoc_fid_err=float(1.0 - F_best))
    finally:
        # never leave the caller's coupler holding the optimizer's pulse
        tone.envelope, tone.drag = env_saved, drag_saved


def optimize_pulse(cpl, a: int, b: int, t_g: float, *, n_ctrl: int = 24,
                   cutoff_GHz: float = 1.0, drag_beat_GHz: Optional[float] = None,
                   warmstart_beat_GHz: Optional[float] = None,
                   backend: str = "qutip", alg: str = "CRAB", n_basis: int = 6,
                   maxiter: int = 200, carrier_resolution: float = 0.3,
                   crab_restarts: int = 1, crab_seed: Optional[int] = None,
                   crab_score: str = "qutip", crab_method: str = "Nelder-Mead",
                   atol: float = 1e-10, rtol: float = 1e-8, nsteps: int = 500000,
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
        'qutip' (default) optimizes with QuTiP: ``alg='JOPT'`` goes through
        ``qutip-qoc``'s JAX analytic-control gradient method, ``alg='CRAB'``
        runs the Chopped RAndom Basis algorithm with QuTiP's compiled
        ``sesolve`` as the black-box objective. Both handle the
        eta/eta^2/eta^3 control-nonlinearity of the SNAIL gate. 'reduced' is the
        in-house scipy L-BFGS-B optimizer over the rotating-frame reduced model
        (fast, no QuTiP).
    alg : {'JOPT', 'CRAB'}
        Algorithm for ``backend='qutip'``. JOPT differentiates the exact
        propagator by JAX autodiff (needs ``qutip-qoc`` + ``qutip-jax``/``jax``);
        CRAB is gradient-FREE over a randomized Fourier basis, so it needs
        neither ``qutip-qoc`` nor gradients -- only ``qutip`` itself -- and is the
        most robust choice when the gradient method stalls or when the JAX stack
        is unavailable. It costs many exact propagations, though.
    n_basis : int
        Basis size: sin() functions per quadrature (JOPT) or randomized
        harmonics (CRAB, 4 real coefficients each). ``n_ctrl`` is the
        piecewise-constant control count for the 'reduced' backend and the
        tlist/sample resolution otherwise.
    crab_restarts : int
        CRAB super-iterations. Each appends a fresh randomized basis and
        warm-starts from the previous optimum, so fidelity is monotone (DCRAB).
    crab_seed : int, optional
        Seed for the random basis, for reproducible pulses.
    crab_score : {'qutip', 'reduced'}
        CRAB objective: the exact QuTiP propagator (default) or the fast reduced
        model for exploration.
    crab_method : str
        Gradient-free scipy method for CRAB ('Nelder-Mead', as in qutip-qtrl, or
        e.g. 'Powell').
    drag_beat_GHz : float or None
        Beat for the DRAG BASELINE quadrature (None -> plain raised cosine). Sets
        F_baseline / the gate the improvement is measured against.
    warmstart_beat_GHz : float or None
        Seed the optimizer from a DRAG raised cosine at THIS beat (all backends);
        baseline/dF unchanged, only the start moves. None -> start from baseline.
    maxiter : int
        Optimizer iteration cap (per CRAB super-iteration).
    carrier_resolution : float
        Max Omega*dt_fine (rad) -- sets fine sub-steps per control slice.
    atol, rtol, nsteps
        QuTiP ODE controls for the exact objective (CRAB with crab_score='qutip').

    Returns
    -------
    dict
        eta_baseline, F_baseline, leak_baseline, eta_opt, F_grape, leak_grape,
        n_ctrl, n_sub, cutoff_GHz, nfev, warmstart_beat_GHz, backend/alg (+
        qoc_fid_err, and crab_freqs/crab_params for CRAB). For CRAB with
        crab_score='qutip', F_baseline and F_grape are FULL-QuTiP numbers, so they
        are directly comparable to the sweep's F_avg; the other paths score on the
        reduced model.
    """
    if backend == "qutip" and alg.upper() == "CRAB":
        return _optimize_crab(cpl, a, b, t_g, n_basis=n_basis, cutoff_GHz=cutoff_GHz,
                              drag_beat_GHz=drag_beat_GHz,
                              warmstart_beat_GHz=warmstart_beat_GHz,
                              maxiter=maxiter, restarts=crab_restarts,
                              seed=crab_seed, score=crab_score, method=crab_method,
                              atol=atol, rtol=rtol, nsteps=nsteps, verbose=verbose)
    if backend == "qutip":
        return _optimize_jax(cpl, a, b, t_g, n_basis=n_basis, cutoff_GHz=cutoff_GHz,
                             drag_beat_GHz=drag_beat_GHz,
                             warmstart_beat_GHz=warmstart_beat_GHz,
                             maxiter=maxiter, n_time=n_ctrl, verbose=verbose)
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
    ap.add_argument("--alg", choices=["JOPT", "CRAB"], default="CRAB",
                    help="qutip optimizer: JOPT is the qutip-qoc JAX gradient method "
                         "(needs qutip-qoc + qutip-jax/jax); CRAB is gradient-free "
                         "over a randomized basis and needs only qutip")
    ap.add_argument("--crab-restarts", type=int, default=1,
                    help="[CRAB] DCRAB super-iterations; each appends a fresh random "
                         "basis and warm-starts from the previous optimum (monotone)")
    ap.add_argument("--crab-seed", type=int, default=None,
                    help="[CRAB] RNG seed for the random basis (reproducible pulses)")
    ap.add_argument("--crab-score", choices=["qutip", "reduced"], default="qutip",
                    help="[CRAB] objective: exact QuTiP propagator (default) or the "
                         "fast reduced model for exploration")
    ap.add_argument("--crab-method", default="Nelder-Mead",
                    help="[CRAB] gradient-free scipy method (Nelder-Mead, Powell, ...)")
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
                  backend=args.backend, alg=args.alg, n_basis=args.n_basis,
                  crab_restarts=args.crab_restarts, crab_seed=args.crab_seed,
                  crab_score=args.crab_score, crab_method=args.crab_method)
    if out.get("alg") == "CRAB":
        tag = (f"qutip/CRAB score={out.get('crab_score')} "
               f"{out.get('crab_freqs', np.zeros(0)).size} harmonics x "
               f"{out.get('crab_restarts')} super-iteration(s)")
    elif out.get("backend") == "qutip":
        tag = (f"qutip-qoc/{out.get('alg','JOPT')} "
               f"(fid_err={out.get('qoc_fid_err', float('nan')):.2e})")
    else:
        tag = f"reduced (cutoff={out['cutoff_GHz']} GHz)"
    print(f"{tag}, {out['n_ctrl']} pts, {out['nfev']} iters:")
    print(f"  DRAG raised-cosine : F = {out['F_baseline']:.5f}  leak = {out['leak_baseline']:.4f}")
    print(f"  GRAPE optimized    : F = {out['F_grape']:.5f}  leak = {out['leak_grape']:.4f}")
    print(f"  improvement dF = {out['F_grape'] - out['F_baseline']:+.5f}")
    print("  (validate the GRAPE pulse in the full QuTiP iswap_fidelity on the cluster)")


if __name__ == "__main__":
    main()