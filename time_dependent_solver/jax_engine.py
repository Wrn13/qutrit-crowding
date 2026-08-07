"""
jax_engine.py
=============

Batched, traceable propagator for the Zhou coupler -- the engine behind
``--engine jax``.

Why this exists
---------------
``ZhouCoupler.propagator_columns`` hands QuTiP a few hundred pure-Python scalar
closures (one ``_eta(t)`` call per term per ODE step) and then runs FOUR separate
``sesolve`` calls, one per computational column. That is fine as a reference but
it is slow, it cannot be differentiated, and it cannot be batched -- so a
150-point Stark chevron or a 2-D calibration map pays the whole cost 150 or N
times over, in separate processes.

This module solves the same equation with the same operators, but:

* **one frozen structure per device.** ``ZhouCoupler.expand_terms_symbolic``
  returns a term structure that does NOT depend on any frequency, so an entire
  ``(w_b, w_spec)`` sweep shares one operator stack and varies only a vector::

        Omega = M @ omega_vec

* **sparse.** Each operator is a sum of ladder products, so it has O(dim)
  non-zeros, not dim^2 (measured: 308x fewer at dim = 135). H is never formed;
  ``H.Psi`` is a gather-multiply-scatter over the COO triplets.
* **all four columns at once**, as a single (dim, 4) block.
* **vmapped** over a batch axis, so a whole grid is one XLA program.

Batch axes
----------
Both knobs a caller actually scans are batchable, and uniformly so:

* ``omega_vec`` -- mode frequencies AND the pump frequency live in the same
  vector, so a pump-offset scan (calibration map, Stark chevron) and a
  mode-frequency sweep are the same operation. There is no need for the
  ``Omega + (n_pos - n_neg) * offset`` special case used in ``grape._H``: the
  pump column of ``M`` already IS ``n_pos - n_neg``.
* ``params`` -- the pulse: amplitude, chirp coefficients, I/Q coefficients.

Integrators
-----------
``method="scan"`` is a branch-free fixed-step Magnus/Taylor propagator: every
batch element executes an identical step schedule, which is what makes ``vmap``
efficient. ``method="diffrax"`` is adaptive and is the accuracy oracle. See
``validate_engines.py``, which measures both against QuTiP rather than assuming.

Precision
---------
The binding constraint is the ARGUMENT of ``exp(-i Omega t)``, not the matrix
arithmetic: at cutoff = inf the fastest carrier is ~2 pi * 20 GHz, so a 77 ns
gate accumulates ``Omega t ~ 1e4`` rad. In float32 the spacing there is ~1e-3 rad
-- a 1e-3 phase error on every coefficient at every step. So ``precision="f32"``
means MIXED precision: time and the coefficient pass stay float64, and only the
state/operator arithmetic drops to complex64. Pure-float32 phases are not
offered, and float16 is not implementable at all (near t = 77 its spacing is
0.0625 ns while the required step is ~0.0024 ns, i.e. ``t + dt == t``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi


# ===========================================================================
# Frozen, frequency-independent structure
# ===========================================================================
class Engine:
    """A frozen operator structure plus the pulse spec, ready to propagate.

    Build with :func:`build_engine`. The instance holds only numpy arrays and
    plain Python, so it is picklable and cheap to ship to workers; the JAX arrays
    are materialized lazily on first use.

    Attributes
    ----------
    Omega0 : ndarray (n_terms,)
        Carrier of each kept term at the coupler's own frequencies (rad/ns).
    M : ndarray (n_terms, n_freq) int
        Integer carrier rows; ``Omega = M @ omega_vec`` for any frequencies.
    n_pos, n_neg : ndarray (n_terms, n_tones) int
        Pump exponents: the term carries ``prod eta^n_pos conj(eta)^n_neg``.
    term, row, col, val : ndarray (nnz,)
        Stacked COO triplets of the kept operators.
    """

    def __init__(self, cpl, cutoff_GHz: float = np.inf, a: int = 0, b: int = 1,
                 precision: str = "f64") -> None:
        if precision not in ("f64", "f32"):
            raise ValueError(f"precision must be 'f64' or 'f32', got {precision!r}")
        self.cpl = cpl
        self.cutoff_GHz = float(cutoff_GHz)
        self.precision = precision
        self.dim = int(cpl.dim)
        self.idx = list(cpl._subspace_indices(a, b))

        S = cpl.expand_terms_symbolic()
        self.omega_vec0 = cpl.frequency_vector()
        Omega_all = S["M"] @ self.omega_vec0

        # Prune by carrier. The cutoff is applied at the coupler's OWN frequencies;
        # a batch that moves them far enough to change which terms matter must be
        # built with a correspondingly larger cutoff (validate_engines quantifies
        # the residual). inf keeps everything and reproduces hamiltonian_matrix.
        keep = np.abs(Omega_all) <= abs(self.cutoff_GHz) * TWO_PI
        term_map = -np.ones(S["n_terms"], dtype=np.int64)
        term_map[keep] = np.arange(int(keep.sum()))

        self.M = S["M"][keep]
        self.n_pos = S["n_pos"][keep]
        self.n_neg = S["n_neg"][keep]
        self.Omega0 = Omega_all[keep]
        self.n_terms = int(keep.sum())
        self.n_dropped = int((~keep).sum())

        nz = keep[S["term"]]
        self.term = term_map[S["term"][nz]]
        self.row = S["row"][nz]
        self.col = S["col"][nz]
        self.val = S["val"][nz]
        self.H_anh = np.asarray(cpl._anharm_op, dtype=complex)

        # per-term operator infinity-norm, for the propagator's step bound
        acc = np.zeros((self.n_terms, self.dim))
        np.add.at(acc, (self.term, self.row), np.abs(self.val))
        self._op_norm = acc.max(axis=1) if self.n_terms else np.zeros(0)
        self._anh_norm = float(np.abs(self.H_anh).sum(axis=1).max())

        self.spec = pulse_spec(cpl)

    def __repr__(self) -> str:  # noqa: D105
        return (f"Engine(dim={self.dim}, n_terms={self.n_terms}, nnz={self.val.size}, "
                f"cutoff_GHz={self.cutoff_GHz}, dropped={self.n_dropped}, "
                f"precision={self.precision})")

    # -- Hamiltonian action ------------------------------------------------
    def coeffs(self, t: Any, omega_vec: Any, params: Dict[str, Any], xp: Any = np) -> Any:
        """Per-term scalar coefficient c_j(t) = e^{-i Omega_j t} prod eta^n_pos ...

        Always evaluated in the working float precision (float64 even when the
        state arithmetic is complex64) -- see the module docstring on why the
        phase argument is the precision-critical quantity.
        """
        Omega = xp.tensordot(xp.asarray(self.M, dtype=omega_vec.dtype), omega_vec,
                             axes=(1, 0))
        c = xp.exp(-1j * Omega * t)
        for p in range(self.spec["n_tones"]):
            eta = eta_at(self.spec, params, t, p, xp)
            c = c * eta ** self.n_pos[:, p] * xp.conj(eta) ** self.n_neg[:, p]
        return c

    def H_apply(self, t: Any, Psi: Any, omega_vec: Any, params: Dict[str, Any],
                xp: Any = np) -> Any:
        """H(t) . Psi for a (dim, k) block, without ever forming H.

        gather (Psi[col]) -> scale (c_term * val) -> scatter-add into rows.
        """
        c = self.coeffs(t, omega_vec, params, xp)
        contrib = (c[self.term] * self.val)[:, None] * Psi[self.col, :]
        if xp is np:
            out = np.zeros((self.dim,) + Psi.shape[1:], dtype=complex)
            np.add.at(out, self.row, contrib)
        else:
            out = xp.zeros((self.dim,) + Psi.shape[1:], dtype=Psi.dtype).at[self.row].add(
                contrib)
        return out + self.H_anh @ Psi

    # -- step schedule ------------------------------------------------------
    def peak_eta(self, params: Optional[Dict[str, Any]] = None, n: int = 257) -> float:
        """Largest |eta_p(t)| over the gate, across tones, for these parameters.

        Sampled rather than assumed: the pump amplitude is set by
        ``normalize_iswap``, which scales as ~1/t_g, so it is NOT order 1. On a
        short gate it easily reaches |eta| ~ 7, and a hardcoded ceiling of 2 there
        under-bounds ||H|| badly enough to diverge the propagator's Taylor series
        (see `step_plan`).
        """
        params = pulse_params(self.cpl) if params is None else params
        t_g = self.spec["tones"][0]["t_g"] if self.spec["n_tones"] else 1.0
        ts = np.linspace(0.0, t_g, n)
        peak = 0.0
        for p in range(self.spec["n_tones"]):
            peak = max(peak, float(np.max(np.abs(eta_at(self.spec, params, ts, p, np)))))
        return peak

    def step_plan(self, t_g: float, omega_vec: Optional[np.ndarray] = None,
                  eta_max: Optional[float] = None, carrier_resolution: float = 0.3,
                  n_steps: Optional[int] = None,
                  params: Optional[Dict[str, Any]] = None) -> Tuple[int, int, int]:
        """Choose (n_steps, taylor_order, n_squarings) from real bounds.

        The step must resolve the fastest KEPT carrier, and the squaring count
        must be read off an actual ``||H||`` bound -- not hardcoded. Too few
        squarings does not raise; it silently returns a diverged Taylor series
        (the same failure that motivated this bound in ``grape.py``, and which
        showed up here as a "fidelity" of 1e290 when `eta_max` defaulted to 2 on a
        20 ns gate whose pump peaks near 7).

        `eta_max` defaults to 1.25x the measured peak |eta|. Pass it explicitly
        when batching over amplitude, so the bound covers the LARGEST pulse in the
        batch -- every batch element shares this one schedule.
        """
        omega_vec = self.omega_vec0 if omega_vec is None else np.asarray(omega_vec)
        Omega = np.abs(self.M @ omega_vec)
        max_Omega = float(Omega.max()) if Omega.size else 0.0
        if n_steps is None:
            dt_max = carrier_resolution / max_Omega if max_Omega > 0 else t_g
            n_steps = max(1, int(np.ceil(t_g / dt_max)))
        dt = t_g / n_steps
        if eta_max is None:
            eta_max = 1.25 * max(self.peak_eta(params), 1e-12)
        pump_quanta = (self.n_pos + self.n_neg).sum(axis=1) if self.n_terms else np.zeros(0)
        h_norm = self._anh_norm + float((self._op_norm * eta_max ** pump_quanta).sum())
        sq = int(max(0, np.ceil(np.log2(max(h_norm * dt, 1e-12) / 0.5))))
        return int(n_steps), 12, sq


# ===========================================================================
# Pulse: static structure vs batchable values
# ===========================================================================
def pulse_spec(cpl) -> Dict[str, Any]:
    """Everything about the pump that is STRUCTURE, not a number to scan over.

    Split this way so the numbers (`pulse_params`) can be a vmap/grad axis while
    the structure stays a Python constant baked into the traced program.
    """
    tones = []
    for tone in cpl._pump_tones:
        env = tone.envelope
        kind = type(env).__name__
        omega_p = tone.w_p_GHz * TWO_PI
        omega_s = cpl.omega[cpl.coupler_index]
        tones.append({
            "kind": kind,
            "t_g": float(env.t_g),
            "freqs": np.asarray(getattr(env, "freqs", np.zeros(0)), dtype=float),
            "prefactor": (1.0 if tone.is_eta
                          else 2 * omega_p / (omega_p ** 2 - omega_s ** 2)),
            "phi_p": float(tone.phi_p),
            "drag": bool(tone.drag and tone.delta_drag_GHz not in (None, 0.0)),
            "drag_rad": float((tone.delta_drag_GHz or 0.0) * TWO_PI),
        })
    return {"n_tones": len(tones), "tones": tones}


def pulse_params(cpl) -> Dict[str, Any]:
    """The scannable/differentiable pump values, as a flat pytree of arrays."""
    amps, chirps, iqs = [], [], []
    for tone in cpl._pump_tones:
        env = tone.envelope
        amps.append(float(env.amp))
        chirps.append(np.asarray(tone.chirp.coeffs_GHz if tone.chirp is not None
                                 else np.zeros(0), dtype=float))
        iqs.append(np.asarray(env.get_params(), dtype=float))
    return {"amp": np.asarray(amps, dtype=float), "chirp": chirps, "iq": iqs}


def _shape_at(spec_tone: Dict[str, Any], params: Dict[str, Any], t: Any,
              p: int, xp: Any, deriv: bool):
    """Envelope shape (or its derivative) at `t`, in units of `amp`."""
    t_g = spec_tone["t_g"]
    support = xp.where((t >= 0.0) & (t <= t_g), 1.0, 0.0)
    if spec_tone["kind"] == "ConstantPulse":
        return (0.0 * xp.asarray(t)) if deriv else support
    # Hann shape, shared by RaisedCosine and IQFourierEnvelope
    S = 0.5 * (1.0 - xp.cos(TWO_PI * t / t_g))
    dS = 0.5 * (TWO_PI / t_g) * xp.sin(TWO_PI * t / t_g)
    freqs = spec_tone["freqs"]
    if freqs.size == 0:
        return (dS if deriv else S) * support
    q = params["iq"][p]
    n = freqs.size
    sI, sQ, cI, cQ = q[:n], q[n:2 * n], q[2 * n:3 * n], q[3 * n:]
    arg = xp.asarray(t)[..., None] * freqs
    s, c = xp.sin(arg), xp.cos(arg)
    M = ((1.0 + xp.sum(s * sI, axis=-1) + xp.sum(c * cI, axis=-1))
         + 1j * (xp.sum(s * sQ, axis=-1) + xp.sum(c * cQ, axis=-1)))
    if not deriv:
        return S * M * support
    dM = ((xp.sum(c * (sI * freqs), axis=-1) - xp.sum(s * (cI * freqs), axis=-1))
          + 1j * (xp.sum(c * (sQ * freqs), axis=-1) - xp.sum(s * (cQ * freqs), axis=-1)))
    return (dS * M + S * dM) * support


def _chirp_phase(params: Dict[str, Any], t: Any, p: int, t_g: float, xp: Any):
    """Accumulated chirp phase Phi(t); mirrors envelope.Chirp.phase.

    Duplicated here (rather than calling the Chirp object) only because the
    coefficients must come from `params` to be a vmap/grad axis. The two are
    pinned together by a test.
    """
    coeffs = params["chirp"][p]
    n = coeffs.shape[0] if hasattr(coeffs, "shape") else len(coeffs)
    if n == 0:
        return 0.0 * xp.asarray(t)
    u = xp.clip(2.0 * xp.asarray(t) / t_g - 1.0, -1.0, 1.0)
    P = [xp.ones_like(u), u]
    for k in range(1, n):
        P.append(((2 * k + 1) * u * P[k] - k * P[k - 1]) / (k + 1))
    total = coeffs[0] * (u + 1.0)
    for k in range(1, n):
        total = total + coeffs[k] * (P[k + 1] - P[k - 1]) / (2 * k + 1)
    return np.pi * t_g * total


def eta_at(spec: Dict[str, Any], params: Dict[str, Any], t: Any, p: int, xp: Any):
    """Pump amplitude eta_p(t) -- the traceable twin of ``ZhouCoupler._eta_at``.

    Same ordering: DRAG differentiates the BASE envelope, then the chirp phase
    multiplies the result.
    """
    st = spec["tones"][p]
    amp = params["amp"][p]
    a = amp * _shape_at(st, params, t, p, xp, deriv=False)
    if st["drag"]:
        a = a - 1j * amp * _shape_at(st, params, t, p, xp, deriv=True) / st["drag_rad"]
    a = a * xp.exp(-1j * _chirp_phase(params, t, p, st["t_g"], xp))
    return st["prefactor"] * a * xp.exp(1j * st["phi_p"])


# ===========================================================================
# Public builder
# ===========================================================================
def build_engine(cpl, cutoff_GHz: float = np.inf, a: int = 0, b: int = 1,
                 precision: str = "f64") -> Engine:
    """Freeze a coupler into a batched propagator structure.

    Parameters
    ----------
    cpl : ZhouCoupler
        A coupler with its pump already attached (``set_pump``), since the pulse
        structure is read off the tone.
    cutoff_GHz : float, default inf
        Keep only terms with ``|Omega| <= 2 pi cutoff``. inf reproduces the exact
        Hamiltonian and is the RIGHT DEFAULT: a finite cutoff is a rotating-wave
        reduction that is only valid at weak drive, and this gate is often not.
        Measured on a 15 ns gate (|eta| = 9.3), a 3 GHz cutoff was off by
        max|dU| ~ 0.9 while inf agreed to 5e-4 -- and inf still ran 174x faster
        than QuTiP, because the speed comes from the CF4 integrator, the sparse
        operators and the column/grid batching, NOT from pruning. Lower it only
        with a `validate_engines.py --cutoff-scan` to back the choice.
    a, b : int
        Target-qubit mode indices (fixes the 4 computational columns).
    precision : {'f64', 'f32'}
        'f32' is MIXED -- float64 phases, complex64 state. See the module
        docstring.
    """
    return Engine(cpl, cutoff_GHz=cutoff_GHz, a=a, b=b, precision=precision)


# ===========================================================================
# Propagation
# ===========================================================================
def _expm_taylor(A, order: int, sq: int, xp):
    """exp(A) by scaling-and-squaring with the counts fixed at BUILD time.

    The order/squaring counts are arguments rather than being read off ``||A||``
    at runtime, because data-dependent control flow cannot compile inside a
    scanned loop. `Engine.step_plan` derives them from a real norm bound.
    """
    As = A / (2 ** sq)
    eye = xp.eye(As.shape[-1], dtype=As.dtype)
    term, out = eye, eye
    for k in range(1, order + 1):
        term = term @ As / k
        out = out + term
    for _ in range(sq):
        out = out @ out
    return out


# Commutator-free Magnus, 4th order (Blanes & Moan). Two exponentials per step at
# the Gauss-Legendre nodes. The 2nd-order exponential midpoint rule converges as
# O(dt^2), which measured out at 1e-4 for a 10 ns gate at carrier_resolution=0.4
# and would need ~2.6M steps to reach 1e-10; CF4 gets there in ~80k. The extra
# exponential per step is bought back many times over.
# Kept as PYTHON floats, not numpy scalars: a np.float64 is strongly typed under
# JAX and would promote a complex64 carry to complex128 mid-scan (which lax.scan
# then rejects, since the carry type must be invariant). Python scalars are weakly
# typed and adopt the array's dtype.
_C4 = float(np.sqrt(3.0) / 6.0)
_CF4_NODES = (0.5 - _C4, 0.5 + _C4)
_CF4_W = ((0.25 + _C4, 0.25 - _C4), (0.25 - _C4, 0.25 + _C4))


def propagator_columns(eng: Engine, t_g: float, omega_vec=None, params=None,
                       n_steps: Optional[int] = None, eta_max: Optional[float] = None,
                       carrier_resolution: float = 0.3, scheme: str = "cf4",
                       xp: Any = np) -> Any:
    """4x4 projected propagator via a fixed-step, branch-free scan.

    Propagates all four computational columns together as one (dim, 4) block --
    where ``ZhouCoupler.propagator_columns`` runs four independent ``sesolve``
    calls. The step schedule is identical for every batch element, which is what
    makes the ``vmap`` in :func:`propagator_columns_batched` efficient.

    Parameters
    ----------
    scheme : {'cf4', 'midpoint'}
        'cf4' is the 4th-order commutator-free Magnus integrator (default);
        'midpoint' is the cheaper 2nd-order exponential midpoint rule. Both
        converge to the same answer -- cf4 just gets there in far fewer steps.
    carrier_resolution : float
        Caps ``max|Omega| * dt``. This is the accuracy knob; halving it cuts the
        cf4 error ~16x.
    """
    omega_vec = eng.omega_vec0 if omega_vec is None else omega_vec
    params = pulse_params(eng.cpl) if params is None else params
    # the schedule is derived from the ENGINE's own reference parameters, not the
    # (possibly traced) batch ones, so every batch element shares one step plan
    n_steps, order, sq = eng.step_plan(t_g, np.asarray(eng.omega_vec0), eta_max,
                                       carrier_resolution, n_steps)
    dt = t_g / n_steps

    cdtype = complex if (xp is np or eng.precision == "f64") else xp.complex64
    Psi = xp.asarray(np.eye(eng.dim, dtype=complex)[:, eng.idx]).astype(cdtype)

    def H_at(t):
        return _dense_H(eng, t, omega_vec, params, xp).astype(cdtype)

    if scheme == "midpoint":
        def step(Psi, t0):
            return _expm_taylor(-1j * H_at(t0 + 0.5 * dt) * dt, order, sq, xp) @ Psi
    elif scheme == "cf4":
        def step(Psi, t0):
            H1 = H_at(t0 + _CF4_NODES[0] * dt)
            H2 = H_at(t0 + _CF4_NODES[1] * dt)
            for w1, w2 in _CF4_W:
                Psi = _expm_taylor(-1j * (w1 * H1 + w2 * H2) * dt, order, sq, xp) @ Psi
            return Psi
    else:
        raise ValueError(f"unknown scheme {scheme!r}; use 'cf4' or 'midpoint'")

    ts = np.arange(n_steps) * dt
    if xp is np:
        for t in ts:
            Psi = step(Psi, float(t))
    else:
        import jax
        Psi, _ = jax.lax.scan(lambda P, t: (step(P, t), None), Psi, xp.asarray(ts))
    return Psi[xp.asarray(eng.idx), :]


def _dense_H(eng: Engine, t, omega_vec, params, xp):
    """Materialize H(t) from the COO stack (needed by the expm propagator).

    ``H_apply`` is the cheap path and is what an ODE integrator wants; the
    scaling-and-squaring propagator needs the matrix itself, so this scatters the
    same triplets into a dense (dim, dim).
    """
    c = eng.coeffs(t, omega_vec, params, xp)
    vals = c[eng.term] * eng.val
    if xp is np:
        H = np.zeros((eng.dim, eng.dim), dtype=complex)
        np.add.at(H, (eng.row, eng.col), vals)
    else:
        # scatter in the accumulation dtype: under precision='f32' jax_enable_x64 is
        # off, so asking for complex128 here would silently downcast anyway
        cdtype = xp.complex128 if eng.precision == "f64" else xp.complex64
        H = xp.zeros((eng.dim, eng.dim), dtype=cdtype).at[eng.row, eng.col].add(
            vals.astype(cdtype))
    return H + eng.H_anh


def propagator_columns_batched(eng: Engine, t_g: float, omega_batch=None,
                               params_batch=None, **kw):
    """``propagator_columns`` vmapped over a batch of frequencies and/or pulses.

    Pass an (B, n_freq) ``omega_batch`` to sweep frequencies (mode allocation, or
    a pump-offset scan -- the pump lives in the same vector), and/or a batched
    ``params_batch`` to sweep the pulse. Returns (B, 4, 4).
    """
    import jax
    import jax.numpy as jnp

    base_omega = jnp.asarray(eng.omega_vec0)
    base_params = _to_jax(pulse_params(eng.cpl))
    in_axes = (0 if omega_batch is not None else None,
               0 if params_batch is not None else None)

    def one(omega_vec, params):
        return propagator_columns(eng, t_g,
                                  base_omega if omega_vec is None else omega_vec,
                                  base_params if params is None else params,
                                  xp=jnp, **kw)

    f = jax.jit(jax.vmap(one, in_axes=in_axes))
    return f(None if omega_batch is None else jnp.asarray(omega_batch),
             None if params_batch is None else params_batch)


def _to_jax(params):
    """Move a pulse-parameter pytree onto JAX arrays."""
    import jax.numpy as jnp
    return {"amp": jnp.asarray(params["amp"]),
            "chirp": [jnp.asarray(c) for c in params["chirp"]],
            "iq": [jnp.asarray(q) for q in params["iq"]]}


def check_propagator(U, tol: float = 1e-6) -> None:
    """Fail loudly if the propagator is not sub-unitary.

    ``U`` is a projection of a unitary onto the computational subspace, so every
    entry must satisfy |U_ij| <= 1. A scaling-and-squaring propagator whose
    squaring count was under-bounded does not raise -- it returns a diverged
    Taylor series, which then scores as a perfectly plausible-looking (or absurd)
    fidelity. This turns that silent corruption into an error.
    """
    U = np.asarray(U)
    peak = float(np.max(np.abs(U))) if U.size else 0.0
    if not np.isfinite(peak) or peak > 1.0 + tol:
        raise FloatingPointError(
            f"propagator diverged (max|U| = {peak:.3e} > 1): the expm squaring count "
            f"was under-bounded. Pass a larger eta_max, or reduce carrier_resolution.")


# ===========================================================================
# Ready-made scans (the workloads that actually dominate wall-clock)
# ===========================================================================
def scan_amp_offset(cpl, t_g: float, amps: Sequence[float], offsets_MHz: Sequence[float],
                    *, cutoff_GHz: float = np.inf, carrier_resolution: float = 0.1,
                    precision: str = "f64", batch: int = 64,
                    metric: str = "fidelity", a: int = 0, b: int = 1) -> Dict[str, Any]:
    """Amplitude x pump-offset grid in one batched program.

    This is the calibration-map / Stark-chevron workload. Both axes are batch
    axes here: the amplitude enters ``params``, and the pump offset is just the
    last entry of ``omega_vec`` -- the device is fixed, so all points share one
    operator stack.

    `cpl` must already carry the pump at ``amp_scale = 1`` and zero extra offset;
    the grid is applied relative to it. Returns the same fields as
    ``calibration_map.scan_qutip`` so it can drop in.

    Returns
    -------
    dict
        ``Z`` (n_amp, n_off) score, ``leak`` (n_amp, n_off), plus ``amps``,
        ``offsets_MHz`` and the engine settings actually used.
    """
    import jax
    import jax.numpy as jnp

    from zhou_coupler import ZhouCoupler

    # Always ENABLE x64 and let the dtype carry the precision choice. Toggling this
    # flag off mid-process is unreliable once arrays exist, and the f32 path needs
    # float64 phases anyway (see the module docstring) -- so f32 is expressed by the
    # complex64 accumulation dtype in `_dense_H`, not by a global downgrade.
    jax.config.update("jax_enable_x64", True)
    eng = build_engine(cpl, cutoff_GHz=cutoff_GHz, a=a, b=b, precision=precision)
    base_p = pulse_params(cpl)
    base_w = np.asarray(eng.omega_vec0)
    amps = np.asarray(amps, dtype=float)
    offsets_MHz = np.asarray(offsets_MHz, dtype=float)

    # flatten the grid so one batch axis covers both scan axes
    AA, OO = np.meshgrid(amps, offsets_MHz, indexing="ij")
    flat_amp, flat_off = AA.ravel(), OO.ravel()
    n = flat_amp.size

    # One step schedule serves the whole grid, so its ||H|| bound must cover the
    # LARGEST amplitude on it -- not the nominal one.
    eta_max = 1.25 * eng.peak_eta(base_p) * float(np.max(np.abs(amps)))

    Z = np.zeros(n)
    leak = np.zeros(n)
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        w = np.repeat(base_w[None, :], hi - lo, axis=0)
        w[:, -1] = base_w[-1] + flat_off[lo:hi] * 1e-3 * TWO_PI   # MHz -> rad/ns
        p = {"amp": jnp.asarray(base_p["amp"][None, :] * flat_amp[lo:hi, None]),
             "chirp": [jnp.repeat(jnp.asarray(c)[None, :], hi - lo, axis=0)
                       for c in base_p["chirp"]],
             "iq": [jnp.repeat(jnp.asarray(q)[None, :], hi - lo, axis=0)
                    for q in base_p["iq"]]}
        U = np.asarray(propagator_columns_batched(
            eng, t_g, omega_batch=w, params_batch=p, eta_max=eta_max,
            carrier_resolution=carrier_resolution))
        check_propagator(U)
        for k in range(hi - lo):
            fid, lk = ZhouCoupler._iswap_fidelity_from_U(U[k], True)
            Z[lo + k] = abs(U[k][2, 1]) ** 2 if metric == "transfer" else fid
            leak[lo + k] = lk

    Z = Z.reshape(AA.shape)
    leak = leak.reshape(AA.shape)
    bi, bj = np.unravel_index(int(np.nanargmax(Z)), Z.shape)
    # key names match scan()/scan_qutip() exactly -- run() consumes `best` the same
    # way for every engine
    best = dict(amp_scale=float(amps[bi]), wp_offset_MHz=float(offsets_MHz[bj]),
                score=float(Z[bi, bj]), leakage=float(leak[bi, bj]))
    return {"amps": amps, "offsets_MHz": offsets_MHz, "Z": Z, "leakage": leak,
            "best": best, "metric": metric, "engine": "jax",
            "cutoff_GHz": cutoff_GHz, "carrier_resolution": carrier_resolution,
            "precision": precision, "n_terms": eng.n_terms,
            "n_dropped": eng.n_dropped}

