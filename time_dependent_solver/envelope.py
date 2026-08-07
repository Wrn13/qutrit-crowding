"""
envelope.py
===========

Pump envelopes, frequency chirps, and the pump-tone container.

This module is the SINGLE SOURCE OF TRUTH for these classes; `zhou_coupler`
re-exports every public name here, so ``from zhou_coupler import RaisedCosine``
keeps working unchanged.

Split of responsibilities
-------------------------
* :class:`Envelope` carries the pulse SHAPE and its peak scale ``amp``. Its
  ``value`` may be complex (the DRAG quadrature and the CRAB I/Q ansatz both
  need that), but ``|value|`` is what the iSWAP normalization integrates.
* :class:`Chirp` carries a time-dependent FREQUENCY offset delta(t) about the
  tone's fixed carrier ``w_p_GHz``, as an accumulated phase Phi(t) = int_0^t delta.
* :class:`PumpTone` binds the two to a carrier.

Why a chirp needs no solver changes
-----------------------------------
A pump letter enters X(t) as ``eta_p(t) e^{-i w_p t}``. Chirping the carrier,
w_p -> w_p + delta(t), is therefore ALGEBRAICALLY IDENTICAL to multiplying the
complex envelope by a phase::

    eta_p(t) e^{-i(w_p t + Phi(t))} = [eta_p(t) e^{-i Phi(t)}] e^{-i w_p t}

so the fixed carrier w_p stays the expansion reference and the chirp rides in the
envelope. A Hamiltonian term carrying k net pump quanta picks up ``e^{-i k Phi(t)}``
automatically, which is exactly the k-quanta net-carrier shift. Nothing in
``_flux_letters``, ``expand_terms`` or ``to_qutip_hamiltonian`` has to know.
This is the time-dependent generalization of the constant ``offset_rad`` trick in
``grape._H``.

Array API and tracing
---------------------
Every shape function is available in an ``xp``-generic form -- ``value_at(t, xp)``,
``deriv_at(t, xp)``, ``Chirp.phase(t, xp)`` -- accepting scalar OR array ``t`` and
built only from ``xp`` primitives, following the convention already used by
``grape._iq_ansatz``. With ``xp=jax.numpy`` these stay trace-clean: no ``float()``
and no ``asarray(..., dtype=)`` is applied to any value that can carry a tracer,
either of which raises (TracerArrayConversionError / ConcretizationTypeError).
Domain guards are ``xp.where`` masks rather than Python ``if``, since a branch on a
traced value cannot compile.

The scalar ``value(t)`` / ``deriv(t)`` methods remain as thin wrappers so that the
existing per-time solver callbacks are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

# numpy>=2 renamed trapz -> trapezoid; fall back only if needed (trapz is gone in 2.x).
_trapezoid: Callable[..., float] = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
TWO_PI = 2.0 * np.pi


# ===========================================================================
# Pump envelopes  (shape eps(t) on [0, t_g]; peak scale carried in `amp`)
# ===========================================================================
class Envelope:
    """Base class for a pump-amplitude envelope eps(t) on [0, t_g].

    Subclasses implement `value_at` and `deriv_at` (the ``xp``-generic forms);
    `value` / `deriv` are scalar wrappers around them and need no overriding.
    `area` integrates |eps| over the gate (subclasses with closed forms override
    it).

    Parameters
    ----------
    amp : float
        Peak amplitude. Carries |eps| in rad/ns, or directly |eta| when the
        owning PumpTone has ``is_eta=True``.
    t_g : float
        Gate duration in ns; the envelope is supported on [0, t_g].

    Attributes
    ----------
    is_complex : bool
        Class attribute; True if `value` can return a complex amplitude. Only
        controls the scalar wrapper's cast.
    """

    is_complex: bool = False

    def __init__(self, amp: float, t_g: float) -> None:
        self.amp: float = float(amp)
        self.t_g: float = float(t_g)

    # -- xp-generic core (subclasses implement these) ----------------------
    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Envelope amplitude at time(s) `t` (ns), built from `xp` primitives.

        Accepts a scalar or an array and returns the same shape. Must not branch
        on `t` (use ``xp.where``) so the body stays vectorizable and traceable.
        """
        raise NotImplementedError

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Time derivative d eps/dt at time(s) `t` (ns), built from `xp`."""
        raise NotImplementedError

    # -- scalar wrappers (the per-time solver callback path) ---------------
    def value(self, t: float) -> Any:
        """Envelope amplitude at a single time `t` (ns)."""
        v = self.value_at(float(t), np)
        return complex(v) if self.is_complex else float(v)

    def deriv(self, t: float) -> Any:
        """Time derivative d eps/dt at a single time `t` (ns)."""
        v = self.deriv_at(float(t), np)
        return complex(v) if self.is_complex else float(v)

    # -- integrated quantities ---------------------------------------------
    def area(self) -> float:
        """Return integral_0^{t_g} value(t) dt by quadrature (override if a closed
        form exists).

        This is the quantity `set_pump(..., normalize_iswap=...)` divides by to
        calibrate the pi/2 rotation. It is unaffected by a chirp: a chirp lives on
        the PumpTone, not the envelope, and is a pure phase that leaves |eta|
        alone. Do not add a chirp correction here.
        """  # noqa: D205
        ts = np.linspace(0.0, self.t_g, 4001)
        return float(_trapezoid(self.value_at(ts, np), ts))

    def samples(self, n: int) -> np.ndarray:
        """`n` midpoint samples of eps(t) across the gate (for plotting/storage)."""
        ts = (np.arange(int(n)) + 0.5) * (self.t_g / int(n))
        return np.asarray(self.value_at(ts, np), dtype=complex)

    # -- optimizer parameter interface (no free parameters by default) -----
    @property
    def n_params(self) -> int:
        """Number of free real shape parameters."""
        return 0

    def get_params(self) -> np.ndarray:
        """Free shape parameters as a flat real vector."""
        return np.zeros(0)

    def set_params(self, p: Any) -> None:
        """Load a flat real vector as produced by :meth:`get_params`."""
        if np.asarray(p).size != 0:
            raise ValueError(f"{type(self).__name__} takes no shape parameters")

    # -- shared helper ------------------------------------------------------
    def _support(self, t: Any, xp: Any) -> Any:
        """1.0 inside [0, t_g], 0.0 outside -- as a mask, never a branch."""
        return xp.where((t >= 0.0) & (t <= self.t_g), 1.0, 0.0)


class ConstantPulse(Envelope):
    """Flat pump with instantaneous on/off; handy for steady-state rate checks."""

    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Amplitude `amp` for t in [0, t_g], else 0."""
        return self.amp * self._support(t, xp)

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Zero everywhere (flat pulse)."""
        return 0.0 * xp.asarray(t)

    def area(self) -> float:
        """Closed form: amp * t_g."""
        return self.amp * self.t_g


class RaisedCosine(Envelope):
    """Hann (raised-cosine) envelope: smooth turn-on/off, vanishing endpoints,.

    value(t) = amp/2 [1 - cos(2 pi t / t_g)] ,  t in [0, t_g] .
    """

    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Hann amplitude at `t` (ns); 0 outside [0, t_g]."""
        return (self.amp * 0.5 * (1.0 - xp.cos(TWO_PI * t / self.t_g))
                * self._support(t, xp))

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Analytic derivative of the Hann window at `t` (ns); 0 outside [0, t_g]."""
        return (self.amp * 0.5 * (TWO_PI / self.t_g) * xp.sin(TWO_PI * t / self.t_g)
                * self._support(t, xp))

    def area(self) -> float:
        """Closed form: amp * t_g / 2 (exact integral of the Hann window)."""
        return self.amp * self.t_g / 2.0


class IQFourierEnvelope(Envelope):
    r"""Complex I/Q envelope: a fixed shape times a truncated Fourier modulation.

    .. math::
        \eta(t) = \mathrm{amp}\;S(t)\;\Big[1
            + \sum_k \big(a^I_k + i\,a^Q_k\big)\sin(\omega_k t)
            + \sum_k \big(b^I_k + i\,b^Q_k\big)\cos(\omega_k t)\Big]

    with :math:`S(t) = \tfrac12[1 - \cos(2\pi t/t_g)]` the Hann shape function.

    This is the ansatz container used by the CRAB / JOPT optimizers in
    ``grape.py``. Two properties make it safe to drop into the existing pump
    machinery:

    * ``value`` returns a COMPLEX amplitude. ``ZhouCoupler._eta`` already treats
      the envelope value as complex (it has to, for the DRAG quadrature), so the
      full QuTiP path -- ``to_qutip_hamiltonian`` -> ``propagator_columns`` /
      ``iswap_fidelity`` -- evaluates an arbitrary shaped pulse with no changes.
      That means an optimizer can use the compiled QuTiP solver as its black box
      instead of a reduced model.
    * With all coefficients zero it reduces EXACTLY to
      :class:`RaisedCosine`, so ``set_pump(..., normalize_iswap=...)`` performed
      with a raised cosine stays valid, ``peak_eta`` is unchanged, and the
      optimizer starts from the sweep's baseline gate.

    The Hann prefactor enforces :math:`\eta(0) = \eta(t_g) = 0` for ANY
    coefficients, so the optimizer cannot produce a pulse with a discontinuous
    turn-on -- the usual CRAB "shape function" role.

    Parameters
    ----------
    amp : float
        Peak amplitude of the unmodulated shape (see :class:`Envelope`).
    t_g : float
        Gate duration (ns).
    freqs : array_like, optional
        Angular frequencies :math:`\omega_k` (rad/ns) of the basis. For CRAB these
        are randomized about the harmonics; ``None`` (default) gives no
        modulation, i.e. a plain raised cosine.
    sin_I, sin_Q, cos_I, cos_Q : array_like, optional
        Coefficients of the sin/cos basis in the in-phase (I) and quadrature (Q)
        components. Each must match ``freqs`` in length; omitted arrays are zero.

    Notes
    -----
    ``area`` integrates the REAL part, matching the leading-order iSWAP
    normalization convention (the in-phase component is what drives the swap);
    with zero coefficients it is exactly ``amp * t_g / 2``.
    """

    is_complex = True

    def __init__(self, amp: float, t_g: float, freqs=None,
                 sin_I=None, sin_Q=None, cos_I=None, cos_Q=None) -> None:
        super().__init__(amp, t_g)
        self.freqs: np.ndarray = (np.zeros(0) if freqs is None
                                  else np.asarray(freqs, dtype=float))
        n = self.freqs.size

        def _co(v):
            if v is None:
                return np.zeros(n)
            arr = np.asarray(v, dtype=float)
            if arr.size != n:
                raise ValueError(f"coefficient array has {arr.size} entries, "
                                 f"expected {n} to match freqs")
            return arr

        self.sin_I, self.sin_Q = _co(sin_I), _co(sin_Q)
        self.cos_I, self.cos_Q = _co(cos_I), _co(cos_Q)

    # -- the modulation factor and its derivative --------------------------
    def _basis(self, t: Any, xp: Any):
        """sin/cos of every basis frequency, broadcast over `t`.

        ``t[..., None] * freqs`` puts the basis on a trailing axis so a scalar and
        an array of times take the same code path; the contractions below then sum
        over that axis.
        """
        arg = xp.asarray(t)[..., None] * self.freqs
        return xp.sin(arg), xp.cos(arg)

    def _mod(self, t: Any, xp: Any = np) -> Any:
        """Modulation M(t); 1 when there is no basis (plain raised cosine)."""
        if self.freqs.size == 0:
            return 1.0 + 0.0j
        s, c = self._basis(t, xp)
        # sum over the trailing basis axis -- NOT float(), which would concretize
        # a tracer and break the JAX gradient path.
        return ((1.0 + xp.sum(s * self.sin_I, axis=-1) + xp.sum(c * self.cos_I, axis=-1))
                + 1j * (xp.sum(s * self.sin_Q, axis=-1) + xp.sum(c * self.cos_Q, axis=-1)))

    def _dmod(self, t: Any, xp: Any = np) -> Any:
        """dM/dt."""
        if self.freqs.size == 0:
            return 0.0 + 0.0j
        w = self.freqs
        s, c = self._basis(t, xp)
        return ((xp.sum(c * (self.sin_I * w), axis=-1)
                 - xp.sum(s * (self.cos_I * w), axis=-1))
                + 1j * (xp.sum(c * (self.sin_Q * w), axis=-1)
                        - xp.sum(s * (self.cos_Q * w), axis=-1)))

    def _shape(self, t: Any, xp: Any = np) -> Any:
        """Hann shape function S(t)."""
        return 0.5 * (1.0 - xp.cos(TWO_PI * t / self.t_g))

    def _dshape(self, t: Any, xp: Any = np) -> Any:
        """dS/dt."""
        return 0.5 * (TWO_PI / self.t_g) * xp.sin(TWO_PI * t / self.t_g)

    def value_at(self, t: Any, xp: Any = np) -> Any:
        """Complex envelope amplitude at `t` (ns); 0 outside [0, t_g]."""
        return self.amp * self._shape(t, xp) * self._mod(t, xp) * self._support(t, xp)

    def deriv_at(self, t: Any, xp: Any = np) -> Any:
        """Analytic d(eta)/dt at `t` (ns); 0 outside [0, t_g]."""
        return (self.amp * (self._dshape(t, xp) * self._mod(t, xp)
                            + self._shape(t, xp) * self._dmod(t, xp))
                * self._support(t, xp))

    def area(self) -> float:
        """Integral of Re[eta] over the gate (see Notes); closed form when the
        basis is empty."""
        if self.freqs.size == 0:
            return self.amp * self.t_g / 2.0
        ts = np.linspace(0.0, self.t_g, 4001)
        return float(_trapezoid(np.real(self.value_at(ts, np)), ts))

    # -- flat parameter-vector interface for optimizers --------------------
    @property
    def n_params(self) -> int:
        """Number of free real coefficients (4 per basis frequency)."""
        return 4 * self.freqs.size

    def get_params(self) -> np.ndarray:
        """Coefficients as one flat real vector [sin_I, sin_Q, cos_I, cos_Q]."""
        return np.concatenate([self.sin_I, self.sin_Q, self.cos_I, self.cos_Q])

    def set_params(self, p) -> None:
        """Load a flat real vector as produced by :meth:`get_params`."""
        p = np.asarray(p, dtype=float)
        n = self.freqs.size
        if p.size != 4 * n:
            raise ValueError(f"expected {4 * n} parameters, got {p.size}")
        self.sin_I, self.sin_Q, self.cos_I, self.cos_Q = (
            p[:n].copy(), p[n:2 * n].copy(), p[2 * n:3 * n].copy(), p[3 * n:].copy())


# ===========================================================================
# Frequency chirp
# ===========================================================================
def _legendre_stack(u: Any, degree: int, xp: Any):
    """Shifted-Legendre values P_0(u) .. P_degree(u), by the standard recurrence.

    Built with the recurrence rather than ``numpy.polynomial`` so the same code
    runs under ``jax.numpy`` (and so it vectorizes over an array of `u`).
    """
    ones = xp.ones_like(u)
    out = [ones]
    if degree >= 1:
        out.append(u)
    for k in range(1, degree):
        # (k+1) P_{k+1} = (2k+1) u P_k - k P_{k-1}
        out.append(((2 * k + 1) * u * out[k] - k * out[k - 1]) / (k + 1))
    return out


class Chirp:
    r"""A time-dependent pump-frequency offset delta(t) about a fixed carrier.

    The offset is a Legendre polynomial in the normalized gate time
    :math:`u = 2t/t_g - 1 \in [-1, 1]`:

    .. math::
        \delta(t) = 2\pi \sum_k c_k P_k(u) ,\qquad
        \Phi(t) = \int_0^t \delta(t')\,dt'

    The Legendre basis is used (rather than raw powers of `t`) because its terms
    are orthogonal over the gate, which keeps an optimizer's parameters from
    fighting each other: `c_0` is the mean detuning, `c_1` the linear chirp rate,
    and higher terms add structure without shifting the mean.

    :math:`\Phi` is obtained in CLOSED FORM, not by quadrature, from
    :math:`\int_{-1}^{u} P_k = (P_{k+1} - P_{k-1})/(2k+1)` for k >= 1 and
    :math:`\int_{-1}^{u} P_0 = u + 1`:

    .. math::
        \Phi(t) = \pi t_g \Big[ c_0 (u+1)
                  + \sum_{k\ge 1} c_k \frac{P_{k+1}(u) - P_{k-1}(u)}{2k+1} \Big]

    Special cases worth knowing:

    * ``coeffs_GHz = [c0]`` is a CONSTANT detuning, and is exactly equivalent to
      building the tone at ``w_p_GHz + c0`` (see the module docstring). This is a
      free regression test and is asserted in ``test_physics``.
    * ``coeffs_GHz = [c0, c1]`` is a linear chirp sweeping from ``c0 - c1`` to
      ``c0 + c1`` GHz across the gate.

    Parameters
    ----------
    coeffs_GHz : sequence of float
        Legendre coefficients of delta(t)/2pi, in GHz. Empty (or all-zero) means
        no chirp.
    t_g : float
        Gate duration (ns); sets the normalization of `u`.

    Notes
    -----
    A chirp is a pure PHASE: it leaves ``|eta(t)|`` untouched. Therefore it does
    not change ``Envelope.area()``, the ``normalize_iswap`` amplitude calibration,
    or ``peak_eta``. Do not "fix" those to account for it.

    `t` is clipped to [0, t_g] before evaluation. The envelope is zero outside the
    gate so this changes no observable, but it stops a high-degree polynomial from
    reporting a meaningless multi-thousand-radian phase just outside the support.
    """

    def __init__(self, coeffs_GHz: Sequence[float], t_g: float) -> None:
        self.coeffs_GHz: np.ndarray = np.asarray(coeffs_GHz, dtype=float).ravel()
        self.t_g: float = float(t_g)

    def __repr__(self) -> str:  # noqa: D105
        return f"Chirp(coeffs_GHz={list(self.coeffs_GHz)}, t_g={self.t_g})"

    @property
    def is_trivial(self) -> bool:
        """True if this chirp is identically zero (so callers can skip the phase)."""
        return self.coeffs_GHz.size == 0 or not np.any(self.coeffs_GHz)

    def _u(self, t: Any, xp: Any):
        """Normalized gate time u = 2t/t_g - 1, clipped to [-1, 1]."""
        return xp.clip(2.0 * xp.asarray(t) / self.t_g - 1.0, -1.0, 1.0)

    def detuning(self, t: Any, xp: Any = np) -> Any:
        """Instantaneous frequency offset delta(t) in rad/ns."""
        n = self.coeffs_GHz.size
        if n == 0:
            return 0.0 * xp.asarray(t)
        u = self._u(t, xp)
        P = _legendre_stack(u, n - 1, xp)
        total = sum(c * P[k] for k, c in enumerate(self.coeffs_GHz))
        return TWO_PI * total

    def phase(self, t: Any, xp: Any = np) -> Any:
        """Accumulated chirp phase Phi(t) = int_0^t delta(t') dt', in radians."""
        n = self.coeffs_GHz.size
        if n == 0:
            return 0.0 * xp.asarray(t)
        u = self._u(t, xp)
        # need P up to degree n (the k = n-1 term uses P_{k+1} = P_n)
        P = _legendre_stack(u, n, xp)
        total = self.coeffs_GHz[0] * (u + 1.0)
        for k in range(1, n):
            total = total + self.coeffs_GHz[k] * (P[k + 1] - P[k - 1]) / (2 * k + 1)
        return np.pi * self.t_g * total

    # -- flat parameter-vector interface for optimizers --------------------
    @property
    def n_params(self) -> int:
        """Number of free chirp coefficients."""
        return int(self.coeffs_GHz.size)

    def get_params(self) -> np.ndarray:
        """Chirp coefficients (GHz) as a flat real vector."""
        return self.coeffs_GHz.copy()

    def set_params(self, p) -> None:
        """Load a flat real vector as produced by :meth:`get_params`."""
        p = np.asarray(p, dtype=float).ravel()
        if p.size != self.coeffs_GHz.size:
            raise ValueError(f"expected {self.coeffs_GHz.size} chirp coefficients, "
                             f"got {p.size}")
        self.coeffs_GHz = p.copy()


def make_chirp(coeffs_GHz: Optional[Sequence[float]], t_g: float) -> Optional[Chirp]:
    """Build a :class:`Chirp`, or None when there is nothing to apply.

    Returning None for an absent/zero chirp keeps the un-chirped solver path
    byte-identical to before this feature existed (``_eta`` skips the phase
    entirely), which is what makes the "chirp is inert when unset" regression
    test meaningful.
    """
    if coeffs_GHz is None:
        return None
    chirp = Chirp(coeffs_GHz, t_g)
    return None if chirp.is_trivial else chirp


# ===========================================================================
# Pump tone
# ===========================================================================
@dataclass
class PumpTone:
    """One pump tone applied to the coupler.

    Parameters
    ----------
    w_p_GHz : float
        Pump frequency f_p (GHz); the dressed amplitude oscillates at this rate
        inside X(t). This is the FIXED reference carrier -- any time dependence
        of the frequency belongs in `chirp`, not here.
    envelope : Envelope
        Shape eps_p(t). Its `amp` carries |eps_p| (rad/ns), or directly the
        dimensionless displaced amplitude |eta_p| when `is_eta` is True.
    phi_p : float, default 0.0
        Pump phase (rad); enters as eta_p -> |eta_p| e^{i phi_p}.
    is_eta : bool, default True
        If True the envelope amplitude is already eta_p; if False it is the bare
        pump eps_p and eta_p is computed from eta = 2 w_p/(w_p^2 - w_s^2) eps
        (Eq. 50).
    drag : bool, default False
        If True, add the first-order DRAG quadrature (Motzoi et al., PRL 103,
        110501 (2009)): eta(t) -> eta(t) - i d/dt[eta(t)] / delta. The derivative
        quadrature cancels, to leading order, the adiabatic excitation of a
        process detuned by `delta_drag_GHz`. It suppresses an OFF-resonant
        spectator and is singular as the detuning -> 0 (an on-resonant collision
        needs frequency allocation, not DRAG).
    delta_drag_GHz : float, optional
        Detuning delta_s (GHz) of the targeted off-resonant process; the
        quadrature is -d eta/dt / (2 pi delta_drag_GHz). Required if `drag`.
    chirp : Chirp, optional
        Time-dependent offset delta(t) of the carrier. Applied as the phase
        e^{-i Phi(t)} on the pump amplitude AFTER the DRAG quadrature -- see
        `ZhouCoupler._eta`. None (default) means an un-chirped tone.

    Notes
    -----
    Sign conventions differ between the two phases here, for historical reasons:
    `phi_p` enters as ``e^{+i phi_p}`` while the chirp enters as ``e^{-i Phi(t)}``,
    which is the sign that matches the ``e^{-i w_p t}`` carrier in X(t). A constant
    chirp `c0` therefore shifts the carrier to ``w_p + c0``, not ``w_p - c0``.
    """

    w_p_GHz: float
    envelope: Envelope
    phi_p: float = 0.0
    is_eta: bool = True
    drag: bool = False
    delta_drag_GHz: Optional[float] = None
    chirp: Optional[Chirp] = None
