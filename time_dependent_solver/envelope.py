# ===========================================================================
# Pump envelopes  (real shape eps(t) on [0, t_g]; peak scale carried in `amp`)
# ===========================================================================
from typing import Callable

import numpy as np
# numpy>=2 renamed trapz -> trapezoid; fall back only if needed (trapz is gone in 2.x).
_trapezoid: Callable[..., float] = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
TWO_PI = 2.0 * np.pi

class Envelope:
    """Base class for a real pump-amplitude envelope eps(t) on [0, t_g].

    Subclasses implement `value` and `deriv`; `area` integrates eps over the gate
    (subclasses with closed forms override it).

    Parameters
    ----------
    amp : float
        Peak amplitude. Carries |eps| in rad/ns, or directly |eta| when the
        owning PumpTone has ``is_eta=True``.
    t_g : float
        Gate duration in ns; the envelope is supported on [0, t_g].
    """

    def __init__(self, amp: float, t_g: float) -> None:
        self.amp: float = float(amp)
        self.t_g: float = float(t_g)

    def value(self, t: float) -> float:
        """Envelope amplitude at time `t` (ns). Subclasses must implement."""
        raise NotImplementedError

    def deriv(self, t: float) -> float:
        """Time derivative d eps/dt at time `t` (ns). Subclasses must implement."""
        raise NotImplementedError

    def area(self) -> float:
        """Return integral_0^{t_g} value(t) dt by quadrature (override if a closed
        form exists).
        """  # noqa: D205
        ts = np.linspace(0.0, self.t_g, 4001)
        return float(_trapezoid([self.value(t) for t in ts], ts))


class ConstantPulse(Envelope):
    """Flat pump with instantaneous on/off; handy for steady-state rate checks."""

    def value(self, t: float) -> float:
        """Amplitude `amp` for t in [0, t_g], else 0."""
        return self.amp if 0.0 <= t <= self.t_g else 0.0

    def deriv(self, t: float) -> float:
        """Zero everywhere (flat pulse)."""
        return 0.0

    def area(self) -> float:
        """Closed form: amp * t_g."""
        return self.amp * self.t_g


class RaisedCosine(Envelope):
    """Hann (raised-cosine) envelope: smooth turn-on/off, vanishing endpoints,.

    value(t) = amp/2 [1 - cos(2 pi t / t_g)] ,  t in [0, t_g] .
    """

    def value(self, t: float) -> float:
        """Hann amplitude at `t` (ns); 0 outside [0, t_g]."""
        if not (0.0 <= t <= self.t_g):
            return 0.0
        return self.amp * 0.5 * (1.0 - np.cos(TWO_PI * t / self.t_g))

    def deriv(self, t: float) -> float:
        """Analytic derivative of the Hann window at `t` (ns); 0 outside [0, t_g]."""
        if not (0.0 <= t <= self.t_g):
            return 0.0
        return self.amp * 0.5 * (TWO_PI / self.t_g) * np.sin(TWO_PI * t / self.t_g)

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

    This is the ansatz container used by the CRAB / GOAT optimizers in
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
    def _mod(self, t: float) -> complex:
        """Modulation M(t); 1 when there is no basis (plain raised cosine)."""
        if self.freqs.size == 0:
            return 1.0 + 0.0j
        s, c = np.sin(self.freqs * t), np.cos(self.freqs * t)
        return (1.0 + float(self.sin_I @ s) + float(self.cos_I @ c)
                + 1j * (float(self.sin_Q @ s) + float(self.cos_Q @ c)))

    def _dmod(self, t: float) -> complex:
        """dM/dt."""
        if self.freqs.size == 0:
            return 0.0 + 0.0j
        w = self.freqs
        s, c = np.sin(w * t), np.cos(w * t)
        return (float((self.sin_I * w) @ c) - float((self.cos_I * w) @ s)
                + 1j * (float((self.sin_Q * w) @ c) - float((self.cos_Q * w) @ s)))

    def _shape(self, t: float) -> float:
        """Hann shape function S(t)."""
        return 0.5 * (1.0 - np.cos(TWO_PI * t / self.t_g))

    def _dshape(self, t: float) -> float:
        """dS/dt."""
        return 0.5 * (TWO_PI / self.t_g) * np.sin(TWO_PI * t / self.t_g)

    def value(self, t: float) -> complex:
        """Complex envelope amplitude at `t` (ns); 0 outside [0, t_g]."""
        if not (0.0 <= t <= self.t_g):
            return 0.0 + 0.0j
        return self.amp * self._shape(t) * self._mod(t)

    def deriv(self, t: float) -> complex:
        """Analytic d(eta)/dt at `t` (ns); 0 outside [0, t_g]."""
        if not (0.0 <= t <= self.t_g):
            return 0.0 + 0.0j
        return self.amp * (self._dshape(t) * self._mod(t)
                           + self._shape(t) * self._dmod(t))

    def area(self) -> float:
        """Integral of Re[eta] over the gate (see Notes); closed form when the
        basis is empty."""
        if self.freqs.size == 0:
            return self.amp * self.t_g / 2.0
        ts = np.linspace(0.0, self.t_g, 4001)
        return float(_trapezoid([self.value(t).real for t in ts], ts))

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

    def samples(self, n: int):
        """`n` midpoint samples of eta(t) across the gate (for plotting/storage)."""
        ts = (np.arange(int(n)) + 0.5) * (self.t_g / int(n))
        return np.array([self.value(float(t)) for t in ts], dtype=complex)