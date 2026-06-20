# ===========================================================================
# Pump envelopes  (real shape eps(t) on [0, t_g]; peak scale carried in `amp`)
# ===========================================================================
from typing import Callable

import numpy as np
# numpy>=2 renamed trapz -> trapezoid; fall back only if needed (trapz is gone in 2.x).
_trapezoid: Callable[..., float] = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

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