import numpy as np

TWO_PI = 2 * np.pi
# ---------------------------------------------------------------------------
# Pump envelopes (numpy-only; same value/deriv/area API as snail_parametric_sim)
# ---------------------------------------------------------------------------
class Envelope:
    """Real pump-amplitude envelope eps(t) on [0, t_g]. Peak-normalized shapes
    carry their scale in `amp`."""

    def __init__(self, amp: float, t_g: float):
        self.amp = float(amp)
        self.t_g = float(t_g)

    def value(self, t: float) -> float:
        raise NotImplementedError

    def deriv(self, t: float) -> float:
        raise NotImplementedError

    def area(self) -> float:
        ts = np.linspace(0.0, self.t_g, 4001)
        return float(np.trapz([self.value(t) for t in ts], ts))


class ConstantPulse(Envelope):
    """Flat pump (with optional instantaneous on/off). Useful for rate checks."""

    def value(self, t):
        return self.amp if 0.0 <= t <= self.t_g else 0.0

    def deriv(self, t):
        return 0.0

    def area(self):
        return self.amp * self.t_g


class RaisedCosine(Envelope):
    """Hann (raised-cosine) envelope: smooth turn-on/off, zero endpoints."""

    def value(self, t):
        if t < 0.0 or t > self.t_g:
            return 0.0
        return self.amp * 0.5 * (1.0 - np.cos(TWO_PI * t / self.t_g))

    def deriv(self, t):
        if t < 0.0 or t > self.t_g:
            return 0.0
        return self.amp * 0.5 * (TWO_PI / self.t_g) * np.sin(TWO_PI * t / self.t_g)

    def area(self):
        return self.amp * self.t_g / 2.0  # exact integral of the Hann window

class GaussianFlat(Envelope):
    r"""Truncated Gaussian, baseline-subtracted so it vanishes at the endpoints.

    Omega(t) = amp * (G(t) - G(0)) / (1 - G(0)),  G(t) = exp(-(t - t_g/2)^2 / (2 sigma^2))
    """

    def __init__(self, amp, t_g, sigma_frac: float = 0.25):
        super().__init__(amp, t_g)
        self.sigma = sigma_frac * t_g
        self._g0 = np.exp(-((0.0 - t_g / 2) ** 2) / (2 * self.sigma**2))
        self._norm = 1.0 / (1.0 - self._g0)

    def _G(self, t):
        return np.exp(-((t - self.t_g / 2) ** 2) / (2 * self.sigma**2))

    def value(self, t):
        if t <= 0.0 or t >= self.t_g:
            return 0.0
        return self.amp * (self._G(t) - self._g0) * self._norm

    def deriv(self, t):
        if t <= 0.0 or t >= self.t_g:
            return 0.0
        dG = -((t - self.t_g / 2) / self.sigma**2) * self._G(t)
        return self.amp * dG * self._norm
