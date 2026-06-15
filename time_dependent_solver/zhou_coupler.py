"""
zhou_coupler.py
===============

Charge-pumped parametric coupler in the dressed-mode framework of

    Chao Zhou, "Quantum Operations with Charge-pumped Parametric Interactions,"
    PhD thesis, University of Pittsburgh (2023), Chapter 2.

This module builds the *interaction-picture* Hamiltonian directly from
EXPERIMENTAL inputs -- the coupler's intrinsic non-linearity g_n (g3 for a
SNAIL three-wave mixer) and the measured participations

        lambda_is = g_is / Delta_is                                   (Eq. 57)

of each mode in the dressed coupler mode. No effective transmon-transmon edge
is assumed: the star connectivity is encoded entirely in the participations,
because the dressed coupler operator is (Eq. 56)

        s' ~= s + sum_i lambda_is a_i .

Master Hamiltonian (Zhou Eqs. 54 / 60; qubit form Eq. 72)
---------------------------------------------------------
        H_I(t)/hbar = sum_n  g_n  X(t)^n ,

        X(t) = s e^{-i w_s t}
             + sum_i lambda_is a_i e^{-i w_i t}
             + sum_p eta_p(t) e^{-i w_p t}
             + h.c. ,

with the displaced pump amplitude (Eq. 50)

        eta_p(t) = 2 w_p / (w_p^2 - w_s^2) * eps_p(t) .

Expanding X(t)^n generates EVERY multi-wave-mixing process at once (iSWAP,
bSWAP, sub-harmonic single-qubit drive, cross-Kerr, and all spectators /
higher orders). Pumping at  k w_p = w_rot  (Eq. 61) makes one process static
under the RWA; its strength is (Eq. 62)

        g_eff = C g_n eta^k prod_i lambda_is ,

with C the multinomial coefficient. The canonical two-qubit example: pumping a
three-wave coupler at w_p = w_b - w_a gives the iSWAP family with (Eqs. 55/73)

        g_eff = 6 g3 lambda_as lambda_bs |eta| .          (verified in __main__)

g_n is the OVERALL prefactor of the dynamics: set g3 = 0 and there is no gate.
This is the key correction relative to a phenomenological effective-edge model.

Conventions
-----------
* Frequencies/non-linearities are entered in GHz and converted to rad/ns
  internally (omega = 2 pi f).  Time is in ns.
* A mode with 2 levels reproduces Zhou's sigma_- (qubit) form (Eq. 72); use
  >= 3 levels to keep it an oscillator/transmon (then leakage is captured).
* The coupler S is kept as a dynamical mode (it can be virtually/really excited
  -- the qubit-coupler spectator channel is therefore automatic).

This module is numpy/scipy only so it is self-contained and testable; call
`to_qutip_hamiltonian()` for a QobjEvo that plugs into a QuTiP solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Callable

import numpy as np
from scipy.integrate import solve_ivp

from envelope import Envelope, ConstantPulse, RaisedCosine, GaussianFlat

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------------------
# Pump tones
# ---------------------------------------------------------------------------
@dataclass
class PumpTone:
    """One pump tone applied to the coupler.

    w_p_GHz   : pump frequency f_p (GHz). The dressed amplitude oscillates at
                this rate inside X(t).
    envelope  : real shape eps_p(t); its `amp` carries |eps_p| (rad/ns) OR, if
                `is_eta` is True, directly |eta_p| (dimensionless displaced amp).
    phi_p     : pump phase (rad); enters as eta_p -> |eta_p| e^{i phi_p}.
    is_eta    : if True the envelope amplitude is already the displaced eta_p;
                if False it is the bare pump eps_p and eta_p is computed from
                eta = 2 w_p/(w_p^2 - w_s^2) eps (Eq. 50).
    drag      : if True, add the first-order DRAG quadrature (Motzoi et al.,
                PRL 103, 110501 (2009)): eta(t) -> eta(t) - i d/dt[eta(t)]/delta.
                This is the in-phase amplitude plus a derivative quadrature that
                cancels, to leading order, the adiabatic (inertial) excitation of
                an off-resonant process detuned by `delta_drag_GHz`. Useful for
                suppressing a spectator that is OFF resonance; it is singular as
                the detuning -> 0 (an on-resonant collision needs frequency
                allocation, not DRAG).
    delta_drag_GHz : detuning delta_s (GHz) of the targeted off-resonant process;
                the quadrature is -d eta/dt / (2 pi delta_drag_GHz).
    """

    w_p_GHz: float
    envelope: Envelope
    phi_p: float = 0.0
    is_eta: bool = True
    drag: bool = False
    delta_drag_GHz: Optional[float] = None


# ---------------------------------------------------------------------------
# Pure-numpy gate-metric helpers (shared conventions with snail_parametric_sim)
# ---------------------------------------------------------------------------
def _ideal_iswap() -> np.ndarray:
    U = np.eye(4, dtype=complex)
    U[1, 1] = 0.0
    U[2, 2] = 0.0
    U[1, 2] = 1j
    U[2, 1] = 1j
    return U


def _fit_virtual_z(U: np.ndarray, U_id: np.ndarray) -> np.ndarray:
    """Maximize |Tr(U_id^d Z U)| over single-qubit virtual-Z phases (free in
    software; McKay et al., PRA 96, 022330 (2017)). Coarse grid + local refine."""
    def Z(pa, pb):
        return np.diag([1.0, np.exp(1j * pb), np.exp(1j * pa), np.exp(1j * (pa + pb))])

    def score(pa, pb):
        return np.abs(np.trace(U_id.conj().T @ (Z(pa, pb) @ U))) ** 2

    best, best_s = (0.0, 0.0), -1.0
    coarse = np.linspace(0, TWO_PI, 48, endpoint=False)
    for pa in coarse:
        for pb in coarse:
            s = score(pa, pb)
            if s > best_s:
                best_s, best = s, (pa, pb)
    span = TWO_PI / 48
    for _ in range(3):
        for pa in np.linspace(best[0] - span, best[0] + span, 21):
            for pb in np.linspace(best[1] - span, best[1] + span, 21):
                s = score(pa, pb)
                if s > best_s:
                    best_s, best = s, (pa, pb)
        span /= 10.0
    return Z(*best) @ U


# ---------------------------------------------------------------------------
# The coupler
# ---------------------------------------------------------------------------
class ZhouCoupler:
    r"""Parametric coupler built from experimental g_n and participations.

    Parameters
    ----------
    mode_freqs_GHz : sequence of float
        Frequencies of ALL modes, INCLUDING the coupler S (GHz). After the
        Bogoliubov dressing these are the dressed mode frequencies (Zhou writes
        omega' -> omega), i.e. the *measured* mode frequencies.
    coupler_index : int
        Index of the coupler mode S in the lists above.
    participations : dict[int, float]
        lambda_is = g_is/Delta_is for each non-coupler mode that participates in
        the dressed coupler (Eq. 57; use Eq. 58 for chained modes and pass the
        product). Modes omitted here do not appear in X(t) (decoupled). The
        coupler's own participation is 1 by definition.
    nonlinearities : dict[int, float]
        {n: g_n} in GHz: the coupler's non-linear coefficients. {3: g3} for a
        pure three-wave SNAIL; add {4: g4} to include four-wave mixing.
    levels : int | sequence[int]
        Per-mode truncation. 2 -> qubit (sigma_-, Eq. 72); >=3 -> oscillator.
    """

    def __init__(
        self,
        mode_freqs_GHz: Sequence[float],
        coupler_index: int,
        participations: Dict[int, float],
        nonlinearities: Dict[int, float],
        levels=3,
    ):
        self.w = np.asarray(mode_freqs_GHz, dtype=float) * TWO_PI  # rad/ns
        self.N = self.w.size
        self.coupler = int(coupler_index)
        if not (0 <= self.coupler < self.N):
            raise ValueError("coupler_index out of range.")

        if np.isscalar(levels):
            self.dims = [int(levels)] * self.N
        else:
            self.dims = [int(x) for x in levels]
            if len(self.dims) != self.N:
                raise ValueError("per-mode levels length must equal #modes.")
        self.D = int(np.prod(self.dims))

        # participation vector: 1 on the coupler, lambda_is on coupled modes, 0 else
        self.lam = np.zeros(self.N, dtype=float)
        self.lam[self.coupler] = 1.0
        for i, val in participations.items():
            if i == self.coupler:
                raise ValueError("Do not list the coupler in participations (it is 1).")
            if not (0 <= i < self.N):
                raise ValueError(f"participation index {i} out of range.")
            self.lam[i] = float(val)

        self.gn = {int(n): float(g) * TWO_PI for n, g in nonlinearities.items()}  # rad/ns
        if not self.gn:
            raise ValueError("Provide at least one non-linearity, e.g. {3: g3_GHz}.")

        # dense ladder operators (mixed radix), built once
        self.a = [self._embed(self._destroy(self.dims[i]), i) for i in range(self.N)]
        self.ad = [op.conj().T for op in self.a]
        self.Id = np.eye(self.D, dtype=complex)

        # components of X(t): (operator, signed_frequency, constant_amplitude)
        # built once; the pump components are added at solve time (time-dependent).
        self._static_components = []
        for i in range(self.N):
            if self.lam[i] == 0.0:
                continue
            self._static_components.append((self.a[i], -self.w[i], self.lam[i]))
            self._static_components.append((self.ad[i], +self.w[i], self.lam[i]))

        self._pumps: List[PumpTone] = []

    # -- mode-space helpers -------------------------------------------------
    @staticmethod
    def _destroy(d):
        return np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)

    def _embed(self, op, m):
        mats = [np.eye(d, dtype=complex) for d in self.dims]
        mats[m] = op
        out = mats[0]
        for x in mats[1:]:
            out = np.kron(out, x)
        return out

    def fock_index(self, occ: Sequence[int]) -> int:
        k = 0
        for n, d in zip(occ, self.dims):
            if not (0 <= n < d):
                raise ValueError("occupation out of range.")
            k = k * d + n
        return k

    def decode_index(self, k: int) -> list:
        occ = []
        for d in reversed(self.dims):
            k, r = divmod(k, d)
            occ.append(r)
        return occ[::-1]

    def mean_occupation(self, probs: np.ndarray, mode: int) -> float:
        return float(sum(probs[k] * self.decode_index(k)[mode] for k in range(self.D)))

    def basis_state(self, occ: Sequence[int]) -> np.ndarray:
        psi = np.zeros(self.D, dtype=complex)
        psi[self.fock_index(occ)] = 1.0
        return psi

    def delta(self, i: int, j: int) -> float:
        return float(self.w[i] - self.w[j])

    # -- pump ---------------------------------------------------------------
    def set_pump(self, tones, normalize_iswap: Optional[tuple] = None) -> None:
        """Attach one or more PumpTones. If `normalize_iswap=(a,b)` is given, the
        amplitude of the FIRST tone is rescaled so that, pumped at w_b - w_a, the
        time-integrated rate gives a full iSWAP on the (a,b) pair (theta=pi/2)."""
        if isinstance(tones, PumpTone):
            tones = [tones]
        self._pumps = list(tones)

        if normalize_iswap is not None:
            a, b = normalize_iswap
            g3 = self.gn.get(3, 0.0)
            if g3 == 0.0:
                raise ValueError("normalize_iswap needs a three-wave g3.")
            tone = self._pumps[0]
            ws = self.w[self.coupler]
            wp = tone.w_p_GHz * TWO_PI
            # |eta(t)| = pref * eps(t); full iSWAP: integral 6 g3 la lb |eta| dt = pi/2
            pref = 1.0 if tone.is_eta else abs(2 * wp / (wp ** 2 - ws ** 2))
            la, lb = self.lam[a], self.lam[b]
            if la == 0.0 or lb == 0.0:
                raise ValueError("Both qubits must have nonzero participation.")
            target_area = (np.pi / 2) / (6 * g3 * la * lb)   # required integral of |eta| dt
            cur_area = pref * tone.envelope.area()
            if cur_area == 0.0:
                raise ValueError("Pump envelope has zero area.")
            tone.envelope.amp *= target_area / cur_area

    def _eta(self, tone: PumpTone, t: float) -> complex:
        ws = self.w[self.coupler]
        wp = tone.w_p_GHz * TWO_PI
        pref = 1.0 if tone.is_eta else (2 * wp / (wp ** 2 - ws ** 2))
        amp = tone.envelope.value(t)
        if tone.drag and tone.delta_drag_GHz not in (None, 0.0):
            delta = tone.delta_drag_GHz * TWO_PI                  # rad/ns
            amp = amp - 1j * tone.envelope.deriv(t) / delta       # 1st DRAG quadrature
        return pref * amp * np.exp(1j * tone.phi_p)

    # -- the dressed flux and the Hamiltonian -------------------------------
    def dressed_flux(self, t: float) -> np.ndarray:
        r"""X(t) = s e^{-i w_s t} + sum_i lambda_is a_i e^{-i w_i t}
                   + sum_p eta_p(t) e^{-i w_p t} + h.c.   (Hermitian)."""
        X = (self.a[self.coupler] * np.exp(-1j * self.w[self.coupler] * t)
             + self.ad[self.coupler] * np.exp(1j * self.w[self.coupler] * t))
        for (op, freq, amp) in self._static_components:
            X = X + amp * op * np.exp(1j * freq * t)
        for tone in self._pumps:
            wp = tone.w_p_GHz * TWO_PI
            eta = self._eta(tone, t)
            X = X + self.Id * (eta * np.exp(-1j * wp * t)
                               + np.conj(eta) * np.exp(1j * wp * t))
        return X

    def hamiltonian_matrix(self, t: float) -> np.ndarray:
        """H_I(t)/hbar = sum_n g_n X(t)^n  (dense)."""
        X = self.dressed_flux(t)
        H = np.zeros((self.D, self.D), dtype=complex)
        # accumulate powers once: X^1, X^2, ... up to the highest non-linearity
        powers = {1: X}
        cur = X
        for n in range(2, max(self.gn) + 1):
            cur = cur @ X
            powers[n] = cur
        for n, g in self.gn.items():
            H = H + g * powers[n]
        return H

    # -- analytic effective-rate estimator (Eq. 62) ------------------------
    def effective_rate(self, modes: Sequence[int], n: int,
                        C: Optional[int] = None, eta: Optional[float] = None) -> float:
        r"""RWA process strength g_eff = C g_n eta^k prod_i lambda_is (Eq. 62).
        `modes` lists the participating non-coupler modes (with multiplicity);
        k = n - len(modes) is the number of pump quanta. Pass C explicitly for
        non-standard processes; default uses the multinomial count for distinct
        single-mode + pump factors."""
        gn = self.gn.get(n)
        if gn is None:
            raise ValueError(f"no g_{n} defined.")
        k = n - len(modes)
        if k < 0:
            raise ValueError("len(modes) cannot exceed n.")
        if eta is None:
            if not self._pumps:
                raise ValueError("Set a pump or pass eta explicitly.")
            eta = abs(self._eta(self._pumps[0], self._pumps[0].envelope.t_g / 2))
        prod_lam = float(np.prod([self.lam[m] for m in modes])) if modes else 1.0
        if C is None:
            from math import factorial
            C = factorial(n)  # distinct factors (e.g. s*,a,b + k pumps) -> n! for n distinct
        return float(C * gn * (eta ** k) * prod_lam)

    def iswap_rate(self, a: int, b: int, eta: Optional[float] = None) -> float:
        """g_eff = 6 g3 lambda_as lambda_bs |eta|  (Eqs. 55/73)."""
        return self.effective_rate([a, b], n=3, C=6, eta=eta)

    def peak_eta(self, tone_index: int = 0) -> float:
        """Magnitude of the displaced pump amplitude |eta| at the envelope peak."""
        tone = self._pumps[tone_index]
        return abs(self._eta(tone, tone.envelope.t_g / 2.0))

    # -- evolution (dense scipy; no QuTiP needed) ---------------------------
    def evolve(self, psi0: np.ndarray, t_span, rtol=1e-9, atol=1e-11, max_step=np.inf):
        sol = solve_ivp(lambda t, y: -1j * (self.hamiltonian_matrix(t) @ y),
                        (float(t_span[0]), float(t_span[-1])), psi0,
                        t_eval=t_span, rtol=rtol, atol=atol, max_step=max_step,
                        method="RK45")
        return sol

    def propagator_columns(self, a: int, b: int, t_g: float,
                           rtol=1e-9, atol=1e-11, max_step=np.inf) -> np.ndarray:
        """4x4 projection of the realized propagator onto the (a,b) computational
        subspace, all other modes initialized in |0> (Pedersen/Wood convention)."""
        comp = [(0, 0), (0, 1), (1, 0), (1, 1)]
        idx = []
        for (ba, bb) in comp:
            occ = [0] * self.N
            occ[a], occ[b] = ba, bb
            idx.append(self.fock_index(occ))
        U = np.zeros((4, 4), dtype=complex)
        for col, k0 in enumerate(idx):
            psi0 = np.zeros(self.D, dtype=complex)
            psi0[k0] = 1.0
            sol = solve_ivp(lambda t, y: -1j * (self.hamiltonian_matrix(t) @ y),
                            (0.0, t_g), psi0, rtol=rtol, atol=atol,
                            max_step=max_step, method="RK45")
            psif = sol.y[:, -1]
            for row, kr in enumerate(idx):
                U[row, col] = psif[kr]
        return U

    def iswap_fidelity(self, a: int, b: int, t_g: float, fit_virtual_z: bool = True,
                       **kw):
        """Average gate fidelity vs ideal iSWAP on (a,b), leakage-aware
        (Pedersen PLA 367, 47 (2007); Wood & Gambetta PRA 97, 032306 (2018)),
        single-qubit virtual-Z fitted out by default."""
        U_proj = self.propagator_columns(a, b, t_g, **kw)
        d = 4
        U_id = _ideal_iswap()
        U_s = _fit_virtual_z(U_proj, U_id) if fit_virtual_z else U_proj
        overlap = np.abs(np.trace(U_id.conj().T @ U_s)) ** 2
        trUU = np.real(np.trace(U_s.conj().T @ U_s))
        F = (overlap + trUU) / (d * (d + 1))
        leak = 1.0 - trUU / d
        return float(F), float(leak), U_proj

    # -- QuTiP export (lazy import; plugs into existing solver tooling) -----
    def to_qutip_hamiltonian(self):
        """Return a QuTiP QobjEvo H(t) = sum_n g_n X(t)^n for use with QuTiP
        solvers. Imported lazily so this module has no hard QuTiP dependency."""
        import qutip as qt

        dims = [self.dims, self.dims]

        def H_func(t, args=None):
            return qt.Qobj(self.hamiltonian_matrix(t), dims=dims)

        try:
            return qt.QobjEvo(H_func)          # QuTiP 5 accepts a callable
        except Exception:
            return H_func                       # QuTiP 4: pass callable as H


# ---------------------------------------------------------------------------
# Demo / self-test: reproduce Zhou's iSWAP coupling g_eff = 6 g3 la lb |eta|
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # two qubits (a, b) + a SNAIL coupler (s); qubits are 2-level (Eq. 72 form)
    cpl = ZhouCoupler(
        mode_freqs_GHz=[4.0, 5.0, 7.0],
        coupler_index=2,
        participations={0: 0.15, 1: 0.15},     # lambda_as, lambda_bs = g_is/Delta_is
        nonlinearities={3: 0.10},              # g3 = 100 MHz
        levels=[2, 2, 6],
    )
    # one pump tone at w_b - w_a activates the iSWAP family (Eq. 73).
    # Use a flat pump long enough to cover the DC-averaging window below.
    wp_GHz = cpl.delta(1, 0) / TWO_PI          # f_b - f_a
    cpl.set_pump(PumpTone(w_p_GHz=wp_GHz, envelope=ConstantPulse(0.06, 50.0), is_eta=True))

    # DC (time-averaged) <ge|H|eg> should equal g_eff = 6 g3 la lb |eta|
    eg, ge = cpl.fock_index([1, 0, 0]), cpl.fock_index([0, 1, 0])
    T = TWO_PI / cpl.delta(1, 0)                # pump period
    ts = np.linspace(0, 40 * T, 8000, endpoint=False)   # within pump-on window
    M = np.mean([cpl.hamiltonian_matrix(t)[ge, eg] for t in ts])
    print(f"DC <ge|H|eg|/2pi          = {abs(M)/TWO_PI*1e3:.4f} MHz")
    print(f"Zhou 6 g3 la lb |eta| /2pi = {cpl.iswap_rate(0, 1)/TWO_PI*1e3:.4f} MHz")
    print(f"ratio                      = {abs(M)/cpl.iswap_rate(0,1):.5f}  (expect 1.0)")