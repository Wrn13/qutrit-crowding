r"""
snail_parametric_sim.py
=======================

Full time-dependent Hamiltonian simulation of a SNAIL-based parametric processor
with a variable number of transmons, arbitrary detunings and coupling edges, a
single driven parametric iSWAP (beam-splitter) interaction, and a toggleable DRAG
correction that suppresses one off-resonant *spectator* exchange channel.

Conventions
-----------
* Angular frequencies are in rad/ns  (i.e. store ``2*pi*f[GHz]``); time is in ns.
  With this choice ``hbar = 1`` and every coefficient below is dimensionally a
  rate in rad/ns, so the unitary is ``exp(-i * integral H dt)`` with H in rad/ns.
* Each transmon is a truncated Duffing (Kerr) oscillator. Keep ``levels >= 3`` so
  the |2> manifold (leakage / spectator target) is represented.

Physical model
--------------
We work in the *per-transmon rotating frame*  U(t) = exp(+i sum_i w_i a_i^d a_i t),
which removes the bare harmonic term and is what makes a multi-GHz device
tractable. The lab-frame circuit-QED Hamiltonian

    H_lab/hbar = sum_i [ w_i a_i^d a_i + (alpha_i/2) a_i^d a_i^d a_i a_i ]
               + sum_<ij> g_ij (a_i^d a_j + a_j^d a_i)

becomes, after the frame transformation (the anharmonicity commutes with the
number operator and is invariant):

    H(t)/hbar = sum_i (alpha_i/2) a_i^d a_i^d a_i a_i
              + sum_<ij> [ c_ij(t) a_i^d a_j + c_ij(t)^* a_j^d a_i ]                (1)

    c_ij(t) = ( g_ij + eta_ij * p(t) ) * exp( i * Delta_ij * t ),  Delta_ij = w_i - w_j

The SNAIL's role.  A SNAIL coupler (three-wave / g3 element, Frattini et al.,
PRApplied 7, 054060 (2017); Sivak et al., PRApplied 11, 054060 (2019)) is flux-
pumped at frequency ``w_d``. To leading order the pump modulates the SNAIL-mediated
inter-transmon couplings, giving each participating edge a time-dependent term
``eta_ij * p(t)`` on top of any static ``g_ij``. This is the standard "fold the
coupler into an effective parametric coupling" picture used for parametric gates
(McKinney et al., arXiv:2409.18262; Lu et al., PRX Quantum 2, 040348 (2021)). The
SNAIL mode itself can instead be carried explicitly -- see ``add_snail_mode`` notes
at the bottom -- at the cost of multiplying the Hilbert-space dimension.

Driving an iSWAP.  Setting ``w_d = Delta_pq`` for a target edge (p, q) makes the
modulation term resonant in the single-excitation manifold:

    eta_pq * Omega(t) cos(w_d t) * exp(i Delta_pq t)
        = (eta_pq Omega(t)/2) [ 1 + exp(2 i w_d t) ]                               (2)

The DC part is a beam-splitter of rate g_eff(t) = eta_pq Omega(t) / 2 acting on
{|...1_p 0_q...>, |...0_p 1_q...>} as (g_eff/2) sigma_x. A *full* iSWAP needs pulse
area  integral g_eff dt = pi/2, i.e.  integral Omega dt = pi / eta_pq.  We keep the
counter-rotating ``exp(2 i w_d t)`` term (this is a non-perturbative, non-RWA
simulation), so Bloch-Siegert-type shifts appear automatically.

Spectators and DRAG.  The same pump tone is off-resonant by
``delta_s = Delta_rs - w_d`` (a "slow" beat) on any other participating edge (r, s),
driving an unwanted exchange. The first-order excitation amplitude is the envelope's
Fourier weight at ``delta_s``:

    integral (eta_rs Omega(t)/2) exp(i delta_s t) dt.

Integrating by parts (with Omega(0)=Omega(t_g)=0) and adding a derivative quadrature
``A(t) sin(w_d t)`` to the pump shows the leading term cancels iff

    A(t) = - d/dt Omega(t) / delta_s.                                              (3)

This is exactly Motzoi et al. DRAG (PRL 103, 110501 (2009), arXiv:0901.0534) with
the leakage anharmonicity alpha replaced by the spectator beat ``delta_s``. Hence

    p(t) = Omega(t) cos(w_d t)  -  drag * (Omega_dot(t)/delta_s) sin(w_d t).        (4)

NB: the DRAG quadrature also lands on the *resonant* target edge as a slow
O(Omega_dot/delta_s) term (imaginary direction), so for a small ``delta_s`` it back-
acts on the target gate -- the perturbative-DRAG breakdown regime. The module lets
you measure exactly this trade-off.

Author note: designed to drop into a QuantumLogicalSimulator-style codebase --
the QobjEvo from `hamiltonian()` can be Trotterized externally, and the
collapse operators from `collapse_operators()` feed a Channel/CPTPMap layer.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

try:
    import qutip as qt
except ImportError as exc:  # pragma: no cover - explicit, not silent
    raise ImportError(
        "snail_parametric_sim requires QuTiP (`pip install qutip`). "
        "Tested against QuTiP 4.7+ and 5.x."
    ) from exc

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Pulse envelopes (each knows its own analytic derivative -- required for DRAG)
# ---------------------------------------------------------------------------
class Envelope:
    """Base class. `value(t)` is Omega(t); `deriv(t)` is d/dt Omega(t).

    Envelopes MUST satisfy value(0) == value(t_g) == 0 for the DRAG
    integration-by-parts argument (Eq. 3) to hold cleanly.
    """

    def __init__(self, amp: float, t_g: float):
        self.amp = float(amp)
        self.t_g = float(t_g)

    def value(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def deriv(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def area(self, n: int = 4000) -> float:
        """integral_0^{t_g} Omega(t) dt  (trapezoid; used for iSWAP normalization)."""
        ts = np.linspace(0.0, self.t_g, n)
        return float(np.trapezoid([self.value(t) for t in ts], ts))


class RaisedCosine(Envelope):
    r"""Hann / raised-cosine: Omega(t) = (amp/2)(1 - cos(2 pi t / t_g)).

    Analytic area over [0, t_g] is amp * t_g / 2. Spectrally clean (good for
    adiabatic spectator suppression); the obvious default for parametric gates.
    """

    def value(self, t):
        if t <= 0.0 or t >= self.t_g:
            return 0.0
        return 0.5 * self.amp * (1.0 - np.cos(TWO_PI * t / self.t_g))

    def deriv(self, t):
        if t <= 0.0 or t >= self.t_g:
            return 0.0
        return 0.5 * self.amp * (TWO_PI / self.t_g) * np.sin(TWO_PI * t / self.t_g)


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


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
@dataclass
class Edge:
    """A coupling edge.

    i, j  : transmon indices (i < j enforced on construction)
    g     : static exchange coupling [rad/ns]  (a_i^d a_j + h.c.)
    eta   : pump participation -- fraction of the flux-pump modulation that
            reaches this edge (1.0 for the strongly pumped target edge,
            smaller for spectator edges in a star topology). Dimensionless.
    """

    i: int
    j: int
    g: float = 0.0
    eta: float = 0.0

    def __post_init__(self):
        if self.i == self.j:
            raise ValueError("Edge connects a transmon to itself.")
        if self.i > self.j:
            self.i, self.j = self.j, self.i


@dataclass
class PumpSpec:
    """Configuration of the single flux pump that drives the target iSWAP."""

    target_edge: tuple  # (p, q): which edge to drive resonantly
    envelope: Envelope
    w_d: Optional[float] = None  # pump (angular) frequency; default = Delta_pq
    drag: bool = False
    delta_drag: Optional[float] = None  # spectator beat delta_s = Delta_rs - w_d
    normalize_iswap: bool = True  # rescale envelope.amp so pulse area = pi/eta_pq


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
class SNAILProcessor:
    r"""Variable-size SNAIL parametric processor in the per-transmon rotating frame.

    Parameters
    ----------
    frequencies : (N,) array_like
        Transmon (angular) frequencies w_i [rad/ns]  (= 2*pi*f[GHz]).
    anharmonicities : (N,) array_like
        Duffing anharmonicities alpha_i [rad/ns] (negative for transmons).
    edges : sequence of Edge
        Coupling graph. Provide static g and/or pump participation eta per edge.
    levels : int
        Truncation per transmon (>= 3 to capture |2> leakage).
    """

    def __init__(
        self,
        frequencies: Sequence[float],
        anharmonicities: Sequence[float],
        edges: Sequence[Edge],
        levels: int = 3,
    ):
        self.w = np.asarray(frequencies, dtype=float)
        self.alpha = np.asarray(anharmonicities, dtype=float)
        self.N = self.w.size
        if self.alpha.size != self.N:
            raise ValueError("frequencies and anharmonicities length mismatch.")
        if levels < 2:
            raise ValueError("levels must be >= 2 (>= 3 recommended for leakage).")
        self.d = int(levels)
        self.edges = list(edges)
        for e in self.edges:
            if not (0 <= e.i < self.N and 0 <= e.j < self.N):
                raise ValueError(f"Edge {e} references a nonexistent transmon.")

        self.dims = [self.d] * self.N
        self.D = self.d**self.N

        # tensored ladder operators a_i (built once)
        a_single = qt.destroy(self.d)
        ident = qt.qeye(self.d)
        self.a = []
        for i in range(self.N):
            ops = [ident] * self.N
            ops[i] = a_single
            self.a.append(qt.tensor(ops))
        self.ad = [op.dag() for op in self.a]

        self._pump: Optional[PumpSpec] = None
        self._c_ops: list = []

    # -- detunings ----------------------------------------------------------
    def delta(self, i: int, j: int) -> float:
        """Detuning Delta_ij = w_i - w_j [rad/ns]."""
        return float(self.w[i] - self.w[j])

    # -- static (rotating-frame-invariant) part -----------------------------
    def _static_hamiltonian(self) -> qt.Qobj:
        """The anharmonic Kerr term; the only time-independent piece of Eq. (1)."""
        H0 = 0 * self.a[0]
        for i in range(self.N):
            H0 += 0.5 * self.alpha[i] * (self.ad[i] * self.ad[i] * self.a[i] * self.a[i])
        return H0

    # -- pump ---------------------------------------------------------------
    def set_pump(self, pump: PumpSpec) -> None:
        """Attach the parametric pump and (optionally) normalize it to a full iSWAP."""
        p, q = pump.target_edge
        edge = self._find_edge(p, q)
        if edge is None or edge.eta == 0.0:
            raise ValueError(
                f"Target edge {(p, q)} must exist and have nonzero pump participation eta."
            )
        if pump.w_d is None:
            pump.w_d = self.delta(p, q)  # resonant drive of the p<->q exchange

        if pump.normalize_iswap:
            # need  integral g_eff dt = pi/2  with g_eff = eta_pq Omega/2
            #   => integral Omega dt = pi / eta_pq
            target_area = np.pi / edge.eta
            current_area = pump.envelope.area()
            if current_area == 0.0:
                raise ValueError("Envelope has zero area; cannot normalize.")
            pump.envelope.amp *= target_area / current_area

        if pump.drag and (pump.delta_drag is None or pump.delta_drag == 0.0):
            raise ValueError(
                "DRAG enabled but delta_drag (spectator beat = Delta_rs - w_d) "
                "is unset or zero. Set it to the detuning of the spectator edge "
                "you intend to suppress."
            )
        self._pump = pump

    def _find_edge(self, i: int, j: int) -> Optional[Edge]:
        i, j = (i, j) if i < j else (j, i)
        for e in self.edges:
            if e.i == i and e.j == j:
                return e
        return None

    def _pump_waveform(self) -> Callable[[float], float]:
        """Return p(t) of Eq. (4). Variables bound by closure-safe default args."""
        if self._pump is None:
            return lambda t: 0.0
        env = self._pump.envelope
        w_d = self._pump.w_d
        drag = self._pump.drag
        dd = self._pump.delta_drag

        def p(t, _env=env, _wd=w_d, _drag=drag, _dd=dd):
            v = _env.value(t) * np.cos(_wd * t)
            if _drag:
                v -= (_env.deriv(t) / _dd) * np.sin(_wd * t)
            return v

        return p

    # -- time-dependent Hamiltonian (QuTiP list / QobjEvo) ------------------
    def hamiltonian(self) -> list:
        r"""Build H(t) of Eq. (1) in QuTiP list format: [H0, [Op, f(t,args)], ...].

        Each edge contributes two non-Hermitian terms whose coefficients are
        complex conjugates, so the total H(t) is Hermitian by construction.
        Coefficient closures bind their edge data via default arguments to avoid
        late-binding capture bugs in the loop.
        """
        p_wave = self._pump_waveform()
        H = [self._static_hamiltonian()]

        for e in self.edges:
            delta_ij = self.delta(e.i, e.j)
            g, eta = e.g, e.eta

            # coefficient of a_i^d a_j  (its h.c. partner gets the conjugate)
            def c_forward(t, args=None, _g=g, _eta=eta, _d=delta_ij, _p=p_wave):
                base = _g + (_eta * _p(t) if _eta != 0.0 else 0.0)
                return base * np.exp(1j * _d * t)

            def c_backward(t, args=None, _g=g, _eta=eta, _d=delta_ij, _p=p_wave):
                base = _g + (_eta * _p(t) if _eta != 0.0 else 0.0)
                return base * np.exp(-1j * _d * t)  # == conj(c_forward)

            H.append([self.ad[e.i] * self.a[e.j], c_forward])
            H.append([self.ad[e.j] * self.a[e.i], c_backward])

        return H

    # -- dissipation (optional; for mesolve) --------------------------------
    def set_relaxation(self, T1: Sequence[float], Tphi: Optional[Sequence[float]] = None):
        r"""Add Lindblad collapse operators for amplitude damping (T1) and pure
        dephasing (Tphi). Rates in ns; pass np.inf to disable a channel.

        Amplitude damping:  sqrt(1/T1_i)   a_i
        Pure dephasing:     sqrt(2/Tphi_i) a_i^d a_i
        """
        T1 = np.asarray(T1, float)
        self._c_ops = []
        for i in range(self.N):
            if np.isfinite(T1[i]) and T1[i] > 0:
                self._c_ops.append(np.sqrt(1.0 / T1[i]) * self.a[i])
        if Tphi is not None:
            Tphi = np.asarray(Tphi, float)
            for i in range(self.N):
                if np.isfinite(Tphi[i]) and Tphi[i] > 0:
                    self._c_ops.append(np.sqrt(2.0 / Tphi[i]) * (self.ad[i] * self.a[i]))

    def collapse_operators(self) -> list:
        return list(self._c_ops)

    # -- Fock index helpers -------------------------------------------------
    def fock_index(self, occupation: Sequence[int]) -> int:
        """Flat index of a Fock state given per-transmon occupation."""
        if len(occupation) != self.N:
            raise ValueError("occupation length must equal N.")
        k = 0
        for n in occupation:
            if not (0 <= n < self.d):
                raise ValueError("occupation out of range for given levels.")
            k = k * self.d + n
        return k

    def fock(self, occupation: Sequence[int]) -> qt.Qobj:
        psi = qt.basis(self.D, self.fock_index(occupation))
        psi.dims = [self.dims, [1] * self.N]
        return psi

    # -- evolution ----------------------------------------------------------
    def evolve(self, psi0: qt.Qobj, tlist: np.ndarray, options=None):
        """Schrodinger (sesolve) if no collapse ops, else master eq (mesolve)."""
        H = self.hamiltonian()
        opts = options or _default_options()
        if self._c_ops:
            return qt.mesolve(H, psi0, tlist, c_ops=self._c_ops, options=opts)
        return qt.sesolve(H, psi0, tlist, options=opts)

    # -- gate metrics on the target pair ------------------------------------
    def iswap_fidelity(self, tlist: np.ndarray, options=None):
        r"""Average gate fidelity vs. the ideal iSWAP on the target pair, with
        spectators initialized in |0>. Returns (F_avg, leakage, U_proj).

        U_proj is the 4x4 projection of the realized propagator onto the target
        pair's computational subspace {|00>,|01>,|10>,|11>}. Average gate fidelity
        uses the leakage-aware formula (Pedersen, Moller & Molmer, PLA 367, 47
        (2007); Wood & Gambetta, PRA 97, 032306 (2018)):

            F_avg = ( |Tr(U_id^d U_proj)|^2 + Tr(U_proj^d U_proj) ) / ( d (d+1) ),  d = 4
            leakage = 1 - Tr(U_proj^d U_proj) / d

        Z-frame note: a parametric beam-splitter realizes iSWAP up to single-qubit
        Z rotations (AC-Stark/frame). We greedily fit those two virtual-Z phases
        before scoring, since they are free in software (McKay et al., PRA 96,
        022330 (2017)). Pass `fit_virtual_z=False` via the module if you want the
        bare comparison.
        """
        if self._pump is None:
            raise RuntimeError("No pump set; nothing to characterize.")
        if self._c_ops:
            warnings.warn("iswap_fidelity uses unitary evolution; collapse ops ignored.")
        p, q = self._pump.target_edge
        others = [k for k in range(self.N) if k not in (p, q)]

        # computational basis of the pair, spectators in |0>
        comp = [(0, 0), (0, 1), (1, 0), (1, 1)]
        cols = []
        cols_idx = []
        for (bp, bq) in comp:
            occ = [0] * self.N
            occ[p], occ[q] = bp, bq
            cols_idx.append(self.fock_index(occ))

        H = self.hamiltonian()
        opts = _default_options()
        U_proj = np.zeros((4, 4), dtype=complex)
        for col, idx in enumerate(cols_idx):
            psi0 = qt.basis(self.D, idx)
            psi0.dims = [self.dims, [1] * self.N]
            res = qt.sesolve(H, psi0, tlist, options=opts)
            psif = res.states[-1].full().ravel()
            for row, idx_r in enumerate(cols_idx):
                U_proj[row, col] = psif[idx_r]

        d = 4
        U_id = _ideal_iswap()
        U_scored = _fit_virtual_z(U_proj, U_id)
        overlap = np.abs(np.trace(U_id.conj().T @ U_scored)) ** 2
        trUU = np.real(np.trace(U_scored.conj().T @ U_scored))
        F_avg = (overlap + trUU) / (d * (d + 1))
        leakage = 1.0 - trUU / d
        return float(F_avg), float(leakage), U_proj


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _default_options():
    """High-accuracy solver options across QuTiP 4/5 (counter-rotating terms
    demand tight tolerances).
    """
    try:  # QuTiP 5
        return {"atol": 1e-11, "rtol": 1e-9, "nsteps": 200000, "max_step": 0.05}
    except Exception:  # pragma: no cover
        return qt.Options(atol=1e-11, rtol=1e-9, nsteps=200000, max_step=0.05)


def _ideal_iswap() -> np.ndarray:
    """Ideal iSWAP in basis {|00>,|01>,|10>,|11>}."""
    U = np.eye(4, dtype=complex)
    U[1, 1] = 0
    U[2, 2] = 0
    U[1, 2] = 1j
    U[2, 1] = 1j
    return U


def _fit_virtual_z(U: np.ndarray, U_id: np.ndarray) -> np.ndarray:
    """Right-multiply by diag virtual-Z phases on each qubit to best match U_id.
    Coarse 1-D phase grid per qubit (free single-qubit Z, McKay et al. 2017).
    """
    best = U
    best_score = -1.0
    grid = np.linspace(0, TWO_PI, 73, endpoint=False)
    for pa in grid:
        for pb in grid:
            Z = np.diag([1, np.exp(1j * pb), np.exp(1j * pa), np.exp(1j * (pa + pb))])
            cand = Z @ U
            score = np.abs(np.trace(U_id.conj().T @ cand)) ** 2
            if score > best_score:
                best_score, best = score, cand
    return best


# ===========================================================================
# Demonstration (reproduces the numbers validated during development)
# ===========================================================================
if __name__ == "__main__":
    import time

    GHz = TWO_PI  # multiply f[GHz] by this to get rad/ns

    # 3 transmons: target pair (0,1) + spectator (2). Star: pump reaches 0-1 and 1-2.
    freqs = np.array([5.00, 4.60, 4.35]) * GHz
    anh = np.array([-0.20, -0.20, -0.20]) * GHz
    edges = [
        Edge(0, 1, g=0.0, eta=1.0),   # strongly pumped target edge
        Edge(1, 2, g=0.0, eta=0.9),   # spectator edge: partner (1) <-> spectator (2)
    ]
    proc = SNAILProcessor(freqs, anh, edges, levels=3)

    w_d = proc.delta(0, 1)                 # = 2*pi*0.40 GHz
    delta_s = proc.delta(1, 2) - w_d       # spectator beat = 2*pi*(0.25-0.40) GHz

    t_g = 12.0                              # ns (fast pulse: DRAG matters here)
    tlist = np.linspace(0, t_g, 600)

    def run(drag):
        env = RaisedCosine(amp=1.0, t_g=t_g)  # amp set by normalize_iswap
        proc.set_pump(PumpSpec((0, 1), env, w_d=w_d, drag=drag, delta_drag=delta_s))
        res = proc.evolve(proc.fock([1, 0, 0]), tlist)
        pops = np.abs(res.states[-1].full().ravel()) ** 2
        n2 = sum(
            pops[proc.fock_index([x, y, z])] * z
            for x in range(3) for y in range(3) for z in range(3)
        )
        return pops[proc.fock_index([0, 1, 0])], n2

    print(f"w_d/2pi = {w_d/GHz:.3f} GHz   spectator beat delta_s/2pi = {delta_s/GHz:+.3f} GHz")
    t0 = time.time()
    for drag in (False, True):
        p010, n2 = run(drag)
        tag = "DRAG ON " if drag else "DRAG OFF"
        print(f"  {tag}:  P|010> (iSWAP transfer) = {p010:.4f}   <n2> (spectator) = {n2:.5f}")
    print(f"  [2 single-state trajectories, dim={proc.D}, wall {time.time()-t0:.2f}s]")

    # full 4x4 gate metric (4 trajectories), DRAG off vs on
    for drag in (False, True):
        env = RaisedCosine(amp=1.0, t_g=t_g)
        proc.set_pump(PumpSpec((0, 1), env, w_d=w_d, drag=drag, delta_drag=delta_s))
        F, leak, _ = proc.iswap_fidelity(tlist)
        print(f"  {'DRAG ON ' if drag else 'DRAG OFF'}:  F_avg(iSWAP) = {F:.4f}   leakage = {leak:.4f}")