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
from envelope import Envelope, RaisedCosine

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
class PumpTone:
    r"""One tone of the flux-pump comb at angular frequency ``mult * w_d``.

    A real SNAIL pump is not a pure cos(w_d t): the cubic/quartic nonlinearity
    mixes the strong pump up into harmonics (mult = 2, 3, ... from g3, g4) and,
    via higher-order/degenerate processes, subharmonics (mult = 1/2, ...). Each
    tone can independently drive a spectator transition that sits near
    ``mult * w_d``, so a comb of tones + a broad detuning sweep maps the full
    spectator ladder.

    mult       : frequency multiplier m (1.0 = fundamental that drives the iSWAP)
    amp        : relative amplitude c_m (the m=1 tone is what gets normalized)
    drag       : apply a DRAG quadrature on THIS tone
    delta_drag : beat to cancel, delta = Delta_rs - mult*w_d  [rad/ns]
    """

    mult: float
    amp: float = 1.0
    drag: bool = False
    delta_drag: Optional[float] = None


@dataclass
class PumpSpec:
    """Configuration of the single flux pump (a comb of tones) driving the iSWAP.

    Backward compatible: leaving `tones=None` and setting `drag`/`delta_drag`
    builds a single fundamental tone (mult=1) carrying that DRAG, exactly like
    the original single-tone behavior.
    """

    target_edge: tuple                       # (p, q): edge driven resonantly by m=1
    envelope: Envelope
    w_d: Optional[float] = None              # base pump frequency; default = Delta_pq
    tones: Optional[list] = None             # list[PumpTone]; default -> one m=1 tone
    drag: bool = False                       # legacy: DRAG on the fundamental
    delta_drag: Optional[float] = None       # legacy: its beat
    normalize_iswap: bool = True             # scale envelope so the m=1 tone -> full iSWAP

    def resolved_tones(self) -> list:
        if self.tones is not None:
            return list(self.tones)
        return [PumpTone(mult=1.0, amp=1.0, drag=self.drag, delta_drag=self.delta_drag)]


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
        An edge may connect any two modes, transmon-transmon OR transmon-SNAIL.
    levels : int | sequence of int
        Truncation PER MODE. Pass an int to broadcast to every mode, or a
        per-mode sequence (e.g. [3, 3, 4] for two transmons + a 4-level SNAIL).
        Use >= 3 on transmons to capture |2> leakage; SNAIL modes that are
        parasitically driven usually want >= 4.
    snail_index : int, optional
        Index of the mode that is the SNAIL (purely a *label*: a SNAIL is just
        another Kerr-anharmonic bosonic mode here, with `anharmonicities[snail]`
        playing the role of the SNAIL Kerr). Used by diagnostics to report
        qubit-SNAIL leakage distinctly. The model treats it on equal footing.

    Modeling note (qubit-SNAIL spectators)
    --------------------------------------
    When the SNAIL is carried explicitly, the *target* iSWAP is still the
    calibrated effective transmon-transmon coupling (eta on the target edge),
    while a transmon-SNAIL edge with nonzero eta represents the SAME flux pump
    parasitically driving a transmon<->SNAIL conversion at beat
    delta = (w_r - w_snail) - w_d. This "augmented effective" model isolates the
    qubit-SNAIL residual channel (matching the SW channel decomposition) without
    re-deriving the gate from the bare g3 three-wave term.
    """

    def __init__(
        self,
        frequencies: Sequence[float],
        anharmonicities: Sequence[float],
        edges: Sequence[Edge],
        levels=3,
        snail_index: Optional[int] = None,
        snail_modes: Optional[dict] = None,
        snail_self_rwa: bool = True,
    ):
        self.w = np.asarray(frequencies, dtype=float)
        self.alpha = np.asarray(anharmonicities, dtype=float)
        self.N = self.w.size
        if self.alpha.size != self.N:
            raise ValueError("frequencies and anharmonicities length mismatch.")

        if np.isscalar(levels):
            self.dims = [int(levels)] * self.N
        else:
            self.dims = [int(x) for x in levels]
            if len(self.dims) != self.N:
                raise ValueError("per-mode `levels` length must equal number of modes.")
        if any(d < 2 for d in self.dims):
            raise ValueError("every mode needs levels >= 2 (>= 3 recommended).")

        # SNAIL modes: {index: {"g3":..., "g4":...}} in rad/ns. These carry a
        # cubic (+quartic) self-Hamiltonian instead of the transmon Duffing Kerr;
        # their entry in `anharmonicities` is ignored (the SNAIL has no alpha/2
        # a^d a^d a a term -- its leading nonlinearity is the three-wave g3).
        self.snail_modes = dict(snail_modes or {})
        self.snail_self_rwa = bool(snail_self_rwa)
        for s in self.snail_modes:
            if not (0 <= s < self.N):
                raise ValueError(f"snail_modes index {s} out of range.")
            if self.alpha[s] != 0.0:
                warnings.warn(
                    f"mode {s} is a SNAIL: its anharmonicities[{s}] (Duffing Kerr) "
                    "is ignored; nonlinearity comes from g3/g4."
                )
        if snail_index is None and len(self.snail_modes) == 1:
            snail_index = next(iter(self.snail_modes))
        self.snail_index = snail_index
        if snail_index is not None and not (0 <= snail_index < self.N):
            raise ValueError("snail_index out of range.")

        self.edges = list(edges)
        for e in self.edges:
            if not (0 <= e.i < self.N and 0 <= e.j < self.N):
                raise ValueError(f"Edge {e} references a nonexistent mode.")

        self.D = int(np.prod(self.dims))

        # tensored ladder operators a_i (built once); per-mode truncation.
        idents = [qt.qeye(d) for d in self.dims]
        self.a = []
        for i in range(self.N):
            ops = list(idents)
            ops[i] = qt.destroy(self.dims[i])
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
        r"""Time-independent piece in the rotating frame:
          * transmon modes: Duffing Kerr  (alpha/2) a^d a^d a a
          * SNAIL modes:    the number-conserving part of g4 (b+b^d)^4, i.e.
                            g4 (6 b^d2 b^2 + 12 b^d b);  the cubic g3 (b+b^d)^3
                            has NO static part (it is added time-dependently).
        The constant 3*g4 is dropped (global phase). The 12*g4 b^d b piece is a
        small SNAIL-frequency renormalization, kept for fidelity.
        """
        H0 = 0 * self.a[0]
        for i in range(self.N):
            if i in self.snail_modes:
                g4 = self.snail_modes[i].get("g4", 0.0)
                if g4 != 0.0:
                    b, bd = self.a[i], self.ad[i]
                    H0 += g4 * (6 * (bd * bd * b * b) + 12 * (bd * b))
            else:
                H0 += 0.5 * self.alpha[i] * (self.ad[i] * self.ad[i] * self.a[i] * self.a[i])
        return H0

    def _snail_self_terms(self) -> list:
        r"""Rotating-frame time-dependent SNAIL self-Hamiltonian terms.

        From the frame transform b -> b e^{-i w_s t}, the cubic/quartic split by
        photon-number change Delta n, each band rotating at Delta n * w_s:

          g3 (b+b^d)^3 :  T+1 e^{i w_s t} + T+3 e^{3i w_s t} + h.c.,
                          T+1 = 3 b^d2 b + 3 b^d,   T+3 = b^d3
          g4 (b+b^d)^4 :  (static, in H0) + Q+2 e^{2i w_s t} + Q+4 e^{4i w_s t} + h.c.,
                          Q+2 = 4 b^d3 b + 6 b^d2,  Q+4 = b^d4

        With snail_self_rwa=True (default) only the slowest, leading terms are
        kept: the cubic T+1 (rotates at w_s) and the static g4 Kerr. The dropped
        bands rotate at >= 2 w_s (~8-18 GHz here) and are far off-resonant, so
        their leading effect is a small static shift already implicit at this
        order. Keeping them (rwa=False) is exact but forces the integrator to
        resolve up to 4 w_s, raising the step count several-fold.
        """
        terms = []
        for s, nl in self.snail_modes.items():
            g3 = nl.get("g3", 0.0)
            g4 = nl.get("g4", 0.0)
            ws = self.w[s]
            b, bd = self.a[s], self.ad[s]
            if g3 != 0.0:
                T1 = 3 * (bd * bd * b) + 3 * bd          # Delta n = +1
                terms += _rotating_pair(g3 * T1, ws)     # +/- w_s
                if not self.snail_self_rwa:
                    T3 = bd * bd * bd                    # Delta n = +3
                    terms += _rotating_pair(g3 * T3, 3 * ws)
            if g4 != 0.0 and not self.snail_self_rwa:
                Q2 = 4 * (bd * bd * bd * b) + 6 * (bd * bd)   # Delta n = +2
                Q4 = bd * bd * bd * bd                        # Delta n = +4
                terms += _rotating_pair(g4 * Q2, 2 * ws)
                terms += _rotating_pair(g4 * Q4, 4 * ws)
        return terms

    # -- pump ---------------------------------------------------------------
    def set_pump(self, pump: PumpSpec) -> None:
        """Attach the parametric pump (a comb of tones) and normalize the m=1
        tone to a full iSWAP."""
        p, q = pump.target_edge
        edge = self._find_edge(p, q)
        if edge is None or edge.eta == 0.0:
            raise ValueError(
                f"Target edge {(p, q)} must exist and have nonzero pump participation eta."
            )
        if pump.w_d is None:
            pump.w_d = self.delta(p, q)  # resonant drive of the p<->q exchange

        tones = pump.resolved_tones()
        # fundamental tone (mult ~ 1) amplitude sets the iSWAP normalization
        fund = min(tones, key=lambda t: abs(t.mult - 1.0))
        if pump.normalize_iswap:
            if abs(fund.mult - 1.0) > 1e-9 or fund.amp == 0.0:
                raise ValueError(
                    "normalize_iswap needs a fundamental tone (mult=1, amp!=0)."
                )
            # resonant rate on target = eta_pq * amp_1 * Omega/2 -> area = pi/(eta amp_1)
            target_area = np.pi / (edge.eta * fund.amp)
            current_area = pump.envelope.area()
            if current_area == 0.0:
                raise ValueError("Envelope has zero area; cannot normalize.")
            pump.envelope.amp *= target_area / current_area

        for t in tones:
            if t.drag and (t.delta_drag is None or t.delta_drag == 0.0):
                raise ValueError(
                    f"DRAG on tone mult={t.mult} needs a nonzero delta_drag "
                    "(= Delta_rs - mult*w_d, the spectator beat to cancel)."
                )
        self._pump = pump

    def _find_edge(self, i: int, j: int) -> Optional[Edge]:
        i, j = (i, j) if i < j else (j, i)
        for e in self.edges:
            if e.i == i and e.j == j:
                return e
        return None

    def _pump_waveform(self) -> Callable[[float], float]:
        r"""Multi-tone comb p(t) = sum_m amp_m Omega cos(m w_d t)
                            - sum_{m in drag} amp_m (Omega'/delta_m) sin(m w_d t).
        Tone data bound via default args (closure-safe)."""
        if self._pump is None:
            return lambda t: 0.0
        env = self._pump.envelope
        w_d = self._pump.w_d
        tones = self._pump.resolved_tones()
        # pre-extract to plain tuples so the closure captures immutables
        spec = tuple((float(t.mult), float(t.amp), bool(t.drag),
                      float(t.delta_drag) if t.delta_drag else 0.0) for t in tones)

        def p(t, _env=env, _wd=w_d, _spec=spec):
            val = _env.value(t)
            dval = _env.deriv(t)
            out = 0.0
            for (m, c, drag, dd) in _spec:
                out += c * val * np.cos(m * _wd * t)
                if drag:
                    out -= c * (dval / dd) * np.sin(m * _wd * t)
            return out

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

        # SNAIL cubic/quartic self-terms (time-dependent in the rotating frame)
        H.extend(self._snail_self_terms())

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

    # -- Fock index helpers (mixed-radix over per-mode dims) ----------------
    def fock_index(self, occupation: Sequence[int]) -> int:
        """Flat index of a Fock state given per-mode occupation."""
        if len(occupation) != self.N:
            raise ValueError("occupation length must equal number of modes.")
        k = 0
        for n, d in zip(occupation, self.dims):
            if not (0 <= n < d):
                raise ValueError("occupation out of range for given levels.")
            k = k * d + n
        return k

    def decode_index(self, k: int) -> list:
        """Inverse of fock_index: flat index -> per-mode occupation list."""
        occ = []
        for d in reversed(self.dims):
            k, r = divmod(k, d)
            occ.append(r)
        return occ[::-1]

    def mean_occupation(self, probs: np.ndarray, mode: int) -> float:
        """<n_mode> from a flat population vector (length D)."""
        return float(sum(probs[k] * self.decode_index(k)[mode] for k in range(self.D)))

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
    def iswap_fidelity(self, tlist: np.ndarray, options=None, fit_virtual_z: bool = True):
        r"""Average gate fidelity vs. the ideal iSWAP on the target pair, with
        spectators initialized in |0>. Returns (F_avg, leakage, U_proj).

        U_proj is the 4x4 projection of the realized propagator onto the target
        pair's computational subspace {|00>,|01>,|10>,|11>}. Average gate fidelity
        uses the leakage-aware formula (Pedersen, Moller & Molmer, PLA 367, 47
        (2007); Wood & Gambetta, PRA 97, 032306 (2018)):

            F_avg = ( |Tr(U_id^d U_proj)|^2 + Tr(U_proj^d U_proj) ) / ( d (d+1) ),  d = 4
            leakage = 1 - Tr(U_proj^d U_proj) / d

        Virtual-Z (fit_virtual_z=True, default): a parametric beam-splitter
        realizes iSWAP only up to single-qubit Z rotations, because the rotating
        frame accumulates deterministic per-qubit phases (AC-Stark / Lamb shift,
        the pump-vs-transition frame offset, Bloch-Siegert). Those phases are
        corrected for FREE in software by redefining the phase reference of
        subsequent pulses (McKay et al., PRA 96, 022330 (2017)), so the physically
        meaningful score is the fidelity to iSWAP modulo local Z. We therefore
        right-multiply U_proj by diag(1, e^{i phi_q}, e^{i phi_p}, e^{i(phi_p+phi_q)})
        and maximize over (phi_p, phi_q) before scoring. Set fit_virtual_z=False
        to score the bare propagator (penalizes those calibratable phases).
        Note: the Z fit only removes single-qubit *phase*; entangling-angle error,
        population/swap error, and leakage are unaffected and still count.
        """
        if self._pump is None:
            raise RuntimeError("No pump set; nothing to characterize.")
        if self._c_ops:
            warnings.warn("iswap_fidelity uses unitary evolution; collapse ops ignored.")
        p, q = self._pump.target_edge

        # computational basis of the pair, spectators (incl. any SNAIL) in |0>
        comp = [(0, 0), (0, 1), (1, 0), (1, 1)]
        cols_idx = []
        for (bp, bq) in comp:
            occ = [0] * self.N
            occ[p], occ[q] = bp, bq
            cols_idx.append(self.fock_index(occ))

        H = self.hamiltonian()
        opts = options or _default_options()
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
        U_scored = _fit_virtual_z(U_proj, U_id) if fit_virtual_z else U_proj
        overlap = np.abs(np.trace(U_id.conj().T @ U_scored)) ** 2
        trUU = np.real(np.trace(U_scored.conj().T @ U_scored))
        F_avg = (overlap + trUU) / (d * (d + 1))
        leakage = 1.0 - trUU / d
        return float(F_avg), float(leakage), U_proj


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _rotating_pair(op_plus: "qt.Qobj", omega: float) -> list:
    """Given a raising-type operator A (Delta n > 0) and its rotation rate omega,
    return the QuTiP terms [A e^{i omega t}, A^d e^{-i omega t}] whose sum is
    Hermitian. Coefficients bound via default args (closure-safe)."""
    def c_up(t, args=None, _w=omega):
        return np.exp(1j * _w * t)

    def c_dn(t, args=None, _w=omega):
        return np.exp(-1j * _w * t)

    return [[op_plus, c_up], [op_plus.dag(), c_dn]]


def _default_options():
    """High-accuracy solver options across QuTiP 4/5 (counter-rotating terms
    demand tight tolerances)."""
    try:  # QuTiP 5
        return {"atol": 1e-11, "rtol": 1e-9, "nsteps": 200000, "max_step": 0.05}
    except Exception:  # pragma: no cover
        return qt.Options(atol=1e-11, rtol=1e-9, nsteps=200000, max_step=0.05)


def _ideal_iswap() -> np.ndarray:
    """iSWAP in basis {|00>,|01>,|10>,|11>}."""
    U = np.eye(4, dtype=complex)
    U[1, 1] = 0
    U[2, 2] = 0
    U[1, 2] = 1j
    U[2, 1] = 1j
    return U


def _fit_virtual_z(U: np.ndarray, U_id: np.ndarray) -> np.ndarray:
    """Right-multiply by diag virtual-Z phases on each qubit to best match U_id.
    Free single-qubit Z (McKay et al. 2017). Coarse grid then local refinement so
    the recovered phase is not resolution-limited for high-fidelity gates."""
    def Z(pa, pb):
        return np.diag([1.0, np.exp(1j * pb), np.exp(1j * pa), np.exp(1j * (pa + pb))])

    def score(pa, pb):
        return np.abs(np.trace(U_id.conj().T @ (Z(pa, pb) @ U))) ** 2

    best = (0.0, 0.0)
    best_s = -1.0
    coarse = np.linspace(0, TWO_PI, 48, endpoint=False)
    for pa in coarse:
        for pb in coarse:
            s = score(pa, pb)
            if s > best_s:
                best_s, best = s, (pa, pb)
    # local refinement around the coarse optimum
    span = TWO_PI / 48
    for _ in range(3):
        fa = np.linspace(best[0] - span, best[0] + span, 21)
        fb = np.linspace(best[1] - span, best[1] + span, 21)
        for pa in fa:
            for pb in fb:
                s = score(pa, pb)
                if s > best_s:
                    best_s, best = s, (pa, pb)
        span /= 10.0
    return Z(*best) @ U


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

    t_g = 100.0                              # ns (fast pulse: DRAG matters here)
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

    # -----------------------------------------------------------------------
    # Demo 2: explicit SNAIL (cubic g3, NOT a transmon Kerr) as a spectator,
    # driven at the SECOND HARMONIC of a 2-tone pump comb. DRAG targets m=2.
    # -----------------------------------------------------------------------
    print("\nSNAIL + 2-tone comb (2nd-harmonic spectator):")
    fs = np.array([5.00, 4.60, 3.95]) * GHz          # SNAIL near 2*w_d below qubit 1
    anh2 = np.array([-0.20, -0.20, 0.00]) * GHz       # SNAIL slot anharmonicity unused
    edges2 = [Edge(0, 1, eta=1.0), Edge(1, 2, eta=0.9)]
    snail = {2: {"g3": 0.03 * GHz, "g4": 0.0}}        # cubic three-wave; no Duffing
    procS = SNAILProcessor(fs, anh2, edges2, levels=[3, 3, 4],
                           snail_modes=snail, snail_self_rwa=True)
    w_d2 = procS.delta(0, 1)
    beat2 = procS.delta(1, 2) - 2 * w_d2              # beat against the m=2 tone
    comb = [PumpTone(1.0, 1.0), PumpTone(2.0, 0.5)]   # fundamental + 2nd harmonic
    for drag in (False, True):
        tones = [PumpTone(1.0, 1.0),
                 PumpTone(2.0, 0.5, drag=drag, delta_drag=(beat2 if drag else None))]
        procS.set_pump(PumpSpec((0, 1), RaisedCosine(amp=1.0, t_g=t_g),
                                w_d=w_d2, tones=tones))
        res = procS.evolve(procS.fock([1, 0, 0]), tlist)
        pops = np.abs(res.states[-1].full().ravel()) ** 2
        nS = procS.mean_occupation(pops, 2)
        print(f"  {'DRAG ON ' if drag else 'DRAG OFF'} (m=2):  "
              f"<n_snail> = {nS:.5f}   beat(m=2)/2pi = {beat2/GHz:+.3f} GHz")