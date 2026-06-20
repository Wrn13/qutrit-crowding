"""zhou_coupler.py.
===============

Charge-pumped parametric coupler in the dressed-mode framework of

    Chao Zhou, "Quantum Operations with Charge-pumped Parametric Interactions,"
    PhD thesis, University of Pittsburgh (2023), Chapter 2.

The Hamiltonian is built directly from EXPERIMENTAL inputs -- the coupler's
intrinsic non-linearity g_n (g3 for a SNAIL three-wave mixer) and the measured
participations

        lambda_is = g_is / Delta_is                                   (Eq. 57)

of each mode in the dressed coupler. No effective transmon-transmon edge is
assumed: the star connectivity lives entirely in the participations, because the
dressed coupler operator is (Eq. 56)

        s' ~= s + sum_i lambda_is a_i .

Master Hamiltonian (Zhou Eqs. 54 / 60; two-level/qubit form Eq. 72)
-------------------------------------------------------------------
        H_I(t) / hbar = sum_n  g_n  X(t)^n ,

        X(t) = s e^{-i w_s t}
             + sum_i lambda_is a_i e^{-i w_i t}
             + sum_p eta_p(t) e^{-i w_p t}
             + h.c. ,

with the displaced pump amplitude (Eq. 50)

        eta_p(t) = 2 w_p / (w_p^2 - w_s^2) * eps_p(t) .

Expanding X(t)^n generates every multi-wave-mixing process at once (iSWAP,
bSWAP, sub-harmonic drive, cross-Kerr, and all spectators / higher orders).
Pumping at  k w_p = w_rot  (Eq. 61) makes one process static under the RWA; its
strength is (Eq. 62)

        g_eff = C g_n eta^k prod_i lambda_is ,

with C a multinomial coefficient. The canonical two-qubit example -- pumping a
three-wave coupler at w_p = w_b - w_a -- gives the iSWAP family with (Eqs. 55/73)

        g_eff = 6 g3 lambda_as lambda_bs |eta| .          (verified in __main__)

g_n is the OVERALL prefactor of the dynamics: with g3 = 0 there is no gate.

Conventions
-----------
* Frequencies and non-linearities are entered in GHz and converted to angular
  units (rad/ns) internally: omega = 2 pi f. Time is in ns.
* A mode with 2 levels reproduces Zhou's sigma_- (qubit) form (Eq. 72); use
  >= 3 levels to keep it an oscillator/transmon so that leakage is captured.
* The coupler S is kept as a dynamical mode, so the qubit-coupler spectator
  channel is automatic.

Time evolution
--------------
Gate dynamics are integrated with QuTiP's compiled solver on the EXACT
Hamiltonian (no terms pruned): `to_qutip_hamiltonian()` returns the operator as a
sparse list-format QobjEvo, and `evolve_state`, `propagator_columns`, and
`iswap_fidelity` drive it through `qt.sesolve`. QuTiP is imported lazily, so the
model builders (`dressed_flux`, `hamiltonian_matrix`, `expand_terms`) and the
analytic estimators stay importable and testable without QuTiP. The dense
`hamiltonian_matrix` is retained only as a lightweight correctness oracle (the
self-test below, and a one-point cross-check against `iswap_fidelity` on a QuTiP
node); it is not part of the production solve path.
"""  # noqa: D205

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import factorial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from envelope import Envelope, RaisedCosine, ConstantPulse
import numpy as np

TWO_PI: float = 2.0 * np.pi



# --- type aliases (documented shapes used throughout) ----------------------
# A "letter" of X(t): (operator, signed_frequency_rad, constant_amplitude,
# pump_key). The term's time factor is exp(-i * signed_frequency * t). pump_key
# is None for a mode operator, or (tone_index, is_conjugate) for a pump factor
# whose (time-dependent) amplitude eta_p(t) is supplied at evaluation time.
FluxLetter = Tuple[np.ndarray, float, float, Optional[Tuple[int, bool]]]

# A grouped term of the expanded Hamiltonian: (Omega_rad, pump_signature,
# operator). pump_signature is the sorted tuple of (tone_index, is_conjugate)
# pump factors carried by the term.
HamiltonianTerm = Tuple[float, Tuple[Tuple[int, bool], ...], np.ndarray]

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
        inside X(t).
    envelope : Envelope
        Real shape eps_p(t). Its `amp` carries |eps_p| (rad/ns), or directly the
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
    """

    w_p_GHz: float
    envelope: Envelope
    phi_p: float = 0.0
    is_eta: bool = True
    drag: bool = False
    delta_drag_GHz: Optional[float] = None


# ===========================================================================
# Gate-metric helpers (pure numpy)
# ===========================================================================
def _ideal_iswap() -> np.ndarray:
    """Target iSWAP on the two-qubit subspace (basis |00>, |01>, |10>, |11>)."""
    U = np.eye(4, dtype=complex)
    U[1, 1] = U[2, 2] = 0.0
    U[1, 2] = U[2, 1] = 1j
    return U


def _fit_virtual_z(U: np.ndarray, U_ideal: np.ndarray) -> np.ndarray:
    """Apply the single-qubit virtual-Z rotation that best aligns `U` with the
    target. Virtual-Z phases are free in software (McKay et al., PRA 96, 022330
    (2017)) and should not be charged as gate error.

    Parameters
    ----------
    U : ndarray, shape (4, 4)
        Realised propagator projected onto the computational subspace.
    U_ideal : ndarray, shape (4, 4)
        Target unitary (here the ideal iSWAP).

    Returns:
    -------
    ndarray, shape (4, 4)
        Z(pa*, pb*) U, with the phases maximising |Tr(U_ideal^dag Z U)|^2 (coarse
        grid plus local refinement).
    """  # noqa: D205
    def z_rotation(phase_a: float, phase_b: float) -> np.ndarray:
        return np.diag([1.0,
                        np.exp(1j * phase_b),
                        np.exp(1j * phase_a),
                        np.exp(1j * (phase_a + phase_b))])

    def overlap(phase_a: float, phase_b: float) -> float:
        return np.abs(np.trace(U_ideal.conj().T @ (z_rotation(phase_a, phase_b) @ U))) ** 2

    best_phases, best_overlap = (0.0, 0.0), -1.0
    coarse = np.linspace(0.0, TWO_PI, 48, endpoint=False)
    for phase_a in coarse:
        for phase_b in coarse:
            score = overlap(phase_a, phase_b)
            if score > best_overlap:
                best_overlap, best_phases = score, (phase_a, phase_b)

    span = TWO_PI / 48
    for _ in range(3):                      # local refinement around the best cell
        for phase_a in np.linspace(best_phases[0] - span, best_phases[0] + span, 21):
            for phase_b in np.linspace(best_phases[1] - span, best_phases[1] + span, 21):
                score = overlap(phase_a, phase_b)
                if score > best_overlap:
                    best_overlap, best_phases = score, (phase_a, phase_b)
        span /= 10.0

    return z_rotation(*best_phases) @ U


# ===========================================================================
# The coupler
# ===========================================================================
class ZhouCoupler:
    r"""Parametric coupler built from experimental g_n and participations.

    Parameters
    ----------
    mode_freqs_GHz : sequence of float
        Frequencies of ALL modes, INCLUDING the coupler S (GHz). After the
        Bogoliubov dressing these are the dressed (i.e. measured) frequencies.
    coupler_index : int
        Index of the coupler mode S within `mode_freqs_GHz`.
    participations : dict[int, float]
        lambda_is = g_is/Delta_is for each non-coupler mode that participates in
        the dressed coupler (Eq. 57; for chained modes use Eq. 58 and pass the
        product). Omitted modes do not appear in X(t). The coupler's own
        participation is 1 by definition and must not be listed.
    nonlinearities : dict[int, float]
        {n: g_n} in GHz. {3: g3} for a pure three-wave SNAIL; add {4: g4} for
        four-wave mixing.
    levels : int or sequence of int, default 3
        Per-mode Fock truncation. 2 -> qubit (sigma_-, Eq. 72); >= 3 -> oscillator
        (captures leakage). A scalar applies to every mode.

    Attributes:
    ----------
    omega : ndarray
        Mode angular frequencies (rad/ns).
    n_modes, dim : int
        Number of modes and total Hilbert-space dimension.
    coupler_index : int
        Index of the coupler mode.
    dims : list[int]
        Per-mode truncations.
    participation : ndarray
        Participation vector (1 on the coupler, lambda_is on coupled modes).
    g_n : dict[int, float]
        Non-linear coefficients in rad/ns.
    a_ops, ad_ops : list[ndarray]
        Embedded annihilation / creation operators.
    identity : ndarray
        Identity on the full space.
    """

    def __init__(
        self,
        mode_freqs_GHz: Sequence[float],
        coupler_index: int,
        participations: Dict[int, float],
        nonlinearities: Dict[int, float],
        levels: Union[int, Sequence[int]] = 3,
    ) -> None:
        self.omega: np.ndarray = np.asarray(mode_freqs_GHz, dtype=float) * TWO_PI  # rad/ns
        self.n_modes: int = self.omega.size

        self.coupler_index: int = int(coupler_index)
        if not (0 <= self.coupler_index < self.n_modes):
            raise ValueError("coupler_index out of range.")

        # per-mode truncation
        if isinstance(levels, int):
            self.dims: List[int] = [levels] * self.n_modes
        else:
            self.dims = [int(d) for d in levels]
            if len(self.dims) != self.n_modes:
                raise ValueError("len(levels) must equal the number of modes.")
        self.dim: int = int(np.prod(self.dims))   # total Hilbert-space dimension

        # participation vector: 1 on the coupler, lambda_is on coupled modes, 0 else
        self.participation: np.ndarray = np.zeros(self.n_modes, dtype=float)
        self.participation[self.coupler_index] = 1.0
        for index, value in participations.items():
            if index == self.coupler_index:
                raise ValueError("Do not list the coupler in participations (it is 1).")
            if not (0 <= index < self.n_modes):
                raise ValueError(f"participation index {index} out of range.")
            self.participation[index] = float(value)

        # non-linear coefficients, GHz -> rad/ns
        self.g_n: Dict[int, float] = {int(n): float(g) * TWO_PI
                                      for n, g in nonlinearities.items()}
        if not self.g_n:
            raise ValueError("Provide at least one non-linearity, e.g. {3: g3_GHz}.")

        # embedded ladder operators (dense, mixed-radix), built once
        self.a_ops: List[np.ndarray] = [self._embed(self._annihilation(self.dims[i]), i)
                                        for i in range(self.n_modes)]
        self.ad_ops: List[np.ndarray] = [op.conj().T for op in self.a_ops]
        self.identity: np.ndarray = np.eye(self.dim, dtype=complex)

        # time-independent letters of X(t) (mode operators only); the pump letters
        # are time-dependent and are added in _flux_letters(). Single source of
        # truth shared by dressed_flux() and the term expansion.
        self._mode_letters: List[FluxLetter] = []
        for i in range(self.n_modes):
            lam = self.participation[i]
            if lam == 0.0:
                continue
            # a_i ~ e^{-i w_i t} (signed_freq +w_i); a_i^dag ~ e^{+i w_i t} (-w_i)
            self._mode_letters.append((self.a_ops[i], +self.omega[i], lam, None))
            self._mode_letters.append((self.ad_ops[i], -self.omega[i], lam, None))

        self._pump_tones: List[PumpTone] = []

    # -- mode-space construction -------------------------------------------
    @staticmethod
    def _annihilation(d: int) -> np.ndarray:
        """Annihilation operator on a single d-level mode."""
        return np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)

    def _embed(self, op: np.ndarray, mode: int) -> np.ndarray:
        """Embed a single-mode operator `op` acting on `mode` into the full
        tensor-product space.
        """  # noqa: D205
        factors = [np.eye(d, dtype=complex) for d in self.dims]
        factors[mode] = op
        embedded = factors[0]
        for factor in factors[1:]:
            embedded = np.kron(embedded, factor)
        return embedded

    # -- Fock-space indexing ------------------------------------------------
    def fock_index(self, occupations: Sequence[int]) -> int:
        """Flat Hilbert-space index of a Fock state.

        Parameters
        ----------
        occupations : sequence of int
            Per-mode photon numbers |n_0, n_1, ...> (mode 0 most significant).

        Returns:
        -------
        int
            The mixed-radix flat index.
        """
        index = 0
        for occ, d in zip(occupations, self.dims):
            if not (0 <= occ < d):
                raise ValueError("occupation out of range.")
            index = index * d + occ
        return index

    def decode_index(self, index: int) -> List[int]:
        """Inverse of `fock_index`.

        Parameters
        ----------
        index : int
            Flat Hilbert-space index.

        Returns:
        -------
        list of int
            Per-mode occupations.
        """
        occupations: List[int] = []
        for d in reversed(self.dims):
            index, occ = divmod(index, d)
            occupations.append(occ)
        return occupations[::-1]

    def basis_state(self, occupations: Sequence[int]) -> np.ndarray:
        """Unit state vector for a Fock state.

        Parameters
        ----------
        occupations : sequence of int
            Per-mode photon numbers.

        Returns:
        -------
        ndarray, shape (dim,)
            The corresponding computational-basis ket.
        """
        psi = np.zeros(self.dim, dtype=complex)
        psi[self.fock_index(occupations)] = 1.0
        return psi

    def mean_occupation(self, probabilities: np.ndarray, mode: int) -> float:
        """Expected photon number of one mode.

        Parameters
        ----------
        probabilities : ndarray, shape (dim,)
            Probability vector over the Fock basis (e.g. |psi|^2).
        mode : int
            Mode index whose occupation is wanted.

        Returns:
        -------
        float
            <n_mode> = sum_k p_k * occ_mode(k).
        """
        return float(sum(probabilities[k] * self.decode_index(k)[mode]
                         for k in range(self.dim)))

    def delta(self, i: int, j: int) -> float:
        """Angular detuning between two modes.

        Parameters
        ----------
        i, j : int
            Mode indices.

        Returns:
        -------
        float
            w_i - w_j (rad/ns).
        """
        return float(self.omega[i] - self.omega[j])

    # -- pump ---------------------------------------------------------------
    def set_pump(self, tones: Union[PumpTone, Sequence[PumpTone]],
                 normalize_iswap: Optional[Tuple[int, int]] = None) -> None:
        """Attach one or more pump tones to the coupler.

        Parameters
        ----------
        tones : PumpTone or sequence of PumpTone
            The pump tone(s). A single tone may be passed directly.
        normalize_iswap : tuple(int, int), optional
            If given as (a, b), the FIRST tone's amplitude is rescaled so that,
            pumped at w_b - w_a, the time-integrated rate realises a full iSWAP on
            the (a, b) pair (rotation angle pi/2). The condition is
            integral 6 g3 lambda_as lambda_bs |eta(t)| dt = pi/2 (Eqs. 55/73).

        Returns:
        -------
        None
        """
        self._pump_tones = [tones] if isinstance(tones, PumpTone) else list(tones)
        if normalize_iswap is None:
            return

        a, b = normalize_iswap
        g3 = self.g_n.get(3, 0.0)
        if g3 == 0.0:
            raise ValueError("normalize_iswap needs a three-wave g3.")
        lam_a, lam_b = self.participation[a], self.participation[b]
        if lam_a == 0.0 or lam_b == 0.0:
            raise ValueError("Both qubits must have nonzero participation.")

        tone = self._pump_tones[0]
        omega_p = tone.w_p_GHz * TWO_PI
        omega_s = self.omega[self.coupler_index]
        # |eta(t)| = eta_prefactor * eps(t); see _eta and Eq. 50.
        eta_prefactor = 1.0 if tone.is_eta else abs(2 * omega_p / (omega_p ** 2 - omega_s ** 2))

        target_eta_area = (np.pi / 2) / (6 * g3 * lam_a * lam_b)   # required integral |eta| dt
        current_eta_area = eta_prefactor * tone.envelope.area()
        if current_eta_area == 0.0:
            raise ValueError("Pump envelope has zero area.")
        tone.envelope.amp *= target_eta_area / current_eta_area

    def _eta(self, tone: PumpTone, t: float) -> complex:
        """Displaced pump amplitude eta_p(t), complex when DRAG is on (Eq. 50;
        Motzoi PRL 103, 110501 (2009)).
        """  # noqa: D205
        omega_p = tone.w_p_GHz * TWO_PI
        omega_s = self.omega[self.coupler_index]
        prefactor = 1.0 if tone.is_eta else (2 * omega_p / (omega_p ** 2 - omega_s ** 2))
        amplitude: complex = tone.envelope.value(t)
        if tone.drag and tone.delta_drag_GHz not in (None, 0.0):
            detuning = tone.delta_drag_GHz * TWO_PI            # rad/ns
            amplitude = amplitude - 1j * tone.envelope.deriv(t) / detuning
        return prefactor * amplitude * np.exp(1j * tone.phi_p)

    def _flux_letters(self) -> List[FluxLetter]:
        """All letters of X(t): the static mode operators plus the current pump
        tones. A tone contributes eta_p e^{-i w_p t} (key (i, False)) and its
        conjugate eta_p* e^{+i w_p t} (key (i, True)).
        """  # noqa: D205
        letters = list(self._mode_letters)
        for tone_index, tone in enumerate(self._pump_tones):
            omega_p = tone.w_p_GHz * TWO_PI
            letters.append((self.identity, +omega_p, 1.0, (tone_index, False)))
            letters.append((self.identity, -omega_p, 1.0, (tone_index, True)))
        return letters

    # -- the dressed flux and the exact Hamiltonian (reference builders) ----
    def dressed_flux(self, t: float) -> np.ndarray:
        r"""Dressed flux operator X(t) (Hermitian).

        X(t) = sum_i lambda_is a_i e^{-i w_i t} + sum_p eta_p(t) e^{-i w_p t} + h.c.
        The coupler (lambda = 1) is one of the modes.

        Parameters
        ----------
        t : float
            Time (ns).

        Returns:
        -------
        ndarray, shape (dim, dim)
            The dense X(t).
        """
        etas = [self._eta(tone, t) for tone in self._pump_tones]
        X = np.zeros((self.dim, self.dim), dtype=complex)
        for op, signed_freq, amp, pump_key in self._flux_letters():
            coeff: complex = amp * np.exp(-1j * signed_freq * t)
            if pump_key is not None:
                tone_index, is_conjugate = pump_key
                eta = etas[tone_index]
                coeff *= np.conj(eta) if is_conjugate else eta
            X = X + coeff * op
        return X

    def hamiltonian_matrix(self, t: float) -> np.ndarray:
        """Exact interaction-picture Hamiltonian H_I(t) = sum_n g_n X(t)^n (dense).

        Reference builder used by the self-test and as a one-point cross-check
        against the QuTiP solver; it is not used in the production solve path.

        Parameters
        ----------
        t : float
            Time (ns).

        Returns:
        -------
        ndarray, shape (dim, dim)
            The dense Hamiltonian (rad/ns).
        """
        X = self.dressed_flux(t)
        highest_order = max(self.g_n)
        powers: Dict[int, np.ndarray] = {1: X}
        for n in range(2, highest_order + 1):       # X^2, X^3, ... accumulated once
            powers[n] = powers[n - 1] @ X
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for n, g in self.g_n.items():
            H = H + g * powers[n]
        return H

    # -- exact operator decomposition (engine of the QuTiP export) ----------
    def expand_terms(self, cutoff_GHz: float = np.inf) -> List[HamiltonianTerm]:
        r"""Expand sum_n g_n X(t)^n into constant operators grouped by net carrier
        frequency Omega = sum (+/- w_i) + sum (+/- w_p).

        Each returned (Omega, pump_signature, O) satisfies

            H(t) = sum_terms  exp(-i Omega t) * prod_p eta_p(t)^(...) * O ,

        where O folds in g_n and the participation products and the pump factors
        are supplied at evaluation time (so envelope and DRAG stay exact).

        Parameters
        ----------
        cutoff_GHz : float, default inf
            Keep only terms with |Omega| <= 2 pi cutoff. The default (inf) prunes
            NOTHING -- the sum reproduces `hamiltonian_matrix(t)` exactly and is
            the form handed to QuTiP. A finite cutoff drops the fast carriers,
            leaving a rotating-wave / average-Hamiltonian reduction.

        Returns:
        -------
        list of (float, tuple, ndarray)
            (exact Omega in rad/ns, pump signature, constant operator) per group.
        """  # noqa: D205
        cutoff_rad = abs(cutoff_GHz) * TWO_PI
        letters = self._flux_letters()

        # group by (rounded Omega, pump signature); accumulate the operator and
        # remember the exact Omega for phase-accurate evaluation.
        groups: Dict[Tuple[float, Tuple[Tuple[int, bool], ...]], List] = {}
        for order, g in self.g_n.items():
            for combo in itertools.product(letters, repeat=order):
                net_freq = sum(letter[1] for letter in combo)
                if abs(net_freq) > cutoff_rad:
                    continue
                amplitude = 1.0
                operator: Optional[np.ndarray] = None
                pump_signature: List[Tuple[int, bool]] = []
                for op, _signed_freq, amp, pump_key in combo:
                    amplitude *= amp
                    operator = op if operator is None else operator @ op
                    if pump_key is not None:
                        pump_signature.append(pump_key)
                key = (round(net_freq, 6), tuple(sorted(pump_signature)))
                contribution = (g * amplitude) * operator
                if key in groups:
                    groups[key][0] += contribution
                else:
                    groups[key] = [contribution, net_freq]   # [operator_sum, exact Omega]

        return [(exact_omega, key[1], operator_sum)
                for key, (operator_sum, exact_omega) in groups.items()]

    # -- analytic effective-rate estimator (Eq. 62) -------------------------
    def effective_rate(self, modes: Sequence[int], n: int,
                       C: Optional[int] = None, eta: Optional[float] = None) -> float:
        r"""RWA process strength g_eff = C g_n eta^k prod_i lambda_is (Eq. 62).

        Parameters
        ----------
        modes : sequence of int
            Participating non-coupler modes, with multiplicity. The pump-quanta
            count is k = n - len(modes).
        n : int
            Non-linear order of the process (must have a defined g_n).
        C : int, optional
            Multinomial coefficient of the process. Defaults to n! (n distinct
            factors); pass it explicitly for degenerate factors.
        eta : float, optional
            Pump amplitude |eta|. Defaults to |eta| at the envelope peak of the
            first tone.

        Returns:
        -------
        float
            The effective rate g_eff (rad/ns).
        """
        g = self.g_n.get(n)
        if g is None:
            raise ValueError(f"no g_{n} defined.")
        n_pump_quanta = n - len(modes)
        if n_pump_quanta < 0:
            raise ValueError("len(modes) cannot exceed n.")
        if eta is None:
            if not self._pump_tones:
                raise ValueError("Set a pump or pass eta explicitly.")
            tone = self._pump_tones[0]
            eta = abs(self._eta(tone, tone.envelope.t_g / 2))
        participation_product = float(np.prod([self.participation[m] for m in modes])) if modes else 1.0
        if C is None:
            C = factorial(n)
        return float(C * g * (eta ** n_pump_quanta) * participation_product)

    def iswap_rate(self, a: int, b: int, eta: Optional[float] = None) -> float:
        """ISWAP coupling g_eff = 6 g3 lambda_as lambda_bs |eta| (Eqs. 55/73).

        Parameters
        ----------
        a, b : int
            The two qubit-mode indices.
        eta : float, optional
            Pump amplitude |eta|; defaults to the envelope peak of the first tone.

        Returns:
        -------
        float
            The iSWAP rate (rad/ns).
        """
        return self.effective_rate([a, b], n=3, C=6, eta=eta)

    def peak_eta(self, tone_index: int = 0) -> float:
        """|eta| at the envelope peak -- the perturbative-pump diagnostic (the
        dressed-mode expansion needs |eta| << 1).

        Parameters
        ----------
        tone_index : int, default 0
            Which attached tone to evaluate.

        Returns:
        -------
        float
            |eta| at t_g/2.
        """  # noqa: D205
        tone = self._pump_tones[tone_index]
        return abs(self._eta(tone, tone.envelope.t_g / 2.0))

    # -- gate-metric helper -------------------------------------------------
    @staticmethod
    def _iswap_fidelity_from_U(U: np.ndarray, fit_virtual_z: bool) -> Tuple[float, float]:
        """Leakage-aware average gate fidelity and leakage from a 4x4 projected
        propagator (Pedersen PLA 367, 47 (2007); Wood & Gambetta PRA 97, 032306
        (2018)); virtual-Z phases optionally fitted out (free in software).
        """  # noqa: D205
        d = 4
        U_ideal = _ideal_iswap()
        U_fit = _fit_virtual_z(U, U_ideal) if fit_virtual_z else U
        overlap = np.abs(np.trace(U_ideal.conj().T @ U_fit)) ** 2
        trace_UU = np.real(np.trace(U_fit.conj().T @ U_fit))
        fidelity = (overlap + trace_UU) / (d * (d + 1))
        leakage = 1.0 - trace_UU / d
        return float(fidelity), float(leakage)

    def _subspace_indices(self, a: int, b: int) -> List[int]:
        """Flat indices of |00>, |01>, |10>, |11> on the (a, b) pair, all other
        modes in |0>.
        """  # noqa: D205
        indices = []
        for occ_a, occ_b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            occupations = [0] * self.n_modes
            occupations[a], occupations[b] = occ_a, occ_b
            indices.append(self.fock_index(occupations))
        return indices

    # -- QuTiP Hamiltonian export (exact, sparse, fast) ---------------------
    def to_qutip_hamiltonian(self, cutoff_GHz: float = np.inf,
                             sparse: bool = True) -> Any:
        r"""Build the QuTiP list-format QobjEvo, [[O_j, c_j(t)], ...], with CONSTANT
        operators O_j (sparse CSR by default) and scalar coefficients
        c_j(t) = exp(-i Omega_j t) prod_p eta_p(t)^(...). g_n and the participation
        factors are folded into O_j.

        Because the operators are built ONCE, QuTiP's compiled solver does sparse
        matrix-vector products each step instead of rebuilding the dense X^n -- the
        fast path. With cutoff_GHz = inf (default) it prunes NOTHING:
        sum_j c_j(t) O_j reproduces hamiltonian_matrix(t) exactly.

        Parameters
        ----------
        cutoff_GHz : float, default inf
            Passed to `expand_terms`; inf keeps the full, exact Hamiltonian. Pass
            a finite value only to obtain the RWA-reduced model.
        sparse : bool, default True
            Store each constant operator as a SciPy CSR matrix (the fast path);
            False keeps dense arrays.

        Returns:
        -------
        qutip.QobjEvo or list
            A QobjEvo on QuTiP 5; the raw list on QuTiP 4 (both accepted by the
            solvers). Feed to qt.sesolve (closed system) or qt.mesolve /
            qt.propagator with collapse operators (T1/T2, coupler loss) for the
            open-system run that hardware adds on top of Zhou's unitary model.
        """  # noqa: D205
        import cmath
        import qutip as qt
        import scipy.sparse as sp

        dims = [self.dims, self.dims]
        pump_tones = list(self._pump_tones)

        def make_coeff(omega: float, pump_signature: Tuple[Tuple[int, bool], ...]
                       ) -> Callable[[float, Any], complex]:
            def coeff(t: float, args: Any = None) -> complex:
                value = cmath.exp(-1j * omega * t)
                for tone_index, is_conjugate in pump_signature:
                    eta = self._eta(pump_tones[tone_index], t)
                    value *= eta.conjugate() if is_conjugate else eta
                return value
            return coeff

        H: List[Any] = []
        for omega, pump_signature, operator in self.expand_terms(cutoff_GHz):
            matrix = sp.csr_matrix(operator) if sparse else np.asarray(operator, dtype=complex)
            qobj = qt.Qobj(matrix, dims=dims)
            if abs(omega) < 1e-9 and not pump_signature:
                H.append(qobj)                                  # genuinely static term
            else:
                H.append([qobj, make_coeff(float(omega), pump_signature)])
        try:
            return qt.QobjEvo(H)
        except Exception:
            return H                                            # QuTiP 4: pass the list

    # -- time evolution & gate metrics (QuTiP compiled solver) --------------
    def _qutip_options(self, atol: float, rtol: float, nsteps: int) -> Any:
        """Return a solver-options object compatible with QuTiP 4 or 5."""
        import qutip as qt
        try:
            return qt.Options(atol=atol, rtol=rtol, nsteps=nsteps)     # QuTiP 4
        except AttributeError:
            return {"atol": atol, "rtol": rtol, "nsteps": nsteps}       # QuTiP 5

    def _sesolve_final(self, H: Any, start_index: int, t_g: float,
                       options: Any) -> np.ndarray:
        """Closed-system evolve a single computational-basis ket to t_g and return
        the final state vector (numpy).
        """  # noqa: D205
        import qutip as qt
        vector = np.zeros(self.dim, dtype=complex)
        vector[start_index] = 1.0
        psi0 = qt.Qobj(vector.reshape(-1, 1), dims=[self.dims, [1] * self.n_modes])
        result = qt.sesolve(H, psi0, [0.0, t_g], options=options)
        return result.states[-1].full().ravel()

    def evolve_state(self, init_occupations: Sequence[int], t_g: float,
                     atol: float = 1e-10, rtol: float = 1e-8,
                     nsteps: int = 500000) -> np.ndarray:
        """Evolve a Fock initial state to t_g on the EXACT Hamiltonian via QuTiP's
        compiled sesolve.

        Parameters
        ----------
        init_occupations : sequence of int
            Per-mode photon numbers of the initial state.
        t_g : float
            Final time (ns).
        atol, rtol : float
            Absolute / relative ODE tolerances.
        nsteps : int
            Maximum internal solver steps between outputs.

        Returns:
        -------
        ndarray, shape (dim,)
            Final state vector |psi(t_g)>.
        """  # noqa: D205
        H = self.to_qutip_hamiltonian()
        options = self._qutip_options(atol, rtol, nsteps)
        return self._sesolve_final(H, self.fock_index(init_occupations), t_g, options)

    def propagator_columns(self, a: int, b: int, t_g: float,
                           atol: float = 1e-10, rtol: float = 1e-8,
                           nsteps: int = 500000) -> np.ndarray:
        """4x4 projection of the realised propagator onto the (a, b) computational
        subspace (Pedersen/Wood convention; all spectators start in |0>), via
        QuTiP's compiled sesolve on the EXACT Hamiltonian. The Hamiltonian is built
        once and reused across the four columns.

        Parameters
        ----------
        a, b : int
            The two target-qubit mode indices.
        t_g : float
            Gate duration (ns).
        atol, rtol : float
            Absolute / relative ODE tolerances.
        nsteps : int
            Maximum internal solver steps between outputs.

        Returns:
        -------
        ndarray, shape (4, 4)
            The projected propagator (columns evolve |00>, |01>, |10>, |11>).
        """  # noqa: D205
        H = self.to_qutip_hamiltonian()
        options = self._qutip_options(atol, rtol, nsteps)
        indices = self._subspace_indices(a, b)
        U = np.zeros((4, 4), dtype=complex)
        for col, start in enumerate(indices):
            psi_final = self._sesolve_final(H, start, t_g, options)
            for row, end in enumerate(indices):
                U[row, col] = psi_final[end]
        return U

    def iswap_fidelity(self, a: int, b: int, t_g: float, fit_virtual_z: bool = True,
                       atol: float = 1e-10, rtol: float = 1e-8,
                       nsteps: int = 500000) -> Tuple[float, float, np.ndarray]:
        """Leakage-aware average iSWAP fidelity on (a, b) via QuTiP's compiled
        sesolve on the EXACT full Hamiltonian (no terms pruned).

        Parameters
        ----------
        a, b : int
            The two target-qubit mode indices.
        t_g : float
            Gate duration (ns).
        fit_virtual_z : bool, default True
            If True, divide out the optimal single-qubit virtual-Z phases (free in
            software) before scoring.
        atol, rtol : float
            Absolute / relative ODE tolerances.
        nsteps : int
            Maximum internal solver steps between outputs.

        Returns:
        -------
        fidelity : float
            Leakage-aware average gate fidelity vs the ideal iSWAP.
        leakage : float
            Population lost from the computational subspace, 1 - Tr(U^dag U)/4.
        U : ndarray, shape (4, 4)
            The projected propagator.

        Notes:
        -----
        Open system ('what hardware achieves'): build collapse operators from the
        embedded ladder ops -- e.g. sqrt(1/T1) qt.Qobj(self.a_ops[i]) for
        relaxation, sqrt(1/(2 Tphi)) qt.Qobj(2 self.ad_ops[i] @ self.a_ops[i]) for
        dephasing -- propagate the superoperator with
        qt.propagator(self.to_qutip_hamiltonian(), [0, t_g], c_ops=...), restrict
        to the computational subspace, and use qt.average_gate_fidelity.
        """  # noqa: D205
        U = self.propagator_columns(a, b, t_g, atol=atol, rtol=rtol, nsteps=nsteps)
        fidelity, leakage = self._iswap_fidelity_from_U(U, fit_virtual_z)
        return fidelity, leakage, U


# ===========================================================================
# Demo / self-test: reproduce Zhou's iSWAP coupling g_eff = 6 g3 la lb |eta|
# (uses the dense reference builder; runs without QuTiP)
# ===========================================================================
if __name__ == "__main__":
    # two qubits (a, b) + a SNAIL coupler (s); qubits are 2-level (Eq. 72 form)
    coupler = ZhouCoupler(
        mode_freqs_GHz=[4.0, 5.0, 7.0],
        coupler_index=2,
        participations={0: 0.15, 1: 0.15},      # lambda_as, lambda_bs = g_is/Delta_is
        nonlinearities={3: 0.10},               # g3 = 100 MHz
        levels=[2, 2, 6],
    )
    # one flat pump at w_b - w_a activates the iSWAP family (Eq. 73)
    pump_freq_GHz = coupler.delta(1, 0) / TWO_PI
    coupler.set_pump(PumpTone(w_p_GHz=pump_freq_GHz,
                              envelope=ConstantPulse(0.06, 50.0), is_eta=True))

    # the DC (time-averaged) matrix element <ge|H|eg> should equal g_eff
    eg = coupler.fock_index([1, 0, 0])
    ge = coupler.fock_index([0, 1, 0])
    pump_period = TWO_PI / coupler.delta(1, 0)
    times = np.linspace(0.0, 40 * pump_period, 8000, endpoint=False)
    dc_matrix_element = np.mean([coupler.hamiltonian_matrix(t)[ge, eg] for t in times])

    print(f"DC <ge|H|eg|/2pi          = {abs(dc_matrix_element) / TWO_PI * 1e3:.4f} MHz")
    print(f"Zhou 6 g3 la lb |eta| /2pi = {coupler.iswap_rate(0, 1) / TWO_PI * 1e3:.4f} MHz")
    print(f"ratio                      = {abs(dc_matrix_element) / coupler.iswap_rate(0, 1):.5f}  (expect 1.0)")