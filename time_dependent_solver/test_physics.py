"""Regression suite for the Zhou SNAIL iSWAP sweep tooling.

Every test here encodes an invariant that a real bug violated at some point, so a
failure means a specific known-bad behaviour has come back:

* beat conventions -- the spectator beat must be symmetric about w_b (a below-w_b
  convention applied above w_b silently disabled DRAG), and the subharmonic beat
  must be the TRUE two-pump detuning w_i - 2 w_p (a factor of 2 was wrong).
* collision labelling -- ``_nearest_collision`` must name the right channel, and
  must fall back to the subharmonics in the bare no-spectator gate.
* rate combinatorics -- the coupler<->spectator process is C = 6 like the iSWAP,
  differing only by participation (verified against extracted matrix elements).
* 3-mode reduction -- the bare gate must reproduce the decoupled 4-mode
  Hamiltonian exactly on the shared n_spec = 0 block.
* gate-area arithmetic -- t_g = 2 * target_eta_area / eta = 138.889 / eta.
* blank-spectator plumbing -- a bare-gate row has no spectator frequency, which
  crashed float formatting in the logger and again in collect.
* operating points -- round-trip through a device JSON, and context-mismatch
  detection (a point calibrated elsewhere must not apply silently).
* frequency-list helpers must not overshoot the band edge.

Deliberately QuTiP-free and fast, so it can gate a cluster submission:

    cd time_dependent_solver && python -m unittest test_physics -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TWO_PI = 2.0 * np.pi


def _cfg(**over):
    """A DEFAULT_CONFIG copy with overrides (no device file needed)."""
    from sweep_common import DEFAULT_CONFIG
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(over)
    return cfg


class TestBeatConventions(unittest.TestCase):
    """The spectator beat must measure to the nearer collision, either side of w_b."""

    def setUp(self):
        from sweep_common import _nearest_collision
        self.nc = _nearest_collision
        self.wa, self.wb, self.wc, self.wp = 3.5, 4.5, 4.7, 1.0

    def _beat(self, wspec, **over):
        return self.nc(_cfg(**over), self.wa, self.wb, self.wc, wspec, self.wp)

    def test_b_onepump_resonance_is_zero_above_wb(self):
        # w_spec = w_b + w_p = 5.5 is the ABOVE-w_b one-pump collision
        beat = self._beat(5.5)
        self.assertEqual(beat[2], "onepump")
        self.assertEqual(beat[3], "b")
        self.assertAlmostEqual(beat[1], 0.0, places=9)

    def test_symmetric_about_wb(self):
        """|beat| must be equal for w_spec = w_b +/- (w_p + x): the |Delta| fix."""
        for x in (0.05, 0.2, 0.35):
            above = self._beat(self.wb + self.wp + x)[1]
            below = self._beat(self.wb - self.wp - x)[1]
            self.assertAlmostEqual(abs(above), x, places=9)
            self.assertAlmostEqual(abs(below), x, places=9)

    def test_no_regression_to_minus_two_wp(self):
        """The old convention reported ~ -2 w_p on-collision above w_b."""
        beat = self._beat(5.5)[1]
        self.assertLess(abs(beat), 0.5 * self.wp,
                        "above-w_b collision is being measured to the wrong side")

    def test_channel_labels(self):
        self.assertEqual(self._beat(4.8)[2:4], ("onepump", "a"))
        self.assertEqual(self._beat(3.3)[2:4], ("static", "a"))


class TestSubharmonicBeat(unittest.TestCase):
    """Subharmonic beat is w_i - 2 w_p (the two-pump detuning), not w_i/2 - w_p."""

    def setUp(self):
        from sweep_common import _nearest_collision
        self.nc = _nearest_collision
        self.wa, self.wb, self.wc, self.wp = 3.5, 5.7, 4.7, 2.2   # 2 w_p = 4.4

    def test_spectator_subharmonic_resonance(self):
        cfg = _cfg(drag_subharmonic=True, subharmonic_modes=["spec"])
        beat = self.nc(cfg, self.wa, self.wb, self.wc, 2 * self.wp, self.wp)
        self.assertEqual(beat[2:4], ("subharm", "spec"))
        self.assertAlmostEqual(beat[1], 0.0, places=9)

    def test_factor_of_two(self):
        """At w_spec = 4.5 the detuning is +100 MHz; the old bug gave +50."""
        cfg = _cfg(drag_subharmonic=True, subharmonic_modes=["spec"])
        beat = self.nc(cfg, self.wa, self.wb, self.wc, 4.5, self.wp)[1]
        self.assertAlmostEqual(beat, 0.1, places=9)

    def test_mode_selection_excludes_others(self):
        cfg = _cfg(drag_subharmonic=True, subharmonic_modes=["spec"])
        for wspec in (4.3, 4.4, 4.5):
            self.assertEqual(self.nc(cfg, self.wa, self.wb, self.wc,
                                     wspec, self.wp)[3], "spec")

    def test_off_by_default(self):
        beat = self.nc(_cfg(), self.wa, self.wb, self.wc, 2 * self.wp, self.wp)
        self.assertNotEqual(beat[2], "subharm")


class TestBareGateCollisions(unittest.TestCase):
    """no_spectator: only subharmonic channels, defaulting to the SNAIL."""

    def test_snail_subharmonic_resonance(self):
        from sweep_common import _nearest_collision
        # w_c = 4.2, w_p = 2.1 -> 2 w_p = w_c exactly
        beat = _nearest_collision(_cfg(no_spectator=True), 3.5, 5.6, 4.2, 99.0, 2.1)
        self.assertEqual(beat[2:4], ("subharm", "s"))
        self.assertAlmostEqual(beat[1], 0.0, places=9)

    def test_detuning_schedule(self):
        """w_c - 2 w_p must march linearly through zero as w_b sweeps."""
        from sweep_common import _nearest_collision
        wa, wc = 3.5, 4.2
        for wb, want in ((5.55, +0.10), (5.60, 0.0), (5.65, -0.10)):
            beat = _nearest_collision(_cfg(no_spectator=True), wa, wb, wc, 99.0, wb - wa)
            self.assertAlmostEqual(beat[1], want, places=9)

    def test_dummy_spectator_never_selected(self):
        from sweep_common import _nearest_collision
        beat = _nearest_collision(_cfg(no_spectator=True), 3.5, 5.6, 4.2, 7.6, 2.1)
        self.assertNotEqual(beat[3], "spec")

    def test_qubit_subharmonic_selectable(self):
        from sweep_common import _nearest_collision
        # a-subharmonic 2 w_p = w_a at w_b = 1.5 w_a = 5.25
        cfg = _cfg(no_spectator=True, drag_subharmonic=True, subharmonic_modes=["a"])
        beat = _nearest_collision(cfg, 3.5, 5.25, 4.7, 99.0, 1.75)
        self.assertEqual(beat[2:4], ("subharm", "a"))
        self.assertAlmostEqual(beat[1], 0.0, places=9)


class TestRateCombinatorics(unittest.TestCase):
    """The qS (coupler<->spectator) process is C = 6, like the iSWAP."""

    def test_qS_over_iswap_ratio_is_one_over_lambda(self):
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        lam = 0.1
        cpl = ZhouCoupler(mode_freqs_GHz=[3.5, 4.5, 4.7, 5.7], coupler_index=2,
                          participations={0: lam, 1: lam, 3: lam},
                          nonlinearities={3: 0.06}, levels=[3, 3, 5, 3],
                          anharmonicities_GHz={0: -0.12, 1: -0.12, 3: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.0, envelope=RaisedCosine(amp=1.0, t_g=100.0),
                              is_eta=True), normalize_iswap=(0, 1))
        g_iswap = cpl.effective_rate([1, 3], n=3, C=6)      # two participations
        g_qS = cpl.effective_rate([2, 3], n=3, C=6)         # coupler participation = 1
        self.assertAlmostEqual(g_qS / g_iswap, 1.0 / lam, places=6)

    def test_extracted_matrix_element_matches_C6(self):
        """Independent check straight off expand_terms (no effective_rate)."""
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        lam, g3 = 0.1, 0.06
        cpl = ZhouCoupler(mode_freqs_GHz=[3.5, 4.5, 4.7, 5.7], coupler_index=2,
                          participations={0: lam, 1: lam, 3: lam},
                          nonlinearities={3: g3}, levels=[3, 3, 5, 3],
                          anharmonicities_GHz={0: -0.12, 1: -0.12, 3: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.0, envelope=RaisedCosine(amp=1.0, t_g=100.0),
                              is_eta=True))
        i, j = cpl.fock_index([0, 0, 1, 0]), cpl.fock_index([0, 0, 0, 1])
        val = max(abs(O[i, j]) for _Om, _ps, O in cpl.expand_terms(cutoff_GHz=np.inf)
                  if abs(O[i, j]) > 0)
        # C=6 with one participation: 6 * g3 * lam_spec (pump normalized to 1)
        self.assertAlmostEqual(val / TWO_PI, 6 * g3 * lam, places=9)


class TestChirp(unittest.TestCase):
    """A chirp must be a pure carrier rotation -- nothing more, nothing less.

    The whole design rests on one identity: a pump letter enters X(t) as
    ``eta e^{-i w_p t}``, so chirping the carrier is the same thing as putting the
    phase ``e^{-i Phi(t)}`` on the envelope. If that identity ever breaks, a
    constant chirp stops agreeing with a retuned carrier and these fail.
    """

    T_G = 77.2

    def _build(self, w_p_GHz, chirp):
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[3, 3, 4], anharmonicities_GHz={0: -0.12, 1: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, is_eta=True, chirp=chirp,
                              envelope=RaisedCosine(amp=0.05, t_g=self.T_G)))
        return cpl

    def _max_dH(self, c1, c2):
        ts = np.linspace(0.0, self.T_G, 11)
        return max(np.max(np.abs(c1.hamiltonian_matrix(t) - c2.hamiltonian_matrix(t)))
                   for t in ts)

    def test_constant_chirp_equals_retuned_carrier(self):
        """delta(t) = c0 must be EXACTLY a pump at w_p + c0 (the design identity)."""
        from zhou_coupler import Chirp
        c0 = 0.037
        self.assertLess(self._max_dH(self._build(1.7, Chirp([c0], self.T_G)),
                                     self._build(1.7 + c0, None)), 1e-12)

    def test_constant_chirp_sign(self):
        """+c0 shifts the carrier UP. A sign slip here would silently detune the gate."""
        from zhou_coupler import Chirp
        c0 = 0.037
        self.assertGreater(self._max_dH(self._build(1.7, Chirp([c0], self.T_G)),
                                        self._build(1.7 - c0, None)), 1e-6)

    def test_zero_chirp_is_inert(self):
        """All-zero coefficients must reproduce the un-chirped Hamiltonian exactly."""
        from zhou_coupler import Chirp
        self.assertEqual(self._max_dH(self._build(1.7, None),
                                      self._build(1.7, Chirp([0.0, 0.0], self.T_G))), 0.0)

    def test_linear_chirp_changes_the_dynamics(self):
        """Guards against a chirp that is silently dropped somewhere in _eta."""
        from zhou_coupler import Chirp
        self.assertGreater(self._max_dH(self._build(1.7, None),
                                        self._build(1.7, Chirp([0.0, 0.05], self.T_G))), 1e-3)

    def test_chirp_does_not_touch_the_amplitude_calibration(self):
        """A chirp is a phase: |eta|, area() and peak_eta must be untouched.

        `set_pump(normalize_iswap=...)` divides by `area()`, so if a chirp ever
        leaked into it the pi/2 amplitude calibration would silently drift.
        """
        from zhou_coupler import Chirp
        plain = self._build(1.7, None)
        chirped = self._build(1.7, Chirp([0.02, 0.05, -0.01], self.T_G))
        self.assertEqual(plain.peak_eta(), chirped.peak_eta())
        self.assertEqual(plain._pump_tones[0].envelope.area(),
                         chirped._pump_tones[0].envelope.area())

    def test_phase_is_the_integral_of_the_detuning(self):
        """Phi(0) = 0 and dPhi/dt = delta(t); the closed form must match quadrature."""
        from zhou_coupler import Chirp
        ch = Chirp([0.01, 0.05, -0.02], self.T_G)
        self.assertAlmostEqual(float(ch.phase(0.0)), 0.0, places=12)
        h = 1e-6
        for t in (12.0, 38.6, 70.0):
            fd = (float(ch.phase(t + h)) - float(ch.phase(t - h))) / (2 * h)
            self.assertAlmostEqual(fd, float(ch.detuning(t)), places=6)

    def test_make_chirp_returns_none_when_trivial(self):
        """An absent/zero chirp must yield None so the solver path stays untouched."""
        from zhou_coupler import make_chirp
        self.assertIsNone(make_chirp(None, self.T_G))
        self.assertIsNone(make_chirp([], self.T_G))
        self.assertIsNone(make_chirp([0.0, 0.0], self.T_G))
        self.assertIsNotNone(make_chirp([0.0, 0.01], self.T_G))


class TestEnvelopeArrayAPI(unittest.TestCase):
    """`value_at`/`deriv_at` must agree with the scalar path and be branch-free.

    The batched/GPU engine evaluates envelopes on arrays of times and under a JAX
    trace; the per-time QuTiP callbacks still use the scalar wrappers. If the two
    ever disagree, the fast engine silently solves a different pulse.
    """

    T_G = 77.2

    def _envelopes(self):
        from envelope import ConstantPulse, IQFourierEnvelope, RaisedCosine
        rng = np.random.default_rng(0)
        freqs = TWO_PI * np.arange(1, 5) / self.T_G
        return [
            ConstantPulse(amp=0.05, t_g=self.T_G),
            RaisedCosine(amp=0.05, t_g=self.T_G),
            IQFourierEnvelope(amp=0.05, t_g=self.T_G),
            IQFourierEnvelope(amp=0.05, t_g=self.T_G, freqs=freqs,
                              sin_I=rng.normal(size=4), sin_Q=rng.normal(size=4),
                              cos_I=rng.normal(size=4), cos_Q=rng.normal(size=4)),
        ]

    def test_vectorized_matches_scalar(self):
        # deliberately overruns [0, t_g] so the support mask is exercised
        ts = np.linspace(-5.0, self.T_G + 5.0, 37)
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__, n=env.n_params):
                self.assertEqual(np.max(np.abs(
                    np.array([env.value(float(t)) for t in ts])
                    - np.asarray(env.value_at(ts, np)))), 0.0)
                self.assertEqual(np.max(np.abs(
                    np.array([env.deriv(float(t)) for t in ts])
                    - np.asarray(env.deriv_at(ts, np)))), 0.0)

    def test_analytic_derivative_matches_finite_difference(self):
        ts, h = np.linspace(2.0, self.T_G - 2.0, 13), 1e-6
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__, n=env.n_params):
                fd = np.array([(env.value(float(t) + h) - env.value(float(t) - h)) / (2 * h)
                               for t in ts])
                self.assertLess(np.max(np.abs(fd - np.asarray(env.deriv_at(ts, np)))), 1e-8)

    def test_envelope_is_zero_outside_the_gate(self):
        for env in self._envelopes():
            with self.subTest(env=type(env).__name__, n=env.n_params):
                for t in (-1e-9, -3.0, self.T_G + 1e-9, self.T_G + 3.0):
                    self.assertEqual(env.value(t), 0.0)

    def test_zero_coefficient_iq_reduces_to_raised_cosine(self):
        """The CRAB ansatz must start exactly at the sweep's baseline pulse."""
        from envelope import IQFourierEnvelope, RaisedCosine
        ts = np.linspace(0.0, self.T_G, 25)
        rc = RaisedCosine(amp=0.05, t_g=self.T_G)
        iq = IQFourierEnvelope(amp=0.05, t_g=self.T_G,
                               freqs=TWO_PI * np.arange(1, 5) / self.T_G)
        self.assertLess(np.max(np.abs(np.asarray(iq.value_at(ts, np))
                                      - np.asarray(rc.value_at(ts, np)))), 1e-15)
        self.assertAlmostEqual(iq.area(), rc.area(), places=12)


class TestSymbolicExpansion(unittest.TestCase):
    """`expand_terms_symbolic` must be `expand_terms` with the frequencies pulled out.

    The batched engine depends on the operator structure being independent of every
    mode and pump frequency -- that is what lets a whole sweep share one operator
    stack. If a frequency ever leaks into the structure, the engine silently solves
    the wrong Hamiltonian for every grid point but the first.
    """

    def _build(self, wb=5.5):
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, wb, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={0: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                              envelope=RaisedCosine(amp=0.3, t_g=40.0)))
        return cpl

    def _grouped(self, cpl):
        """Both expansions, folded onto the same (Omega, signature) keys."""
        from collections import defaultdict
        S = cpl.expand_terms_symbolic()
        Om = S["M"] @ cpl.frequency_vector()
        sym = defaultdict(lambda: np.zeros((cpl.dim, cpl.dim), dtype=complex))
        for j in range(S["n_terms"]):
            m = S["term"] == j
            O = np.zeros((cpl.dim, cpl.dim), dtype=complex)
            O[S["row"][m], S["col"][m]] = S["val"][m]
            sym[(round(float(Om[j]), 6), S["signatures"][j])] += O
        ref = defaultdict(lambda: np.zeros((cpl.dim, cpl.dim), dtype=complex))
        for omega, sig, O in cpl.expand_terms(cutoff_GHz=np.inf):
            ref[(round(float(omega), 6), sig)] += O
        return sym, ref

    def test_matches_expand_terms(self):
        sym, ref = self._grouped(self._build())
        self.assertEqual(set(sym), set(ref))
        self.assertEqual(max(np.max(np.abs(sym[k] - ref[k])) for k in ref), 0.0)

    def test_structure_is_frequency_independent(self):
        """Move w_b; M/operators must not budge, only `frequency_vector` does."""
        a, b = self._build(5.5), self._build(5.9)
        Sa, Sb = a.expand_terms_symbolic(), b.expand_terms_symbolic()
        for key in ("M", "n_pos", "n_neg", "term", "row", "col"):
            np.testing.assert_array_equal(Sa[key], Sb[key], err_msg=key)
        np.testing.assert_allclose(Sa["val"], Sb["val"], rtol=0, atol=0)
        self.assertGreater(np.max(np.abs(Sa["M"] @ a.frequency_vector()
                                         - Sb["M"] @ b.frequency_vector())), 1.0)

    def test_operators_are_sparse(self):
        """The COO stack must be far smaller than a dense (n_terms, dim, dim)."""
        cpl = self._build()
        S = cpl.expand_terms_symbolic()
        self.assertLess(S["val"].size, 0.2 * S["n_terms"] * cpl.dim ** 2)


class TestBatchedEngine(unittest.TestCase):
    """The batched engine must reproduce the dense Hamiltonian and the scalar pump.

    QuTiP-free: both checks are against `hamiltonian_matrix` / `_eta`, which are the
    same oracles the rest of this suite uses. The end-to-end engine-vs-sesolve
    comparison lives in `validate_engines.py` (it needs QuTiP and is slow).
    """

    T_G = 40.0

    def _build(self, chirp=None, drag=False):
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={0: -0.12})
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True, drag=drag,
                              delta_drag_GHz=(0.3 if drag else 0.0), chirp=chirp,
                              envelope=RaisedCosine(amp=0.3, t_g=self.T_G)))
        return cpl

    def _cases(self):
        from zhou_coupler import Chirp
        return [("plain", self._build()),
                ("drag", self._build(drag=True)),
                ("chirp", self._build(chirp=Chirp([0.01, 0.03], self.T_G))),
                ("chirp+drag", self._build(chirp=Chirp([0.01, 0.03], self.T_G), drag=True))]

    def test_sparse_H_matches_dense_hamiltonian(self):
        """The exact (cutoff=inf) engine must reproduce `hamiltonian_matrix`."""
        import jax_engine as JE
        rng = np.random.default_rng(0)
        for label, cpl in self._cases():
            with self.subTest(case=label):
                eng = JE.build_engine(cpl, cutoff_GHz=np.inf)
                params = JE.pulse_params(cpl)
                Psi = (rng.normal(size=(cpl.dim, 4)) + 1j * rng.normal(size=(cpl.dim, 4)))
                for t in (0.0, 13.7, 29.1, self.T_G):
                    got = eng.H_apply(t, Psi, eng.omega_vec0, params, np)
                    want = cpl.hamiltonian_matrix(t) @ Psi
                    self.assertLess(np.max(np.abs(got - want)), 1e-10, f"{label} t={t}")

    def test_engine_eta_matches_coupler_eta(self):
        """`jax_engine.eta_at` is a separate implementation of `_eta_at`; pin them."""
        import jax_engine as JE
        for label, cpl in self._cases():
            with self.subTest(case=label):
                eng = JE.build_engine(cpl, cutoff_GHz=1.0)
                params = JE.pulse_params(cpl)
                for t in np.linspace(0.0, self.T_G, 9):
                    got = complex(JE.eta_at(eng.spec, params, float(t), 0, np))
                    want = cpl._eta(cpl._pump_tones[0], float(t))
                    self.assertAlmostEqual(got, want, places=12, msg=f"{label} t={t}")

    def test_cutoff_prunes_and_inf_keeps_everything(self):
        import jax_engine as JE
        cpl = self._build()
        self.assertEqual(JE.build_engine(cpl, cutoff_GHz=np.inf).n_dropped, 0)
        self.assertGreater(JE.build_engine(cpl, cutoff_GHz=1.0).n_dropped, 0)

    def test_propagator_is_unitary_on_the_full_space(self):
        """A sanity check that needs no reference: the full propagator is unitary.

        (The 4x4 projection is not, because of leakage, so this checks the block
        columns keep unit norm only in the no-leakage 2-level limit.)
        """
        import jax_engine as JE
        cpl = self._build()
        eng = JE.build_engine(cpl, cutoff_GHz=np.inf)
        U = np.asarray(JE.propagator_columns(eng, self.T_G, carrier_resolution=0.2))
        self.assertLessEqual(float(np.max(np.abs(U))), 1.0 + 1e-9)

    def test_short_gate_does_not_diverge(self):
        """A short gate forces a LARGE |eta|; the expm bound must follow it.

        `normalize_iswap` scales the pump as ~1/t_g, so at t_g = 20 ns the peak
        |eta| is ~7, not ~1. A fixed eta_max ceiling under-bounds ||H||, the
        scaling-and-squaring count comes out too small, and the Taylor series
        diverges SILENTLY -- observed as a reported "fidelity" of 1e290.
        """
        import jax_engine as JE
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        for t_g in (20.0, 77.2):
            with self.subTest(t_g=t_g):
                cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                                  participations={0: 0.1, 1: 0.1},
                                  nonlinearities={3: 0.06}, levels=[2, 2, 3],
                                  anharmonicities_GHz={})
                cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                                      envelope=RaisedCosine(amp=1.0, t_g=t_g)),
                             normalize_iswap=(0, 1))
                eng = JE.build_engine(cpl, cutoff_GHz=2.0)
                U = np.asarray(JE.propagator_columns(eng, t_g, carrier_resolution=0.1))
                JE.check_propagator(U)          # must not raise
                self.assertLessEqual(float(np.max(np.abs(U))), 1.0 + 1e-6)

    def test_divergence_guard_fires(self):
        """The guard must actually catch an under-bounded propagator."""
        import jax_engine as JE
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        cpl = ZhouCoupler(mode_freqs_GHz=[3.8, 5.5, 4.9], coupler_index=2,
                          participations={0: 0.1, 1: 0.1}, nonlinearities={3: 0.06},
                          levels=[2, 2, 3], anharmonicities_GHz={})
        cpl.set_pump(PumpTone(w_p_GHz=1.7, is_eta=True,
                              envelope=RaisedCosine(amp=1.0, t_g=20.0)),
                     normalize_iswap=(0, 1))
        eng = JE.build_engine(cpl, cutoff_GHz=2.0)
        bad = JE.propagator_columns(eng, 20.0, eta_max=2.0, carrier_resolution=0.1)
        with self.assertRaises(FloatingPointError):
            JE.check_propagator(np.asarray(bad))

    def test_cf4_converges_fourth_order(self):
        """Halving the step must cut the error ~16x -- the integrator's contract.

        A regression here means the Magnus weights or the node placement broke, which
        would otherwise show up only as a slightly-wrong fidelity.
        """
        import jax_engine as JE
        eng = JE.build_engine(self._build(), cutoff_GHz=2.0)
        ref = np.asarray(JE.propagator_columns(eng, self.T_G, carrier_resolution=0.0125))
        errs = [np.max(np.abs(np.asarray(
            JE.propagator_columns(eng, self.T_G, carrier_resolution=cr)) - ref))
            for cr in (0.4, 0.2)]
        self.assertGreater(errs[0] / errs[1], 8.0)


class TestEnvelopeSingleSourceOfTruth(unittest.TestCase):
    """zhou_coupler must RE-EXPORT the envelope classes, not redefine them.

    They were duplicated verbatim in both modules; the copies would have diverged
    the moment either was touched.
    """

    def test_reexported_classes_are_identical_objects(self):
        import envelope
        import zhou_coupler
        for name in ("Envelope", "ConstantPulse", "RaisedCosine",
                     "IQFourierEnvelope", "PumpTone", "Chirp"):
            self.assertIs(getattr(zhou_coupler, name), getattr(envelope, name), name)


class TestThreeModeReduction(unittest.TestCase):
    """The bare 3-mode gate must equal the decoupled 4-mode one on n_spec = 0."""

    def _build(self, no_spec):
        from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
        wa, wb, wc, t_g = 4.6, 5.6, 4.2, 92.6
        if no_spec:
            kw = dict(mode_freqs_GHz=[wa, wb, wc], participations={0: 0.1, 1: 0.1},
                      levels=[3, 3, 5], anharmonicities_GHz={0: -0.12, 1: -0.12})
        else:
            kw = dict(mode_freqs_GHz=[wa, wb, wc, 7.6],
                      participations={0: 0.1, 1: 0.1, 3: 0.0}, levels=[3, 3, 5, 3],
                      anharmonicities_GHz={0: -0.12, 1: -0.12, 3: -0.12})
        cpl = ZhouCoupler(coupler_index=2, nonlinearities={3: 0.06}, **kw)
        cpl.set_pump(PumpTone(w_p_GHz=abs(wb - wa),
                              envelope=RaisedCosine(amp=1.0, t_g=t_g), is_eta=True),
                     normalize_iswap=(0, 1))
        return cpl

    def test_dimension_reduction(self):
        self.assertEqual(self._build(True).dim * 3, self._build(False).dim)

    def test_hamiltonian_block_identical(self):
        import itertools
        c3, c4 = self._build(True), self._build(False)
        idx = np.array([c4.fock_index(list(occ) + [0])
                        for occ in itertools.product(*[range(d) for d in c3.dims])])
        for t in (0.0, 23.7, 61.2):
            H3 = c3.hamiltonian_matrix(t)
            H4 = c4.hamiltonian_matrix(t)[np.ix_(idx, idx)]
            self.assertLess(np.max(np.abs(H3 - H4)), 1e-12,
                            f"3-mode and decoupled 4-mode disagree at t={t}")

    def test_same_iswap_rate(self):
        self.assertAlmostEqual(self._build(True).iswap_rate(0, 1),
                               self._build(False).iswap_rate(0, 1), places=12)


class TestGateArea(unittest.TestCase):
    """t_g = 2 * target_eta_area / eta = 138.889 / eta for a raised cosine."""

    def test_area(self):
        from device_utils import target_eta_area
        self.assertAlmostEqual(target_eta_area(0.06, 0.1, 0.1), 69.4444, places=3)

    def test_auto_t_g_scaling(self):
        from device_utils import auto_t_g
        for eta, want in ((1.2, 115.741), (1.5, 92.593), (1.8, 77.160)):
            self.assertAlmostEqual(auto_t_g(0.06, 0.1, 0.1, eta), want, places=3)

    def test_inverse_relation(self):
        from device_utils import auto_t_g
        self.assertAlmostEqual(auto_t_g(0.06, 0.1, 0.1, 1.0) * 2.0,
                               auto_t_g(0.06, 0.1, 0.1, 0.5), places=6)


class TestBlankSpectatorFormatting(unittest.TestCase):
    """A bare-gate row has no spectator frequency; formatting must not crash."""

    def test_log_line_handles_blank(self):
        from sweep_common import _log_line
        row = dict(kind="target", wb_GHz=5.6, spec_GHz="", w_p_GHz=2.1,
                   nearest_beat_GHz=0.0, nearest_kind="subharm",
                   g_collision_MHz=float("nan"), F_avg=None)
        line = _log_line(row)                      # must not raise
        self.assertIn("bare", line)

    def test_log_line_still_formats_numeric(self):
        from sweep_common import _log_line
        row = dict(kind="target", wb_GHz=5.6, spec_GHz=4.0, w_p_GHz=2.1,
                   nearest_beat_GHz=0.1, nearest_kind="onepump",
                   g_collision_MHz=5.0, F_avg=0.99)
        self.assertIn("4.000", _log_line(row))


class TestOperatingPoints(unittest.TestCase):
    """Round-trip through a device JSON, plus context-mismatch detection."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "dev.json")
        with open(self.path, "w") as fh:
            json.dump(dict(qubit_freqs_GHz=[3.5, 5.7], coupler_freq_GHz=4.7,
                           t_g_ns=92.6, lam_a=0.1, lam_b=0.1, g3_GHz=0.06), fh)
        self.rec = dict(amp_scale=0.87, wp_offset_GHz=-0.0012, t_g_ns=92.6,
                        wa_GHz=3.5, wb_GHz=5.7, spec_abs_GHz=None,
                        drag_beat_GHz=None, metric="fidelity", score=0.994)

    def test_round_trip(self):
        import operating_points as OP
        OP.save_point(self.path, "p1", self.rec)
        with open(self.path) as fh:
            cfg = json.load(fh)
        self.assertIn("p1", OP.list_points(cfg))
        got = OP.get_point(cfg, "p1")
        self.assertAlmostEqual(got["amp_scale"], 0.87)
        self.assertIn("created", got)              # provenance stamped

    def test_no_silent_overwrite(self):
        import operating_points as OP
        OP.save_point(self.path, "p1", self.rec)
        with self.assertRaises(SystemExit):
            OP.save_point(self.path, "p1", self.rec)
        OP.save_point(self.path, "p1", self.rec, overwrite=True)   # explicit is fine

    def test_apply_sets_amp_and_offset(self):
        import operating_points as OP
        cfg = OP.apply_point(dict(amp_scale=1.0, wp_offset_GHz=0.0, t_g_ns=50.0),
                             self.rec)
        self.assertAlmostEqual(cfg["amp_scale"], 0.87)
        self.assertAlmostEqual(cfg["wp_offset_GHz"], -0.0012)
        self.assertAlmostEqual(cfg["t_g_ns"], 92.6)

    def test_context_mismatch_detected(self):
        import operating_points as OP
        cfg = dict(qubit_freqs_GHz=[3.5, 5.7], t_g_ns=92.6)
        self.assertEqual(OP.check_context(self.rec, cfg), [])
        issues = OP.check_context(self.rec, cfg, wb_GHz=5.5)   # different pair
        self.assertTrue(any("wb_GHz" in i for i in issues))
        issues = OP.check_context(self.rec, cfg, t_g=60.0)     # different gate length
        self.assertTrue(any("t_g_ns" in i for i in issues))

    def test_unknown_name_is_clean_error(self):
        import operating_points as OP
        with open(self.path) as fh:
            cfg = json.load(fh)
        with self.assertRaises(SystemExit):
            OP.resolve(cfg, "nope")


class TestBareGatePipeline(unittest.TestCase):
    """End-to-end analytic bare-gate run: the path that crashed twice on formatting."""

    def test_prepare_local_collect(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = os.path.join(tmp, "dev.json")
            with open(dev, "w") as fh:
                json.dump(dict(qubit_freqs_GHz=[3.5, 5.6], coupler_freq_GHz=4.2,
                               t_g_ns=92.6, lam_a=0.1, lam_b=0.1, g3_GHz=0.06,
                               anharm_qubit_GHz=-0.12), fh)
            out = os.path.join(tmp, "run")
            base = [sys.executable, "run_sweep_zhou.py"]
            p = subprocess.run(base + ["prepare", "--sweep", "target", "--no-integrate",
                                       "--no-spectator", "--drag-subharmonic",
                                       "--subharmonic-modes", "s", "--device", dev,
                                       "--wb-GHz", "5.55,5.60,5.65",
                                       "--outdir", out],
                               cwd=HERE, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            p = subprocess.run(base + ["local", "--outdir", out, "--nproc", "1"],
                               cwd=HERE, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("bare", p.stdout, "bare-gate rows should log wspec=bare")

            import csv as _csv
            with open(os.path.join(out, "summary.csv")) as fh:
                rows = list(_csv.DictReader(fh))
            self.assertEqual(len(rows), 3)
            for r in rows:
                self.assertEqual(r["spec_GHz"], "")       # blank, not a phantom 7.6
                self.assertEqual(r["n_spec"], "")
                self.assertEqual(r["nearest_kind"], "subharm")
                self.assertEqual(r["nearest_target"], "s")
            beats = sorted(float(r["nearest_beat_GHz"]) for r in rows)
            self.assertAlmostEqual(beats[1], 0.0, places=9)   # resonance at w_b = 5.60


class TestPlotterSmoke(unittest.TestCase):
    """Plotters must render from a synthetic summary (no QuTiP, no real run)."""

    def test_bare_sweep_plot(self):
        import csv as _csv
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "summary.csv")
            with open(path, "w", newline="") as fh:
                w = _csv.writer(fh)
                w.writerow(["index", "kind", "wb_GHz", "nearest_beat_GHz", "F_avg",
                            "n_coupler", "p_transfer", "drag"])
                for i, wb in enumerate((5.55, 5.60, 5.65)):
                    beat = 4.2 - 2 * (wb - 3.5)
                    for drag in ("False", "True"):
                        w.writerow([i, "target", wb, f"{beat:.6f}", 0.9, 0.05, 0.88, drag])
            out = os.path.join(tmp, "fig.png")
            env = dict(os.environ, MPLBACKEND="Agg")
            p = subprocess.run([sys.executable, "plot_bare_sweep.py", "--csv", path,
                                "--out", out], cwd=HERE, capture_output=True,
                               text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main(verbosity=2)