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


class TestFrequencyListHelpers(unittest.TestCase):
    """Campaign frequency lists must stay inside the band."""

    def test_wb_list_never_overshoots(self):
        import campaign
        for step in (0.10, 0.4, 0.0125, 0.3):
            vals = [float(v) for v in campaign._wb_list(3.8, 5.7, step).split(",")]
            self.assertLessEqual(max(vals), 5.7 + 1e-9,
                                 f"step={step} overshoots the band edge")
            self.assertAlmostEqual(min(vals), 3.8, places=9)
            self.assertAlmostEqual(max(vals), 5.7, places=9)

    def test_spec_list_centered(self):
        import campaign
        vals = [float(v) for v in campaign._spec_list(2.2, 0.2, 25).split(",")]
        self.assertEqual(len(vals), 25)
        self.assertAlmostEqual(np.mean(vals), 2.2, places=9)
        self.assertAlmostEqual(max(vals) - min(vals), 0.2, places=9)


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