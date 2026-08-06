"""
run_sweep_zhou.py
=================

Batch driver for ``zhou_coupler.ZhouCoupler`` -- the dressed-mode (first-
principles) counterpart of ``run_sweep.py``. Where ``run_sweep.py`` drives the
adiabatically-eliminated effective model (``SNAILProcessor``), this driver builds
the Hamiltonian directly from EXPERIMENTAL inputs in Chao Zhou's framework
(thesis Ch. 2): the coupler non-linearity g3 (+ optional g4) and the measured
participations lambda_is = g_is/Delta_is. The central SNAIL is an explicit mode,
not an effective edge.

System (4 modes)
----------------
    index 0 : target qubit a
    index 1 : target qubit b   (anchor for the spectator-frequency sweep)
    index 2 : coupler S (the SNAIL; carries g3/g4; participation 1)
    index 3 : spectator         (the "free" mode that is swept)

The pump sits on the coupler at  w_p = w_b - w_a  so the target iSWAP on (a,b)
is fixed; the pump amplitude is normalized to a full iSWAP at t_g. Only the
spectator frequency moves, so the target gate is untouched as the sweep runs:

    w_spec = w_b - (spec_freq_GHz * 2pi).

Sweep axes  (spec_freq x drag)
------------------------------
* spec_freq : spectator transition frequency Delta = w_b - w_spec (GHz), swept
              broadly. For a single-tone, pure-g3 coupler the spectator's only
              collision in the qubit band is the one-pump exchange with the
              anchor, resonant at |Delta| = w_p (beat = |Delta| - w_p); the |.|
              picks the nearer collision, so the spectator may sit below w_b
              (Delta = +w_p) or above it (Delta = -w_p). Two-mode higher-order
              collisions require g4 or a second pump tone; the FULL sim still
              captures whatever is actually present.
* drag      : first-order DRAG quadrature (Motzoi et al., PRL 103, 110501 (2009))
              on the pump, eta(t) -> eta(t) - i d eta/dt / (2 pi * beat), tuned to
              the spectator's beat detuning. Suppresses the OFF-resonant spectator;
              skipped (drag_applied=False) when |beat| < 5e-4 GHz, since an
              on-resonant collision needs frequency allocation, not DRAG.

The spectator is a single fixed 3-level anharmonic transmon (config ``spec_levels``
= 3, ``anharm_spec_GHz``) whose participation equals the qubits' ``lam_b``. Neither
the participation nor a "channel"/level count is a sweep axis.

Frequency-allocation sweep  (--sweep target)
--------------------------------------------
The spectator sweep above fixes the target pair and moves the spectator. The
allocation sweep instead fixes the SNAIL frequency w_s and target qubit a, and
scans a 2-D grid over the partner frequency w_b (``--wb-GHz``) and the spectator's
ABSOLUTE frequency w_spec (``--spec-GHz``) to find the placement with the best
iSWAP fidelity. Moving w_b moves the pump w_p=|w_b-w_a|, so each (w_b, w_spec)
sees a different collision landscape. For every point the analytic pass reports
the NEAREST collision over both target qubits q in {a,b} vs the spectator,
considering the pump-assisted exchange (resonant at |w_q-w_spec|=w_p,
``nearest_kind=onepump``) and the direct exchange (resonant at w_q=w_spec,
``nearest_kind=static``). A good allocation maximizes |nearest_beat_GHz|; the full
sim then reports F_avg, and ``collect`` prints the best-fidelity (w_b, w_spec).
Placements with |w_b-w_a| < ``min_detuning_GHz`` (pump too slow) are dropped.

Pass ``--drag-compare`` to add the effect of DRAG in the near-collision regime:
for every placement whose |nearest_beat| is below ``--drag-compare-below-MHz``
(default 100 MHz -- the off-resonant-but-close window where DRAG helps), the gate
is run BOTH without and with DRAG (Motzoi et al., tuned to that beat) in one point,
adding F_avg_drag / leakage_drag / dF_drag (= F_drag - F_off). Outside the window
DRAG is left off (well-detuned spectators do not need it; on-resonant ones need
allocation, not DRAG). ``collect`` then also reports the mean/best DRAG gain.

Two metrics per point
----------------------
* ANALYTIC (always, free -- no integration): nearest collision order m, its beat
  detuning, and the Eq.-62 effective rate g_spec_eff together with the target
  rate g_iswap_eff. This alone is the frequency-allocation collision map.
* FULL (config ``integrate=true``, default): evolve the EXACT dressed-mode
  Hamiltonian sum_n g_n X(t)^n with QuTiP's compiled solver and report the
  leakage-aware iSWAP fidelity on (a,b) plus the spectator and coupler
  occupations. For broad first passes set ``integrate=false`` (or
  --no-integrate) for the instant analytic map.

Solver
------
Time evolution uses QuTiP exclusively: ``zhou_coupler`` builds the exact
Hamiltonian as a sparse list-format QobjEvo (no terms pruned) and drives it
through ``qt.sesolve``. QuTiP must therefore be importable wherever
``integrate`` is True; the analytic ``--no-integrate`` path needs only
numpy/scipy. Tune the integrator with ``rtol``/``atol`` and ``nsteps``.

Workflow (identical CLI shape to run_sweep.py)
----------------------------------------------
    python run_sweep_zhou.py prepare --outdir results_zhou/
    RUNNER=run_sweep_zhou.py OUTDIR=results_zhou sbatch --array=0-<M-1> slurm/snail_sweep.slurm
    python run_sweep_zhou.py local --outdir results_zhou/ --nproc 8
    python run_sweep_zhou.py collect --outdir results_zhou/

Modes 'prepare' and 'collect' do NOT import the solver; 'point'/'local' import
``zhou_coupler`` lazily.
"""

from __future__ import annotations

import os
import argparse
import glob
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Shared machinery (constants, DEFAULT_*, Point, grid IO, _nearest_collision,
# _stark_offset_GHz, collect, CLI helpers) and the two sweep implementations.
from sweep_common import *                     # noqa: F401,F403  (API re-exported)
from sweep_common import (_drag_skip_GHz, _nearest_collision, _stark_offset_GHz,
                          _parse_list, _bool_list, _log_line, _print_submit_hint,
                          _compress_ranges)
from sweep_spectator import build_grid, run_spectator_point
from sweep_target import build_target_grid, _run_target_point


def run_point(pt: "Point", config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one grid point, dispatching by ``pt.kind`` to the spectator or the
    target evaluator. Kept as the single entry the CLI (and external callers) use.

    Parameters
    ----------
    pt : Point
        Grid point; ``kind`` selects the evaluator (\"spectator\" or \"target\").
    config : dict
        Resolved device/simulation configuration.

    Returns
    -------
    dict
        The per-point result row (same contract for both sweep kinds).
    """
    if getattr(pt, "kind", "spectator") == "target":
        return _run_target_point(pt, config)
    return run_spectator_point(pt, config)


def main() -> None:
    """Command-line entry point.

    Subcommands: ``prepare`` (write grid.json), ``point`` (run one --index),
    ``local`` (run the whole grid with a process pool, then collect), and
    ``collect`` (gather points into summary.csv / combined.npz). Run
    ``--help`` for the full option list.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prepare", "point", "local", "collect", "chevrons",
                                     "missing"])
    ap.add_argument("--outdir", default="results_zhou")
    ap.add_argument("--device", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--index", type=int, help="point index (mode=point); with --chunk N "
                    "this is the CHUNK index and the task runs points [index*N, (index+1)*N)")
    ap.add_argument("--chunk", type=int, default=1,
                    help="points per array task (mode=point). Use >1 when the number of "
                         "points exceeds SLURM MaxArraySize: array size becomes ceil(N/chunk).")
    ap.add_argument("--grid", help="path to grid.json (default: <outdir>/grid.json)")
    ap.add_argument("--nproc", type=int, default=4, help="processes for mode=local")
    ap.add_argument("--stark-jobs", type=int, default=None,
                    help="processes for the per-point --stark chevron's offset scan. "
                         "mode=point: defaults to SLURM_CPUS_PER_TASK (parallelize the "
                         "chevron across the task's cores); mode=local: forced to 1 "
                         "(points are already pooled).")
    ap.add_argument("--sweep", choices=["spectator", "target"], default="spectator",
                    help="spectator sweep (default) or target-frequency allocation sweep")
    ap.add_argument("--specfreqs", help="comma list of spectator freqs Delta = w_b - w_spec (GHz)")
    ap.add_argument("--beats", help="[spectator] comma list of BEAT detunings delta = Delta - w_p "
                                    "(GHz); 0 = on the collision. Sets spec_freq = w_p + delta.")
    ap.add_argument("--beat-span-MHz", type=float, default=None,
                    help="[spectator] auto beat sweep: +/-span/2 about the collision (delta=0)")
    ap.add_argument("--beat-points", type=int, default=13,
                    help="[spectator] number of points for --beat-span-MHz (default 13)")
    ap.add_argument("--clip-band", action="store_true",
                    help="[spectator] drop points whose ABSOLUTE spectator frequency "
                         "w_spec = w_b - Delta falls outside the physical band [w_a, w_b] "
                         "(a spectator qubit lives between the computational qubits)")
    ap.add_argument("--drags", help="comma list of bools, e.g. false,true")
    ap.add_argument("--t-g-ns", type=float, default=None,
                    help="gate duration t_g (ns); overrides the device/default value")
    ap.add_argument("--resume-list", default=None,
                    help="[point] file of point indices (one per line, e.g. missing.txt); "
                         "--index selects the chunk-th slice of this list instead of the "
                         "full grid range. Written by `missing`.")
    ap.add_argument("--target-eta", type=float, default=None,
                    help="set t_g via auto_t_g so the raised-cosine full-iSWAP pump has "
                         "peak |eta| = TARGET_ETA (t_g = 2*area/eta; on this device "
                         "eta=1.2 -> ~115.7 ns, eta=1.5 -> ~92.6 ns); takes precedence "
                         "over --t-g-ns")
    ap.add_argument("--wb-GHz", help="[target] comma list of partner (w_b) freqs (GHz)")
    ap.add_argument("--spec-GHz", help="[target] comma list of spectator ABSOLUTE freqs (GHz)")
    ap.add_argument("--spec-min-GHz", type=float, default=None,
                    help="[target] lower edge of the auto spectator band (GHz); "
                         "default min(w_a, min w_b) - 0.40")
    ap.add_argument("--spec-max-GHz", type=float, default=None,
                    help="[target] upper edge of the auto spectator band (GHz); "
                         "default max(w_a, max w_b) + 0.40, i.e. above w_b. Raise this "
                         "to place the spectator well above the partner qubit.")
    ap.add_argument("--spec-step-GHz", type=float, default=None,
                    help="[target] spectator grid step for the auto band (GHz); default 0.10")
    ap.add_argument("--drag-compare", action="store_true",
                    help="[target] also run DRAG-on where |nearest beat| < --drag-compare-below-MHz")
    ap.add_argument("--drag-compare-below-MHz", type=float, default=None,
                    help="[target] near-collision window for the DRAG comparison (default 100)")
    ap.add_argument("--no-integrate", action="store_true",
                    help="analytic collision map only (no time integration)")
    ap.add_argument("--stark", action="store_true",
                    help="drive each point at its AC-Stark-shifted resonance, located "
                         "by a per-point chevron that INCLUDES the spectator (so the "
                         "offset varies with detuning); needs the integrated run and "
                         "is markedly slower (one extra chevron per point)")
    ap.add_argument("--stark-match-pulse", action="store_true",
                    help="make the --stark chevron use the ACTUAL gate pulse (raised-"
                         "cosine, plus DRAG on DRAG-on points) instead of a constant "
                         "probe, so DRAG-on points calibrate onto the DRAG-on resonance "
                         "(captures the DRAG-quadrature Stark shift)")
    ap.add_argument("--calibrate", action="store_true",
                    help="per-point amplitude+Stark tune-up (calibrate_gate) before each "
                         "gate, with the SPECTATOR LOADED and DRAG on if the point uses it "
                         "(hardware-style per-point calibration); slowest, most faithful")
    ap.add_argument("--calibrate-iters", type=int, default=None,
                    help="amplitude/frequency rounds per point for --calibrate (default 1)")
    ap.add_argument("--grape", action="store_true",
                    help="per-point GRAPE optimal-control run (grape.optimize_pulse) on "
                         "top of the integrated gate; records grape_baseline_F / F_grape / "
                         "dF_grape / leak_grape and stores the optimized envelope. Opt-in "
                         "and costly (an L-BFGS-B optimization per point); needs --integrate")
    ap.add_argument("--grape-backend", choices=["qutip", "reduced"], default=None,
                    help="GRAPE engine: 'qutip' (qutip-qoc optimal control, default) "
                         "or 'reduced' (in-house scipy over the reduced model)")
    ap.add_argument("--grape-alg", choices=["GOAT", "JOPT"], default=None,
                    help="qutip-qoc analytic-control algorithm (JOPT needs JAX)")
    ap.add_argument("--grape-nbasis", type=int, default=None,
                    help="sin() basis functions per quadrature (qutip GRAPE backend)")
    ap.add_argument("--grape-nctrl", type=int, default=None,
                    help="GRAPE piecewise-constant control points (default 24)")
    ap.add_argument("--grape-cutoff-GHz", type=float, default=None,
                    help="GRAPE reduced-model carrier cutoff (default 1.0 GHz)")
    ap.add_argument("--grape-maxiter", type=int, default=None,
                    help="GRAPE L-BFGS-B iteration cap (default 200)")
    ap.add_argument("--grape-warmstart-drag", action="store_true",
                    help="on DRAG-off points, seed GRAPE from a DRAG raised cosine at the "
                         "nearest beat (better start near a collision; baseline/dF "
                         "unchanged; skipped inside the drag-skip window)")
    ap.add_argument("--drag", action="store_true",
                    help="force DRAG on for every allocation point (tuned to the nearest beat)")
    ap.add_argument("--operating-point", default=None,
                    help="use a calibrated operating point stored in the device JSON "
                         "(amp_scale + wp_offset, and t_g unless --t-g-ns/--target-eta "
                         "is given). Create one with calibration_map.py --save-point; "
                         "list them with operating_points.py --device <dev>")
    ap.add_argument("--operating-point-strict", action="store_true",
                    help="fail instead of warning when the operating point was "
                         "calibrated in a different context (w_a/w_b/t_g/spectator)")
    ap.add_argument("--drag-subharmonic", action="store_true",
                    help="let DRAG target the nearest SUBHARMONIC collision (the pump's "
                         "2nd harmonic 2 w_p, generated by the SNAIL, drives mode i at "
                         "w_i = 2 w_p, for i in --subharmonic-modes) in addition to the "
                         "one-pump swap; DRAG follows whichever channel is closest")
    ap.add_argument("--subharmonic-modes", default=None,
                    help="comma list selecting which subharmonic channels to include with "
                         "--drag-subharmonic: subset of {a,b,spec,s}. e.g. 'spec' isolates the "
                         "swept spectator subharmonic; 's' the SNAIL/coupler one. Default all.")
    ap.add_argument("--no-spectator", action="store_true",
                    help="[target] sweep the BARE a-b-coupler gate with NO spectator "
                         "(lam_spec=0). Vary --wb-GHz so 2 w_p scans the SNAIL subharmonic "
                         "w_c = 2 w_p; nearest-collision then reports/DRAGs w_c - 2 w_p. "
                         "NB w_c = 2 w_p needs w_p = w_c/2; with w_a=3.5 and coupler 4.7 that "
                         "is out of band (needs w_b=5.85) -- lower the coupler to <=4.4.")
    ap.add_argument("--stark-span-MHz", type=float, default=None,
                    help="per-point Stark chevron width (default 60)")
    ap.add_argument("--stark-points", type=int, default=None,
                    help="per-point Stark chevron offset samples (default 21)")
    ap.add_argument("--gpu", action="store_true",
                    help="run the solver on GPU via qutip-jax/diffrax (see zhou_coupler.use_gpu)")
    args = ap.parse_args()

    if args.gpu:
        import zhou_coupler
        zhou_coupler.use_gpu(True)

    from paths import resolve_device, in_results
    args.outdir = in_results(args.outdir)                 # bare name -> results/
    config = dict(DEFAULT_CONFIG)
    if args.device:
        with open(resolve_device(args.device)) as f:      # bare name -> devices/
            config.update(json.load(f))
    # gate duration: explicit --t-g-ns, or --target-eta -> auto_t_g (raised-cosine peak
    # |eta|). target_eta wins if both are given. Baked into the grid config at prepare.
    if args.t_g_ns is not None:
        config["t_g_ns"] = float(args.t_g_ns)
    if args.target_eta is not None:
        from device_utils import auto_t_g
        config["t_g_ns"] = float(auto_t_g(float(config["g3_GHz"]), float(config["lam_a"]),
                                          float(config["lam_b"]), float(args.target_eta)))
        _env = config.get("envelope", "raised_cosine")
        print(f"target_eta={args.target_eta} -> t_g = {config['t_g_ns']:.3f} ns "
              f"(auto_t_g, raised-cosine peak |eta|)")
        if _env != "raised_cosine":
            print(f"  warning: auto_t_g assumes a raised-cosine (Hann) pump; envelope is "
                  f"'{_env}', so the actual peak |eta| will differ (constant pulse: "
                  f"eta = area/t_g, not the Hann 2*area/t_g).")
    # a stored operating point supplies the calibrated (amp_scale, wp_offset) -- and
    # t_g unless an explicit --t-g-ns/--target-eta was given. Baked into the grid
    # config here, so every worker inherits it instead of re-deriving a tune-up.
    if getattr(args, "operating_point", None):
        from operating_points import resolve as _resolve_op
        _explicit_tg = (args.t_g_ns is not None) or (args.target_eta is not None)
        config, _pt = _resolve_op(config, args.operating_point,
                                  t_g=float(config["t_g_ns"]),
                                  strict=bool(getattr(args, "operating_point_strict", False)),
                                  set_t_g=not _explicit_tg)
        print(f"operating point '{args.operating_point}': amp_scale={config['amp_scale']}, "
              f"wp_offset={config['wp_offset_GHz']} GHz, t_g={config['t_g_ns']:.3f} ns"
              + ("  (t_g from CLI)" if _explicit_tg else ""))
    if args.no_integrate:
        config["integrate"] = False
    if args.stark:
        config["stark_drive"] = True
    if args.stark_match_pulse:
        config["stark_match_pulse"] = True
        if not config.get("stark_drive"):
            print("note: --stark-match-pulse also needs --stark to do anything "
                  "(it shapes the per-point Stark chevron).")
    if args.calibrate:
        config["calibrate_points"] = True
        if not config.get("integrate", True):
            print("note: --calibrate needs the integrated run; it has no effect with "
                  "--no-integrate (analytic map only).")
    if args.calibrate_iters is not None:
        config["calibrate_iters"] = int(args.calibrate_iters)
    if args.drag:
        config["drag_always"] = True
    if args.drag_subharmonic:
        config["drag_subharmonic"] = True
    if args.subharmonic_modes:
        config["subharmonic_modes"] = [m.strip() for m in
                                       args.subharmonic_modes.split(",") if m.strip()]
    if args.stark_span_MHz is not None:
        config["stark_span_MHz"] = float(args.stark_span_MHz)
    if args.stark_points is not None:
        config["stark_points"] = int(args.stark_points)
    if args.drag_compare:
        config["drag_compare"] = True
    if args.drag_compare_below_MHz is not None:
        config["drag_compare_below_MHz"] = float(args.drag_compare_below_MHz)
    if args.grape:
        config["grape"] = True
        if not config.get("integrate", True):
            print("note: --grape needs the integrated run; it has no effect with "
                  "--no-integrate (nothing to optimize against).")
    if args.grape_nctrl is not None:
        config["grape_nctrl"] = int(args.grape_nctrl)
    if args.grape_backend is not None:
        config["grape_backend"] = args.grape_backend
    if args.grape_alg is not None:
        config["grape_alg"] = args.grape_alg
    if args.grape_nbasis is not None:
        config["grape_nbasis"] = int(args.grape_nbasis)
    if args.grape_cutoff_GHz is not None:
        config["grape_cutoff_GHz"] = float(args.grape_cutoff_GHz)
    if args.grape_maxiter is not None:
        config["grape_maxiter"] = int(args.grape_maxiter)
    if args.grape_warmstart_drag:
        config["grape_warmstart_drag"] = True
        if not args.grape:
            print("note: --grape-warmstart-drag has no effect without --grape.")
    if args.no_spectator:
        config["no_spectator"] = True

    if args.mode == "prepare":
        if args.sweep == "target":
            nominal = config["qubit_freqs_GHz"]
            wa_GHz = float(nominal[0])                                   # fixed
            wb_default = [round(float(nominal[1]) - 0.30 + 0.02 * k, 3) for k in range(31)]
            wb_list = _parse_list(args.wb_GHz, float) or wb_default
            # default spectator band spans from below the lower qubit to ABOVE the
            # highest w_b, so spectators higher than the partner qubit are included.
            # Override the window with --spec-min-GHz/--spec-max-GHz/--spec-step-GHz,
            # or give an explicit list with --spec-GHz.
            wb_lo = min(float(w) for w in wb_list)
            wb_hi = max(float(w) for w in wb_list)
            band_lo = min(wa_GHz, wb_lo) - 0.40
            band_hi = max(wa_GHz, wb_hi) + 0.40
            spec_lo = band_lo if args.spec_min_GHz is None else float(args.spec_min_GHz)
            spec_hi = band_hi if args.spec_max_GHz is None else float(args.spec_max_GHz)
            spec_step = 0.10 if args.spec_step_GHz is None else float(args.spec_step_GHz)
            if spec_step <= 0:
                sys.exit("--spec-step-GHz must be > 0")
            if spec_hi < spec_lo:
                sys.exit("--spec-max-GHz must be >= --spec-min-GHz")
            n_spec = int(round((spec_hi - spec_lo) / spec_step)) + 1
            spec_default = [round(spec_lo + spec_step * k, 6) for k in range(max(n_spec, 1))]
            spec_list = _parse_list(args.spec_GHz, float) or spec_default
            if config.get("no_spectator"):
                # bare a-b-coupler gate: one decoupled dummy spectator (lam_spec=0 at
                # run time), placed far out of band so it never collides; only w_b sweeps.
                spec_list = [round(max(band_hi, 5.7) + 1.5, 3)]
                print(f"  no-spectator: bare a-b-coupler gate, sweeping w_b only "
                      f"({len(wb_list)} pts); 2*w_p scans the SNAIL subharmonic w_c=2w_p")
            if config.get("drag_compare"):
                if args.drags:
                    print("note: --drag-compare runs DRAG on/off per point; ignoring --drags.")
                drags = [False]
            else:
                drags = _bool_list(args.drags) or DEFAULT_TARGET_DRAGS
            points = build_target_grid(wa_GHz, wb_list, spec_list, drags,
                                       float(config.get("min_detuning_GHz", 0.05)))
            path = write_grid(args.outdir, config, points)
            m = len(points)
            print(f"Wrote {m} allocation points -> {path}")
            print(f"  fixed: w_a={wa_GHz:.4f} GHz, w_snail={config['coupler_freq_GHz']:.4f} GHz; "
                  f"spectator lam={config['lam_b']} (3-level anharmonic)")
            print(f"  scan: w_b ({len(wb_list)}) x w_spec ({len(spec_list)}) x drag ({len(drags)}) "
                  f"[dropped |detuning|<{config.get('min_detuning_GHz', 0.05)} GHz]")
            _above = sum(1 for s in spec_list if s > wb_hi)
            print(f"  spectator band: {min(spec_list):.3f}..{max(spec_list):.3f} GHz "
                  f"({_above} point(s) above the highest w_b = {wb_hi:.3f} GHz)")
            if config.get("drag_compare"):
                print(f"  DRAG comparison ON for |nearest beat| < "
                      f"{config.get('drag_compare_below_MHz', 100.0):.0f} MHz")
            print(f"integrate = {config['integrate']}  "
                  f"({'FULL sim per point' if config['integrate'] else 'analytic map only'})")
            _print_submit_hint(args.outdir, m)
            return
        # Detuning axis. Most intuitive is the BEAT delta = spec_freq - w_p (0 = on
        # the collision); convert to spec_freq = w_p + delta. Fall back to explicit
        # --specfreqs (Delta = w_b - w_spec) or the default broad axis.
        wa_g, wb_g = (float(x) for x in config["qubit_freqs_GHz"])
        w_p = abs(wb_g - wa_g) + float(config.get("wp_offset_GHz", 0.0))
        if args.beat_span_MHz is not None:
            half = float(args.beat_span_MHz) / 2000.0            # MHz full-width -> GHz half
            beats = np.linspace(-half, half, int(args.beat_points))
            specfreqs = [round(w_p + float(b), 6) for b in beats]
            print(f"beat-centered sweep: collision at spec_freq = w_p = {w_p:.4f} GHz; "
                  f"delta in +/-{args.beat_span_MHz/2:.0f} MHz, {args.beat_points} points")
        elif args.beats:
            beats = _parse_list(args.beats, float)
            specfreqs = [round(w_p + float(b), 6) for b in beats]
            print(f"beat-centered sweep: collision at spec_freq = w_p = {w_p:.4f} GHz")
        else:
            specfreqs = _parse_list(args.specfreqs, float) or DEFAULT_SPECFREQS_GHz
        drags = _bool_list(args.drags) or DEFAULT_DRAGS
        # Physical spectator band: a spectator qubit sits BETWEEN the computational
        # qubits, so w_spec in [w_a, w_b]. spec_freq is the DETUNING Delta = w_b - w_spec,
        # hence w_spec = w_b - Delta; beat = Delta - w_p > 0 places w_spec below w_a.
        band_lo, band_hi = min(wa_g, wb_g), max(wa_g, wb_g)
        wspec = [round(wb_g - float(sf), 6) for sf in specfreqs]
        oob = [(sf, ws) for sf, ws in zip(specfreqs, wspec)
               if not (band_lo <= ws <= band_hi)]
        if oob:
            print(f"WARNING: {len(oob)}/{len(specfreqs)} spectator points are OUTSIDE the "
                  f"physical band [{band_lo:.3f}, {band_hi:.3f}] GHz: absolute w_spec in "
                  f"[{min(w for _, w in oob):.3f}, {max(w for _, w in oob):.3f}] GHz "
                  f"(w_spec = w_b - Delta). Visualize with plot_allocation.py.")
            if args.clip_band:
                kept = [sf for sf, ws in zip(specfreqs, wspec) if band_lo <= ws <= band_hi]
                print(f"         --clip-band: keeping {len(kept)} in-band points, "
                      f"dropping {len(specfreqs) - len(kept)}.")
                specfreqs = kept
            else:
                print("         (pass --clip-band to drop them, or --allow it by ignoring "
                      "this warning if you are deliberately probing out-of-band.)")
        points = build_grid(specfreqs, drags)
        path = write_grid(args.outdir, config, points)
        m = len(points)
        print(f"Wrote {m} points -> {path}")
        print(f"integrate = {config['integrate']}  "
              f"({'FULL sim per point' if config['integrate'] else 'analytic map only'})")
        _print_submit_hint(args.outdir, m)
        return

    if args.mode == "collect":
        collect(args.outdir)
        return

    if args.mode == "missing":
        find_missing(args.outdir)
        return

    if args.mode == "chevrons":
        idxs = [args.index] if args.index is not None else None
        paths = plot_chevrons(args.outdir, idxs)
        dest = os.path.join(args.outdir, "figs", "chevrons")
        if paths:
            print(f"Rendered {len(paths)} chevron figure(s) -> {dest}")
        else:
            print(f"No saved chevrons found in {args.outdir}/points (run with --stark).")
        return

    grid_path = args.grid or os.path.join(args.outdir, "grid.json")
    if not os.path.exists(grid_path):
        sys.exit(f"grid.json not found at {grid_path}; run `prepare` first.")
    config, points = load_grid(args.outdir if not args.grid
                               else os.path.dirname(args.grid) or ".")

    # load_grid replaces `config` with the grid's copy, so execution flags meant to
    # be settable at run time (not just at prepare) must be RE-APPLIED here. Only
    # force-on when explicitly passed (store_true), so a value baked in at prepare is
    # preserved when the flag is absent. This removes the trap where --stark-match-pulse
    # passed at point/local time was silently dropped.
    if args.stark:
        config["stark_drive"] = True
    if args.stark_match_pulse:
        config["stark_match_pulse"] = True
    if args.drag_subharmonic:
        config["drag_subharmonic"] = True
    if args.subharmonic_modes:
        config["subharmonic_modes"] = [m.strip() for m in
                                       args.subharmonic_modes.split(",") if m.strip()]
    if config.get("stark_match_pulse") and not config.get("stark_drive"):
        print("note: stark_match_pulse has no effect without --stark (no per-point chevron).")

    # GRAPE is a RUN-TIME choice too (it costs an optimization per point, so you may
    # want it on a rerun of an existing grid, not baked in at prepare). Same
    # convention: store_true only forces ON; a value already in the grid survives an
    # absent flag, and the tunables override whenever explicitly passed.
    if args.grape:
        config["grape"] = True
    if args.grape_warmstart_drag:
        config["grape_warmstart_drag"] = True
    if args.grape_backend is not None:
        config["grape_backend"] = args.grape_backend
    if args.grape_alg is not None:
        config["grape_alg"] = args.grape_alg
    if args.grape_nbasis is not None:
        config["grape_nbasis"] = int(args.grape_nbasis)
    if args.grape_nctrl is not None:
        config["grape_nctrl"] = int(args.grape_nctrl)
    if args.grape_cutoff_GHz is not None:
        config["grape_cutoff_GHz"] = float(args.grape_cutoff_GHz)
    if args.grape_maxiter is not None:
        config["grape_maxiter"] = int(args.grape_maxiter)
    if config.get("grape") and not config.get("integrate", True):
        print("note: grape needs the integrated run; it has no effect with integrate=false.")
    if config.get("grape_warmstart_drag") and not config.get("grape"):
        print("note: grape_warmstart_drag has no effect without --grape.")

    # Per-point --stark chevron parallelism is a RUN-TIME choice (it depends on the
    # mode and the node allocation), not something baked into the grid: parallelize
    # in mode=point (one task = one point, so use the task's cores), keep it serial
    # in mode=local (points are already pooled -> avoid nested oversubscription).
    if args.mode == "point":
        config["stark_jobs"] = int(args.stark_jobs if args.stark_jobs is not None
                                   else os.environ.get("SLURM_CPUS_PER_TASK", 1))
    elif args.mode == "local":
        if args.stark_jobs and int(args.stark_jobs) > 1:
            print("note: mode=local pools points; forcing stark_jobs=1 (no nested pools).")
        config["stark_jobs"] = 1

    if args.mode == "point":
        if args.index is None:
            sys.exit("mode=point requires --index")
        chunk = max(1, int(args.chunk))
        if args.resume_list:
            # --index is the CHUNK position into the resume list, not a point index;
            # the actual point indices come from the file (one per line). This keeps the
            # array size = ceil(len(list)/chunk) regardless of how large/scattered the
            # missing indices are.
            with open(args.resume_list) as f:
                order = [int(x) for x in f.read().split()]
            lo, hi = args.index * chunk, min(args.index * chunk + chunk, len(order))
            if lo >= len(order):
                print(f"[resume {args.index}] no entries in [{lo}, {hi}) of {len(order)}; "
                      f"nothing to do.")
                return
            todo = order[lo:hi]
        else:
            # array task K runs points [K*chunk, (K+1)*chunk); chunk=1 -> one point.
            lo, hi = args.index * chunk, min(args.index * chunk + chunk, len(points))
            if lo >= len(points):
                print(f"[chunk {args.index}] no points in [{lo}, {hi}) (N={len(points)}); "
                      f"nothing to do.")
                return
            todo = list(range(lo, hi))
        for i in todo:
            if not (0 <= i < len(points)):
                print(f"[point {i}] out of range 0..{len(points)-1}; skipping.")
                continue
            res = run_point(points[i], config)
            path = save_point(res, args.outdir)
            print(f"[point {i}] {_log_line(res)} ({res['wall_s']:.1f}s) -> {path}")
        return

    if args.mode == "local":
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        os.makedirs(os.path.join(args.outdir, "points"), exist_ok=True)
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.nproc, mp_context=ctx) as ex:
            futs = [ex.submit(run_point, pt, config) for pt in points]
            for done, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                save_point(res, args.outdir)
                print(f"[{done}/{len(points)}] idx={res['index']:>4} {_log_line(res)}")
        print(f"Local sweep done: {len(points)} points in {time.time()-t0:.1f}s")
        collect(args.outdir)
        return


if __name__ == "__main__":
    main()