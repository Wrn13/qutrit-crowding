#!/usr/bin/env python3
r"""
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

Sweep axes  (channel x spec_freq x lam_spec x drag)
---------------------------------------------------
* channel   : "qubit_qubit" (2-level neighbor) or "qubit_sub" (3-level neighbor,
              captures |1>->|2> leakage). In the dressed-mode picture the SNAIL
              is the coupler, so the old "qubit_snail" channel is reported on
              EVERY point as the coupler-occupation column ``n_coupler``.
* spec_freq : spectator transition frequency Delta = w_b - w_spec (GHz), swept
              broadly. For a single-tone, pure-g3 coupler the spectator's only
              collision in the qubit band is the one-pump exchange with the
              anchor, resonant at Delta = w_p (beat = Delta - w_p). Two-mode
              higher-order collisions require g4 or a second pump tone; the FULL
              sim still captures whatever is actually present.
* lam_spec  : spectator participation lambda_spec = g_spec,s/Delta_spec,s. This
              replaces the old phenomenological spectator ``eta``.
* drag      : first-order DRAG quadrature (Motzoi et al., PRL 103, 110501 (2009))
              on the pump, eta(t) -> eta(t) - i d eta/dt / (2 pi * beat), tuned to
              the spectator's beat detuning. Suppresses the OFF-resonant spectator;
              skipped (drag_applied=False) when |beat| < 5e-4 GHz, since an
              on-resonant collision needs frequency allocation, not DRAG.

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
    RUNNER=run_sweep_zhou.py OUTDIR=results_zhou sbatch --array=0-<M-1> snail_sweep.slurm
    python run_sweep_zhou.py local --outdir results_zhou/ --nproc 8
    python run_sweep_zhou.py collect --outdir results_zhou/

Modes 'prepare' and 'collect' do NOT import the solver; 'point'/'local' import
``zhou_coupler`` lazily.
"""
from __future__ import annotations

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported, so many single-point
# processes (array tasks / pool workers) do not oversubscribe cores.
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi
_CHANNELS = ("qubit_qubit", "qubit_sub")
_DELTA_EPS_GHz = 5e-4   # |beat| below this: collision is essentially on-resonance

# ---------------------------------------------------------------------------
# Default device + simulation configuration (override with --device cfg.json).
# Frequencies / nonlinearities in GHz; the runner converts to rad/ns.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # target qubits a, b
    "qubit_freqs_GHz": [5.00, 4.60],
    "qubit_levels":    3,            # >=3 so target-pair leakage is captured
    "lam_a":           0.10,         # participation lambda_as = g_as/Delta_as
    "lam_b":           0.10,         # participation lambda_bs = g_bs/Delta_bs
    # coupler S (the SNAIL)
    "coupler_freq_GHz": 7.00,
    "coupler_levels":   5,
    "g3_GHz":           0.10,        # measured cubic (engine of every process)
    "g4_GHz":           0.0,         # optional quartic (four-wave mixing)
    # spectator mode
    "spec_levels_qubit": 2,          # "qubit_qubit" channel
    "spec_levels_sub":   3,          # "qubit_sub" channel
    "anchor":            1,          # spectator freq measured below qubit b
    "anharm_qubit_GHz": -0.12,       # transmon anharmonicity of qubits a & b (0 = harmonic)
    "anharm_spec_GHz":  -0.12,       # spectator anharmonicity (qubit_qubit channel)
    # fixed neighbour bath at ABSOLUTE frequencies (used by --sweep target)
    # target-allocation sweep (--sweep target): fixed w_a & w_s, scan (w_b, w_spec)
    "spec_lam":          0.10,       # spectator participation for the allocation sweep
    "spec_channel":      "qubit_qubit",  # "qubit_qubit" or "qubit_sub"
    "min_detuning_GHz":  0.05,       # drop placements with |w_b-w_a| below this
    "drag_compare":         False,   # target sweep: also run DRAG-on in the near-collision window
    "drag_compare_below_MHz": 100.0, # |nearest beat| window (MHz) for the DRAG comparison
    # pulse / solver
    "t_g_ns":   60.0,
    "envelope": "raised_cosine",
    "amp_scale":      1.0,           # calibrated pump-amplitude correction (see calibrate_iswap.py)
    "wp_offset_GHz":  0.0,           # calibrated pump-frequency offset from w_b - w_a
    "stark_drive":    False,         # drive each point at its AC-Stark-shifted resonance
    "stark_span_MHz": 60.0,          # per-point chevron scan width (see find_stark_resonance.py)
    "stark_points":   21,            # per-point chevron offset samples
    "stark_window_factor": 2.0,      # chevron time window = factor * t_g
    "stark_time_points":   120,      # chevron time samples
    "integrate": True,               # set False for the instant analytic map only
    "rtol": 1e-8, "atol": 1e-10,     # QuTiP ODE tolerances
    "nsteps": 500000,                # max internal solver steps between outputs
}

# Default sweep axes (override with --channels/--specfreqs/--lams).
DEFAULT_CHANNELS = ["qubit_qubit", "qubit_sub"]
# Broad spectator-frequency sweep (GHz). With w_p = 0.40 this crosses the
# fundamental exchange (m=1, 0.40) and the two-pump process (m=2, 0.80).
DEFAULT_SPECFREQS_GHz = [round(0.20 + 0.05 * k, 3) for k in range(15)]  # 0.20..0.90
DEFAULT_LAMS = [0.10, 0.20]
DEFAULT_DRAGS = [False, True]
# For --sweep target, w_a and w_s are fixed while (w_b, w_spec) are scanned; DRAG
# can only null one collision, so it defaults off for allocation maps.
DEFAULT_TARGET_DRAGS = [False]


@dataclass
class Point:
    """One sweep point (a single coupler configuration to simulate).

    Attributes
    ----------
    index : int
        Position in the grid; also the output filename suffix.
    channel : str
        Spectator channel, "qubit_qubit" or "qubit_sub" (spectator sweep).
    spec_freq_GHz : float
        Spectator transition frequency Delta = w_b - w_spec (GHz; spectator sweep).
    lam_spec : float
        Spectator participation lambda_spec = g_spec,s / Delta_spec,s (spectator sweep).
    drag : bool
        Whether the first-order DRAG quadrature is requested for this point.
    kind : str
        "spectator" (default; move one spectator against a fixed pair) or "target"
        (fixed w_a & w_s, scan w_b and the spectator -- frequency allocation).
    wa_GHz, wb_GHz : float or None
        Target-qubit frequencies for the allocation sweep (kind="target"); None for
        the spectator sweep, which reads the pair from the config.
    spec_abs_GHz : float or None
        Spectator ABSOLUTE frequency (GHz) for the allocation sweep (kind="target").
    """

    index: int
    channel: str = "qubit_qubit"
    spec_freq_GHz: float = 0.0
    lam_spec: float = 0.0
    drag: bool = False
    kind: str = "spectator"
    wa_GHz: Optional[float] = None
    wb_GHz: Optional[float] = None
    spec_abs_GHz: Optional[float] = None


# ---------------------------------------------------------------------------
# Grid construction (deterministic, solver-free)
# ---------------------------------------------------------------------------
def build_grid(channels: Sequence[str], specfreqs: Sequence[float],
               lams: Sequence[float], drags: Sequence[bool]) -> List[Point]:
    """Build the Cartesian sweep grid in a stable order.

    Parameters
    ----------
    channels : sequence of str
        Spectator channels; each must be one of ("qubit_qubit", "qubit_sub").
    specfreqs : sequence of float
        Spectator frequencies Delta = w_b - w_spec (GHz).
    lams : sequence of float
        Spectator participations lambda_spec.
    drags : sequence of bool
        DRAG on/off settings.

    Returns
    -------
    list of Point
        Points ordered channel -> spec_freq -> lam_spec -> drag, indexed 0..M-1.
    """
    points: List[Point] = []
    index = 0
    for channel in channels:
        if channel not in _CHANNELS:
            raise ValueError(f"unknown channel {channel!r}; choose from {_CHANNELS}")
        for spec_freq in specfreqs:
            for lam in lams:
                for drag in drags:
                    points.append(Point(index=index, channel=channel,
                                        spec_freq_GHz=float(spec_freq),
                                        lam_spec=float(lam), drag=bool(drag)))
                    index += 1
    return points


def build_target_grid(wa_GHz: float, wb_list: Sequence[float], spec_list: Sequence[float],
                      channel: str, drags: Sequence[bool],
                      min_detuning_GHz: float = 0.05) -> List[Point]:
    """Build the allocation grid: fixed w_a (and w_s), Cartesian w_b x w_spec x drag.
    Placements with |w_b - w_a| < ``min_detuning_GHz`` (pump too slow) are dropped.

    Parameters
    ----------
    wa_GHz : float
        Fixed qubit-a frequency (GHz).
    wb_list : sequence of float
        Partner-qubit frequencies w_b to scan (GHz).
    spec_list : sequence of float
        Spectator ABSOLUTE frequencies w_spec to scan (GHz).
    channel : str
        Spectator channel, "qubit_qubit" or "qubit_sub".
    drags : sequence of bool
        DRAG on/off settings.
    min_detuning_GHz : float, default 0.05
        Minimum |w_b - w_a|; smaller placements are skipped.

    Returns
    -------
    list of Point
        Points (kind="target") ordered w_b -> w_spec -> drag, indexed 0..M-1.
    """
    if channel not in _CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; choose from {_CHANNELS}")
    points: List[Point] = []
    index = 0
    for wb in wb_list:
        if abs(float(wb) - float(wa_GHz)) < float(min_detuning_GHz):
            continue
        for spec in spec_list:
            for drag in drags:
                points.append(Point(index=index, kind="target", channel=channel,
                                    wa_GHz=float(wa_GHz), wb_GHz=float(wb),
                                    spec_abs_GHz=float(spec), drag=bool(drag)))
                index += 1
    if not points:
        raise ValueError("empty target grid; check --wb-GHz/--spec-GHz and min_detuning_GHz.")
    return points


def write_grid(outdir: str, config: Dict[str, Any], points: List[Point]) -> str:
    """Write the grid and resolved config to ``<outdir>/grid.json``.

    Parameters
    ----------
    outdir : str
        Output directory; a ``points/`` subdirectory is created.
    config : dict
        The resolved device/simulation configuration to persist.
    points : list of Point
        The grid to serialise.

    Returns
    -------
    str
        Path to the written grid.json.
    """
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    path = os.path.join(outdir, "grid.json")
    with open(path, "w") as f:
        json.dump({"config": config, "points": [asdict(p) for p in points]},
                  f, indent=2)
    return path


def load_grid(outdir: str) -> Tuple[Dict[str, Any], List[Point]]:
    """Load the config and points from ``<outdir>/grid.json``.

    Parameters
    ----------
    outdir : str
        Directory containing grid.json.

    Returns
    -------
    (dict, list of Point)
        The persisted config and the reconstructed points.
    """
    with open(os.path.join(outdir, "grid.json")) as f:
        blob = json.load(f)
    return blob["config"], [Point(**p) for p in blob["points"]]


# ---------------------------------------------------------------------------
# Single-point computation (imports zhou_coupler lazily)
# ---------------------------------------------------------------------------
def _stark_offset_GHz(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                      t_g: float, amp_scale: float, solver: Dict[str, Any]) -> float:
    """Per-point AC-Stark-shifted iSWAP resonance offset (GHz) for the pump.

    Runs the spectator-free chevron of find_stark_resonance.py at this point's
    (w_a, w_b) and operating amplitude, returning the offset from |w_b - w_a| that
    maximises swap contrast. The spectator is excluded on purpose: the drive
    frequency is a property of the a<->b gate, calibrated as on hardware without
    the neighbour. Serial (n_jobs=1) because the caller may already be in a pool.

    Parameters
    ----------
    config : dict
        Merged device configuration (supplies g3, participations, levels,
        anharmonicity, envelope, and the stark_* chevron resolution keys).
    wa_GHz, wb_GHz : float
        Qubit frequencies of this point (GHz).
    t_g : float
        Operating gate time (ns); sets the probe |eta| via the normalization.
    amp_scale : float
        Amplitude-scale correction from a prior amplitude calibration.
    solver : dict
        QuTiP tolerances (atol, rtol, nsteps).

    Returns
    -------
    float
        Stark resonance offset (GHz) to add to |w_b - w_a|.
    """
    import find_stark_resonance as FS
    sub = dict(config)
    sub["qubit_freqs_GHz"] = [float(wa_GHz), float(wb_GHz)]
    span = float(config.get("stark_span_MHz", 60.0)) / 1000.0
    n_pts = int(config.get("stark_points", 21))
    offsets = np.linspace(-span / 2.0, span / 2.0, n_pts)
    window = float(config.get("stark_window_factor", 2.0)) * float(t_g)
    n_time = int(config.get("stark_time_points", 120))
    res = FS.scan(sub, float(t_g), float(amp_scale), offsets, window, n_time,
                  solver, n_jobs=1)
    return float(res["resonance_offset_GHz"])


def run_point(pt: Point, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute one sweep point: the analytic collision prediction always, and the
    full QuTiP gate simulation when ``config['integrate']`` is True.

    Parameters
    ----------
    pt : Point
        The grid point (channel, spectator frequency, participation, DRAG flag).
    config : dict
        Resolved device/simulation configuration (see DEFAULT_CONFIG). Imports
        ``zhou_coupler`` (and, for the full path, QuTiP) lazily.

    Returns
    -------
    dict
        Result row with analytic fields (beat, eta_peak, effective rates, ...) and,
        if integrated, F_avg / leakage / occupations / the 4x4 ``U_proj``.
    """
    if getattr(pt, "kind", "spectator") == "target":
        return _run_target_point(pt, config)

    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

    t0 = time.time()
    a, b, coupler, spec = 0, 1, 2, 3
    anchor = int(config["anchor"])

    # --- frequencies (rad/ns) ---------------------------------------------
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float) * TWO_PI)
    ws = config["coupler_freq_GHz"] * TWO_PI
    w_p = abs(wb - wa)                              # iSWAP pump = |detuning| (fixed)
    w_p_GHz = w_p / TWO_PI + float(config.get("wp_offset_GHz", 0.0))   # calibrated offset
    # optional: drive at the AC-Stark-shifted resonance (needs the integrated run).
    # In the spectator sweep w_a, w_b are fixed, so this offset is identical for every
    # point -- for efficiency you can instead precompute it once with
    # find_stark_resonance.py and set wp_offset_GHz.
    stark_offset_GHz = 0.0
    if bool(config.get("integrate", True)) and bool(config.get("stark_drive", False)):
        _sv = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                   nsteps=int(config.get("nsteps", 500000)))
        stark_offset_GHz = _stark_offset_GHz(config, wa / TWO_PI, wb / TWO_PI,
                                             float(config["t_g_ns"]),
                                             float(config.get("amp_scale", 1.0)), _sv)
        w_p_GHz += stark_offset_GHz
    w_spec = wb - pt.spec_freq_GHz * TWO_PI         # move ONLY the spectator
    freqs_GHz = [wa / TWO_PI, wb / TWO_PI, ws / TWO_PI, w_spec / TWO_PI]

    # --- levels per channel ------------------------------------------------
    if pt.channel == "qubit_qubit":
        spec_levels = int(config["spec_levels_qubit"])
    elif pt.channel == "qubit_sub":
        spec_levels = int(config["spec_levels_sub"])
    else:
        raise ValueError(f"unknown channel {pt.channel!r}")
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"]), spec_levels]

    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    aq = float(config.get("anharm_qubit_GHz", 0.0))
    anharm = {a: aq, b: aq}
    if pt.channel == "qubit_qubit":                    # spectator is a transmon too
        anharm[spec] = float(config.get("anharm_spec_GHz", 0.0))

    cpl = ZhouCoupler(
        mode_freqs_GHz=freqs_GHz,
        coupler_index=coupler,
        participations={a: float(config["lam_a"]),
                        b: float(config["lam_b"]),
                        spec: float(pt.lam_spec)},
        nonlinearities=nonlin,
        levels=levels,
        anharmonicities_GHz=anharm,
    )

    # DRAG targets the off-resonant spectator<->anchor exchange, detuned from the
    # pump by beat = spec_freq - w_p. It is singular as beat -> 0 (an on-resonant
    # collision needs frequency allocation, not DRAG), so skip it there.
    beat_GHz = pt.spec_freq_GHz - w_p_GHz
    use_drag = bool(pt.drag)
    status_drag = "ok"
    if pt.drag and abs(beat_GHz) < _DELTA_EPS_GHz:
        use_drag = False
        status_drag = "drag_skipped_resonant_spectator"

    # pump at w_b - w_a, amplitude normalized to a full iSWAP on (a,b)
    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse
    env = EnvCls(amp=1.0, t_g=float(config["t_g_ns"]))
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=env, is_eta=True,
                          drag=use_drag,
                          delta_drag_GHz=(beat_GHz if use_drag else None)),
                 normalize_iswap=(a, b))
    # calibrated amplitude correction (1.0 = raw analytic pi/2 normalization)
    cpl.scale_pump_amplitude(float(config.get("amp_scale", 1.0)))

    # --- ANALYTIC collision prediction (free; Eq. 62) ---------------------
    # For a single-tone, pure-g3 (n=3) coupler the spectator's only collision in
    # the qubit band is the ONE-pump exchange with the anchor, resonant at
    # spec_freq = w_p. (Higher-order/two-mode collisions require g4 or a second
    # pump tone; the FULL sim below still captures whatever is actually there.)
    eta_peak = cpl.peak_eta()
    g_iswap = cpl.iswap_rate(a, b)                  # 6 g3 la lb |eta|
    g_spec = cpl.effective_rate([anchor, spec], n=3, C=6)   # 6 g3 l_anchor l_spec |eta|

    out = {
        "index": pt.index,
        "channel": pt.channel,
        "spec_freq_GHz": pt.spec_freq_GHz,
        "lam_spec": pt.lam_spec,
        "drag": bool(pt.drag),
        "drag_applied": bool(use_drag),
        "beat_GHz": round(float(beat_GHz), 6),
        "eta_peak": round(float(eta_peak), 5),
        "g_iswap_eff_MHz": round(float(g_iswap / TWO_PI * 1e3), 4),
        "g_spec_eff_MHz": round(float(g_spec / TWO_PI * 1e3), 4),
        "w_p_GHz": round(float(w_p_GHz), 6),
        "stark_offset_MHz": round(float(stark_offset_GHz) * 1e3, 4),
        "w_spec_GHz": round(float(w_spec / TWO_PI), 6),
        "t_g_ns": float(config["t_g_ns"]),
        "status": status_drag if status_drag != "ok" else "analytic",
        "F_avg": "", "leakage": "", "n_spec": "", "n_coupler": "", "p_transfer": "",
        "U_proj": None,
    }

    if not config.get("integrate", True):
        out["wall_s"] = time.time() - t0
        return out

    # --- FULL non-perturbative evolution (QuTiP, exact Hamiltonian) -------
    t_g = float(config["t_g_ns"])
    solver = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                  nsteps=int(config.get("nsteps", 500000)))

    # (a) spectator diagnostic: excite qubit a, see where population ends up.
    occ0 = [0, 0, 0, 0]; occ0[a] = 1
    pops = np.abs(cpl.evolve_state(occ0, t_g, **solver)) ** 2
    occ_b = [0, 0, 0, 0]; occ_b[b] = 1
    out["n_spec"] = float(cpl.mean_occupation(pops, spec))
    out["n_coupler"] = float(cpl.mean_occupation(pops, coupler))
    out["p_transfer"] = float(pops[cpl.fock_index(occ_b)])

    # (b) leakage-aware iSWAP fidelity on the target pair (4 trajectories).
    F_avg, leakage, U_proj = cpl.iswap_fidelity(a, b, t_g, fit_virtual_z=True,
                                                **solver)
    out["F_avg"] = float(F_avg)
    out["leakage"] = float(leakage)
    out["U_proj"] = U_proj
    out["status"] = status_drag        # "ok", or the drag-skip note if it fired
    out["wall_s"] = time.time() - t0
    return out


def _run_target_point(pt: Point, config: Dict[str, Any]) -> Dict[str, Any]:
    """Allocation point: fixed w_a and w_s; the partner sits at w_b and a single
    spectator at absolute w_spec. Analytic collision search always; full iSWAP
    fidelity when ``config['integrate']``. Same return-row contract as `run_point`.

    Parameters
    ----------
    pt : Point
        A kind="target" point carrying wb_GHz, spec_abs_GHz, channel, drag (w_a is
        read from the config, held fixed).
    config : dict
        Resolved configuration; reads ``qubit_freqs_GHz[0]`` (fixed w_a),
        ``coupler_freq_GHz`` (fixed w_s), ``lam_a``/``lam_b``, ``spec_lam``,
        ``g3_GHz``, pulse and solver keys.

    Returns
    -------
    dict
        Row with allocation fields (nearest_beat_GHz, nearest_kind, g_collision_MHz,
        ...) and, if integrated, F_avg / leakage / n_spec / n_coupler / U_proj.
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

    t0 = time.time()
    a, b, coupler, spec = 0, 1, 2, 3
    wa_GHz = float(config["qubit_freqs_GHz"][0])       # fixed
    ws_GHz = float(config["coupler_freq_GHz"])         # fixed SNAIL
    wb_GHz = float(pt.wb_GHz)                           # swept
    wspec_GHz = float(pt.spec_abs_GHz)                  # swept (absolute)
    w_p_GHz = abs(wb_GHz - wa_GHz) + float(config.get("wp_offset_GHz", 0.0))
    lam_spec = float(config.get("spec_lam", 0.20))
    channel = pt.channel
    integrate = bool(config.get("integrate", True))

    # optional: drive at the per-point AC-Stark-shifted resonance (w_b varies across
    # the allocation sweep, so Delta_Stark differs point to point). Needs the
    # integrated run; shifts w_p for both the collision classification and the gate.
    stark_offset_GHz = 0.0
    if integrate and bool(config.get("stark_drive", False)):
        _sv = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                   nsteps=int(config.get("nsteps", 500000)))
        stark_offset_GHz = _stark_offset_GHz(config, wa_GHz, wb_GHz,
                                             float(config["t_g_ns"]),
                                             float(config.get("amp_scale", 1.0)), _sv)
        w_p_GHz += stark_offset_GHz

    # Rates/eta are level-independent, so the analytic-only build uses 2 levels
    # everywhere (tiny Hilbert space); the full build uses the configured levels.
    if integrate:
        q_lv, c_lv = int(config["qubit_levels"]), int(config["coupler_levels"])
        s_lv = int(config["spec_levels_sub"] if channel == "qubit_sub"
                   else config["spec_levels_qubit"])
    else:
        q_lv = c_lv = s_lv = 2

    freqs_GHz = [wa_GHz, wb_GHz, ws_GHz, wspec_GHz]
    participations = {a: float(config["lam_a"]), b: float(config["lam_b"]),
                      spec: lam_spec}
    levels = [q_lv, q_lv, c_lv, s_lv]

    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    aq = float(config.get("anharm_qubit_GHz", 0.0))
    anharm = {a: aq, b: aq}
    if channel == "qubit_qubit":
        anharm[spec] = float(config.get("anharm_spec_GHz", 0.0))

    cpl = ZhouCoupler(mode_freqs_GHz=freqs_GHz, coupler_index=coupler,
                      participations=participations, nonlinearities=nonlin, levels=levels,
                      anharmonicities_GHz=anharm)

    # --- nearest collision: spectator vs {a, b}, pump-assisted and direct -----
    # pump-assisted exchange a_q^d a_spec is resonant at |w_q - w_spec| = w_p; the
    # direct (no-pump) exchange is resonant at w_q = w_spec. Track the closest.
    nearest = None   # (|beat|, signed beat, kind, q_label, q_idx)
    for q_idx, q_freq, q_label in ((a, wa_GHz, "a"), (b, wb_GHz, "b")):
        sep = abs(q_freq - wspec_GHz)
        for beat, kind in ((sep - w_p_GHz, "onepump"), (sep, "static")):
            if nearest is None or abs(beat) < nearest[0]:
                nearest = (abs(beat), float(beat), kind, q_label, q_idx)

    # near-collision window where DRAG is meant to help (off-resonant but close).
    # Round the beat to 1 kHz so placements exactly at the threshold classify
    # deterministically (strict < threshold, i.e. genuinely below the cutoff).
    drag_beat = nearest[1]
    beat_abs = round(abs(drag_beat), 6)
    thr_GHz = float(config.get("drag_compare_below_MHz", 100.0)) / 1000.0
    drag_compare = bool(config.get("drag_compare", False))
    in_window = (_DELTA_EPS_GHz < beat_abs < thr_GHz)

    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse
    amp_scale = float(config.get("amp_scale", 1.0))

    def _configure(use_drag: bool) -> None:
        """(Re)set the pump with DRAG on/off (tuned to the nearest beat) and
        renormalize the iSWAP. A fresh unit-amplitude envelope each call means the
        normalization is not applied cumulatively."""
        cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz,
                              envelope=EnvCls(amp=1.0, t_g=float(config["t_g_ns"])),
                              is_eta=True, drag=use_drag,
                              delta_drag_GHz=(drag_beat if use_drag else None)),
                     normalize_iswap=(a, b))
        cpl.scale_pump_amplitude(amp_scale)

    # Baseline pump. With --drag-compare the baseline is DRAG-off and we ALSO run
    # DRAG-on inside the window; otherwise honour the point's own drag flag
    # (skipped on-resonance, where allocation -- not DRAG -- is the fix).
    if drag_compare:
        base_drag = False
        status_drag = "drag_compare" if in_window else "ok"
    else:
        base_drag = bool(pt.drag)
        status_drag = "ok"
        if pt.drag and abs(drag_beat) < _DELTA_EPS_GHz:
            base_drag = False
            status_drag = "drag_skipped_resonant_collision"

    _configure(base_drag)
    eta_peak = cpl.peak_eta()
    g_iswap = cpl.iswap_rate(a, b)
    g_coll = cpl.effective_rate([nearest[4], spec], n=3, C=6)   # 6 g3 l_q l_spec |eta|

    out = {
        "index": pt.index, "kind": "target", "channel": channel,
        "wa_GHz": round(wa_GHz, 6), "wb_GHz": round(wb_GHz, 6),
        "w_snail_GHz": round(ws_GHz, 6), "spec_GHz": round(wspec_GHz, 6),
        "detuning_GHz": round(wb_GHz - wa_GHz, 6), "w_p_GHz": round(w_p_GHz, 6),
        "stark_offset_MHz": round(float(stark_offset_GHz) * 1e3, 4),
        "lam_spec": round(lam_spec, 4),
        "drag": bool(pt.drag), "drag_applied": bool(base_drag),
        "drag_compare_window": bool(drag_compare and in_window),
        "eta_peak": round(float(eta_peak), 5),
        "g_iswap_eff_MHz": round(float(g_iswap / TWO_PI * 1e3), 4),
        "nearest_beat_GHz": round(nearest[1], 6),
        "nearest_kind": nearest[2],
        "nearest_target": nearest[3],
        "g_collision_MHz": round(float(g_coll / TWO_PI * 1e3), 4),
        "t_g_ns": float(config["t_g_ns"]),
        "status": status_drag if status_drag != "ok" else "analytic",
        "F_avg": "", "leakage": "", "n_spec": "", "n_coupler": "", "p_transfer": "",
        "F_avg_drag": "", "leakage_drag": "", "dF_drag": "",
        "U_proj": None,
    }

    if not integrate:
        out["wall_s"] = time.time() - t0
        return out

    # --- FULL non-perturbative evolution (QuTiP, exact Hamiltonian) -----------
    t_g = float(config["t_g_ns"])
    solver = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                  nsteps=int(config.get("nsteps", 500000)))

    occ0 = [0, 0, 0, 0]; occ0[a] = 1
    pops = np.abs(cpl.evolve_state(occ0, t_g, **solver)) ** 2
    occ_b = [0, 0, 0, 0]; occ_b[b] = 1
    out["n_spec"] = float(cpl.mean_occupation(pops, spec))
    out["n_coupler"] = float(cpl.mean_occupation(pops, coupler))
    out["p_transfer"] = float(pops[cpl.fock_index(occ_b)])

    F_avg, leakage, U_proj = cpl.iswap_fidelity(a, b, t_g, fit_virtual_z=True, **solver)
    out["F_avg"] = float(F_avg)
    out["leakage"] = float(leakage)
    out["U_proj"] = U_proj
    out["status"] = status_drag

    # DRAG comparison: rerun the SAME placement with DRAG on, only in the
    # |beat| < threshold window (where DRAG can suppress the off-resonant collision).
    if drag_compare and in_window:
        _configure(True)
        F_d, leak_d, U_d = cpl.iswap_fidelity(a, b, t_g, fit_virtual_z=True, **solver)
        out["F_avg_drag"] = float(F_d)
        out["leakage_drag"] = float(leak_d)
        out["dF_drag"] = float(F_d - F_avg)
        if F_d >= F_avg:                       # keep the better propagator + flag
            out["U_proj"] = U_d
            out["drag_applied"] = True

    out["wall_s"] = time.time() - t0
    return out


def save_point(result: Dict[str, Any], outdir: str) -> str:
    """Persist one result row to ``<outdir>/points/point_XXXXX.npz``.

    Parameters
    ----------
    result : dict
        A row from `run_point`; its ``U_proj`` (if any) is stored separately and
        the remaining scalar metadata is JSON-encoded.
    outdir : str
        Output directory (its ``points/`` subdirectory must exist).

    Returns
    -------
    str
        Path to the written .npz file.
    """
    path = os.path.join(outdir, "points", f"point_{result['index']:05d}.npz")
    U = result.pop("U_proj", None)
    if U is None:
        U = np.zeros((4, 4), dtype=complex)
    np.savez_compressed(path,
                        U_proj_real=np.real(U), U_proj_imag=np.imag(U),
                        meta=json.dumps(result))
    return path


# ---------------------------------------------------------------------------
# Collection -> summary.csv + combined.npz
# ---------------------------------------------------------------------------
def collect(outdir: str) -> None:
    """Gather all per-point .npz files into ``summary.csv`` and ``combined.npz``.

    Parameters
    ----------
    outdir : str
        Sweep directory containing ``points/point_*.npz``.

    Returns
    -------
    None
        Writes ``<outdir>/summary.csv`` (one row per point, sorted by index) and
        ``<outdir>/combined.npz`` (stacked 4x4 propagators).
    """
    files = sorted(glob.glob(os.path.join(outdir, "points", "point_*.npz")))
    if not files:
        print("No point_*.npz found; nothing to collect.", file=sys.stderr)
        return
    rows, U_stack, idx_stack = [], [], []
    for fpath in files:
        with np.load(fpath, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            U = z["U_proj_real"] + 1j * z["U_proj_imag"]
        rows.append(meta); U_stack.append(U); idx_stack.append(meta["index"])

    rows.sort(key=lambda r: r["index"])
    spec_cols = ["index", "channel", "spec_freq_GHz", "lam_spec", "drag", "drag_applied",
                 "beat_GHz", "eta_peak", "g_iswap_eff_MHz", "g_spec_eff_MHz",
                 "status", "F_avg", "leakage", "n_spec", "n_coupler", "p_transfer",
                 "w_p_GHz", "stark_offset_MHz", "w_spec_GHz", "t_g_ns", "wall_s"]
    target_cols = ["index", "kind", "channel", "wa_GHz", "wb_GHz", "w_snail_GHz",
                   "spec_GHz", "detuning_GHz", "w_p_GHz", "stark_offset_MHz", "lam_spec", "drag",
                   "drag_applied", "drag_compare_window", "eta_peak",
                   "g_iswap_eff_MHz", "nearest_beat_GHz", "nearest_kind",
                   "nearest_target", "g_collision_MHz", "status", "F_avg", "leakage",
                   "F_avg_drag", "leakage_drag", "dF_drag", "n_spec", "n_coupler",
                   "p_transfer", "t_g_ns", "wall_s"]
    is_target = bool(rows) and rows[0].get("kind") == "target"
    cols = target_cols if is_target else spec_cols
    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    order = np.argsort(idx_stack)
    np.savez_compressed(os.path.join(outdir, "combined.npz"),
                        index=np.array(idx_stack)[order],
                        U_proj=np.array(U_stack)[order])
    print(f"Collected {len(rows)} points -> {csv_path} and combined.npz")

    # For an integrated allocation sweep, report the best placement (allowing DRAG
    # where it was compared) and, if present, the DRAG gain in the <threshold window.
    if is_target:
        def _best_F(r: Dict[str, Any]) -> float:
            vals = [v for v in (r.get("F_avg"), r.get("F_avg_drag"))
                    if isinstance(v, (int, float))]
            return max(vals) if vals else float("-inf")

        scored = [r for r in rows if _best_F(r) > float("-inf")]
        if scored:
            best = max(scored, key=_best_F)
            with_drag = (isinstance(best.get("F_avg_drag"), (int, float))
                         and best["F_avg_drag"] >= best.get("F_avg", -1))
            print(f"Best allocation: w_b={best['wb_GHz']:.4f} GHz, "
                  f"w_spec={best['spec_GHz']:.4f} GHz "
                  f"({'with' if with_drag else 'no'} DRAG) -> F={_best_F(best):.5f}, "
                  f"nearest_beat={best['nearest_beat_GHz']:+.3f} GHz")
        gains = [r["dF_drag"] for r in rows if isinstance(r.get("dF_drag"), (int, float))]
        if gains:
            g = np.array(gains)
            print(f"DRAG effect over {g.size} points with |beat|<threshold: "
                  f"mean dF={g.mean():+.4f}, best dF={g.max():+.4f}, "
                  f"helped {int((g > 0).sum())}/{g.size}")


# ---------------------------------------------------------------------------
# Modes / CLI
# ---------------------------------------------------------------------------
def _parse_list(s: Optional[str], cast: Callable[[str], Any]) -> Optional[List[Any]]:
    """Parse a comma-separated CLI string into a list via `cast`, or None if empty.

    Parameters
    ----------
    s : str or None
        Raw comma-separated argument (e.g. "0.2,0.25,0.3").
    cast : callable
        Element constructor (e.g. float, str).

    Returns
    -------
    list or None
        The parsed list, or None when `s` is falsy (use the caller's default).
    """
    return [cast(x) for x in s.split(",")] if s else None


def _bool_list(s: Optional[str]) -> Optional[List[bool]]:
    """Parse a comma-separated string of booleans (e.g. "false,true").

    Parameters
    ----------
    s : str or None
        Raw argument; tokens in {1,true,t,yes,on} (case-insensitive) are True.

    Returns
    -------
    list of bool or None
        The parsed flags, or None when `s` is falsy.
    """
    if not s:
        return None
    return [tok.strip().lower() in ("1", "true", "t", "yes", "on")
            for tok in s.split(",")]


def _log_line(res: Dict[str, Any]) -> str:
    """One-line human summary for a result row (both sweep kinds)."""
    f_str = res["F_avg"] if res.get("F_avg", "") != "" else "  --  "
    if res.get("kind") == "target":
        tail = ""
        if isinstance(res.get("dF_drag"), (int, float)):
            tail = f" dF_drag={res['dF_drag']:+.4f}"
        return (f"wb={res['wb_GHz']:.3f} wspec={res['spec_GHz']:.3f} wp={res['w_p_GHz']:.3f} "
                f"nearest={res['nearest_beat_GHz']:+.3f}GHz({res['nearest_kind']}) "
                f"g_coll={res['g_collision_MHz']}MHz F={f_str}{tail}")
    return (f"beat={res['beat_GHz']:+.3f}GHz drag={res['drag_applied']} "
            f"eta={res['eta_peak']:.3f} g_spec={res['g_spec_eff_MHz']:.3f}MHz F={f_str}")


def main() -> None:
    """Command-line entry point.

    Subcommands: ``prepare`` (write grid.json), ``point`` (run one --index),
    ``local`` (run the whole grid with a process pool, then collect), and
    ``collect`` (gather points into summary.csv / combined.npz). Run
    ``--help`` for the full option list.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prepare", "point", "local", "collect"])
    ap.add_argument("--outdir", default="results_zhou")
    ap.add_argument("--device", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--index", type=int, help="point index (mode=point)")
    ap.add_argument("--grid", help="path to grid.json (default: <outdir>/grid.json)")
    ap.add_argument("--nproc", type=int, default=4, help="processes for mode=local")
    ap.add_argument("--sweep", choices=["spectator", "target"], default="spectator",
                    help="spectator sweep (default) or target-frequency allocation sweep")
    ap.add_argument("--channels", help="comma list: qubit_qubit,qubit_sub")
    ap.add_argument("--specfreqs", help="comma list of spectator freqs Delta (GHz)")
    ap.add_argument("--lams", help="comma list of spectator participations")
    ap.add_argument("--drags", help="comma list of bools, e.g. false,true")
    ap.add_argument("--wb-GHz", help="[target] comma list of partner (w_b) freqs (GHz)")
    ap.add_argument("--spec-GHz", help="[target] comma list of spectator ABSOLUTE freqs (GHz)")
    ap.add_argument("--channel", choices=list(_CHANNELS), help="[target] spectator channel")
    ap.add_argument("--lam-spec", type=float, help="[target] spectator participation lambda_spec")
    ap.add_argument("--drag-compare", action="store_true",
                    help="[target] also run DRAG-on where |nearest beat| < --drag-compare-below-MHz")
    ap.add_argument("--drag-compare-below-MHz", type=float, default=None,
                    help="[target] near-collision window for the DRAG comparison (default 100)")
    ap.add_argument("--no-integrate", action="store_true",
                    help="analytic collision map only (no time integration)")
    ap.add_argument("--stark", action="store_true",
                    help="drive each point at its AC-Stark-shifted resonance (per-point "
                         "chevron; needs the integrated run and is markedly slower)")
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

    config = dict(DEFAULT_CONFIG)
    if args.device:
        with open(args.device) as f:
            config.update(json.load(f))
    if args.no_integrate:
        config["integrate"] = False
    if args.stark:
        config["stark_drive"] = True
    if args.stark_span_MHz is not None:
        config["stark_span_MHz"] = float(args.stark_span_MHz)
    if args.stark_points is not None:
        config["stark_points"] = int(args.stark_points)
    if args.channel:
        config["spec_channel"] = args.channel
    if args.lam_spec is not None:
        config["spec_lam"] = float(args.lam_spec)
    if args.drag_compare:
        config["drag_compare"] = True
    if args.drag_compare_below_MHz is not None:
        config["drag_compare_below_MHz"] = float(args.drag_compare_below_MHz)

    if args.mode == "prepare":
        if args.sweep == "target":
            nominal = config["qubit_freqs_GHz"]
            wa_GHz = float(nominal[0])                                   # fixed
            wb_default = [round(float(nominal[1]) - 0.30 + 0.02 * k, 3) for k in range(31)]
            wb_list = _parse_list(args.wb_GHz, float) or wb_default
            # default spectator band: across the qubit band around the pair
            lo = min(wa_GHz, float(nominal[1])) - 0.40
            spec_default = [round(lo + 0.02 * k, 3) for k in range(41)]
            spec_list = _parse_list(args.spec_GHz, float) or spec_default
            channel = config.get("spec_channel", "qubit_qubit")
            if config.get("drag_compare"):
                if args.drags:
                    print("note: --drag-compare runs DRAG on/off per point; ignoring --drags.")
                drags = [False]
            else:
                drags = _bool_list(args.drags) or DEFAULT_TARGET_DRAGS
            points = build_target_grid(wa_GHz, wb_list, spec_list, channel, drags,
                                       float(config.get("min_detuning_GHz", 0.05)))
            path = write_grid(args.outdir, config, points)
            m = len(points)
            print(f"Wrote {m} allocation points -> {path}")
            print(f"  fixed: w_a={wa_GHz:.4f} GHz, w_snail={config['coupler_freq_GHz']:.4f} GHz; "
                  f"spectator lam={config.get('spec_lam', 0.2)} ({channel})")
            print(f"  scan: w_b ({len(wb_list)}) x w_spec ({len(spec_list)}) x drag ({len(drags)}) "
                  f"[dropped |detuning|<{config.get('min_detuning_GHz', 0.05)} GHz]")
            if config.get("drag_compare"):
                print(f"  DRAG comparison ON for |nearest beat| < "
                      f"{config.get('drag_compare_below_MHz', 100.0):.0f} MHz")
            print(f"integrate = {config['integrate']}  "
                  f"({'FULL sim per point' if config['integrate'] else 'analytic map only'})")
            print(f"Submit with:\n  RUNNER=run_sweep_zhou.py OUTDIR={args.outdir} "
                  f"sbatch --array=0-{m-1} snail_sweep.slurm")
            return
        channels = _parse_list(args.channels, str) or DEFAULT_CHANNELS
        specfreqs = _parse_list(args.specfreqs, float) or DEFAULT_SPECFREQS_GHz
        lams = _parse_list(args.lams, float) or DEFAULT_LAMS
        drags = _bool_list(args.drags) or DEFAULT_DRAGS
        points = build_grid(channels, specfreqs, lams, drags)
        path = write_grid(args.outdir, config, points)
        m = len(points)
        print(f"Wrote {m} points -> {path}")
        print(f"integrate = {config['integrate']}  "
              f"({'FULL sim per point' if config['integrate'] else 'analytic map only'})")
        print(f"Submit with:\n  RUNNER=run_sweep_zhou.py OUTDIR={args.outdir} "
              f"sbatch --array=0-{m-1} snail_sweep.slurm")
        return

    if args.mode == "collect":
        collect(args.outdir)
        return

    grid_path = args.grid or os.path.join(args.outdir, "grid.json")
    if not os.path.exists(grid_path):
        sys.exit(f"grid.json not found at {grid_path}; run `prepare` first.")
    config, points = load_grid(args.outdir if not args.grid
                               else os.path.dirname(args.grid) or ".")

    if args.mode == "point":
        if args.index is None:
            sys.exit("mode=point requires --index")
        if not (0 <= args.index < len(points)):
            sys.exit(f"index {args.index} out of range 0..{len(points)-1}")
        res = run_point(points[args.index], config)
        path = save_point(res, args.outdir)
        print(f"[point {args.index}] {_log_line(res)} ({res['wall_s']:.1f}s) -> {path}")
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