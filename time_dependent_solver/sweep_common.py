"""Shared machinery for the Zhou SNAIL iSWAP sweeps.

Constants, the DEFAULT_CONFIG, the Point grid record, grid IO, the analytic
collision search (_nearest_collision), the calibration chevron (_stark_offset_GHz),
result collection, and CLI helpers. Imported by sweep_spectator, sweep_target,
and the run_sweep_zhou entry point.
"""
from __future__ import annotations

import os
import glob
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


TWO_PI = 2.0 * np.pi

_DELTA_EPS_GHz = 5e-4

def _drag_skip_GHz(config: Dict[str, Any]) -> float:
    """|beat| (GHz) below which DRAG is skipped. The first-order quadrature ~ 1/beat
    diverges toward the collision, so it must not be applied within ~a few x g_iswap
    of it. Configurable via ``drag_skip_below_MHz`` (default 5 MHz)."""
    return max(float(config.get("drag_skip_below_MHz", 5.0)) / 1e3, _DELTA_EPS_GHz)

DEFAULT_CONFIG = {
    # target qubits a, b
    "qubit_freqs_GHz": [5.00, 4.60],
    "qubit_levels":    3,            # >=3 so target-pair leakage is captured
    "lam_a":           0.20,         # participation lambda_as = g_as/Delta_as
    "lam_b":           0.20,         # participation lambda_bs = g_bs/Delta_bs
    # coupler S (the SNAIL)
    "coupler_freq_GHz": 7.00,
    "coupler_levels":   5,
    "g3_GHz":           0.10,        # measured cubic (engine of every process)
    "g4_GHz":           0.0,         # optional quartic (four-wave mixing)
    # spectator mode: a single 3-level anharmonic transmon; participation = lam_b
    "spec_levels":       3,          # spectator Hilbert dimension (captures |1>->|2> leakage)
    "anchor":            1,          # spectator freq measured below qubit b
    "anharm_qubit_GHz": -0.20,       # transmon anharmonicity of qubits a & b (0 = harmonic)
    "anharm_spec_GHz":  -0.20,       # spectator anharmonicity (0 = harmonic 3-level ladder)
    # target-allocation sweep (--sweep target): fixed w_a & w_s, scan (w_b, w_spec)
    "min_detuning_GHz":  0.05,       # drop placements with |w_b-w_a| below this
    "drag_compare":         False,   # target sweep: also run DRAG-on in the near-collision window
    "drag_compare_below_MHz": 100.0, # |nearest beat| window (MHz) for the DRAG comparison
    "drag_skip_below_MHz":    5.0,   # |beat| below this -> DRAG is SKIPPED: the first-order
                                     #   quadrature ~ 1/beat diverges as beat -> 0, so applying
                                     #   it inside ~a few x g_iswap of the collision blows the
                                     #   gate up (population dumped into the coupler). Set to a
                                     #   few x g_iswap for your device.
    # pulse / solver
    "t_g_ns":   60.0,
    "envelope": "raised_cosine",
    "amp_scale":      1.0,           # calibrated pump-amplitude correction (see calibrate_gate.py)
    "wp_offset_GHz":  0.0,           # calibrated pump-frequency offset from w_b - w_a
    "stark_drive":    False,         # drive each point at its AC-Stark-shifted resonance (spectator-aware chevron)
    "stark_span_MHz": 60.0,          # per-point chevron scan width (see find_stark_resonance.py)
    "stark_points":   21,            # per-point chevron offset samples
    "stark_window_factor": 2.0,      # chevron time window = factor * t_g
    "stark_time_points":   120,      # chevron time samples
    "stark_jobs":     1,             # processes for the per-point chevron's offset scan
                                     #   (>1 only in mode=point, where one task = one point;
                                     #    mode=local pools points and forces this to 1)
    "stark_match_pulse": False,      # per-point chevron uses the ACTUAL pulse (raised-cosine,
                                     #   + DRAG for DRAG-on points) instead of a constant probe,
                                     #   so DRAG-on points land on the DRAG-on resonance
    "calibrate_points": False,       # per-point amplitude+Stark tune-up (calibrate_gate),
                                     #   spectator-present + DRAG-aware (hardware-style)
    "calibrate_iters":  1,           # amplitude/frequency rounds per point
    "cal_amp_lo":       0.6,         # per-point amplitude-scale search bounds
    "cal_amp_hi":       1.4,
    "cal_amp_points":   9,
    "drag_always":      False,       # force DRAG on for every allocation point
    "drag_subharmonic": False,       # also let DRAG target the nearest SUBHARMONIC
                                     #   collision: the pump's 2nd harmonic (2 w_p, generated
                                     #   by the SNAIL) drives mode i at w_i = 2 w_p, for i in
                                     #   {a, b, spectator, coupler}. The detuning of that
                                     #   two-pump drive is w_i - 2 w_p (NOT w_i/2 - w_p), which
                                     #   is what DRAG must use.
    "subharmonic_modes": ["a", "b", "spec", "s"],  # which modes' subharmonics to include
                                     #   when drag_subharmonic is on; e.g. ["spec"] isolates the
                                     #   swept spectator subharmonic, ["s"] the SNAIL/coupler one
    "no_spectator": False,           # [target] sweep the BARE a-b-coupler gate (no spectator:
                                     #   lam_spec=0), varying w_b so 2 w_p scans the SNAIL
                                     #   subharmonic w_c = 2 w_p; nearest-collision reports w_c-2w_p
    "integrate": True,               # set False for the instant analytic map only
    "rtol": 1e-8, "atol": 1e-10,     # QuTiP ODE tolerances
    "nsteps": 500000,                # max internal solver steps between outputs
}

DEFAULT_SPECFREQS_GHz = [round(0.20 + 0.05 * k, 3) for k in range(15)]

DEFAULT_DRAGS = [False, True]

DEFAULT_TARGET_DRAGS = [False]


@dataclass
class Point:
    """One sweep point (a single coupler configuration to simulate).

    Attributes
    ----------
    index : int
        Position in the grid; also the output filename suffix.
    spec_freq_GHz : float
        Spectator transition frequency Delta = w_b - w_spec (GHz; spectator sweep).
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
    spec_freq_GHz: float = 0.0
    drag: bool = False
    kind: str = "spectator"
    wa_GHz: Optional[float] = None
    wb_GHz: Optional[float] = None
    spec_abs_GHz: Optional[float] = None

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

def _stark_offset_GHz(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                      t_g: float, amp_scale: float, solver: Dict[str, Any],
                      spec_abs_GHz: Optional[float] = None,
                      drag_beat_GHz: Optional[float] = None) -> Dict[str, Any]:
    """Per-point AC-Stark-shifted iSWAP resonance offset (GHz) for the pump.

    Runs the chevron of find_stark_resonance.py at this point's (w_a, w_b) and
    operating amplitude, returning the offset from |w_b - w_a| that maximises swap
    contrast. With ``spec_abs_GHz`` given, the spectator is INCLUDED in the chevron
    at that absolute frequency, so the located resonance carries the spectator's
    (detuning-dependent) dispersive pull -- the offset then varies point to point.
    With ``spec_abs_GHz=None`` it is the bare a<->b resonance. The offset scan
    parallelizes over ``config['stark_jobs']`` processes: keep it at 1 when the
    caller is itself in a pool (mode=local) and raise it to cpus-per-task in
    mode=point, where one task runs a single point.

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
    spec_abs_GHz : float, optional
        Spectator ABSOLUTE frequency (GHz) to include in the chevron. None -> bare
        pair.
    drag_beat_GHz : float, optional
        DRAG beat detuning (GHz). Only used when ``config['stark_match_pulse']`` is
        set: the chevron then uses the ACTUAL raised-cosine gate pulse with the DRAG
        quadrature tuned to this beat, so the located resonance is the DRAG-ON
        resonance (it carries the DRAG-quadrature Stark shift a constant probe
        cannot see). None -> shaped pulse without DRAG.

    Returns
    -------
    dict
        The full find_stark_resonance.scan result (offsets_GHz, times_ns, P10,
        max_transfer, resonance_offset_GHz, eta_op, shape, drag_beat_GHz, ...).
        Callers take ``resonance_offset_GHz`` and may persist the rest.
    """
    import find_stark_resonance as FS
    sub = dict(config)
    sub["qubit_freqs_GHz"] = [float(wa_GHz), float(wb_GHz)]
    span = float(config.get("stark_span_MHz", 60.0)) / 1000.0
    n_pts = int(config.get("stark_points", 21))
    offsets = np.linspace(-span / 2.0, span / 2.0, n_pts)
    window = float(config.get("stark_window_factor", 2.0)) * float(t_g)
    n_time = int(config.get("stark_time_points", 120))
    shaped = bool(config.get("stark_match_pulse", False))
    return FS.scan(sub, float(t_g), float(amp_scale), offsets, window, n_time,
                   solver, n_jobs=int(config.get("stark_jobs", 1)),
                   spec_abs_GHz=(None if spec_abs_GHz is None else float(spec_abs_GHz)),
                   shape=("raised_cosine" if shaped else "constant"),
                   drag_beat_GHz=(float(drag_beat_GHz) if (shaped and drag_beat_GHz is not None)
                                  else None))

def _nearest_collision(config: Dict[str, Any], wa_GHz: float, wb_GHz: float,
                       ws_GHz: float, wspec_GHz: float, w_p_GHz: float):
    """Nearest spectator/mode collision to the pump for a target point.

    Candidates: the per-qubit exchange channels -- one-pump swap (resonant at
    ``|w_q - w_spec| = w_p``) and static exchange (``w_q = w_spec``) -- with
    ``beat = |w_q - w_spec| - n*w_p`` (n in {1, 0}); and, when ``drag_subharmonic``
    is set, the subharmonic drives (the pump's 2nd harmonic, 2 w_p, driving mode i
    at w_i = 2 w_p) for i in ``subharmonic_modes`` (subset of {a, b, spectator,
    coupler}), with ``beat = w_i - 2 w_p`` -- the true detuning of that two-pump
    drive. The beat is the detuning from the channel's resonance (the DRAG detuning).
    Uses target-sweep mode indices a=0, b=1, coupler=2, spectator=3.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    wa_GHz, wb_GHz, ws_GHz, wspec_GHz : float
        Absolute mode frequencies (GHz).
    w_p_GHz : float
        Pump frequency (GHz) to reference the beats against (nominal or calibrated).

    Returns
    -------
    tuple
        ``(|beat|, signed beat, kind, target_label, target_idx)`` for the nearest
        channel; ``kind`` in {"onepump", "static", "subharm"}.
    """
    a, b, coupler, spec = 0, 1, 2, 3
    nearest = None
    no_spec = bool(config.get("no_spectator", False))
    if not no_spec:
        for q_idx, q_freq, q_label in ((a, wa_GHz, "a"), (b, wb_GHz, "b")):
            sep = abs(q_freq - wspec_GHz)
            for kind, harm in (("onepump", w_p_GHz), ("static", 0.0)):
                beat = sep - harm
                if nearest is None or abs(beat) < nearest[0]:
                    nearest = (abs(beat), float(beat), kind, q_label, q_idx)
    if no_spec or bool(config.get("drag_subharmonic", False)):
        # no_spectator: the only channels are the subharmonics; default to the SNAIL one.
        sub_modes = config.get("subharmonic_modes") or (["s"] if no_spec
                                                        else ["a", "b", "spec", "s"])
        for lab, idx, wi in (("a", a, wa_GHz), ("b", b, wb_GHz),
                             ("spec", spec, wspec_GHz), ("s", coupler, ws_GHz)):
            if lab not in sub_modes:
                continue
            if lab == "spec" and no_spec:               # no spectator mode present
                continue
            beat = wi - 2.0 * w_p_GHz                    # detuning of the 2-pump drive: w_i - 2 w_p
            if nearest is None or abs(beat) < nearest[0]:
                nearest = (abs(beat), float(beat), "subharm", lab, idx)
    if nearest is None:
        nearest = (0.0, 0.0, "none", "-", -1)
    return nearest

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
    chev = result.pop("_chevron", None)        # popped before JSON so meta stays scalar
    extra: Dict[str, Any] = {}
    if chev is not None:
        extra = {"chev_offsets_GHz": np.asarray(chev["offsets_GHz"], dtype=float),
                 "chev_times_ns": np.asarray(chev["times_ns"], dtype=float),
                 "chev_P10": np.asarray(chev["P10"], dtype=float),
                 "chev_max_transfer": np.asarray(chev["max_transfer"], dtype=float),
                 "chev_resonance_metric": np.asarray(
                     chev.get("resonance_metric", chev["max_transfer"]), dtype=float),
                 "chev_metric_label": str(chev.get("metric_label", "max-over-time P(|10>)")),
                 "chev_resonance_offset_GHz": float(chev["resonance_offset_GHz"]),
                 "chev_eta_op": float(chev["eta_op"]),
                 "chev_shape": str(chev.get("shape", "constant")),
                 "chev_drag_beat_GHz": float(chev.get("drag_beat_GHz", np.nan)),
                 "chev_spec_abs_GHz": float(chev.get("spec_abs_GHz", np.nan))}
    np.savez_compressed(path,
                        U_proj_real=np.real(U), U_proj_imag=np.imag(U),
                        meta=json.dumps(result), **extra)
    return path

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
    spec_cols = ["index", "spec_freq_GHz", "lam_spec", "drag", "drag_applied",
                 "beat_GHz", "nearest_kind", "nearest_target",
                 "eta_peak", "g_iswap_eff_MHz", "g_spec_eff_MHz",
                 "status", "F_avg", "leakage", "n_spec", "n_coupler", "p_transfer",
                 "w_p_GHz", "stark_offset_MHz", "amp_scale_used", "wp_offset_used_MHz",
                 "w_spec_GHz", "t_g_ns", "wall_s"]
    target_cols = ["index", "kind", "wa_GHz", "wb_GHz", "w_snail_GHz",
                   "spec_GHz", "detuning_GHz", "w_p_GHz", "stark_offset_MHz",
                   "amp_scale_used", "wp_offset_used_MHz", "lam_spec", "drag",
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

    # Flag gaps against the grid so a partial (timed-out) sweep is obvious here, not just
    # in `missing`. Cheap: reads grid.json only if present.
    try:
        _cfg, _pts = load_grid(outdir)
        got = {r["index"] for r in rows}
        gaps = sorted(set(range(len(_pts))) - got)
        if gaps:
            print(f"  WARNING: {len(gaps)}/{len(_pts)} grid points have no file "
                  f"(unfinished). Run `missing` to list and resubmit them.")
    except Exception:
        pass  # no grid.json (e.g. collecting a hand-assembled points dir)

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

def plot_chevrons(outdir: str, indices: Optional[List[int]] = None) -> List[str]:
    """Render the per-point Stark chevrons saved by --stark runs to
    ``<outdir>/figs/chevrons/chevron_XXXXX.png`` (reuses find_stark_resonance's
    renderer, so DRAG-matched chevrons are labelled with their beat).

    Parameters
    ----------
    outdir : str
        Sweep output directory (must contain ``points/point_*.npz``).
    indices : list of int, optional
        Restrict to these point indices; default renders every point that stored a
        chevron.

    Returns
    -------
    list of str
        Paths of the written PNGs.
    """
    import glob
    import find_stark_resonance as FS
    figs = os.path.join(outdir, "figs", "chevrons")
    os.makedirs(figs, exist_ok=True)
    written: List[str] = []
    for npz in sorted(glob.glob(os.path.join(outdir, "points", "point_*.npz"))):
        d = np.load(npz)
        if "chev_P10" not in d.files:
            continue
        meta = json.loads(str(d["meta"]))
        idx = int(meta.get("index", -1))
        if indices is not None and idx not in indices:
            continue
        chev = {"offsets_GHz": d["chev_offsets_GHz"], "times_ns": d["chev_times_ns"],
                "P10": d["chev_P10"], "max_transfer": d["chev_max_transfer"],
                "resonance_offset_GHz": float(d["chev_resonance_offset_GHz"]),
                "eta_op": float(d["chev_eta_op"]),
                "shape": str(d["chev_shape"]),
                "drag_beat_GHz": float(d["chev_drag_beat_GHz"])}
        if "chev_resonance_metric" in d.files:
            chev["resonance_metric"] = d["chev_resonance_metric"]
            chev["metric_label"] = (str(d["chev_metric_label"])
                                    if "chev_metric_label" in d.files else "")
        bits: List[str] = []
        if "spec_freq_GHz" in meta:
            bits.append(rf"$\Delta$={float(meta['spec_freq_GHz']):.3f} GHz")
        if isinstance(meta.get("beat_GHz"), (int, float)):
            bits.append(f"beat={float(meta['beat_GHz'])*1e3:+.0f} MHz")
        bits.append("DRAG on" if meta.get("drag") else "DRAG off")
        png = os.path.join(figs, f"chevron_{idx:05d}.png")
        FS.render_chevron(chev, png, title_suffix=f"pt {idx}  " + "  ".join(bits))
        written.append(png)
    return written

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

def _print_submit_hint(outdir: str, m: int, max_array: int = 1000) -> None:
    """Print the sbatch submission line, chunking the array when the point count would
    exceed a typical SLURM ``MaxArraySize``. One array task then runs CHUNK contiguous
    points, so the array size is ceil(m / CHUNK).

    Parameters
    ----------
    outdir : str
        Sweep output directory.
    m : int
        Number of grid points.
    max_array : int, default 1000
        Conservative MaxArraySize assumption; the true value is site-specific
        (``scontrol show config | grep MaxArraySize``).

    Returns
    -------
    None
    """
    import math
    base = f"RUNNER=run_sweep_zhou.py OUTDIR={outdir}"
    if m <= max_array:
        print(f"Submit with:\n  {base} sbatch --array=0-{m - 1} slurm/snail_sweep.slurm")
        return
    chunk = math.ceil(m / max_array)
    ntasks = math.ceil(m / chunk)
    print(f"Submit with (N={m} exceeds a typical MaxArraySize={max_array}, so CHUNK the array):")
    print(f"  {base} CHUNK={chunk} sbatch --array=0-{ntasks - 1} slurm/snail_sweep.slurm")
    print(f"  -> {ntasks} tasks x {chunk} points/task. Check your site limit with "
          f"`scontrol show config | grep MaxArraySize` and raise --array/lower CHUNK if it allows.")
    print(f"  (raise #SBATCH --time accordingly: each task now runs {chunk} points in series.)")

def _compress_ranges(indices: List[int]) -> str:
    """Collapse a sorted index list into a SLURM ``--array`` spec, e.g.
    ``[3, 7, 8, 9, 20]`` -> ``"3,7-9,20"``.

    Parameters
    ----------
    indices : list of int
        Sorted, unique, non-negative indices.

    Returns
    -------
    str
        Comma-separated runs of contiguous indices.
    """
    if not indices:
        return ""
    parts: List[str] = []
    lo = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{lo}" if lo == prev else f"{lo}-{prev}")
        lo = prev = i
    parts.append(f"{lo}" if lo == prev else f"{lo}-{prev}")
    return ",".join(parts)

def find_missing(outdir: str, max_array: int = 1000) -> None:
    """Report grid points with no saved ``point_XXXXX.npz`` (i.e. they never finished),
    write a resume list, and print ready-to-run resubmission commands.

    A point file is written only after :func:`run_point` returns, so a missing file means
    the task timed out or died before saving. Points that finished but produced a bad
    result (e.g. ``F_avg`` is nan) are *not* caught here -- those have a file; check the
    ``status``/``F_avg`` columns of ``summary.csv`` after ``collect`` for those.

    Parameters
    ----------
    outdir : str
        Sweep directory containing ``grid.json`` and ``points/point_*.npz``.
    max_array : int, optional
        Assumed SLURM ``MaxArraySize`` for choosing between a direct index-list array and
        a chunked resume array. Confirm your site's value with
        ``scontrol show config | grep MaxArraySize``.
    """
    _config, points = load_grid(outdir)
    n = len(points)
    done: set[int] = set()
    for p in glob.glob(os.path.join(outdir, "points", "point_*.npz")):
        stem = os.path.basename(p)[len("point_"):-len(".npz")]
        try:
            done.add(int(stem))
        except ValueError:
            pass
    missing = sorted(set(range(n)) - done)
    extra = sorted(i for i in done if i >= n)  # stray files from a stale/edited grid
    print(f"{n} grid points: {n - len(missing)} finished, {len(missing)} missing.")
    if extra:
        print(f"  note: {len(extra)} saved point file(s) have index >= {n} "
              f"(stale grid?): {_compress_ranges(extra)}")
    if not missing:
        print("nothing to resubmit.")
        return

    listpath = os.path.join(outdir, "missing.txt")
    with open(listpath, "w") as f:
        f.write("\n".join(str(i) for i in missing) + "\n")
    print(f"wrote {listpath}  ({len(missing)} indices)")
    print(f"  indices: {_compress_ranges(missing)}")

    m = len(missing)
    print("\nResubmit (robust for any indices -- array is 0..M-1, points read from the list):")
    if m <= max_array:
        print(f"  RESUME={listpath} OUTDIR={outdir} sbatch "
              f"--array=0-{m - 1} slurm/snail_sweep.slurm")
    else:
        chunk = math.ceil(m / max_array)
        ntasks = math.ceil(m / chunk)
        print(f"  RESUME={listpath} OUTDIR={outdir} CHUNK={chunk} sbatch "
              f"--array=0-{ntasks - 1} slurm/snail_sweep.slurm")
        print(f"  ({ntasks} tasks x {chunk} points/task; raise #SBATCH --time to match.)")
    if missing and missing[-1] < max_array:
        print("Or directly (only if the largest index is below MaxArraySize):")
        print(f"  OUTDIR={outdir} CHUNK=1 sbatch "
              f"--array={_compress_ranges(missing)} slurm/snail_sweep.slurm")
    print("Then re-run `collect` once the resubmitted tasks finish.")


__all__ = [
    'TWO_PI',
    '_DELTA_EPS_GHz',
    '_drag_skip_GHz',
    'DEFAULT_CONFIG',
    'DEFAULT_SPECFREQS_GHz',
    'DEFAULT_DRAGS',
    'DEFAULT_TARGET_DRAGS',
    'Point',
    'write_grid',
    'load_grid',
    '_stark_offset_GHz',
    '_nearest_collision',
    'save_point',
    'collect',
    'plot_chevrons',
    '_parse_list',
    '_bool_list',
    '_log_line',
    '_print_submit_hint',
    '_compress_ranges',
    'find_missing',
]