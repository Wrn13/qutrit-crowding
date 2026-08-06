"""Spectator sweep: fixed a-b pair, a single spectator swept in frequency.

build_grid() builds the Delta = w_b - w_spec grid; run_spectator_point()
evaluates one point (analytic collision + full iSWAP fidelity).
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sweep_common import (Point, TWO_PI, DEFAULT_CONFIG, _drag_skip_GHz,
                          _nearest_collision, _stark_offset_GHz,
                          _grape_augment)


def build_grid(specfreqs: Sequence[float], drags: Sequence[bool]) -> List[Point]:
    """Build the Cartesian sweep grid in a stable order.

    Parameters
    ----------
    specfreqs : sequence of float
        Spectator frequencies Delta = w_b - w_spec (GHz).
    drags : sequence of bool
        DRAG on/off settings.

    Returns
    -------
    list of Point
        Points ordered spec_freq -> drag, indexed 0..M-1.
    """
    points: List[Point] = []
    index = 0
    for spec_freq in specfreqs:
        for drag in drags:
            points.append(Point(index=index, spec_freq_GHz=float(spec_freq),
                                drag=bool(drag)))
            index += 1
    return points

def run_spectator_point(pt: Point, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute one sweep point: the analytic collision prediction always, and the
    full QuTiP gate simulation when ``config['integrate']`` is True.

    Parameters
    ----------
    pt : Point
        The grid point (spectator frequency, DRAG flag).
    config : dict
        Resolved device/simulation configuration (see DEFAULT_CONFIG). Imports
        ``zhou_coupler`` (and, for the full path, QuTiP) lazily.

    Returns
    -------
    dict
        Result row with analytic fields (beat, eta_peak, effective rates, ...) and,
        if integrated, F_avg / leakage / occupations / the 4x4 ``U_proj``.
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

    t0 = time.time()
    a, b, coupler, spec = 0, 1, 2, 3
    anchor = int(config["anchor"])

    # --- frequencies (rad/ns) ---------------------------------------------
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float) * TWO_PI)
    ws = config["coupler_freq_GHz"] * TWO_PI
    w_p = abs(wb - wa)                              # iSWAP pump = |detuning| (fixed)
    w_p_GHz = w_p / TWO_PI + float(config.get("wp_offset_GHz", 0.0))   # calibrated offset
    w_spec = wb - pt.spec_freq_GHz * TWO_PI         # move ONLY the spectator
    freqs_GHz = [wa / TWO_PI, wb / TWO_PI, ws / TWO_PI, w_spec / TWO_PI]

    # --- per-point calibration (needs the integrated run) -------------------
    # calibrate_points -> full amplitude + Stark tune-up (calibrate_gate) with the
    #   spectator LOADED at w_spec, and DRAG on if this point uses it (hardware-style);
    #   sets amp_scale AND wp_offset for THIS point.
    # stark_drive -> frequency only (cheaper): shift w_p to the spectator-aware Stark
    #   resonance at the configured amplitude. The chevron INCLUDES the spectator, so
    #   the offset carries its dispersive pull (largest as the collision beat -> 0); with
    #   stark_match_pulse it uses the ACTUAL pulse (+ DRAG), i.e. the DRAG-on resonance.
    amp_scale_used = float(config.get("amp_scale", 1.0))
    wp_offset_used_GHz = float(config.get("wp_offset_GHz", 0.0))
    stark_offset_GHz = 0.0
    _chevron = None
    _w_p_nom_GHz = w_p / TWO_PI
    _wspec_abs_GHz = w_spec / TWO_PI
    if bool(config.get("integrate", True)) and bool(config.get("calibrate_points", False)):
        import calibrate_gate as CG
        _sv = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                   nsteps=int(config.get("nsteps", 500000)))
        sub = dict(config); sub["qubit_freqs_GHz"] = [wa / TWO_PI, wb / TWO_PI]
        # DRAG on during calibration if this point runs DRAG, tuned to the anchor beat
        # (Delta - w_p) from the nominal pump; skip on-collision (beat -> 0 is singular).
        _cb = _nearest_collision(config, wa / TWO_PI, wb / TWO_PI, ws / TWO_PI,
                                 _wspec_abs_GHz, _w_p_nom_GHz)[1]
        _cal_drag = (_cb if (bool(pt.drag) and abs(_cb) >= _drag_skip_GHz(config)) else None)
        rec = CG.run_calibration(
            sub, float(config["t_g_ns"]),
            iters=int(config.get("calibrate_iters", 1)),
            amp_bounds=(float(config.get("cal_amp_lo", 0.6)),
                        float(config.get("cal_amp_hi", 1.4))),
            amp_points=int(config.get("cal_amp_points", 9)),
            span_MHz=float(config.get("stark_span_MHz", 60.0)),
            chevron_points=int(config.get("stark_points", 21)),
            window_factor=float(config.get("stark_window_factor", 2.0)),
            time_points=int(config.get("stark_time_points", 120)),
            solver=_sv, n_jobs=1,
            spec_abs_GHz=_wspec_abs_GHz, drag_beat_GHz=_cal_drag)["final"]
        amp_scale_used = float(rec["amp_scale"])
        wp_offset_used_GHz = float(rec["wp_offset_GHz"])          # measured from nominal
    elif bool(config.get("integrate", True)) and bool(config.get("stark_drive", False)):
        _sv = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                   nsteps=int(config.get("nsteps", 500000)))
        # DRAG beat for the chevron uses the pre-Stark pump (the ~MHz Stark offset is
        # negligible vs the beat in the DRAG quadrature); skip DRAG on-collision.
        _chev_beat = _nearest_collision(config, wa / TWO_PI, wb / TWO_PI, ws / TWO_PI,
                                        _wspec_abs_GHz,
                                        _w_p_nom_GHz + wp_offset_used_GHz)[1]
        _chev_drag = (_chev_beat if (pt.drag and abs(_chev_beat) >= _drag_skip_GHz(config))
                      else None)
        _chevron = _stark_offset_GHz(config, wa / TWO_PI, wb / TWO_PI,
                                     float(config["t_g_ns"]), amp_scale_used, _sv,
                                     spec_abs_GHz=_wspec_abs_GHz,
                                     drag_beat_GHz=_chev_drag)
        stark_offset_GHz = float(_chevron["resonance_offset_GHz"])
        wp_offset_used_GHz += stark_offset_GHz
    w_p_GHz = _w_p_nom_GHz + wp_offset_used_GHz               # calibrated / configured pump

    # --- spectator: a single 3-level anharmonic transmon --------------------
    spec_levels = int(config["spec_levels"])
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"]), spec_levels]

    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    aq = float(config.get("anharm_qubit_GHz", 0.0))
    lam_b = float(config["lam_b"])
    anharm = {a: aq, b: aq, spec: float(config.get("anharm_spec_GHz", 0.0))}

    cpl = ZhouCoupler(
        mode_freqs_GHz=freqs_GHz,
        coupler_index=coupler,
        participations={a: float(config["lam_a"]),
                        b: lam_b,
                        spec: lam_b},          # spectator participation = lam_b
        nonlinearities=nonlin,
        levels=levels,
        anharmonicities_GHz=anharm,
    )

    # DRAG targets the off-resonant spectator exchange, detuned from the pump by the
    # NEAREST collision across channels (via _nearest_collision, matching the target
    # sweep): the one-pump swaps |w_q - w_spec| = w_p and static exchanges w_q = w_spec
    # for q in {a, b} (and, with --drag-subharmonic, the w_i/2 = w_p subharmonics).
    # Because it uses |w_q - w_spec|, the spectator may sit below OR above w_b; for a
    # below-w_b spectator whose nearest channel is the b<->spec one-pump, this reduces
    # to the old spec_freq - w_p. It is singular as beat -> 0 (an on-resonant collision
    # needs frequency allocation, not DRAG), so skip it there. nearest_kind/target say
    # WHICH channel it is.
    _nearest = _nearest_collision(config, wa / TWO_PI, wb / TWO_PI, ws / TWO_PI,
                                  _wspec_abs_GHz, w_p_GHz)
    beat_GHz = _nearest[1]
    use_drag = bool(pt.drag)
    status_drag = "ok"
    if pt.drag and abs(beat_GHz) < _drag_skip_GHz(config):
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
    cpl.scale_pump_amplitude(amp_scale_used)

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
        "spec_freq_GHz": pt.spec_freq_GHz,
        "lam_spec": round(lam_b, 4),           # spectator participation (= lam_b)
        "drag": bool(pt.drag),
        "drag_applied": bool(use_drag),
        "beat_GHz": round(float(beat_GHz), 6),
        "nearest_kind": _nearest[2],
        "nearest_target": _nearest[3],
        "eta_peak": round(float(eta_peak), 5),
        "g_iswap_eff_MHz": round(float(g_iswap / TWO_PI * 1e3), 4),
        "g_spec_eff_MHz": round(float(g_spec / TWO_PI * 1e3), 4),
        "w_p_GHz": round(float(w_p_GHz), 6),
        "stark_offset_MHz": round(float(stark_offset_GHz) * 1e3, 4),
        "amp_scale_used": round(float(amp_scale_used), 5),
        "wp_offset_used_MHz": round(float(wp_offset_used_GHz) * 1e3, 4),
        "w_spec_GHz": round(float(w_spec / TWO_PI), 6),
        "t_g_ns": float(config["t_g_ns"]),
        "status": status_drag if status_drag != "ok" else "analytic",
        "F_avg": "", "leakage": "", "n_spec": "", "n_coupler": "", "p_transfer": "",
        "grape_baseline_F": "", "F_grape": "", "leak_grape": "", "dF_grape": "",
        "grape_nfev": "", "grape_warmstart_GHz": "",
        "U_proj": None,
    }
    if _chevron is not None:
        out["_chevron"] = _chevron          # persisted by save_point, plotted by mode=chevrons

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

    # GRAPE optimal control on this point (opt-in); baseline = the applied gate.
    if config.get("grape"):
        _grape_augment(out, cpl, a, b, config,
                       drag_beat_GHz=(beat_GHz if use_drag else None),
                       nearest_beat_GHz=beat_GHz)

    out["wall_s"] = time.time() - t0
    return out