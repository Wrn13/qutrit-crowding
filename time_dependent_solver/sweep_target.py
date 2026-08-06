"""Target (frequency-allocation) sweep: fixed w_a and coupler; w_b and a
spectator swept. build_target_grid() builds the grid; _run_target_point()
evaluates one point. Supports the bare-gate no_spectator mode.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sweep_common import (Point, TWO_PI, DEFAULT_CONFIG, _drag_skip_GHz,
                          _nearest_collision, _stark_offset_GHz,
                          _grape_augment)


def build_target_grid(wa_GHz: float, wb_list: Sequence[float], spec_list: Sequence[float],
                      drags: Sequence[bool],
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
    drags : sequence of bool
        DRAG on/off settings.
    min_detuning_GHz : float, default 0.05
        Minimum |w_b - w_a|; smaller placements are skipped.

    Returns
    -------
    list of Point
        Points (kind="target") ordered w_b -> w_spec -> drag, indexed 0..M-1.
    """
    points: List[Point] = []
    index = 0
    for wb in wb_list:
        if abs(float(wb) - float(wa_GHz)) < float(min_detuning_GHz):
            continue
        for spec in spec_list:
            for drag in drags:
                points.append(Point(index=index, kind="target",
                                    wa_GHz=float(wa_GHz), wb_GHz=float(wb),
                                    spec_abs_GHz=float(spec), drag=bool(drag)))
                index += 1
    if not points:
        raise ValueError("empty target grid; check --wb-GHz/--spec-GHz and min_detuning_GHz.")
    return points

def _run_target_point(pt: Point, config: Dict[str, Any]) -> Dict[str, Any]:
    """Allocation point: fixed w_a and w_s; the partner sits at w_b and a single
    spectator at absolute w_spec. Analytic collision search always; full iSWAP
    fidelity when ``config['integrate']``. Same return-row contract as `run_point`.

    Parameters
    ----------
    pt : Point
        A kind="target" point carrying wb_GHz, spec_abs_GHz, drag (w_a is
        read from the config, held fixed).
    config : dict
        Resolved configuration; reads ``qubit_freqs_GHz[0]`` (fixed w_a),
        ``coupler_freq_GHz`` (fixed w_s), ``lam_a``/``lam_b`` (the spectator uses
        ``lam_b``), ``g3_GHz``, pulse and solver keys.

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
    no_spec = bool(config.get("no_spectator", False))  # true 3-mode bare a-b-coupler gate
    lam_spec = 0.0 if no_spec else float(config["lam_b"])   # spectator participation
    integrate = bool(config.get("integrate", True))

    # Per-point calibration (w_b AND the spectator vary across the sweep, so the optimal
    # amplitude and the Stark shift move point to point). Two levels:
    #   calibrate_points -> full amplitude + Stark tune-up (calibrate_gate) with the
    #                       spectator loaded (hardware-style), sets amp_scale + wp_offset;
    #   stark_drive      -> frequency only (cheaper): shift w_p to the Stark
    #                       resonance at the configured amplitude.
    # Both need the integrated run and drive the (a,b) iSWAP.
    amp_scale_used = float(config.get("amp_scale", 1.0))
    wp_offset_used_GHz = float(config.get("wp_offset_GHz", 0.0))
    _chevron = None
    _w_p_nom = abs(wb_GHz - wa_GHz)
    _use_drag = bool(config.get("drag_always", False)) or bool(pt.drag)
    if integrate and bool(config.get("calibrate_points", False)):
        import calibrate_gate as CG
        _sv = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                   nsteps=int(config.get("nsteps", 500000)))
        sub = dict(config); sub["qubit_freqs_GHz"] = [wa_GHz, wb_GHz]
        # Hardware-style: calibrate with the spectator present at wspec_GHz, and with
        # DRAG on (tuned to the nearest-collision beat from the nominal pump) if this
        # point runs DRAG, so the tune-up matches the gate as actually operated.
        _cb = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, _w_p_nom)[1]
        _cal_drag = (_cb if (_use_drag and abs(_cb) >= _drag_skip_GHz(config)) else None)
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
            spec_abs_GHz=(None if no_spec else wspec_GHz), drag_beat_GHz=_cal_drag)["final"]
        amp_scale_used = float(rec["amp_scale"])
        wp_offset_used_GHz = float(rec["wp_offset_GHz"])
        w_p_GHz = abs(wb_GHz - wa_GHz) + wp_offset_used_GHz
    elif integrate and bool(config.get("stark_drive", False)):
        _sv = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                   nsteps=int(config.get("nsteps", 500000)))
        # nearest collision from the pre-Stark pump, for the DRAG-matched chevron
        # (only meaningful when DRAG is forced on every point via drag_always).
        _cb = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, w_p_GHz)[1]
        _chev_drag = (_cb if (bool(config.get("drag_always", False))
                              and abs(_cb) >= _drag_skip_GHz(config)) else None)
        _chevron = _stark_offset_GHz(config, wa_GHz, wb_GHz,
                                     float(config["t_g_ns"]), amp_scale_used, _sv,
                                     spec_abs_GHz=(None if no_spec else wspec_GHz),
                                     drag_beat_GHz=_chev_drag)
        wp_offset_used_GHz += float(_chevron["resonance_offset_GHz"])
        w_p_GHz = abs(wb_GHz - wa_GHz) + wp_offset_used_GHz

    # Rates/eta are level-independent, so the analytic-only build uses 2 levels
    # everywhere (tiny Hilbert space); the full build uses the configured levels.
    if integrate:
        q_lv, c_lv = int(config["qubit_levels"]), int(config["coupler_levels"])
        s_lv = int(config["spec_levels"])
    else:
        q_lv = c_lv = s_lv = 2

    aq = float(config.get("anharm_qubit_GHz", 0.0))
    if no_spec:
        # true 3-mode bare gate [a, b, coupler] -- no spectator Hilbert dimension
        freqs_GHz = [wa_GHz, wb_GHz, ws_GHz]
        participations = {a: float(config["lam_a"]), b: float(config["lam_b"])}
        levels = [q_lv, q_lv, c_lv]
        anharm = {a: aq, b: aq}
    else:
        freqs_GHz = [wa_GHz, wb_GHz, ws_GHz, wspec_GHz]
        participations = {a: float(config["lam_a"]), b: float(config["lam_b"]),
                          spec: lam_spec}
        levels = [q_lv, q_lv, c_lv, s_lv]
        anharm = {a: aq, b: aq, spec: float(config.get("anharm_spec_GHz", 0.0))}

    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    cpl = ZhouCoupler(mode_freqs_GHz=freqs_GHz, coupler_index=coupler,
                      participations=participations, nonlinearities=nonlin, levels=levels,
                      anharmonicities_GHz=anharm)

    # --- nearest collision: spectator vs {a, b}, referenced to the calibrated pump ---
    nearest = _nearest_collision(config, wa_GHz, wb_GHz, ws_GHz, wspec_GHz, w_p_GHz)

    # near-collision window where DRAG is meant to help (off-resonant but close).
    # Round the beat to 1 kHz so placements exactly at the threshold classify
    # deterministically (strict < threshold, i.e. genuinely below the cutoff).
    drag_beat = nearest[1]
    beat_abs = round(abs(drag_beat), 6)
    thr_GHz = float(config.get("drag_compare_below_MHz", 100.0)) / 1000.0
    drag_compare = bool(config.get("drag_compare", False))
    in_window = (_drag_skip_GHz(config) < beat_abs < thr_GHz)

    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse
    amp_scale = amp_scale_used                          # per-point calibrated (or config default)

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
        base_drag = bool(pt.drag) or bool(config.get("drag_always", False))
        status_drag = "ok"
        if base_drag and abs(drag_beat) < _drag_skip_GHz(config):
            base_drag = False
            status_drag = "drag_skipped_resonant_collision"

    _configure(base_drag)
    eta_peak = cpl.peak_eta()
    g_iswap = cpl.iswap_rate(a, b)
    if no_spec or nearest[2] == "subharm":
        g_coll = float("nan")     # no spectator, or a higher-order (g4) subharmonic drive;
                                  # the cubic pair rate below models neither
    else:
        g_coll = cpl.effective_rate([nearest[4], spec], n=3, C=6)   # 6 g3 l_q l_spec |eta|

    out = {
        "index": pt.index, "kind": "target",
        "wa_GHz": round(wa_GHz, 6), "wb_GHz": round(wb_GHz, 6),
        "w_snail_GHz": round(ws_GHz, 6),
        "spec_GHz": ("" if no_spec else round(wspec_GHz, 6)),
        "detuning_GHz": round(wb_GHz - wa_GHz, 6), "w_p_GHz": round(w_p_GHz, 6),
        "stark_offset_MHz": round((wp_offset_used_GHz
                                   - float(config.get("wp_offset_GHz", 0.0))) * 1e3, 4),
        "amp_scale_used": round(float(amp_scale_used), 5),
        "wp_offset_used_MHz": round(float(wp_offset_used_GHz) * 1e3, 4),
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
        "grape_baseline_F": "", "F_grape": "", "leak_grape": "", "dF_grape": "",
        "grape_nfev": "",
        "U_proj": None,
    }
    if _chevron is not None:
        out["_chevron"] = _chevron          # persisted by save_point, plotted by mode=chevrons

    if not integrate:
        out["wall_s"] = time.time() - t0
        return out

    # --- FULL non-perturbative evolution (QuTiP, exact Hamiltonian) -----------
    t_g = float(config["t_g_ns"])
    solver = dict(atol=float(config["atol"]), rtol=float(config["rtol"]),
                  nsteps=int(config.get("nsteps", 500000)))

    n_modes = 3 if no_spec else 4
    occ0 = [0] * n_modes; occ0[a] = 1
    pops = np.abs(cpl.evolve_state(occ0, t_g, **solver)) ** 2
    occ_b = [0] * n_modes; occ_b[b] = 1
    out["n_spec"] = "" if no_spec else float(cpl.mean_occupation(pops, spec))
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

    # (c) GRAPE optimal control on this exact point (opt-in). Baseline = the gate
    # actually applied here, so dF_grape is the headroom over DRAG/raised-cosine.
    if config.get("grape"):
        _grape_augment(out, cpl, a, b, config,
                       drag_beat_GHz=(drag_beat if out["drag_applied"] else None))

    out["wall_s"] = time.time() - t0
    return out