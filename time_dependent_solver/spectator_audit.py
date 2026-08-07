#!/usr/bin/env python3
r"""Spectator channel audit: which parasitic processes exist, and how STRONG they are.

``plot_allocation.py`` answers "WHERE are the collisions" -- it draws the curated
analytic centres (direct exchange, one-pump sidebands, subharmonics) as forbidden
zones on the frequency axis. This tool answers "HOW BAD is each one", by reading
the strengths out of the MODEL instead of a table:

For a spectator placed at w_spec, build the real 4-mode system [a, b, coupler,
spec] and walk every term of ``ZhouCoupler.expand_terms``. Each term carries a
carrier detuning Omega_j (the channel's beat) and an operator O_j; the matrix
elements of O_j that EXCITE the spectator out of |0>, starting from the low-lying
computational states, are the parasitic couplings. That enumeration is complete by
construction -- it cannot miss a channel the way a hand-written list can, and it
automatically includes eta^2 / eta^3 processes (the two-pump subharmonics), the
|2>-involving and anharmonicity-shifted channels, and coupler-mediated
combinations.

Per channel it reports

    g            coupling matrix element (MHz), weighted by |eta|^n_pump
    detuning     signed Omega_j / 2 pi (MHz) -- the DRAG beat for that channel
    g/|detuning| dimensionless first-order excitation amplitude. >= 1 means the
                 channel is NOT perturbative: no DRAG/virtual-Z correction fixes
                 it, and the placement has to move.
    P_exc        off-resonant excitation estimate min(1, 2 (g/detuning)^2), the
                 population parked in the spectator during the gate
    transition    which modes change occupation, and the pump order

and a band scan sweeps the spectator across [w_a, w_b] to show total estimated
error vs placement, i.e. where the quiet windows actually are.

CLI
---
    # audit one placement, ranked table + chart
    python spectator_audit.py --device evan_device.json --spec 4.0

    # scan the whole band and mark the quiet windows
    python spectator_audit.py --device evan_device.json --scan-points 241 \
        --out figs/spectator_audit.png --save-npz figs/spectator_audit.npz

    # table only, no figure (quick terminal audit)
    python spectator_audit.py --device evan_device.json --spec 4.0 --no-plot
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi
A, B, COUPLER, SPEC = 0, 1, 2, 3        # mode order, matching sweep_target.py


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
SPEC_MARKERS = ["o", "v", "P", "X", "h", "<"]      # one per spectator, cycled


def build_with_spectators(config: Dict[str, Any], w_specs_GHz: Sequence[float],
                          t_g_ns: float, *, coupler_levels: Optional[int] = None,
                          spec_levels: Optional[int] = None):
    """Same as :func:`build_with_spectator` but with an arbitrary number of spectators.

    Mode order is [a, b, coupler, spec_1, spec_2, ...] so the spectator indices are
    ``3, 4, ...``; each takes participation ``lam_b`` and ``anharm_spec_GHz``, matching
    the single-spectator convention in ``sweep_target.py``.

    Note the Hilbert dimension multiplies by ``spec_levels`` per spectator, and
    ``expand_terms`` slows accordingly -- with two or more spectators use
    ``spec_levels=2`` (enough to see |0> -> |1> excitation) unless you specifically
    need spectator |2> channels.

    Returns
    -------
    (ZhouCoupler, float, list of int)
        Coupler, peak |eta|, and the spectator mode indices.
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
    wa, wb = (float(f) for f in config["qubit_freqs_GHz"])
    ws = float(config["coupler_freq_GHz"])
    specs = [float(w) for w in w_specs_GHz]
    q_lv = int(config.get("qubit_levels", 3))
    c_lv = int(coupler_levels or config.get("coupler_levels", 5))
    s_lv = int(spec_levels or (3 if len(specs) <= 1 else 2))
    aq = float(config.get("anharm_qubit_GHz", 0.0))
    a_sp = float(config.get("anharm_spec_GHz", 0.0))
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])
    idxs = [SPEC + k for k in range(len(specs))]
    part = {A: float(config["lam_a"]), B: float(config["lam_b"])}
    part.update({i: float(config["lam_b"]) for i in idxs})
    anh = {A: aq, B: aq}
    anh.update({i: a_sp for i in idxs})
    cpl = ZhouCoupler(mode_freqs_GHz=[wa, wb, ws] + specs, coupler_index=COUPLER,
                      participations=part, nonlinearities=nonlin,
                      levels=[q_lv, q_lv, c_lv] + [s_lv] * len(specs),
                      anharmonicities_GHz=anh)
    cpl.set_pump(PumpTone(w_p_GHz=abs(wb - wa),
                          envelope=RaisedCosine(amp=1.0, t_g=float(t_g_ns)),
                          is_eta=True), normalize_iswap=(A, B))
    return cpl, float(cpl.peak_eta()), idxs


def mode_tags(cpl, spec_indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Per-mode identity for labelling: short tag, frequency (GHz) and role."""
    info: Dict[int, Dict[str, Any]] = {}
    for m in range(cpl.n_modes):
        f = float(cpl.omega[m]) / TWO_PI
        if m == A:
            info[m] = dict(tag="a", freq=f, role="qubit a (target)")
        elif m == B:
            info[m] = dict(tag="b", freq=f, role="qubit b (target)")
        elif m == COUPLER:
            info[m] = dict(tag="s", freq=f, role="SNAIL coupler")
        else:
            k = list(spec_indices).index(m) + 1 if m in spec_indices else 0
            info[m] = dict(tag=f"sp{k}", freq=f, role=f"spectator {k}")
    return info


def build_with_spectator(config: Dict[str, Any], w_spec_GHz: float, t_g_ns: float,
                         *, coupler_levels: Optional[int] = None,
                         spec_levels: Optional[int] = None):
    """4-mode [a, b, coupler, spec] system with the iSWAP pump set and normalized.

    Follows the sweep convention: mode order a=0, b=1, coupler=2, spectator=3, and
    the spectator participation equals ``lam_b`` (see ``sweep_target.py``).

    Parameters
    ----------
    config : dict
        Device configuration (``qubit_freqs_GHz``, ``coupler_freq_GHz``, ``g3_GHz``,
        ``lam_a``, ``lam_b``, levels, anharmonicities).
    w_spec_GHz : float
        Absolute spectator frequency (GHz).
    t_g_ns : float
        Gate duration, which fixes |eta| through the pi/2 normalization.
    coupler_levels, spec_levels : int, optional
        Truncation overrides.

    Returns
    -------
    (ZhouCoupler, float)
        The coupler and its peak |eta|.
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine
    wa, wb = (float(f) for f in config["qubit_freqs_GHz"])
    ws = float(config["coupler_freq_GHz"])
    q_lv = int(config.get("qubit_levels", 3))
    c_lv = int(coupler_levels or config.get("coupler_levels", 5))
    s_lv = int(spec_levels or config.get("qubit_levels", 3))
    aq = float(config.get("anharm_qubit_GHz", 0.0))
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])
    cpl = ZhouCoupler(
        mode_freqs_GHz=[wa, wb, ws, float(w_spec_GHz)], coupler_index=COUPLER,
        participations={A: float(config["lam_a"]), B: float(config["lam_b"]),
                        SPEC: float(config["lam_b"])},        # = lam_b, per sweeps
        nonlinearities=nonlin, levels=[q_lv, q_lv, c_lv, s_lv],
        anharmonicities_GHz={A: aq, B: aq,
                             SPEC: float(config.get("anharm_spec_GHz", 0.0))})
    cpl.set_pump(PumpTone(w_p_GHz=abs(wb - wa),
                          envelope=RaisedCosine(amp=1.0, t_g=float(t_g_ns)),
                          is_eta=True), normalize_iswap=(A, B))
    return cpl, float(cpl.peak_eta())


def t_g_for_eta(config: Dict[str, Any], eta: float) -> float:
    """Raised-cosine peak-|eta| gate time: t_g = 1 / (12 g3 la lb eta)."""
    return 1.0 / (12.0 * config["g3_GHz"] * config["lam_a"] * config["lam_b"]
                  * float(eta))


# --------------------------------------------------------------------------- #
# Channel enumeration                                                         #
# --------------------------------------------------------------------------- #
def _label(cpl, i_from: int, i_to: int, n_pump: int,
           info: Optional[Dict[int, Dict[str, Any]]] = None) -> str:
    """Compact transition label, e.g. 'a1->0 sp1 0->1 (2 pump)'."""
    if info is None:
        info = {A: dict(tag="a"), B: dict(tag="b"), COUPLER: dict(tag="s")}
        for m in range(cpl.n_modes):
            info.setdefault(m, dict(tag=f"sp{m - SPEC + 1}"))
    o_from, o_to = cpl.decode_index(i_from), cpl.decode_index(i_to)
    parts = [f"{info[m]['tag']}{o_from[m]}->{o_to[m]}"
             for m in range(cpl.n_modes) if o_from[m] != o_to[m]]
    return " ".join(parts) + (f" ({n_pump} pump)" if n_pump else " (static)")


def _process_name(cpl, i_from: int, i_to: int, n_pump: int,
                  info: Dict[int, Dict[str, Any]], is_target: bool = False) -> str:
    """Physics name for a process, e.g. 'SNAIL subharmonic', 'qubit A subharmonic',
    'A-SNAIL spectator', 'A-B |11>->|02> leakage'.

    The taxonomy follows what the process actually does:

    * ONE mode gains a quantum, driven by n pump photons -> a SUBHARMONIC of that
      mode (``n w_p = w_i``): n=2 is the usual one, n=1 is a direct drive.
    * TWO modes exchange a quantum (one up, one down) -> a SPECTATOR interaction
      between them, pump-assisted when n >= 1. The a<->b version at n=1 is the
      wanted gate, not a parasite.
    * A mode driven into |2> -> LEAKAGE, named by the partner it exchanged with.
    * Two modes both gaining -> PAIR CREATION.
    """
    o_f, o_t = cpl.decode_index(i_from), cpl.decode_index(i_to)
    delta = {m: o_t[m] - o_f[m] for m in range(cpl.n_modes) if o_t[m] != o_f[m]}
    ups = [m for m, d in delta.items() if d > 0]
    downs = [m for m, d in delta.items() if d < 0]

    def short(m: int) -> str:
        return info[m]["tag"].upper() if info[m]["tag"] != "s" else "SNAIL"

    def long(m: int) -> str:
        t = info[m]["tag"]
        if t == "s":
            return "SNAIL"
        if t in ("a", "b"):
            return f"qubit {t.upper()}"
        return f"spectator {t[2:]}" if t.startswith("sp") else t.upper()

    ORD = {0: "static", 1: "direct drive", 2: "subharmonic",
           3: "3rd subharmonic", 4: "4th subharmonic"}

    if is_target:
        return "target iSWAP (A-B exchange)"

    # one mode changes alone: a subharmonic of that mode (n w_p = w_i)
    if len(ups) == 1 and not downs:
        m = ups[0]
        lvl = "" if o_t[m] <= 1 else f"|{o_f[m]}>->|{o_t[m]}> "
        return f"{long(m)} {lvl}{ORD.get(n_pump, f'{n_pump}-pump')}"
    if len(downs) == 1 and not ups:
        m = downs[0]                          # conjugate branch of the same resonance
        return f"{long(m)} {ORD.get(n_pump, f'{n_pump}-pump')} (emission)"

    # one up, one down: an exchange between the two
    if len(ups) == 1 and len(downs) == 1:
        u, d = ups[0], downs[0]
        if o_t[u] >= 2:                       # driven into |2>: leakage
            return f"{short(u)} |2> leakage (via {short(d)})"
        if n_pump == 0:
            return f"{short(d)}-{short(u)} static exchange"
        if n_pump == 1:
            return f"{short(d)}-{short(u)} spectator"
        return f"{short(d)}-{short(u)} {n_pump}-pump spectator"

    if len(ups) == 2 and not downs:
        return f"{short(ups[0])}+{short(ups[1])} pair creation"
    if len(downs) == 2 and not ups:
        return f"{short(downs[0])}+{short(downs[1])} pair annihilation"
    return _label(cpl, i_from, i_to, n_pump, info)


def _describe(cpl, i_from: int, i_to: int, n_pump: int,
              info: Dict[int, Dict[str, Any]]) -> str:
    """Physical sentence for a process, naming each mode AND its frequency, e.g.
    '2w_p -> s@4.700 |0>->|1>'  or  'w_p: b@5.700 1->0, sp1@4.600 0->1'."""
    o_from, o_to = cpl.decode_index(i_from), cpl.decode_index(i_to)
    moved = [m for m in range(cpl.n_modes) if o_from[m] != o_to[m]]
    bits = [f"{info[m]['tag']}@{info[m]['freq']:.3f} "
            f"|{o_from[m]}>->|{o_to[m]}>" for m in moved]
    drive = ("static" if n_pump == 0 else
             ("w_p" if n_pump == 1 else f"{n_pump}w_p"))
    return f"{drive}: " + ", ".join(bits)


def spectator_channels(cpl, *, window_GHz: float = 1.0, min_g_MHz: float = 1e-3,
                       dedupe_MHz: float = 0.5) -> List[Dict[str, Any]]:
    """Every near-resonant process that EXCITES the spectator, with its strength.

    Walks ``expand_terms`` and, for each term, finds the matrix elements that take a
    low-lying computational state (qubits in {0,1}, coupler and spectator in |0>) to
    a state with the spectator excited. The element is weighted by ``|eta|^n_pump``,
    so a two-pump channel is correctly suppressed (or not) by the drive strength.

    Parameters
    ----------
    cpl : ZhouCoupler
        4-mode system from :func:`build_with_spectator`.
    window_GHz : float
        Keep channels detuned by less than this.
    min_g_MHz : float
        Discard channels weaker than this.
    dedupe_MHz : float
        Merge channels whose detunings agree to within this, keeping the strongest.

    Returns
    -------
    list of dict
        Sorted by descending ``ratio``; keys g_MHz, detuning_MHz, ratio, P_exc,
        n_pump, transition, perturbative.
    """
    eta = float(cpl.peak_eta())
    starts = []
    for na in range(min(2, cpl.dims[A])):
        for nb in range(min(2, cpl.dims[B])):
            occ = [0] * cpl.n_modes
            occ[A], occ[B] = na, nb
            starts.append(cpl.fock_index(occ))

    best: Dict[int, Dict[str, Any]] = {}
    for Omega, pump_sig, O in cpl.expand_terms(cutoff_GHz=window_GHz):
        det = Omega / TWO_PI                                   # signed, GHz
        n_pump = len(pump_sig)
        scale = eta ** n_pump
        Oa = np.asarray(O)
        for i in starts:
            col = Oa[:, i]
            for f in np.nonzero(np.abs(col) > 1e-12)[0]:
                if cpl.decode_index(f)[SPEC] <= cpl.decode_index(i)[SPEC]:
                    continue                                   # spectator not excited
                g = abs(col[f]) * scale / TWO_PI                # GHz
                if g * 1e3 < min_g_MHz:
                    continue
                # ON resonance the first-order ratio diverges: flag it rather than
                # printing a meaningless huge number. Resonant means full exchange,
                # so the excitation estimate saturates.
                resonant = abs(det) * 1e3 < 1e-3           # within 1 kHz
                ratio = np.inf if resonant else g / abs(det)
                key = int(round(det * 1e3 / max(dedupe_MHz, 1e-6)))
                rec = dict(g_MHz=g * 1e3, detuning_MHz=det * 1e3, ratio=ratio,
                           P_exc=1.0 if resonant else min(1.0, 2.0 * ratio ** 2),
                           n_pump=n_pump, transition=_label(cpl, i, f, n_pump),
                           perturbative=bool(not resonant and ratio < 1.0),
                           resonant=bool(resonant))
                prev = best.get(key)
                if prev is None or rec["ratio"] > prev["ratio"]:
                    best[key] = rec
    return sorted(best.values(), key=lambda r: -r["ratio"])


def drag_verdict(g_MHz: float, det_MHz: float, t_g_ns: float) -> Dict[str, Any]:
    r"""How well first-order DRAG can suppress a channel at (g, detuning).

    First-order DRAG (Motzoi et al. 2009) adds the quadrature
    :math:`-i\,\dot\eta/(2\pi\delta)`, which cancels the ADIABATIC (transient)
    excitation of a process detuned by :math:`\delta`. That transient has amplitude
    :math:`A_0 \simeq g/|\delta|`; once the leading term is cancelled what remains is
    the next order, :math:`\sim (g/|\delta|)^2`. So the DRAG suppression factor is
    itself :math:`\simeq g/|\delta|`:

        **DRAG works best exactly where you need it least** -- far-detuned, weakly
        coupled channels get suppressed hard, while near-resonant strong ones barely
        improve.

    Two hard failure modes:

    * :math:`g/|\delta| \geq 1` -- the channel is not perturbative at all, so there is
      no leading term to cancel. Frequency allocation, not pulse shaping.
    * :math:`|\delta| \lesssim 1/t_g` -- the detuning is inside the pulse's own
      spectral width, so the drive has real weight ON the transition; the excitation
      is not adiabatic and the DRAG quadrature (which goes as :math:`1/\delta`) is
      both huge and ineffective.

    Parameters
    ----------
    g_MHz : float
        Channel coupling matrix element (MHz).
    det_MHz : float
        Signed channel detuning (MHz); 0 means exactly resonant.
    t_g_ns : float
        Gate duration (ns), which sets the pulse bandwidth ``1/t_g``.

    Returns
    -------
    dict
        ratio (g/|det|), bandwidth_MHz, adiabaticity (|det| * t_g), suppression
        (post/pre amplitude ratio, <1 is good), suppression_dB, verdict, colour.
    """
    bw_MHz = 1e3 / float(t_g_ns)                    # pulse spectral width (MHz)
    ad = abs(float(det_MHz)) / bw_MHz               # |det| * t_g, adiabaticity
    ratio = (np.inf if abs(det_MHz) < 1e-9
             else abs(float(g_MHz)) / abs(float(det_MHz)))
    if not np.isfinite(ratio) or ratio >= 1.0:
        verdict, supp, colour = "fails: not perturbative", 1.0, "#b22222"
    elif ad <= 1.0:
        verdict, supp, colour = "fails: inside pulse bandwidth", 1.0, "#d95f02"
    elif ad <= 3.0:
        verdict, supp, colour = "marginal: barely adiabatic", min(1.0, ratio), "#e6ab02"
    elif ratio >= 0.3:
        verdict, supp, colour = "marginal: weak suppression", ratio, "#7570b3"
    else:
        verdict, supp, colour = "DRAG effective", ratio, "#1b7837"
    supp = float(max(supp, 1e-6))
    return dict(ratio=float(ratio), bandwidth_MHz=float(bw_MHz),
                adiabaticity=float(ad), suppression=supp,
                suppression_dB=float(-20.0 * np.log10(supp)),
                verdict=verdict, colour=colour)


def interaction_channels(cpl, *, window_GHz: float = 1.0, t_g_ns: float = 100.0,
                         min_g_MHz: float = 1e-3, dedupe_MHz: float = 0.5,
                         spec_indices: Optional[Sequence[int]] = None,
                         has_spectator: bool = True) -> List[Dict[str, Any]]:
    """EVERY near-resonant process, classified, with strength, detuning and DRAG verdict.

    Unlike :func:`spectator_channels` (which keeps only spectator excitation), this
    enumerates the full set so the desired gate and its parasites appear on one
    footing. Categories:

    ``target``     the wanted a<->b exchange, |01> <-> |10> (detuning ~ 0)
    ``spectator``  excites the spectator mode
    ``coupler``    excites the SNAIL coupler (heating)
    ``leakage``    drives a or b into |2>
    ``other``      remaining off-diagonal processes in the low-lying manifold

    Each channel also carries ``f_p_res_GHz``, the pump frequency at which it would
    become resonant: for an ``n``-pump process with detuning ``det``, that is
    ``f_p + det/n``. This is what puts the parasites on a common frequency axis with
    the gate -- the distance from ``f_p`` is how far the pump would have to move to
    land on that resonance. Static (0-pump) channels have no such mapping and get
    ``nan``; they are not pump-tunable.
    """
    eta = float(cpl.peak_eta())
    f_p = float(cpl._pump_tones[0].w_p_GHz)
    n_modes = cpl.n_modes
    if spec_indices is None:
        spec_indices = ([SPEC] if (has_spectator and n_modes > SPEC) else [])
    spec_indices = list(spec_indices)
    info = mode_tags(cpl, spec_indices)
    # expand_terms carries the HARMONIC carrier only: the transmon anharmonicity is a
    # separate static diagonal operator. A transition i->f therefore sits at
    #     Omega_total = Omega + (E_anh[f] - E_anh[i]),
    # which is what shifts every |2>-involving channel by the anharmonicity (e.g.
    # |11> -> |02> is resonant in the harmonic expansion but really detuned by alpha).
    E_anh = np.real(np.diag(np.asarray(cpl._anharm_op)))

    starts = []
    for na in range(min(2, cpl.dims[A])):
        for nb in range(min(2, cpl.dims[B])):
            occ = [0] * n_modes
            occ[A], occ[B] = na, nb
            starts.append(cpl.fock_index(occ))

    best: Dict[Tuple, Dict[str, Any]] = {}
    for Omega, pump_sig, O in cpl.expand_terms(cutoff_GHz=window_GHz + 0.5):
        n_pump = len(pump_sig)
        scale = eta ** n_pump
        Oa = np.asarray(O)
        for i in starts:
            oi = cpl.decode_index(i)
            col = Oa[:, i]
            for f in np.nonzero(np.abs(col) > 1e-12)[0]:
                if f == i:
                    continue                                    # diagonal -> Stark shift
                of = cpl.decode_index(f)
                det = (Omega + (E_anh[f] - E_anh[i])) / TWO_PI   # signed GHz, anharm-shifted
                if abs(det) > window_GHz:
                    continue
                g = abs(col[f]) * scale / TWO_PI                 # GHz
                if g * 1e3 < min_g_MHz:
                    continue
                # --- classify, and record WHICH mode is the victim
                excited_spec = [m for m in spec_indices if of[m] > oi[m]]
                victim = None
                if excited_spec:
                    cat, victim = "spectator", excited_spec[0]
                elif of[COUPLER] > oi[COUPLER]:
                    cat, victim = "coupler", COUPLER
                elif max(of[A], of[B]) >= 2 and max(of[A], of[B]) > max(oi[A], oi[B]):
                    cat = "leakage"
                    victim = A if of[A] >= 2 else B
                elif ((oi[A], oi[B]) in ((0, 1), (1, 0))
                      and (of[A], of[B]) == (oi[B], oi[A])
                      and of[COUPLER] == 0
                      and all(of[m] == 0 for m in spec_indices)):
                    cat = "target"          # the wanted |01> <-> |10> exchange
                else:
                    cat = "other"
                spec_id = (spec_indices.index(victim) + 1
                           if victim is not None and victim in spec_indices else 0)
                rec = dict(g_MHz=g * 1e3, detuning_MHz=det * 1e3, n_pump=n_pump,
                           category=cat, transition=_label(cpl, i, f, n_pump, info),
                           name=_process_name(cpl, i, f, n_pump, info,
                                              is_target=(cat == "target")),
                           process=_describe(cpl, i, f, n_pump, info),
                           victim=(-1 if victim is None else int(victim)),
                           victim_tag=("--" if victim is None else info[victim]["tag"]),
                           victim_freq_GHz=(np.nan if victim is None
                                            else info[victim]["freq"]),
                           spectator_id=int(spec_id),
                           f_p_res_GHz=(f_p + det / n_pump if n_pump else np.nan))
                if cat == "target":
                    rec.update(drag_verdict(rec["g_MHz"], rec["detuning_MHz"], t_g_ns))
                    rec.update(verdict="resonant by design (the gate)",
                               colour="#111111", suppression=1.0, suppression_dB=0.0)
                else:
                    rec.update(drag_verdict(rec["g_MHz"], rec["detuning_MHz"], t_g_ns))
                key = (cat, rec["victim"],
                       int(round(det * 1e3 / max(dedupe_MHz, 1e-6))))
                prev = best.get(key)
                if prev is None or rec["g_MHz"] > prev["g_MHz"]:
                    best[key] = rec
    out = list(best.values())
    # target first, then strongest parasites by pre-DRAG excitation amplitude
    out.sort(key=lambda r: (r["category"] != "target",
                            -(r["g_MHz"] / max(abs(r["detuning_MHz"]), 1e-6))))
    return out


CATEGORY_STYLE = {
    "target":    dict(colour="#111111", marker="*", label="target iSWAP  $a\\!\\leftrightarrow\\!b$"),
    "spectator": dict(colour="#d62728", marker="o", label="spectator excitation"),
    "coupler":   dict(colour="#2ca02c", marker="s", label="coupler heating"),
    "leakage":   dict(colour="#9467bd", marker="^", label=r"leakage to $|2\rangle$"),
    "other":     dict(colour="#7f7f7f", marker="d", label="other"),
}


def print_interaction_table(channels: Sequence[Dict[str, Any]], f_p_GHz: float,
                            t_g_ns: float, eta: float, top: int = 20) -> None:
    """Ranked interaction table with the DRAG verdict for each channel."""
    bw = 1e3 / t_g_ns
    print(f"\npump f_p = {f_p_GHz:.4f} GHz,  t_g = {t_g_ns:.1f} ns  ->  pulse bandwidth "
          f"1/t_g = {bw:.1f} MHz,  peak |eta| = {eta:.3f}")
    print(f"{len(channels)} near-resonant process(es); DRAG suppression factor ~ g/|det| "
          f"(smaller is better)\n")
    print(f"  {'interaction':<34} {'g(MHz)':>8} {'det(MHz)':>10} "
          f"{'f_p^res':>9} {'g/|det|':>8} {'DRAG':>8}  verdict")
    for c in channels[:int(top)]:
        rs = "  inf" if not np.isfinite(c["ratio"]) else f"{c['ratio']:8.3f}"
        fp = "   --   " if not np.isfinite(c["f_p_res_GHz"]) else f"{c['f_p_res_GHz']:9.4f}"
        db = ("  0.0" if c["suppression"] >= 1.0 else f"{c['suppression_dB']:5.1f}")
        print(f"  {c['name']:<34} {c['g_MHz']:8.3f} "
              f"{c['detuning_MHz']:+10.1f} {fp} {rs} {db:>6}dB  {c['verdict']}")
        print(f"  {'':<34} -> {c['process']}")
    if len(channels) > top:
        print(f"  ... {len(channels) - top} weaker process(es) not shown")


def _style_for(c: Dict[str, Any]) -> Dict[str, Any]:
    """Colour+marker for a channel: category sets the colour, and each SPECIFIC
    spectator gets its own marker so they are distinguishable on the chart."""
    st = dict(CATEGORY_STYLE[c["category"]])
    if c["category"] == "spectator" and c.get("spectator_id", 0) > 0:
        st["marker"] = SPEC_MARKERS[(c["spectator_id"] - 1) % len(SPEC_MARKERS)]
    return st


def plot_interaction_chart(config: Dict[str, Any], channels: Sequence[Dict[str, Any]],
                           out: str, *, f_p_GHz: float, t_g_ns: float, eta: float,
                           w_spec_GHz: Optional[float] = None,
                           mode_info: Optional[Dict[int, Dict[str, Any]]] = None,
                           max_key: int = 14, device_name: str = "") -> None:
    r"""Two views of the interaction landscape, with a numbered key.

    Channels are NUMBERED on both panels (the target is ``T``); the names, strengths
    and DRAG verdicts live in the key panel on the right, together with the mode
    inventory. Keeping the plot area free of text is what makes a crowded landscape
    readable -- several processes routinely sit at nearly the same
    :math:`(g, |\delta|)`.

    LEFT TOP -- pump-frequency axis. The gate sits at :math:`f_p`; every pump-tunable
    process is a stem at the pump frequency where it becomes resonant,
    :math:`f_p + \delta/n`, with height = coupling :math:`g`. Horizontal distance from
    :math:`f_p` is how far the pump would have to move to hit that resonance, and the
    shaded strip is the pulse bandwidth :math:`1/t_g`.

    LEFT BOTTOM -- the DRAG phase diagram, :math:`g` vs :math:`|\delta|`. The diagonal
    :math:`g=|\delta|` is the perturbative boundary and the vertical line the pulse
    bandwidth; iso-suppression diagonals mark 10/20/30 dB.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bw = 1e3 / float(t_g_ns)                                  # MHz
    # ---- assign numbers: T for the gate, 1..N for parasites by excitation amplitude
    para = [c for c in channels if c["category"] != "target"]
    para = sorted(para, key=lambda r: -(r["g_MHz"] / max(abs(r["detuning_MHz"]), 1e-6)))
    tgt = [c for c in channels if c["category"] == "target"]
    numbers: Dict[int, str] = {id(c): "T" for c in tgt}
    for k, c in enumerate(para, start=1):
        numbers[id(c)] = str(k)
    tunable = [c for c in channels if np.isfinite(c["f_p_res_GHz"])]

    fig = plt.figure(figsize=(14.6, 9.2), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.52], height_ratios=[1.0, 1.05],
                          wspace=0.04, hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    axk = fig.add_subplot(gs[:, 1]); axk.axis("off")

    def place_numbers(axis, items, min_sep_px: float = 15.0):
        """Draw the number badges, choosing an offset for each so they do not collide.

        Channels are routinely degenerate or near-degenerate (the two |2> leakage
        paths share a detuning; the two branches of a subharmonic sit at +/- the same
        offset), so a fixed offset would hide badges under one another. Work in
        DISPLAY pixels and take the first candidate slot that clears every badge
        already placed."""
        fig_ = axis.figure
        fig_.canvas.draw()                     # transforms must be current
        cands = [(0, 9), (0, -17), (14, 5), (-14, 5), (14, -13), (-14, -13),
                 (0, 22), (0, -30), (24, 0), (-24, 0), (24, 14), (-24, 14)]
        placed: List[Tuple[float, float]] = []
        for c, x, y in items:
            px, py = axis.transData.transform((x, y))
            best = cands[0]
            for dx, dy in cands:
                q = (px + dx, py + dy)
                if all((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 > min_sep_px ** 2
                       for p in placed):
                    best = (dx, dy)
                    break
            placed.append((px + best[0], py + best[1]))
            axis.annotate(numbers[id(c)], xy=(x, y), xytext=best,
                          textcoords="offset points", ha="center", va="center",
                          fontsize=7.4, fontweight="bold", color="#222222", zorder=8,
                          bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                                    ec=_style_for(c)["colour"], lw=0.9, alpha=0.95))

    # ---------------- TOP: pump-frequency axis --------------------------------
    gmax = max([c["g_MHz"] for c in channels] + [1.0])
    gmin = min([c["g_MHz"] for c in channels] + [1.0])
    ax.axvspan(f_p_GHz - bw * 1e-3, f_p_GHz + bw * 1e-3, color="#f0c9c9", alpha=0.85,
               zorder=1, label=f"pulse bandwidth $1/t_g$ = {bw:.0f} MHz")
    ax.axvline(f_p_GHz, color="#111111", lw=2.0, zorder=4)
    for c in tunable:
        st = _style_for(c)
        ax.plot([c["f_p_res_GHz"]] * 2, [1e-6, c["g_MHz"]], color=st["colour"],
                lw=1.6, alpha=0.85, zorder=3)
        ax.plot([c["f_p_res_GHz"]], [c["g_MHz"]], marker=st["marker"],
                ms=12 if c["category"] == "target" else 7.5,
                color=st["colour"], mec="white", mew=0.8, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(max(1e-4, 0.3 * gmin), gmax * 12)
    ax.set_xlabel(r"pump frequency at which the process is resonant,"
                  r"  $f_p + \delta/n_{\rm pump}$  (GHz)")
    ax.set_ylabel("coupling $g$ (MHz)")
    ax.text(f_p_GHz, gmax * 6.5, f"  operating pump $f_p$ = {f_p_GHz:.4f} GHz",
            fontsize=8.5, ha="left", va="center")
    hs = []
    for k in ("target", "coupler", "leakage", "other"):
        if any(c["category"] == k for c in channels):
            hs.append(Line2D([], [], color=CATEGORY_STYLE[k]["colour"],
                             marker=CATEGORY_STYLE[k]["marker"], ls="none",
                             label=CATEGORY_STYLE[k]["label"]))
    seen_sp = {}
    for c in channels:
        if c["category"] == "spectator" and c.get("spectator_id", 0) > 0:
            seen_sp.setdefault(c["spectator_id"], (c["victim_tag"], c["victim_freq_GHz"]))
    for sid in sorted(seen_sp):
        tag, fq = seen_sp[sid]
        hs.append(Line2D([], [], color=CATEGORY_STYLE["spectator"]["colour"],
                         marker=SPEC_MARKERS[(sid - 1) % len(SPEC_MARKERS)], ls="none",
                         label=f"{tag} @ {fq:.3f} GHz"))
    ax.legend(handles=hs, fontsize=7.4, loc="lower left", ncol=3, framealpha=0.93)
    place_numbers(ax, sorted([(c, c["f_p_res_GHz"], c["g_MHz"]) for c in tunable],
                             key=lambda t: -t[2]))
    ttl = f"Interaction landscape — {device_name}" if device_name else "Interaction landscape"
    ax.set_title(f"{ttl}\n$t_g$={t_g_ns:.1f} ns,  peak $|\\eta|$={eta:.2f}",
                 fontsize=10.5, pad=8)
    ax.grid(alpha=0.22, which="both")

    # ---------------- BOTTOM: DRAG phase diagram ------------------------------
    if para:
        dets = np.array([max(abs(c["detuning_MHz"]), 1e-3) for c in para])
        gs_ = np.array([c["g_MHz"] for c in para])
        lo_d = max(1e-2, min(dets.min() / 4, bw * 0.5))
        hi_d = dets.max() * 4
        lo_g, hi_g = max(1e-4, gs_.min() / 4), gs_.max() * 4
        dd = np.logspace(np.log10(lo_d), np.log10(hi_d), 200)
        ax2.fill_between(dd, dd, hi_g, color="#b22222", alpha=0.13, zorder=0)
        ax2.axvspan(lo_d, min(bw, hi_d), color="#d95f02", alpha=0.13, zorder=0)
        ax2.plot(dd, dd, color="#b22222", lw=1.4,
                 label=r"$g=|\delta|$  (perturbation theory fails)")
        for r, lab in ((0.3, "10 dB"), (0.1, "20 dB"), (0.03, "30 dB")):
            ax2.plot(dd, r * dd, color="#1b7837", lw=0.9, ls="--", alpha=0.8)
            x_top = hi_g / r
            if x_top <= hi_d:
                xa, ya, va, ha = x_top, hi_g, "top", "right"
            else:
                xa, ya, va, ha = hi_d, r * hi_d, "bottom", "right"
            ax2.annotate(f"{lab}", xy=(xa, ya),
                         xytext=(-3, -3 if va == "top" else 3),
                         textcoords="offset points", ha=ha, va=va,
                         fontsize=7.0, color="#1b7837", clip_on=False)
        ax2.axvline(bw, color="#d95f02", lw=1.4,
                    label=f"pulse bandwidth $1/t_g$ = {bw:.0f} MHz")
        for c in para:
            st = _style_for(c)
            xv = max(abs(c["detuning_MHz"]), 1e-3)
            ax2.plot([xv], [c["g_MHz"]], marker=st["marker"], ms=9, color=st["colour"],
                     mec="white", mew=0.8, zorder=5)
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlim(lo_d, hi_d); ax2.set_ylim(lo_g, hi_g)
        ax2.text(0.018, 0.965, "DRAG cannot help\n($g\\geq|\\delta|$: reallocate)",
                 transform=ax2.transAxes, ha="left", va="top", fontsize=8,
                 color="#8b1a1a",
                 bbox=dict(boxstyle="round", fc="white", ec="#b22222", alpha=0.9))
        ax2.text(0.982, 0.055, "DRAG effective\n(suppression $\\sim g/|\\delta|$)",
                 transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
                 color="#1b7837",
                 bbox=dict(boxstyle="round", fc="white", ec="#1b7837", alpha=0.9))
        ax2.set_xlabel(r"channel detuning $|\delta|$ (MHz)   —   the DRAG beat")
        ax2.set_ylabel("coupling $g$ (MHz)")
        ax2.set_title("Which parasites DRAG can suppress", fontsize=10)
        ax2.legend(fontsize=7.4, loc="lower left", framealpha=0.93)
        ax2.grid(alpha=0.22, which="both")
        place_numbers(ax2, sorted([(c, max(abs(c["detuning_MHz"]), 1e-3), c["g_MHz"])
                                   for c in para], key=lambda t: -t[2]))
    else:
        ax2.text(0.5, 0.5, "no parasitic channel inside the window",
                 transform=ax2.transAxes, ha="center", va="center")
        ax2.set_xticks([]); ax2.set_yticks([])

    # ---------------- RIGHT: mode inventory + numbered interaction key ---------
    y = 0.995
    axk.text(0.0, y, "MODE INVENTORY", fontsize=8.6, fontweight="bold",
             family="monospace", va="top")
    y -= 0.030
    if mode_info:
        for m in sorted(mode_info):
            mi = mode_info[m]
            axk.text(0.0, y, f"  {mi['tag']:<5}{mi['freq']:8.3f} GHz   {mi['role']}",
                     fontsize=7.6, family="monospace", va="top")
            y -= 0.024
    axk.text(0.0, y, f"  {'pump':<5}{f_p_GHz:8.3f} GHz   "
                     f"2w_p = {2 * f_p_GHz:.3f} GHz", fontsize=7.6,
             family="monospace", va="top")
    y -= 0.024
    axk.text(0.0, y, f"  {'1/t_g':<5}{bw:8.1f} MHz   pulse bandwidth",
             fontsize=7.6, family="monospace", va="top")

    y -= 0.045
    axk.text(0.0, y, "INTERACTIONS", fontsize=8.6, fontweight="bold",
             family="monospace", va="top")
    y -= 0.028
    axk.text(0.0, y, f"  {'#':<3}{'process':<30}{'g(MHz)':>8}{'det(MHz)':>10}{'DRAG':>8}",
             fontsize=7.3, family="monospace", va="top", color="#444444")
    y -= 0.023
    axk.plot([0.0, 1.0], [y + 0.010, y + 0.010], color="#bbbbbb", lw=0.8,
             transform=axk.transAxes, clip_on=False)
    for c in (tgt + para)[:int(max_key)]:
        nm = c["name"] if len(c["name"]) <= 30 else c["name"][:29] + "…"
        db = ("  --  " if c["suppression"] >= 1.0 else f"{c['suppression_dB']:5.1f}dB")
        axk.text(0.0, y,
                 f"  {numbers[id(c)]:<3}{nm:<30}{c['g_MHz']:8.2f}"
                 f"{c['detuning_MHz']:+10.1f}{db:>8}",
                 fontsize=7.3, family="monospace", va="top",
                 color=_style_for(c)["colour"])
        y -= 0.0235
    if len(channels) > max_key:
        axk.text(0.0, y, f"  ... {len(channels) - max_key} weaker process(es)",
                 fontsize=7.0, family="monospace", va="top", color="#777777")
        y -= 0.0235
    y -= 0.020
    axk.text(0.0, y, "DRAG column = suppression of that channel,\n"
                     "  ~ g/|det| (blank = cannot help).",
             fontsize=7.0, family="monospace", va="top", color="#555555")
    axk.set_xlim(0, 1); axk.set_ylim(0, 1)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def total_error_estimate(channels: Sequence[Dict[str, Any]]) -> float:
    """Summed excitation estimate over channels, capped at 1 (a rough error budget)."""
    return float(min(1.0, sum(c["P_exc"] for c in channels)))


def scan_band(config: Dict[str, Any], t_g_ns: float, *, n_points: int = 161,
              window_GHz: float = 1.0, pad_GHz: float = 0.0,
              coupler_levels: Optional[int] = None,
              verbose: bool = True) -> Dict[str, Any]:
    """Sweep the spectator across the physical band and record its parasitic load.

    A spectator qubit lives between the computational qubits, so the band is
    [w_a, w_b] (optionally padded). For each placement the worst channel ratio and
    the summed excitation estimate are recorded -- the quiet windows are the minima.

    Returns
    -------
    dict
        w_spec_GHz, worst_ratio, total_P_exc, worst_g_MHz, worst_det_MHz,
        n_nonperturbative, plus t_g_ns / eta / band metadata.
    """
    wa, wb = sorted(float(f) for f in config["qubit_freqs_GHz"])
    lo, hi = wa - float(pad_GHz), wb + float(pad_GHz)
    grid = np.linspace(lo, hi, int(n_points))
    worst_ratio = np.zeros(grid.size)
    total = np.zeros(grid.size)
    worst_g = np.zeros(grid.size)
    worst_det = np.zeros(grid.size)
    n_bad = np.zeros(grid.size, dtype=int)
    eta_nom = np.nan
    for k, w in enumerate(grid):
        cpl, eta_nom = build_with_spectator(config, float(w), t_g_ns,
                                            coupler_levels=coupler_levels)
        ch = spectator_channels(cpl, window_GHz=window_GHz)
        if ch:
            worst_ratio[k] = ch[0]["ratio"]
            worst_g[k] = ch[0]["g_MHz"]
            worst_det[k] = ch[0]["detuning_MHz"]
            n_bad[k] = sum(1 for c in ch if not c["perturbative"])
        total[k] = total_error_estimate(ch)
        if verbose and (k + 1) % max(1, grid.size // 8) == 0:
            print(f"  scanned {k + 1}/{grid.size}  w_spec={w:.3f} GHz  "
                  f"worst ratio={worst_ratio[k]:.2f}")
    return dict(w_spec_GHz=grid, worst_ratio=worst_ratio, total_P_exc=total,
                worst_g_MHz=worst_g, worst_det_MHz=worst_det,
                n_nonperturbative=n_bad, t_g_ns=float(t_g_ns), eta=float(eta_nom),
                band_GHz=(wa, wb))


def quiet_windows(scan: Dict[str, Any], threshold: float = 1e-3
                  ) -> List[Tuple[float, float]]:
    """Contiguous spectator placements whose total excitation estimate is below
    ``threshold`` -- the intervals where a spectator can actually live."""
    w, tot = scan["w_spec_GHz"], scan["total_P_exc"]
    ok = tot < float(threshold)
    out, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = w[i]
        elif not good and start is not None:
            out.append((float(start), float(w[i - 1])))
            start = None
    if start is not None:
        out.append((float(start), float(w[-1])))
    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def print_table(channels: Sequence[Dict[str, Any]], w_spec_GHz: float,
                eta: float, top: int = 12) -> None:
    """Ranked channel table for one spectator placement."""
    print(f"\nspectator at {w_spec_GHz:.4f} GHz, peak |eta| = {eta:.3f}: "
          f"{len(channels)} channel(s) found")
    if not channels:
        print("  (no spectator-exciting channel inside the window)")
        return
    print(f"  {'g (MHz)':>9} {'detuning':>10} {'g/|det|':>8} {'P_exc':>9}  transition")
    for c in channels[:int(top)]:
        if c.get("resonant"):
            flag, ratio_s = "   <-- ON RESONANCE", "     inf"
        elif not c["perturbative"]:
            flag, ratio_s = "   <-- NOT perturbative", f"{c['ratio']:8.3f}"
        else:
            flag, ratio_s = "", f"{c['ratio']:8.3f}"
        print(f"  {c['g_MHz']:9.3f} {c['detuning_MHz']:+10.2f} {ratio_s} "
              f"{c['P_exc']:9.2e}  {c['transition']}{flag}")
    if len(channels) > top:
        print(f"  ... {len(channels) - top} weaker channel(s) not shown")
    print(f"  total excitation estimate: {total_error_estimate(channels):.3e}")


def plot_chart(config: Dict[str, Any], scan: Optional[Dict[str, Any]],
               channels: Optional[Sequence[Dict[str, Any]]], out: str,
               *, w_spec_GHz: Optional[float] = None,
               threshold: float = 1e-3, device_name: str = "") -> None:
    """Frequency chart: mode layout + strength-weighted channels, over a band scan.

    Top panel is the spectral layout in the style of ``plot_allocation.py`` (modes,
    pump, spectator band, curated collision centres) with the MEASURED channel
    strengths overlaid as stems at the audited placement. Bottom panel is the band
    scan: total excitation estimate vs spectator frequency, shaded where a channel
    is non-perturbative, with the quiet windows marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wa, wb = sorted(float(f) for f in config["qubit_freqs_GHz"])
    ws = float(config["coupler_freq_GHz"])
    w_p = abs(wb - wa)

    # curated centres, reused from the existing allocation tool when importable
    try:
        from plot_allocation import allocation_frequencies
        alloc = allocation_frequencies(config)
        families = alloc["families"]
    except Exception:                      # keep the tool standalone if it moves
        families = [
            {"key": "direct", "color": "#555555", "label": "direct",
             "centers": [wa, ws, wb]},
            {"key": "subharm", "color": "#e9a000", "label": "subharmonic",
             "centers": [2.0 * w_p]}]

    n_rows = 2 if scan is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(11.0, 4.2 * n_rows), dpi=200,
                             sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 1.15][:n_rows]))
    ax = axes[0] if n_rows > 1 else axes

    # ---- top: spectral layout -------------------------------------------
    ax.set_ylim(0.0, 1.0)
    ax.axhspan(0.0, 1.0, xmin=0, xmax=0, color="none")          # keep limits
    ax.axvspan(wa, wb, color="#dfe9f5", alpha=0.7, zorder=0)
    ax.text(0.5 * (wa + wb), 0.055,
            r"physical spectator band  $[\omega_a,\omega_b]$",
            ha="center", va="bottom", fontsize=8.5, color="#2c4a72")
    # mode lines, labelled INSIDE the axes so nothing collides with the title
    for w, lab, col in ((wa, r"$\omega_a$", "#d62728"),
                        (wb, r"$\omega_b$", "#1f77b4"),
                        (ws, r"$\omega_s$ (SNAIL)", "#2ca02c")):
        ax.axvline(w, color=col, lw=2.2, zorder=3)
        ax.text(w, 0.985, f" {lab} {w:.3f} ", ha="center", va="top", fontsize=8.5,
                color=col, rotation=90, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # the pump's 2nd harmonic is the channel that bites hardest here -- label it
    if wa - 0.05 <= 2.0 * w_p <= wb + 0.05:
        ax.axvline(2.0 * w_p, color="#e9a000", lw=1.8, ls="--", zorder=3)
        ax.text(2.0 * w_p, 0.985, f" $2\\omega_p$ {2*w_p:.3f} ", ha="center", va="top",
                fontsize=8.5, color="#b37700", rotation=90, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    for fam in families:                   # curated collision centres, in band
        for c in fam["centers"]:
            if wa - 0.05 <= c <= wb + 0.05 and abs(c - 2.0 * w_p) > 1e-9:
                ax.axvline(c, color=fam["color"], lw=1.0, ls=":", alpha=0.75, zorder=2)

    # measured channel strengths at the audited placement, as a readable table
    if channels:
        base = float(w_spec_GHz if w_spec_GHz is not None else 0.5 * (wa + wb))
        ax.axvline(base, color="#000000", lw=1.6, ls="--", zorder=4)
        r0 = float(channels[0]["ratio"])
        head = (r"audited $\omega_{\rm spec}$ = " + f"{base:.3f} GHz    "
                + ("worst channel: ON RESONANCE" if not np.isfinite(r0)
                   else f"worst $g/|\\delta|$ = {r0:.3f}"))
        lines = [head, ""]
        lines.append(f"{'g (MHz)':>8}  {'det (MHz)':>10}  {'g/|det|':>8}   transition")
        for c in channels[:5]:
            rs = "inf" if not np.isfinite(c["ratio"]) else f"{c['ratio']:.3f}"
            lines.append(f"{c['g_MHz']:8.3f}  {c['detuning_MHz']:+10.1f}  {rs:>8}   "
                         f"{c['transition']}")
        tot = total_error_estimate(channels)
        lines.append("")
        lines.append(f"total excitation estimate: {tot:.2e}")
        side = "left" if base > 0.5 * (wa + wb) else "right"
        xa = 0.015 if side == "left" else 0.985
        ax.text(xa, 0.62, "\n".join(lines), transform=ax.transAxes,
                ha=side, va="top", fontsize=7.6, family="monospace", zorder=7,
                bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.94))
    ax.set_yticks([])
    ax.set_ylabel("spectral layout")
    ttl = f"Spectator audit — {device_name}" if device_name else "Spectator audit"
    sub = (f"$\\omega_p$={w_p:.3f} GHz,  $2\\omega_p$={2*w_p:.3f} GHz"
           + (f",  $t_g$={scan['t_g_ns']:.1f} ns,  peak $|\\eta|$={scan['eta']:.2f}"
              if scan else ""))
    ax.set_title(f"{ttl}\n{sub}", fontsize=10.5, pad=10)
    handles = [Line2D([], [], color="#e9a000", lw=1.8, ls="--",
                      label=r"$2\omega_p$ (pump 2nd harmonic)")]
    handles += [Line2D([], [], color=f["color"], lw=1.0, ls=":", label=f["label"])
                for f in families]
    handles += [Line2D([], [], color="k", lw=1.6, ls="--", label="audited placement")]
    ax.legend(handles=handles, fontsize=7.2, loc="lower left", ncol=2, framealpha=0.92)

    # ---- bottom: band scan ----------------------------------------------
    if scan is not None:
        ax2 = axes[1]
        w, tot, bad = scan["w_spec_GHz"], scan["total_P_exc"], scan["n_nonperturbative"]
        ax2.semilogy(w, np.maximum(tot, 1e-12), color="#1f3b73", lw=1.8,
                     label="total excitation estimate")
        ax2.axhline(threshold, color="#888888", ls="--", lw=1.0,
                    label=f"threshold {threshold:g}")
        # shade non-perturbative placements
        inbad = bad > 0
        if inbad.any():
            ax2.fill_between(w, 1e-12, 1.0, where=inbad, color="#b22222", alpha=0.16,
                             step="mid", label="a channel is non-perturbative")
        for lo_w, hi_w in quiet_windows(scan, threshold):
            if hi_w > lo_w:
                ax2.axvspan(lo_w, hi_w, color="#2ca02c", alpha=0.18, zorder=0)
        qw = quiet_windows(scan, threshold)
        if qw:
            ax2.text(0.005, 0.04,
                     "quiet windows (GHz): "
                     + ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in qw[:4])
                     + (" ..." if len(qw) > 4 else ""),
                     transform=ax2.transAxes, fontsize=8, color="#1a6b1a",
                     bbox=dict(boxstyle="round", fc="white", ec="#2ca02c", alpha=0.85))
        for wv, col in ((wa, "#d62728"), (wb, "#1f77b4"), (ws, "#2ca02c")):
            ax2.axvline(wv, color=col, lw=1.6, alpha=0.8)
        if w_spec_GHz is not None:
            ax2.axvline(float(w_spec_GHz), color="k", lw=1.4, ls="--")
        finite = tot[np.isfinite(tot) & (tot > 0)]
        floor = max(1e-12, (finite.min() / 5.0) if finite.size else 1e-9)
        ax2.set_ylim(min(floor, 0.5 * float(threshold)), 1.6)
        ax2.set_ylabel("estimated spectator excitation")
        ax2.set_xlabel(r"spectator frequency $\omega_{\rm spec}$ (GHz)")
        ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax2.grid(alpha=0.25, which="both")
    else:
        ax.set_xlabel(r"frequency (GHz)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def save_npz(path: str, scan: Optional[Dict[str, Any]],
             channels: Optional[Sequence[Dict[str, Any]]],
             meta: Dict[str, Any],
             interactions: Optional[Sequence[Dict[str, Any]]] = None) -> None:
    """Persist the scan arrays and the audited placement's channel table."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {f"meta_{k}": v for k, v in meta.items()}
    if scan is not None:
        for k in ("w_spec_GHz", "worst_ratio", "total_P_exc", "worst_g_MHz",
                  "worst_det_MHz", "n_nonperturbative"):
            payload[k] = scan[k]
        payload["t_g_ns"] = scan["t_g_ns"]
        payload["eta"] = scan["eta"]
    if channels:
        for key, dtype in (("g_MHz", float), ("detuning_MHz", float),
                           ("ratio", float), ("P_exc", float), ("n_pump", int)):
            payload[f"ch_{key}"] = np.array([c[key] for c in channels], dtype=dtype)
        payload["ch_transition"] = np.array([c["transition"] for c in channels])
    if interactions:
        for key, dtype in (("g_MHz", float), ("detuning_MHz", float), ("ratio", float),
                           ("f_p_res_GHz", float), ("adiabaticity", float),
                           ("suppression", float), ("suppression_dB", float),
                           ("n_pump", int)):
            payload[f"int_{key}"] = np.array([c[key] for c in interactions], dtype=dtype)
        payload["int_category"] = np.array([c["category"] for c in interactions])
        payload["int_victim_tag"] = np.array([c["victim_tag"] for c in interactions])
        payload["int_victim_freq_GHz"] = np.array(
            [c["victim_freq_GHz"] for c in interactions], dtype=float)
        payload["int_spectator_id"] = np.array(
            [c["spectator_id"] for c in interactions], dtype=int)
        payload["int_process"] = np.array([c["process"] for c in interactions])
        payload["int_name"] = np.array([c["name"] for c in interactions])
        payload["int_transition"] = np.array([c["transition"] for c in interactions])
        payload["int_verdict"] = np.array([c["verdict"] for c in interactions])
    np.savez_compressed(path, **payload)
    print("saved", path)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True, help="device JSON")
    ap.add_argument("--spec", default=None,
                    help="ABSOLUTE spectator frequency in GHz, or a comma-separated list "
                         "for several spectators (e.g. 4.6,5.2). Each becomes its own "
                         "mode, labelled sp1, sp2, ... on the chart")
    ap.add_argument("--spec-detuning", default=None,
                    help="spectator(s) by the sweep-convention detuning "
                         "Delta = w_b - w_spec (GHz); comma-separated for several")
    ap.add_argument("--spec-levels", type=int, default=None,
                    help="spectator truncation (default 3 for one spectator, 2 for "
                         "several -- enough to see |0>->|1> excitation)")
    ap.add_argument("--eta", type=float, default=None,
                    help="operating point as peak |eta| (sets t_g); default 1.0")
    ap.add_argument("--t-g-ns", type=float, default=None,
                    help="explicit gate duration (ns); overrides --eta")
    ap.add_argument("--scan-points", type=int, default=161,
                    help="band-scan resolution (0 disables the scan)")
    ap.add_argument("--window-GHz", type=float, default=1.0,
                    help="only report channels detuned by less than this")
    ap.add_argument("--pad-GHz", type=float, default=0.0,
                    help="extend the scan this far beyond [w_a, w_b]")
    ap.add_argument("--threshold", type=float, default=1e-3,
                    help="quiet-window cut on the total excitation estimate")
    ap.add_argument("--coupler-levels", type=int, default=None,
                    help="override coupler truncation (raise at strong drive)")
    ap.add_argument("--top", type=int, default=12, help="table rows to print")
    ap.add_argument("--out", default="figs/spectator_audit.png")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--chart", choices=["interactions", "placement", "both"],
                    default="interactions",
                    help="'interactions' (default): the gate and every parasitic process "
                         "on a pump-frequency axis, weighted by coupling, plus the DRAG "
                         "phase diagram. 'placement': total excitation vs where the "
                         "SPECTATOR sits (the band scan). 'both' writes both figures.")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    try:                                   # use the pipeline's resolver when present
        from paths import resolve_device
        device_path = resolve_device(args.device)
    except Exception:
        device_path = args.device
    with open(device_path) as f:
        config = json.load(f)

    t_g = (float(args.t_g_ns) if args.t_g_ns
           else t_g_for_eta(config, args.eta if args.eta else 1.0))
    wa, wb = sorted(float(x) for x in config["qubit_freqs_GHz"])
    print(f"{os.path.basename(device_path)}: w_a={wa} w_b={wb} "
          f"w_s={config['coupler_freq_GHz']} GHz, w_p={abs(wb - wa):.3f} GHz")
    _c, eta_nom = build_with_spectator(config, 0.5 * (wa + wb), t_g,
                                       coupler_levels=args.coupler_levels)
    print(f"t_g = {t_g:.2f} ns  ->  peak |eta| = {eta_nom:.3f}")

    scan = None
    need_scan = (args.chart in ("placement", "both")) and int(args.scan_points) > 0
    if need_scan:
        print(f"scanning the band [{wa - args.pad_GHz:.3f}, {wb + args.pad_GHz:.3f}] "
              f"GHz at {args.scan_points} points ...")
        scan = scan_band(config, t_g, n_points=int(args.scan_points),
                         window_GHz=args.window_GHz, pad_GHz=args.pad_GHz,
                         coupler_levels=args.coupler_levels)
        qw = quiet_windows(scan, args.threshold)
        print(f"\nquiet windows (total excitation < {args.threshold:g}):")
        if qw:
            for lo_w, hi_w in qw:
                print(f"   [{lo_w:.4f}, {hi_w:.4f}] GHz   "
                      f"(width {1e3 * (hi_w - lo_w):.0f} MHz)")
        else:
            print("   NONE -- every placement in the band exceeds the threshold.")
        frac = float(np.mean(scan["n_nonperturbative"] > 0))
        print(f"non-perturbative (g >= detuning) at {100 * frac:.0f}% of placements")

    # which placement(s) to audit in detail
    def _parse_list(txt):
        return [float(x) for x in str(txt).replace(" ", "").split(",") if x]

    w_specs: List[float] = []
    if args.spec is not None:
        w_specs = _parse_list(args.spec)
    elif args.spec_detuning is not None:
        wb_ref = float(config["qubit_freqs_GHz"][1])
        w_specs = [wb_ref - d for d in _parse_list(args.spec_detuning)]
    w_spec = w_specs[0] if w_specs else None
    if w_spec is None and scan is not None:
        w_spec = float(scan["w_spec_GHz"][int(np.argmax(scan["total_P_exc"]))])
        w_specs = [w_spec]
        print(f"\n(no --spec given; auditing the WORST placement found)")
    channels = None
    if w_spec is not None:
        cpl, eta_nom = build_with_spectator(config, float(w_spec), t_g,
                                            coupler_levels=args.coupler_levels)
        channels = spectator_channels(cpl, window_GHz=args.window_GHz)
        print_table(channels, float(w_spec), eta_nom, top=args.top)

    # ---- interaction landscape (the gate + its parasites on a frequency axis) ----
    inter = None
    if args.chart in ("interactions", "both"):
        w_list = w_specs if w_specs else [0.5 * (wa + wb)]
        cpl_i, eta_i, sp_idx = build_with_spectators(
            config, w_list, t_g, coupler_levels=args.coupler_levels,
            spec_levels=args.spec_levels)
        info = mode_tags(cpl_i, sp_idx)
        print("\nmode inventory:")
        for m in sorted(info):
            print(f"   {info[m]['tag']:<5} {info[m]['freq']:8.4f} GHz   {info[m]['role']}")
        inter = interaction_channels(cpl_i, window_GHz=args.window_GHz, t_g_ns=t_g,
                                     spec_indices=sp_idx)
        print_interaction_table(inter, float(cpl_i._pump_tones[0].w_p_GHz), t_g, eta_i,
                                top=args.top)
        if not args.no_plot:
            out_i = (args.out if args.chart == "interactions"
                     else args.out.rsplit(".", 1)[0] + "_interactions.png")
            plot_interaction_chart(config, inter, out_i,
                                   f_p_GHz=float(cpl_i._pump_tones[0].w_p_GHz),
                                   t_g_ns=t_g, eta=eta_i,
                                   w_spec_GHz=(w_list[0] if len(w_list) == 1 else None),
                                   mode_info=info,
                                   device_name=os.path.basename(device_path))

    if args.save_npz:
        save_npz(args.save_npz, scan, channels,
                 dict(device=os.path.basename(device_path), t_g_ns=t_g,
                      eta=eta_nom, w_spec_GHz=(np.nan if w_spec is None else w_spec)),
                 interactions=inter)
    if not args.no_plot and args.chart in ("placement", "both"):
        out_p = (args.out if args.chart == "placement"
                 else args.out.rsplit(".", 1)[0] + "_placement.png")
        plot_chart(config, scan, channels, out_p, w_spec_GHz=w_spec,
                   threshold=args.threshold,
                   device_name=os.path.basename(device_path))


if __name__ == "__main__":
    main()