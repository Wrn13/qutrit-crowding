#!/usr/bin/env python3
"""Frequency-allocation diagram for the SNAIL parametric-coupler system.

Draws the spectral layout on a single frequency axis: the two computational
qubits and the SNAIL coupler as labelled stripes, the parametric pump
w_p = |w_b - w_a| as a separate drive stripe, the PHYSICAL spectator band
[w_a, w_b] (a spectator qubit lives between the two computational qubits), and
the collision zones inside that band that a spectator must avoid.

Collision physics (why the danger frequencies are where they are)
------------------------------------------------------------------
A spectator at w_spec is dangerous when it is resonant with an active process:

* direct exchange with a mode:                w_spec = w_a, w_b, or w_s
* pump-assisted (one-pump) sideband:          |w_q - w_spec| = w_p  for q in {a,b,s}

Because the iSWAP pump is the qubit-qubit difference, w_p = |w_b - w_a|, the
pump-assisted sidebands of the qubits fall exactly on the OTHER qubit:
    w_b - w_p = w_a           and           w_a + w_p = w_b .
So inside the band the collision frequencies are w_a, the coupler w_s, and w_b;
a spectator wants to sit in the gaps between them. This also means the
"spectator resonance" seen in the sweeps (beat = Delta - w_p -> 0) is the point
where the spectator becomes degenerate with qubit a, i.e. the lower band edge.

Note on the sweep axis
----------------------
run_sweep_zhou's ``spec_freq_GHz`` is the DETUNING Delta = w_b - w_spec, not an
absolute frequency; the absolute placement is w_spec = w_b - Delta. This tool
works in ABSOLUTE frequency (GHz) so the layout is unambiguous.

CLI
---
    python plot_allocation.py --device warren_device.json --out figs/allocation.png
    python plot_allocation.py --device warren_device.json --spec 4.0        # mark a placement
    python plot_allocation.py --device warren_device.json --sweep-detuning 0.0,2.2   # show a swept band
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def allocation_frequencies(config: Dict[str, Any],
                           margin_MHz: float = 50.0) -> Dict[str, Any]:
    """Compute the mode frequencies, pump, spectator band and collision zones.

    Parameters
    ----------
    config : dict
        Device configuration; needs ``qubit_freqs_GHz`` (2 entries) and
        ``coupler_freq_GHz``. ``g3_GHz``, ``lam_a``, ``lam_b`` are used only to
        report a representative iSWAP coupling.
    margin_MHz : float, default 50.0
        Half-width (MHz) of the shaded collision zone around each collision
        centre. A few tens of MHz (order 10x the effective coupling) captures
        the region where the gate degrades in the sweeps.

    Returns
    -------
    dict
        w_a_GHz, w_b_GHz, w_s_GHz, w_p_GHz, band_GHz=(lo, hi),
        collisions=list of (centre_GHz, label), margin_GHz, g_iswap_MHz.
    """
    wa, wb = sorted(float(f) for f in config["qubit_freqs_GHz"])   # wa < wb
    ws = float(config["coupler_freq_GHz"])
    w_p = abs(wb - wa)
    margin = float(margin_MHz) / 1e3

    # representative effective iSWAP coupling (GHz): g_eff = 6 g3 lam_a lam_b |eta|
    g3 = float(config.get("g3_GHz", 0.0))
    la = float(config.get("lam_a", 0.0)); lb = float(config.get("lam_b", 0.0))
    g_iswap_MHz = 6.0 * g3 * la * lb * 1e3

    # Spectator interaction channels. A spectator at w_spec is resonant with an
    # active process at these frequencies (the ones to keep it away from):
    #   direct exchange with a mode:        w_spec = w_a, w_s, w_b
    #   pump-assisted swap (one g3 pump):   |w_q - w_spec| = w_p  -> w_spec = w_q +/- w_p
    #   subharmonic (pump's 2nd harmonic):  w_p = w_i/2  for every mode i
    # Because the iSWAP pump is the qubit difference w_p = w_b - w_a, the one-pump
    # sidebands fall on the OTHER qubit (w_a + w_p = w_b, w_b - w_p = w_a). Each mode i
    # also has a subharmonic resonance at the pump frequency w_i/2 (the pump's second
    # harmonic drives mode i); those are marked at w_a/2, w_s/2, w_b/2, and the in-band
    # point 2 w_p is where the CURRENT pump's second harmonic would drive any mode.
    families = [
        {"key": "direct", "color": "#555555",
         "label": r"direct  $w_{\rm spec}=w_q$",
         "centers": [wa, ws, wb]},
        {"key": "a_spec", "color": "#d62728",
         "label": r"$a\!\leftrightarrow\!$spec  $|w_{\rm spec}-w_a|=w_p$",
         "centers": [wa - w_p, wa + w_p]},
        {"key": "b_spec", "color": "#1f77b4",
         "label": r"$b\!\leftrightarrow\!$spec  $|w_{\rm spec}-w_b|=w_p$",
         "centers": [wb - w_p, wb + w_p]},
        {"key": "subharm", "color": "#e9a000",
         "label": r"subharmonic  $w_p=w_i/2$",
         "centers": [0.5 * wa, 0.5 * ws, 0.5 * wb, 2.0 * w_p]},
    ]

    # union of all IN-BAND collision centres (for the forbidden-zone shading), deduped
    lo, hi = wa - 0.02, wb + 0.02
    merged: List[Tuple[float, str]] = []
    for fam in families:
        for c in fam["centers"]:
            if lo <= c <= hi:
                merged.append((round(c, 6), fam["key"]))
    collisions: List[Tuple[float, str]] = []
    for c, key in sorted(merged):
        if collisions and abs(c - collisions[-1][0]) < 1e-6:
            if key not in collisions[-1][1]:
                collisions[-1] = (collisions[-1][0], collisions[-1][1] + "+" + key)
        else:
            collisions.append((c, key))

    return {"w_a_GHz": wa, "w_b_GHz": wb, "w_s_GHz": ws, "w_p_GHz": w_p,
            "band_GHz": (wa, wb), "collisions": collisions, "families": families,
            "margin_GHz": margin, "g_iswap_MHz": g_iswap_MHz}


def _safe_windows(band: Tuple[float, float],
                  collisions: List[Tuple[float, float]],
                  margin: float, min_width: float = 0.0) -> List[Tuple[float, float]]:
    """Sub-intervals of the band left free once each collision +/- margin is removed.

    Windows narrower than ``min_width`` (GHz) are dropped -- a gap thinner than the
    collision margin is not usefully collision-free.
    """
    lo, hi = band
    blocked = sorted((c - margin, c + margin) for c, _ in collisions)
    free: List[Tuple[float, float]] = []
    cursor = lo
    for b0, b1 in blocked:
        if b0 > cursor:
            free.append((cursor, min(b0, hi)))
        cursor = max(cursor, b1)
        if cursor >= hi:
            break
    if cursor < hi:
        free.append((cursor, hi))
    return [(a, b) for a, b in free if (b - a) > max(min_width, 1e-6)]


def in_band(config: Dict[str, Any], w_spec_GHz: float) -> bool:
    """True if an absolute spectator frequency lies within the physical band [w_a, w_b]."""
    wa, wb = sorted(float(f) for f in config["qubit_freqs_GHz"])
    return wa <= float(w_spec_GHz) <= wb


def plot_allocation(config: Dict[str, Any], png_path: str,
                    spec_GHz: Optional[float] = None,
                    sweep_wspec_GHz: Optional[Tuple[float, float]] = None,
                    margin_MHz: float = 50.0,
                    f_min: Optional[float] = None,
                    f_max: Optional[float] = 6.0,
                    title: Optional[str] = None) -> Dict[str, Any]:
    """Render the frequency-allocation diagram to ``png_path``.

    Parameters
    ----------
    config : dict
        Device configuration (see ``allocation_frequencies``).
    png_path : str
        Output PNG path (parent directory is created).
    spec_GHz : float, optional
        A specific ABSOLUTE spectator frequency to mark; annotated green if it
        falls in a safe window, red if inside a collision zone or out of band.
    sweep_wspec_GHz : (float, float), optional
        ABSOLUTE spectator range covered by a sweep, drawn as a bar so out-of-band
        excursions are obvious.
    margin_MHz : float, default 50.0
        Collision-zone half-width (MHz).
    title : str, optional
        Figure title.

    Returns
    -------
    dict
        The ``allocation_frequencies`` result (frequencies + collision zones),
        for programmatic use.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle, FancyBboxPatch, Circle
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    from matplotlib.legend_handler import HandlerBase
    try:
        from plot_results import set_literature_style
        set_literature_style()
    except Exception:
        pass

    class _GradHandler(HandlerBase):
        """Legend handler that draws a horizontal colour gradient swatch (c0 -> c1)
        as a row of thin rectangles, so drive entries show their blended colours."""
        def __init__(self, c0: str, c1: str, n: int = 24) -> None:
            super().__init__()
            self._c0, self._c1, self._n = to_rgb(c0), to_rgb(c1), n
        def create_artists(self, legend, orig, xd, yd, w, h, fs, trans):
            out = []
            for i in range(self._n):
                t = i / (self._n - 1)
                col = tuple((1 - t) * a + t * b for a, b in zip(self._c0, self._c1))
                out.append(Rectangle((xd + w * i / self._n, yd), w / self._n + 0.6, h,
                                     facecolor=col, edgecolor="none", transform=trans))
            return out

    info = allocation_frequencies(config, margin_MHz)
    wa, wb, ws, w_p = info["w_a_GHz"], info["w_b_GHz"], info["w_s_GHz"], info["w_p_GHz"]
    band, collisions, margin = info["band_GHz"], info["collisions"], info["margin_GHz"]

    # Distinct colour per mode; each DRIVE is painted as a vertical gradient blending
    # the two modes it couples, so "which qubit" is unambiguous: the pump blends a->b,
    # the one-pump swaps blend qubit->spectator, the subharmonic blends pump->spectator.
    A, B = "#2a9d8f", "#e76f51"           # qubit a (teal), qubit b (coral)
    SPEC, COUPLER, GOLD = "#7b2cbf", "#6c757d", "#e9a000"
    # paint: ("solid", color) or ("grad", bottom_color, top_color)
    CATS: Dict[str, Dict[str, Any]] = {
        "qubit_a": {"paint": ("solid", A),          "label": r"qubit $a$"},
        "qubit_b": {"paint": ("solid", B),          "label": r"qubit $b$"},
        "coupler": {"paint": ("solid", COUPLER),    "label": r"SNAIL $w_s$"},
        "pump":    {"paint": ("grad", A, B),        "label": r"pump $w_p$ ($a\!\to\!b$)"},
        "a_spec":  {"paint": ("grad", A, SPEC),     "label": r"$a\!\leftrightarrow\!$spec ($a\!\to\!$spec)"},
        "b_spec":  {"paint": ("grad", B, SPEC),     "label": r"$b\!\leftrightarrow\!$spec ($b\!\to\!$spec)"},
        "subharm": {"paint": ("solid", GOLD),        "label": r"subharmonic ($w_p\!=\!w_i/2$)"},
    }
    ORDER = ["qubit_a", "qubit_b", "coupler", "pump", "a_spec", "b_spec", "subharm"]

    # frequency -> categories resonant there. Qubit degeneracies are split by qubit so
    # a-vs-b is distinguishable; one-pump sidebands and the subharmonics are explicit.
    freqmap: Dict[float, set] = {}
    def _add(f: float, cat: str) -> None:
        freqmap.setdefault(round(float(f), 6), set()).add(cat)
    _add(wa, "qubit_a"); _add(wb, "qubit_b"); _add(ws, "coupler"); _add(w_p, "pump")
    for c in (wa - w_p, wa + w_p):
        _add(c, "a_spec")
    for c in (wb - w_p, wb + w_p):
        _add(c, "b_spec")
    # a subharmonic at every w_p = w_i/2: mark w_i/2 for each mode (pump 2nd harmonic
    # drives mode i), plus 2 w_p (in-band, where the CURRENT pump drives any mode).
    sub_marks = [(0.5 * wa, r"$w_a/2$"), (0.5 * ws, r"$w_s/2$"),
                 (0.5 * wb, r"$w_b/2$"), (2.0 * w_p, r"$2w_p$")]
    if spec_GHz is not None:
        sub_marks.append((0.5 * float(spec_GHz), r"$w_{\rm spec}/2$"))
    sub_labels: Dict[float, str] = {}
    for f, lab in sub_marks:
        _add(f, "subharm"); sub_labels[round(float(f), 6)] = lab

    x_lo = min(w_p, wa, min(freqmap)) - 0.3
    x_hi = wb + 0.3
    if f_min is not None:
        x_lo = float(f_min)
    if f_max is not None:
        x_hi = float(f_max)

    fig, (icon_ax, ax) = plt.subplots(
        1, 2, figsize=(11.5, 2.9), gridspec_kw={"width_ratios": [1, 15], "wspace": 0.02})

    # ---- device schematic on the left (qubit a - SNAIL - qubit b), colour-matched ----
    icon_ax.set_xlim(0, 1); icon_ax.set_ylim(0, 1); icon_ax.axis("off")
    icon_ax.add_patch(FancyBboxPatch((0.05, 0.30), 0.90, 0.42,
                                     boxstyle="round,pad=0.02,rounding_size=0.10",
                                     fc="#f4f4f4", ec="0.5", lw=1.2))
    icon_ax.plot([0.24, 0.76], [0.51, 0.51], color="#9a9a9a", lw=1.4, zorder=1)
    for xx, cc, lab in [(0.26, A, "a"), (0.50, COUPLER, "s"), (0.74, B, "b")]:
        icon_ax.add_patch(Circle((xx, 0.51), 0.11, fc=cc, ec="k", lw=0.7, zorder=2))
        icon_ax.annotate(lab, (xx, 0.51), ha="center", va="center", color="white",
                         fontsize=7.5, zorder=3)
    icon_ax.annotate("qubit\u2013SNAIL\u2013qubit", (0.5, 0.22), ha="center", va="top",
                     fontsize=6.3, color="#555")

    # ---- spectral bar (all vertical coordinates in DATA units) ----
    y0, y1 = 0.10, 0.74                                   # stripe vertical extent
    hw = 0.006 * (x_hi - x_lo)                            # stripe half-width (data units)
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(0.0, 1.18)

    def _shade(a0: float, a1: float, color: str, alpha: float) -> None:
        if a1 > a0:
            ax.add_patch(Rectangle((a0, y0), a1 - a0, y1 - y0, fc=color, ec="none",
                                   alpha=alpha, zorder=0))
    _shade(band[0], band[1], "#2ca02c", 0.06)
    for (s0, s1) in _safe_windows(band, collisions, margin, min_width=margin):
        _shade(s0, s1, "#2ca02c", 0.13)
    for c, _k in collisions:
        _shade(c - margin, c + margin, "#d62728", 0.13)

    def _rep_color(paint) -> str:
        return paint[1]                                   # solid colour, or gradient's bottom end

    def _paint_stripe(x: float, ylo: float, yhi: float, paint) -> None:
        if paint[0] == "solid":
            ax.plot([x, x], [ylo, yhi], color=paint[1], lw=3.4, solid_capstyle="butt",
                    zorder=4, clip_on=False)
        else:                                             # ("grad", c_bottom, c_top)
            cmap = LinearSegmentedColormap.from_list("", [paint[1], paint[2]])
            ax.imshow(np.linspace(0, 1, 128).reshape(-1, 1),
                      extent=[x - hw, x + hw, ylo, yhi], origin="lower", cmap=cmap,
                      aspect="auto", zorder=4, interpolation="bilinear")

    # stacked stripes at each frequency (coincident channels split the height); a
    # gradient drive shows the two modes it couples. Off-scale -> edge arrows.
    ymid = 0.5 * (y0 + y1)
    for f, cats in sorted(freqmap.items()):
        cs = [c for c in ORDER if c in cats]
        if f < x_lo or f > x_hi:
            edge = x_lo if f < x_lo else x_hi
            dx = 0.06 * (x_hi - x_lo) * (1 if f < x_lo else -1)
            for j, cat in enumerate(cs):
                ax.annotate("", xy=(edge, ymid - 0.10 + 0.14 * j),
                            xytext=(edge - dx, ymid - 0.10 + 0.14 * j),
                            arrowprops=dict(arrowstyle="->",
                                            color=_rep_color(CATS[cat]["paint"]), lw=1.4))
            ax.annotate(f"{f:.2f}", (edge - dx, ymid + 0.10 + 0.14 * (len(cs) - 1)),
                        ha="center", va="bottom", fontsize=6.0, color="0.35")
            continue
        dy = (y1 - y0) / len(cs)
        for i, cat in enumerate(cs):
            _paint_stripe(f, y0 + i * dy, y0 + (i + 1) * dy, CATS[cat]["paint"])

    # name labels above the mode / pump stripes; frequency below every visible stripe
    for f, name, col in [(wa, r"$a$", A), (wb, r"$b$", B),
                         (ws, r"SNAIL", COUPLER), (w_p, r"pump", "0.35")]:
        if x_lo <= f <= x_hi:
            ax.annotate(name, (f, y1 + 0.03), ha="center", va="bottom", fontsize=8, color=col)
    for f in sorted(freqmap):
        if x_lo <= f <= x_hi:
            ax.annotate(f"{f:.2f}", (f, y0 - 0.03), ha="center", va="top",
                        fontsize=6.4, color="0.35", rotation=90)
    # subharmonic marks: label which mode each w_i/2 (or 2w_p) drives, in gold
    for f, lab in sub_labels.items():
        if x_lo <= f <= x_hi:
            ax.annotate(lab, (f, y1 + 0.02), ha="center", va="bottom", fontsize=6.4,
                        color="#b07800")

    # optional spectator placement + swept range, above the bar
    if spec_GHz is not None:
        wsp = float(spec_GHz)
        safe = in_band(config, wsp) and all(abs(wsp - c) > margin for c, _ in collisions)
        col = SPEC if safe else "#d62728"
        ax.annotate("", xy=(wsp, y1 + 0.02), xytext=(wsp, y1 + 0.15),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6))
        tag = "OK" if safe else ("out of band" if not in_band(config, wsp) else "collision")
        ax.annotate(rf"spectator {wsp:.2f} ({tag})", (wsp, y1 + 0.16),
                    ha="center", va="bottom", fontsize=7.2, color=col)
    if sweep_wspec_GHz is not None:
        s0, s1 = sorted(float(v) for v in sweep_wspec_GHz)
        for a0, a1, col in ((s0, min(s1, band[0]), "#d62728"),
                            (max(s0, band[0]), min(s1, band[1]), "#333333"),
                            (max(s0, band[1]), s1, "#d62728")):
            if a1 > a0:
                ax.plot([a0, a1], [1.08, 1.08], color=col, lw=4.0, solid_capstyle="butt",
                        zorder=6, clip_on=False)
        ax.annotate("swept range", (0.5 * (s0 + s1), 1.10), ha="center", va="bottom",
                    fontsize=6.8)

    ax.add_patch(Rectangle((x_lo, y0), x_hi - x_lo, y1 - y0, fill=False, ec="0.35",
                           lw=1.0, zorder=5))
    ax.set_yticks([])
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(0.0, 1.18)       # re-assert (imshow can nudge limits)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("frequency (GHz)")
    ax.set_title(title or "SNAIL coupler frequency allocation  "
                 rf"($w_p={w_p:.2f}$ GHz, zone $\pm${margin_MHz:.0f} MHz)", pad=6)

    # legend: solid swatches for modes, gradient swatches for the drives
    handles, labels, hmap = [], [], {}
    for c in ORDER:
        paint = CATS[c]["paint"]
        if paint[0] == "solid":
            h = Line2D([0], [0], color=paint[1], lw=3.4)
        else:
            h = Patch()
            hmap[h] = _GradHandler(paint[1], paint[2])
        handles.append(h); labels.append(CATS[c]["label"])
    handles += [Patch(facecolor="#2ca02c", alpha=0.14), Patch(facecolor="#d62728", alpha=0.14)]
    labels += ["safe window", "collision zone"]
    ax.legend(handles, labels, handler_map=hmap, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=5, fontsize=6.9, frameon=False,
              columnspacing=1.1, handlelength=1.7)

    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return info


def _load_config(device: Optional[str]) -> Dict[str, Any]:
    """Load DEFAULT_CONFIG overlaid with a device JSON (bare name -> devices/)."""
    from run_sweep_zhou import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    if device:
        import json
        from paths import resolve_device
        with open(resolve_device(device)) as f:
            config.update(json.load(f))
    return config


def main() -> None:
    ap = argparse.ArgumentParser(description="Frequency-allocation diagram")
    ap.add_argument("--device", help="device JSON (bare name resolves under devices/)")
    ap.add_argument("--out", default="figs/allocation.png", help="output PNG")
    ap.add_argument("--spec", type=float, default=None,
                    help="mark an ABSOLUTE spectator frequency (GHz)")
    ap.add_argument("--spec-detuning", type=float, default=None,
                    help="mark a spectator by DETUNING Delta=w_b-w_spec (GHz); "
                         "converted to absolute w_spec = w_b - Delta")
    ap.add_argument("--sweep-detuning", default=None,
                    help="comma pair 'lo,hi' of spectator DETUNINGS Delta to show as a "
                         "swept band (absolute w_spec = w_b - Delta)")
    ap.add_argument("--margin-MHz", type=float, default=50.0,
                    help="collision-zone half-width (MHz, default 50)")
    ap.add_argument("--f-min", type=float, default=None,
                    help="lower frequency limit (GHz); default auto")
    ap.add_argument("--f-max", type=float, default=6.0,
                    help="upper frequency limit (GHz); default 6.0. Channels beyond the "
                         "limits are shown as edge arrows.")
    args = ap.parse_args()

    config = _load_config(args.device)
    wb = max(float(f) for f in config["qubit_freqs_GHz"])

    spec = args.spec
    if spec is None and args.spec_detuning is not None:
        spec = wb - float(args.spec_detuning)

    sweep = None
    if args.sweep_detuning:
        d0, d1 = (float(v) for v in args.sweep_detuning.split(","))
        sweep = (wb - d0, wb - d1)                # detuning -> absolute w_spec

    info = plot_allocation(config, args.out, spec_GHz=spec,
                           sweep_wspec_GHz=sweep, margin_MHz=args.margin_MHz,
                           f_min=args.f_min, f_max=args.f_max)
    wa, wb = info["w_a_GHz"], info["w_b_GHz"]
    print(f"w_a={wa:.3f}  w_b={wb:.3f}  w_s={info['w_s_GHz']:.3f}  "
          f"w_p={info['w_p_GHz']:.3f} GHz   g_iSWAP~{info['g_iswap_MHz']:.1f} MHz")
    print(f"physical spectator band: [{wa:.3f}, {wb:.3f}] GHz")
    for fam in info["families"]:
        inb = [f"{c:.3f}" for c in sorted(set(round(x, 6) for x in fam["centers"]))
               if wa <= c <= wb]
        allc = [f"{c:.3f}" for c in sorted(set(round(x, 6) for x in fam["centers"]))]
        print(f"  {fam['key']:8s}: all {', '.join(allc)}  |  in-band {', '.join(inb) or '(none)'}")
    margin = info["margin_GHz"]
    windows = _safe_windows(info["band_GHz"], info["collisions"], margin, min_width=margin)
    print("safe windows (GHz): " + (", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in windows)
                                     or "(none wider than the collision margin)"))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
