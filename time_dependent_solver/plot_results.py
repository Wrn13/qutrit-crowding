#!/usr/bin/env python3
r"""
plot_results.py
===============

Read a sweep result directory (``summary.csv`` + ``combined.npz`` produced by
``run_sweep_zhou.py`` or ``run_sweep.py``) and render publication-style figures
in the idiom of the parametric-gate / frequency-collision literature:

  * Fig 1  infidelity 1 - F vs spectator frequency, log-y, DRAG off vs on
           (the canonical DRAG / collision figure).
  * Fig 2  leakage and spectator / coupler occupation vs spectator frequency.
  * Fig 3  2D collision map: log10(1 - F) over (spectator frequency, participation)
           -- the frequency-allocation heatmap (cf. McKinney et al.,
           arXiv:2409.18262).
  * Fig 4  analytic collision predictor: the Eq.-62 exchange rate and the
           "danger ratio" g_spec / |beat|, which is available even for an
           analytic-only sweep (``--no-integrate``).

The module auto-detects what is present: a full sweep yields Figs 1-4, an
analytic-only sweep yields Fig 4. It also adapts to either schema (the Zhou
driver's ``lam_spec`` or the effective-model driver's ``eta`` as the series
variable). Styling is self-contained (no external style files); math is set in
Computer-Modern via mathtext, so no system LaTeX is required (pass ``--usetex``
to switch to a real TeX install if you have one).

Usage
-----
    python plot_results.py --outdir results_zhou/ --figdir results_zhou/figs/
    python plot_results.py --outdir results_zhou/ --figdir figs/ --usetex
    # one figure only:
    python plot_results.py --outdir results_zhou/ --only heatmap --metric leakage
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Wong colorblind-safe palette (Nature Methods 8, 441 (2011)) -- common in
# current superconducting-qubit papers.
WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#F0E442", "#000000"]

# columns that should be parsed as floats (others stay strings / bools)
_FLOAT_COLS = {
    "index", "spec_freq_GHz", "lam_spec", "eta", "beat_GHz", "eta_peak",
    "g_iswap_eff_MHz", "g_spec_eff_MHz", "F_avg", "leakage", "n_spec",
    "n_coupler", "p_transfer", "w_p_GHz", "w_free_GHz", "w_spec_GHz",
    "t_g_ns", "wall_s", "nearest_mult",
}
_BOOL_COLS = {"drag", "drag_applied"}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def set_literature_style(usetex: bool = False) -> None:
    """Apply tight, paper-style Matplotlib rcParams: inward minor ticks on all
    spines, Computer-Modern math, hairline axes, constrained layout.

    Parameters
    ----------
    usetex : bool, default False
        If True, render text with a system LaTeX install (requires TeX +
        amsmath); otherwise mathtext is used and no LaTeX is needed.

    Returns
    -------
    None
    """
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 600, "savefig.bbox": "tight",
        "font.size": 9, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "axes.titlepad": 4.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "legend.handlelength": 1.8, "legend.labelspacing": 0.3,
        "lines.linewidth": 1.4, "lines.markersize": 4.0,
        "lines.markeredgewidth": 1.0,
        "figure.constrained_layout.use": True,
    })
    if usetex:
        mpl.rcParams.update({"text.usetex": True,
                             "text.latex.preamble": r"\usepackage{amsmath}"})


def _panel_label(ax, text: str) -> None:
    ax.text(0.04, 0.93, text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_summary(outdir: str) -> Dict[str, np.ndarray]:
    """Parse ``<outdir>/summary.csv`` into column arrays.

    Parameters
    ----------
    outdir : str
        Sweep directory containing summary.csv.

    Returns
    -------
    dict[str, ndarray]
        One array per column. Numeric columns are floats with empty cells -> NaN
        (so analytic-only rows coexist with full rows), boolean columns are bool,
        and the rest are object (string) arrays.
    """
    path = os.path.join(outdir, "summary.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    cols = rows[0].keys()
    data: Dict[str, np.ndarray] = {}
    for c in cols:
        raw = [r[c] for r in rows]
        if c in _FLOAT_COLS:
            data[c] = np.array([float(x) if x not in ("", None) else np.nan
                                for x in raw], dtype=float)
        elif c in _BOOL_COLS:
            data[c] = np.array([str(x).strip().lower() in ("1", "true", "t", "yes")
                                for x in raw], dtype=bool)
        else:
            data[c] = np.array([str(x) for x in raw], dtype=object)
    # The Zhou sweep no longer varies a spectator "channel"; if that column is
    # absent, synthesize a single-value stand-in so the per-channel faceting
    # collapses to one panel instead of raising KeyError.
    if "channel" not in data:
        n = len(next(iter(data.values())))
        data["channel"] = np.array(["spectator"] * n, dtype=object)
    return data


def series_column(data: Dict[str, np.ndarray]) -> str:
    """Return the name of the swept series variable.

    Parameters
    ----------
    data : dict[str, ndarray]
        Loaded summary columns.

    Returns
    -------
    str
        'lam_spec' (Zhou driver) or 'eta' (effective-model driver).
    """
    if "lam_spec" in data:
        return "lam_spec"
    if "eta" in data:
        return "eta"
    raise KeyError("no series column ('lam_spec' or 'eta') in summary.csv")


# x-axis for the vs-frequency figures: absolute spectator frequency (default) or
# the beat detuning delta = Delta - w_p (0 = on the collision). Set by main().
_XAXIS = "spec_freq_GHz"
_XLABELS = {
    "spec_freq_GHz": r"$\Delta = w_b - w_\mathrm{spec}$ (GHz)",
    "beat_GHz": r"Detuning $\delta = \Delta - w_p$ (GHz)",
}


def _xcol(data: Dict[str, np.ndarray]) -> str:
    """Chosen x column, falling back to spec_freq_GHz when beat is unavailable."""
    if _XAXIS == "beat_GHz" and "beat_GHz" in data:
        return "beat_GHz"
    return "spec_freq_GHz"


def _mark_collision(ax, xcol: str) -> None:
    """On a beat axis, draw delta=0 (the collision, where DRAG cannot help)."""
    if xcol == "beat_GHz":
        ax.axvline(0.0, color="0.5", lw=0.8, ls=":", zorder=0)


def has_full_metrics(data: Dict[str, np.ndarray]) -> bool:
    """Whether the sweep contains integrated results.

    Parameters
    ----------
    data : dict[str, ndarray]
        Loaded summary columns.

    Returns
    -------
    bool
        True if an ``F_avg`` column with at least one finite value is present.
    """
    return "F_avg" in data and np.isfinite(data["F_avg"]).any()


def _series_label(col: str) -> str:
    return r"$\lambda_\mathrm{spec}$" if col == "lam_spec" else r"$\eta$"


def _resonance_GHz(data: Dict[str, np.ndarray]) -> Optional[float]:
    if "w_p_GHz" in data and np.isfinite(data["w_p_GHz"]).any():
        return float(np.nanmedian(data["w_p_GHz"]))
    return None


# ---------------------------------------------------------------------------
# Figure 1: infidelity vs spectator frequency
# ---------------------------------------------------------------------------
def fig_infidelity_vs_frequency(data: Dict[str, np.ndarray],
                                figsize: Optional[Tuple[float, float]] = None) -> "plt.Figure":
    """Figure 1: gate infidelity 1 - F vs spectator frequency (log-y), one panel
    per channel, DRAG off (dashed/open) vs on (solid/filled), coloured by the
    swept participation.

    Parameters
    ----------
    data : dict[str, ndarray]
        Loaded summary columns; must contain finite ``F_avg``.
    figsize : tuple(float, float), optional
        Figure size in inches; a channel-count-dependent default is used if None.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    scol = series_column(data)
    chans = list(dict.fromkeys(data["channel"]))
    lams = sorted(np.unique(data[scol]))
    colors = {lam: WONG[i % len(WONG)] for i, lam in enumerate(lams)}
    wres = _resonance_GHz(data)

    n = len(chans)
    fig, axes = plt.subplots(1, n, figsize=figsize or (3.5 * n + 0.3, 3.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    floor = np.inf
    xc = _xcol(data)
    for ax, ch in zip(axes, chans):
        for lam in lams:
            for drag in (False, True):
                m = ((data["channel"] == ch) & (data[scol] == lam)
                     & (data["drag"] == drag) & np.isfinite(data["F_avg"]))
                if not m.any():
                    continue
                x = data[xc][m]
                y = 1.0 - data["F_avg"][m]
                o = np.argsort(x)
                pos = y[o] > 0
                if pos.any():
                    floor = min(floor, float(np.min(y[o][pos])))
                ax.plot(x[o], y[o], color=colors[lam],
                        ls="-" if drag else "--",
                        marker="o", mfc=(colors[lam] if drag else "white"),
                        mec=colors[lam])
        if wres is not None and xc == "spec_freq_GHz":
            ax.axvline(wres, color="0.5", lw=0.8, ls=":", zorder=0)
            ax.text(wres, 1.0, "  spectator\n  resonance", transform=
                    ax.get_xaxis_transform(), va="top", ha="left",
                    fontsize=6.8, color="0.4")
        _mark_collision(ax, xc)
        ax.set_yscale("log")
        ax.set_xlabel(_XLABELS[xc])
        ax.set_title(ch.replace("_", "$-$"))
    axes[0].set_ylabel(r"infidelity $1-F$")
    for ax, lab in zip(axes, "abcdef"):
        _panel_label(ax, f"({lab})")
    if np.isfinite(floor):
        axes[0].set_ylim(bottom=max(floor / 3, 1e-9))

    # two compact legends: colour = series value, linestyle = DRAG
    from matplotlib.lines import Line2D
    lam_handles = [Line2D([], [], color=colors[l], marker="o", ls="-",
                          label=f"{l:g}") for l in lams]
    drag_handles = [Line2D([], [], color="0.25", ls="--", marker="o",
                           mfc="white", label="DRAG off"),
                    Line2D([], [], color="0.25", ls="-", marker="o",
                           label="DRAG on")]
    leg1 = axes[-1].legend(handles=lam_handles, title=_series_label(scol),
                           loc="upper right", ncol=1)
    axes[-1].add_artist(leg1)
    axes[0].legend(handles=drag_handles, loc="lower right")
    return fig


# ---------------------------------------------------------------------------
# Figure 2: leakage / occupation vs spectator frequency
# ---------------------------------------------------------------------------
def fig_leakage_vs_frequency(data: Dict[str, np.ndarray],
                             figsize: Optional[Tuple[float, float]] = None) -> "plt.Figure":
    """Figure 2: leakage and spectator/coupler occupation vs spectator frequency
    (log-y), for the largest swept participation, DRAG off vs on.

    Parameters
    ----------
    data : dict[str, ndarray]
        Loaded summary columns; uses whichever of leakage / n_spec / n_coupler
        are present and finite.
    figsize : tuple(float, float), optional
        Figure size in inches; a channel-count-dependent default is used if None.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    scol = series_column(data)
    chans = list(dict.fromkeys(data["channel"]))
    # one representative participation (largest) keeps the panel readable
    lam = sorted(np.unique(data[scol]))[-1]
    wres = _resonance_GHz(data)
    metrics = [m for m in ("leakage", "n_spec", "n_coupler") if m in data
               and np.isfinite(data[m]).any()]
    mlabel = {"leakage": r"leakage $L_1$", "n_spec": r"$\langle n_\mathrm{spec}\rangle$",
              "n_coupler": r"$\langle n_\mathrm{coupler}\rangle$"}
    mcolor = {"leakage": WONG[1], "n_spec": WONG[0], "n_coupler": WONG[2]}

    n = len(chans)
    fig, axes = plt.subplots(1, n, figsize=figsize or (3.5 * n + 0.3, 3.0),
                             sharey=True, squeeze=False)
    axes = axes[0]
    xc = _xcol(data)
    for ax, ch in zip(axes, chans):
        for met in metrics:
            for drag in (False, True):
                m = ((data["channel"] == ch) & (data[scol] == lam)
                     & (data["drag"] == drag) & np.isfinite(data[met]))
                if not m.any():
                    continue
                x = data[xc][m]; y = data[met][m]; o = np.argsort(x)
                ax.plot(x[o], np.clip(y[o], 1e-12, None), color=mcolor[met],
                        ls="-" if drag else "--", marker="o",
                        mfc=(mcolor[met] if drag else "white"), mec=mcolor[met])
        if wres is not None and xc == "spec_freq_GHz":
            ax.axvline(wres, color="0.5", lw=0.8, ls=":", zorder=0)
        _mark_collision(ax, xc)
        ax.set_yscale("log")
        ax.set_xlabel(_XLABELS[xc])
        ax.set_title(ch.replace("_", "$-$") + rf"   ({_series_label(scol)}$={lam:g}$)")
    axes[0].set_ylabel("population")
    for ax, lab in zip(axes, "abcdef"):
        _panel_label(ax, f"({lab})")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=mcolor[m], marker="o", label=mlabel[m])
               for m in metrics]
    handles += [Line2D([], [], color="0.25", ls="--", marker="o", mfc="white",
                       label="DRAG off"),
                Line2D([], [], color="0.25", ls="-", marker="o", label="DRAG on")]
    axes[-1].legend(handles=handles, loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# Figure 3: 2D collision heatmap
# ---------------------------------------------------------------------------
def _pivot(data, mask, xcol, ycol, vcol):
    """Build a (ys, xs, Z) grid for pcolormesh from scattered rows."""
    xs = np.array(sorted(np.unique(data[xcol][mask])))
    ys = np.array(sorted(np.unique(data[ycol][mask])))
    Z = np.full((ys.size, xs.size), np.nan)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    for x, y, v in zip(data[xcol][mask], data[ycol][mask], data[vcol][mask]):
        Z[yi[y], xi[x]] = v
    return xs, ys, Z


def _edges(c):
    c = np.asarray(c, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def fig_collision_heatmap(data: Dict[str, np.ndarray], metric: str = "F_avg",
                          figsize: Optional[Tuple[float, float]] = None) -> "plt.Figure":
    """Figure 3: 2D collision map of a metric over (spectator frequency,
    participation), one panel per (channel, DRAG) with a shared colour scale.

    Parameters
    ----------
    data : dict[str, ndarray]
        Loaded summary columns.
    metric : str, default "F_avg"
        Column to map. "F_avg" is shown as log10(1 - F); any other column is shown
        as log10(metric).
    figsize : tuple(float, float), optional
        Figure size in inches; a panel-count-dependent default is used if None.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    scol = series_column(data)
    chans = list(dict.fromkeys(data["channel"]))
    drags = [d for d in (False, True) if (data["drag"] == d).any()]
    wres = _resonance_GHz(data)

    if metric == "F_avg":
        vcol, vlabel, cmap = "F_avg", r"$\log_{10}(1-F)$", "magma"
        def transform(z): return np.log10(np.clip(1.0 - z, 1e-12, None))
    else:
        safe_metric = metric.replace("_", r"\_")        # backslash outside the f-string (pre-3.12 safe)
        vcol, vlabel, cmap = metric, rf"$\log_{{10}}\,${safe_metric}", "magma"
        def transform(z): return np.log10(np.clip(z, 1e-12, None))

    nr, nc = len(drags), len(chans)
    fig, axes = plt.subplots(nr, nc, figsize=figsize or (3.4 * nc + 0.6, 2.7 * nr),
                             squeeze=False, sharex=True, sharey=True)
    # common color scale across panels
    allvals = transform(data[vcol][np.isfinite(data[vcol])])
    vmin, vmax = (np.nanpercentile(allvals, 2), np.nanpercentile(allvals, 98)) \
        if allvals.size else (-6, 0)
    im = None
    xc = _xcol(data)
    for i, drag in enumerate(drags):
        for j, ch in enumerate(chans):
            ax = axes[i][j]
            m = ((data["channel"] == ch) & (data["drag"] == drag)
                 & np.isfinite(data[vcol]))
            if m.any():
                xs, ys, Z = _pivot(data, m, xc, scol, vcol)
                im = ax.pcolormesh(_edges(xs), _edges(ys), transform(Z),
                                   cmap=cmap, vmin=vmin, vmax=vmax,
                                   shading="auto", rasterized=True)
                if wres is not None and xc == "spec_freq_GHz":
                    ax.axvline(wres, color="w", lw=0.8, ls=":")
                elif xc == "beat_GHz":
                    ax.axvline(0.0, color="w", lw=0.8, ls=":")
            if i == nr - 1:
                ax.set_xlabel(_XLABELS[xc])
            if j == 0:
                ax.set_ylabel(_series_label(scol))
            ax.set_title(f"{ch.replace('_', '$-$')}, DRAG {'on' if drag else 'off'}")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
        cb.set_label(vlabel)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: analytic collision predictor (works for analytic-only sweeps)
# ---------------------------------------------------------------------------
def fig_analytic_map(data: Dict[str, np.ndarray],
                     figsize: Optional[Tuple[float, float]] = None) -> Optional["plt.Figure"]:
    """Figure 4: analytic collision predictor -- the Eq.-62 exchange rate
    g_spec_eff and the "danger ratio" g_spec/|beat| vs spectator frequency.
    Available even for an analytic-only sweep (``--no-integrate``).

    Parameters
    ----------
    data : dict[str, ndarray]
        Loaded summary columns; requires ``g_spec_eff_MHz``.
    figsize : tuple(float, float), optional
        Figure size in inches; default (7.2, 3.0) if None.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None if the analytic rate column is absent.
    """
    scol = series_column(data)
    if "g_spec_eff_MHz" not in data:
        return None
    chans = list(dict.fromkeys(data["channel"]))
    lams = sorted(np.unique(data[scol]))
    colors = {lam: WONG[i % len(WONG)] for i, lam in enumerate(lams)}
    wres = _resonance_GHz(data)

    fig, axes = plt.subplots(1, 2, figsize=figsize or (7.2, 3.0))
    ch = chans[0]   # the analytic exchange rate is channel-independent
    xc = _xcol(data)
    for lam in lams:
        m = (data["channel"] == ch) & (data[scol] == lam) & (data["drag"] == False)
        if not m.any():
            continue
        x = data[xc][m]; o = np.argsort(x)
        gspec = data["g_spec_eff_MHz"][m][o]
        beat = data["beat_GHz"][m][o]
        danger = gspec / np.clip(np.abs(beat) * 1e3, 1e-6, None)  # MHz / MHz
        axes[0].plot(x[o], gspec, color=colors[lam], marker="o",
                     label=f"{lam:g}")
        axes[1].plot(x[o], danger, color=colors[lam], marker="o")
    g_is = float(np.nanmedian(data["g_iswap_eff_MHz"])) if "g_iswap_eff_MHz" in data else None
    if g_is is not None:
        axes[0].axhline(g_is, color="0.4", lw=0.9, ls="-.")
        axes[0].text(axes[0].get_xlim()[1], g_is, r" $g^\mathrm{eff}_\mathrm{iSWAP}$",
                     va="center", ha="right", fontsize=7.5, color="0.4")
    for ax in axes:
        if wres is not None and xc == "spec_freq_GHz":
            ax.axvline(wres, color="0.5", lw=0.8, ls=":", zorder=0)
        _mark_collision(ax, xc)
        ax.set_xlabel(_XLABELS[xc])
    axes[0].set_ylabel(r"exchange rate $g^\mathrm{eff}_\mathrm{spec}/2\pi$ (MHz)")
    axes[1].set_ylabel(r"danger ratio $g^\mathrm{eff}_\mathrm{spec}/|\delta_s|$")
    axes[1].set_yscale("log")
    _panel_label(axes[0], "(a)"); _panel_label(axes[1], "(b)")
    axes[0].legend(title=_series_label(scol), loc="best")
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _save(fig, figdir, name, fmts):
    os.makedirs(figdir, exist_ok=True)
    paths = []
    for ext in fmts:
        p = os.path.join(figdir, f"{name}.{ext}")
        fig.savefig(p)
        paths.append(p)
    return paths


def main() -> None:
    """Command-line entry point: load a sweep directory and render the applicable
    figures. A full sweep yields Figs 1-4; an analytic-only sweep yields Fig 4.
    Run ``--help`` for the option list (``--outdir``, ``--figdir``, ``--only``,
    ``--metric``, ``--format``, ``--usetex``).
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True, help="sweep result dir (has summary.csv)")
    ap.add_argument("--figdir", help="where to write figures (default <outdir>/figs)")
    ap.add_argument("--only", choices=["infidelity", "leakage", "heatmap", "analytic"],
                    help="render a single figure instead of all applicable")
    ap.add_argument("--metric", default="F_avg",
                    help="heatmap metric: F_avg (default), leakage, n_spec, n_coupler")
    ap.add_argument("--format", default="pdf,png",
                    help="comma list of output formats (default pdf,png)")
    ap.add_argument("--usetex", action="store_true", help="use a system LaTeX install")
    ap.add_argument("--vs-beat", action="store_true",
                    help="plot vs beat detuning delta = Delta - w_p (0 = collision) "
                         "instead of absolute spectator frequency")
    args = ap.parse_args()

    global _XAXIS
    if args.vs_beat:
        _XAXIS = "beat_GHz"

    figdir = args.figdir or os.path.join(args.outdir, "figs")
    fmts = [s.strip() for s in args.format.split(",") if s.strip()]
    set_literature_style(usetex=args.usetex)
    data = load_summary(args.outdir)
    full = has_full_metrics(data)

    builders = {
        "infidelity": (fig_infidelity_vs_frequency, full),
        "leakage": (fig_leakage_vs_frequency, full and any(
            m in data and np.isfinite(data[m]).any() for m in
            ("leakage", "n_spec", "n_coupler"))),
        "heatmap": (lambda d: fig_collision_heatmap(d, metric=args.metric), full),
        "analytic": (fig_analytic_map, "g_spec_eff_MHz" in data),
    }
    todo = [args.only] if args.only else ["infidelity", "leakage", "heatmap", "analytic"]

    made = []
    for name in todo:
        builder, ok = builders[name]
        if not ok:
            if args.only:
                print(f"Cannot build '{name}': required columns absent "
                      f"({'no finite F_avg' if not full else 'missing metric'}).")
            continue
        fig = builder(data)
        if fig is None:
            continue
        made += _save(fig, figdir, f"fig_{name}", fmts)
        plt.close(fig)

    if not made:
        print("No figures produced. (Analytic-only sweeps yield only the "
              "'analytic' figure; run with integrate=true for the rest.)")
    else:
        print(f"Wrote {len(made)} file(s) to {figdir}/:")
        for p in made:
            print("  " + os.path.basename(p))


if __name__ == "__main__":
    main()