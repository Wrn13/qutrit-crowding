"""Plot fidelity over a 1D bare-gate (no-spectator) target sweep.

The ``--no-spectator`` target sweep varies only w_b, so 2 w_p scans the SNAIL
subharmonic w_c = 2 w_p. This reads results/<outdir>/summary.csv (target-kind
rows with a blank spectator) and plots, DRAG off vs on:

  * top    : infidelity 1 - F   (log axis)
  * bottom : coupler occupation n_coupler  -- the direct snail-subharmonic signal
             (the pump 2nd harmonic loading the coupler)

x-axis is the snail-subharmonic detuning w_c - 2 w_p (== nearest_beat_GHz, 0 on
resonance) by default, or w_b with ``--xaxis wb``. Near resonance DRAG is skipped
(singular), so the two curves merge there -- that is physical, not a plotting bug.

Usage
-----
    python plot_bare_sweep.py --outdir <name>            # reads results/<name>/summary.csv
    python plot_bare_sweep.py --csv path/to/summary.csv --xaxis wb
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, RED = "#2980B9", "#C0392B"


def _f(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(path: str) -> List[Dict[str, object]]:
    """Load integrated target rows (those with a numeric F_avg) from summary.csv.

    Parameters
    ----------
    path : str
        Path to a summary.csv written by ``collect``.

    Returns
    -------
    list of dict
        One entry per integrated row with keys wb, beat, F, ncpl, ptr, drag.
    """
    rows: List[Dict[str, object]] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            F = _f(r.get("F_avg", ""))
            if F is None:                       # skip analytic-only / missing rows
                continue
            rows.append(dict(
                wb=_f(r.get("wb_GHz", "")),
                beat=_f(r.get("nearest_beat_GHz", "")),
                F=F,
                ncpl=_f(r.get("n_coupler", "")),
                ptr=_f(r.get("p_transfer", "")),
                drag=str(r.get("drag", "")).strip().lower() == "true",
            ))
    if not rows:
        raise SystemExit(f"no integrated rows (numeric F_avg) found in {path}")
    return rows


def plot_bare_sweep(rows: List[Dict[str, object]], xaxis: str = "beat",
                    out: str = "figs/bare_sweep.png",
                    title: Optional[str] = None) -> None:
    """Two-panel infidelity + coupler-occupation plot vs detuning (or w_b)."""
    xkey = "beat" if xaxis == "beat" else "wb"
    xscale = 1e3 if xaxis == "beat" else 1.0
    xlabel = (r"snail subharmonic detuning  $\omega_c - 2\omega_p$  (MHz)"
              if xaxis == "beat" else r"partner frequency  $\omega_b$  (GHz)")

    off = sorted([r for r in rows if not r["drag"] and r[xkey] is not None],
                 key=lambda r: r[xkey])
    on = sorted([r for r in rows if r["drag"] and r[xkey] is not None],
                key=lambda r: r[xkey])

    have_ncpl = all(r["ncpl"] is not None for r in rows)
    nrows = 2 if have_ncpl else 1
    fig, axl = plt.subplots(nrows, 1, figsize=(7.0, 6.2 if have_ncpl else 4.2),
                            sharex=True, dpi=200,
                            gridspec_kw=dict(height_ratios=[2, 1]) if have_ncpl else None)
    ax = list(np.atleast_1d(axl))

    for series, color, lab, mfc in ((off, BLUE, "no DRAG", "white"),
                                    (on, RED, "with DRAG", RED)):
        if not series:
            continue
        x = [r[xkey] * xscale for r in series]
        ax[0].plot(x, [1.0 - r["F"] for r in series], color=color, lw=2.2,
                   marker="o", ms=5, mfc=mfc, mec=color, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"infidelity  $1-F$")
    ax[0].legend(frameon=False, fontsize=11)

    if have_ncpl:
        for series, color, mfc in ((off, BLUE, "white"), (on, RED, RED)):
            if not series:
                continue
            x = [r[xkey] * xscale for r in series]
            ax[1].plot(x, [r["ncpl"] for r in series], color=color, lw=2.2,
                       marker="o", ms=5, mfc=mfc, mec=color)
        ax[1].set_ylabel(r"coupler occ.  $n_c$")

    ax[-1].set_xlabel(xlabel)
    if xaxis == "beat":
        for a in ax:
            a.axvline(0.0, color="0.6", ls=":", lw=1.2, zorder=0)
        ax[0].text(0.0, 1.02, "resonance", transform=ax[0].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=9, color="0.4")
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=13)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out, "and", out.rsplit(".", 1)[0] + ".pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", help="results/<outdir>/summary.csv")
    ap.add_argument("--csv", help="explicit path to summary.csv (overrides --outdir)")
    ap.add_argument("--xaxis", choices=["beat", "wb"], default="beat",
                    help="x-axis: snail-subharmonic detuning (default) or w_b")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.csv:
        path = args.csv
    elif args.outdir:
        try:
            from paths import in_results
            path = os.path.join(in_results(args.outdir), "summary.csv")
        except Exception:
            path = os.path.join("results", args.outdir, "summary.csv")
    else:
        ap.error("give --outdir or --csv")

    rows = load_rows(path)
    out = args.out or os.path.join(os.path.dirname(path), "figs", "bare_sweep.png")
    plot_bare_sweep(rows, args.xaxis, out, args.title)


if __name__ == "__main__":
    main()