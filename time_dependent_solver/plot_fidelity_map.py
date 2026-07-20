"""Side-by-side DRAG / no-DRAG fidelity heatmaps over the target allocation grid.

The ``target`` sweep in :mod:`run_sweep_zhou` scans the partner-qubit frequency
:math:`\\omega_b` against the spectator frequency :math:`\\omega_c` (column
``spec_GHz``) and records the average iSWAP gate fidelity per point. This module
pivots ``summary.csv`` onto that grid and renders two heatmaps -- without DRAG and
with DRAG -- sharing one colour scale so the panels are directly comparable, with an
optional dotted contour marking the ``nearest_beat = 0`` collision locus.

DRAG appears in the summary in one of two layouts, both handled here:

* separate ``--drags true,false`` rows: two rows per :math:`(\\omega_b, \\omega_c)`
  cell whose ``F_avg`` is routed by ``drag_applied``;
* an in-row ``--drag-compare``: one row per cell whose ``F_avg`` is the no-DRAG
  result and whose ``F_avg_drag`` is the DRAG result (here ``drag_applied`` is not a
  reliable router, so the presence of ``F_avg_drag`` takes precedence).

Where both layouts are present for a cell, a dedicated DRAG-on row wins over the
in-row compare value.

CLI
---
``python plot_fidelity_map.py --outdir results/<run> [--metric fidelity|infidelity]
[--log/--no-log] [--cmap NAME] [--no-collision-line] [--out fig.png]``
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def _to_float(x: object) -> Optional[float]:
    """Parse a CSV cell into a finite float, or ``None``.

    Parameters
    ----------
    x : object
        Raw CSV field (``str`` such as ``""``, ``"0.97"``, ``"nan"``).

    Returns
    -------
    float or None
        The value if it parses to a finite float, else ``None``.
    """
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _to_bool(x: object) -> bool:
    """Parse a CSV cell into a bool (``True`` for ``true``/``1``/``yes``)."""
    return str(x).strip().lower() in ("true", "1", "yes")


def load_summary_rows(path: str) -> List[Dict[str, str]]:
    """Read a sweep ``summary.csv`` into a list of row dicts.

    Parameters
    ----------
    path : str
        Either a sweep output directory (``summary.csv`` is appended) or a direct
        path to the CSV.

    Returns
    -------
    list of dict
        One dict per row, keyed by column name.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist (``collect`` has not been run).
    """
    csv_path = os.path.join(path, "summary.csv") if os.path.isdir(path) else path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found; run `collect` first.")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_fidelity_grids(
    rows: List[Dict[str, str]],
    metric: str = "fidelity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot target-sweep rows onto the :math:`(\\omega_b, \\omega_c)` grid.

    Parameters
    ----------
    rows : list of dict
        Rows from a *target* ``summary.csv`` (must contain ``wb_GHz`` and
        ``spec_GHz``).
    metric : {"fidelity", "infidelity"}, optional
        ``"fidelity"`` maps each cell to ``F_avg``; ``"infidelity"`` maps it to
        ``1 - F_avg``.

    Returns
    -------
    wb_vals : ndarray, shape (n_wb,)
        Sorted unique :math:`\\omega_b` grid values (GHz), the x-axis.
    spec_vals : ndarray, shape (n_spec,)
        Sorted unique spectator :math:`\\omega_c` grid values (GHz), the y-axis.
    z_nodrag : ndarray, shape (n_spec, n_wb)
        Metric without DRAG; missing cells are ``nan``.
    z_drag : ndarray, shape (n_spec, n_wb)
        Metric with DRAG; missing cells are ``nan``.
    beat_grid : ndarray, shape (n_spec, n_wb)
        ``nearest_beat_GHz`` per cell (for the collision contour); ``nan`` where
        absent.

    Raises
    ------
    KeyError
        If the rows are not from a target sweep (no ``wb_GHz``/``spec_GHz``).
    """
    if not rows or "wb_GHz" not in rows[0] or "spec_GHz" not in rows[0]:
        raise KeyError(
            "these rows are not a target sweep (need 'wb_GHz' and 'spec_GHz'); "
            "this tool plots the (omega_b, omega_c) allocation grid."
        )
    if metric not in ("fidelity", "infidelity"):
        raise ValueError("metric must be 'fidelity' or 'infidelity'")

    def _key(r: Dict[str, str]) -> Optional[Tuple[float, float]]:
        wb, sp = _to_float(r.get("wb_GHz")), _to_float(r.get("spec_GHz"))
        if wb is None or sp is None:
            return None
        return (round(wb, 9), round(sp, 9))

    nodrag: Dict[Tuple[float, float], float] = {}
    drag_ded: Dict[Tuple[float, float], float] = {}   # dedicated DRAG-on rows
    drag_cmp: Dict[Tuple[float, float], float] = {}    # in-row F_avg_drag compare
    beat: Dict[Tuple[float, float], float] = {}

    for r in rows:
        key = _key(r)
        if key is None:
            continue
        f_avg = _to_float(r.get("F_avg"))
        f_drag = _to_float(r.get("F_avg_drag"))
        b = _to_float(r.get("nearest_beat_GHz"))
        if b is not None:
            beat[key] = b
        if f_drag is not None:
            # in-row compare: F_avg is no-DRAG, F_avg_drag is DRAG (ignore
            # drag_applied here -- it is set True in the compare window even though
            # F_avg is the DRAG-off result).
            if f_avg is not None:
                nodrag[key] = f_avg
            drag_cmp[key] = f_drag
        elif _to_bool(r.get("drag_applied")):
            if f_avg is not None:
                drag_ded[key] = f_avg
        else:
            if f_avg is not None:
                nodrag[key] = f_avg

    drag = {**drag_cmp, **drag_ded}  # dedicated DRAG-on wins where both exist

    wb_vals = np.array(sorted({k[0] for k in
                               set(nodrag) | set(drag) | set(beat)}))
    spec_vals = np.array(sorted({k[1] for k in
                                 set(nodrag) | set(drag) | set(beat)}))
    wb_ix = {w: i for i, w in enumerate(wb_vals)}
    sp_ix = {s: i for i, s in enumerate(spec_vals)}

    shape = (spec_vals.size, wb_vals.size)
    z_nodrag = np.full(shape, np.nan)
    z_drag = np.full(shape, np.nan)
    beat_grid = np.full(shape, np.nan)

    def _fill(dst: np.ndarray, src: Dict[Tuple[float, float], float],
              transform: bool) -> None:
        for (w, s), v in src.items():
            dst[sp_ix[s], wb_ix[w]] = (1.0 - v) if (transform and metric ==
                                                    "infidelity") else v

    _fill(z_nodrag, nodrag, transform=True)
    _fill(z_drag, drag, transform=True)
    _fill(beat_grid, beat, transform=False)
    return wb_vals, spec_vals, z_nodrag, z_drag, beat_grid


def plot_fidelity_map(
    source: str,
    *,
    metric: str = "fidelity",
    log: Optional[bool] = None,
    cmap: Optional[str] = None,
    collision_line: bool = True,
    out: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Render the DRAG vs no-DRAG fidelity heatmaps and save a PNG.

    Parameters
    ----------
    source : str
        Sweep output directory (or a direct ``summary.csv`` path).
    metric : {"fidelity", "infidelity"}, optional
        Quantity shown in colour. Default ``"fidelity"`` (as requested);
        ``"infidelity"`` with ``log=True`` usually reveals the collision ridges
        far more clearly.
    log : bool or None, optional
        Logarithmic colour scale. ``None`` auto-selects: log for ``infidelity``,
        linear for ``fidelity``.
    cmap : str or None, optional
        Matplotlib colormap. ``None`` picks ``"viridis"`` for fidelity (bright =
        good) and ``"inferno"`` for infidelity (bright = bad).
    collision_line : bool, optional
        Overlay a dotted contour where ``nearest_beat_GHz`` crosses zero (the
        one-pump spectator resonance). Default ``True``.
    out : str or None, optional
        Output PNG path. ``None`` writes ``<dir>/figs/fidelity_map_<metric>.png``.
    title : str or None, optional
        Figure suptitle. ``None`` uses a sensible default.

    Returns
    -------
    str
        Path to the written PNG.

    Notes
    -----
    Missing grid cells (points that did not finish) render in light grey. A colour
    scale shared across both panels keeps them directly comparable.
    """
    if log is None:
        log = (metric == "infidelity")
    if cmap is None:
        cmap = "inferno" if metric == "infidelity" else "viridis"

    rows = load_summary_rows(source)
    wb, spec, z_off, z_on, beat = build_fidelity_grids(rows, metric=metric)
    if wb.size == 0 or spec.size == 0:
        raise ValueError("no (wb_GHz, spec_GHz) grid found in the summary.")

    finite = np.concatenate([z_off[np.isfinite(z_off)].ravel(),
                             z_on[np.isfinite(z_on)].ravel()])
    if finite.size == 0:
        raise ValueError("summary has no finite fidelities to plot.")
    if log:
        pos = finite[finite > 0]
        vmin = float(pos.min()) if pos.size else 1e-4
        vmax = float(finite.max())
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.0001))
    else:
        norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.85")  # missing cells

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             constrained_layout=True, sharey=True)
    wb_mesh, sp_mesh = np.meshgrid(wb, spec)
    have_beat = np.isfinite(beat).any() and (np.nanmin(beat) < 0 < np.nanmax(beat))

    im = None
    for ax, z, lab, present in ((axes[0], z_off, "no DRAG", np.isfinite(z_off).any()),
                                (axes[1], z_on, "with DRAG", np.isfinite(z_on).any())):
        im = ax.pcolormesh(wb, spec, np.ma.masked_invalid(z), shading="nearest",
                           cmap=cmap_obj, norm=norm)
        if collision_line and have_beat:
            ax.contour(wb_mesh, sp_mesh, beat, levels=[0.0],
                       colors="white", linestyles=":", linewidths=1.1, alpha=0.75)
        ax.set_xlabel(r"$\omega_b$ (GHz)")
        ax.set_title(lab + ("" if present else "  (no data)"), fontsize=11)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel(r"$\omega_c$ spectator (GHz)")

    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label(("infidelity $1-F$" if metric == "infidelity"
                  else r"fidelity $F$") + ("  (log)" if log else ""))
    if title is None:
        title = "iSWAP allocation: DRAG vs no DRAG"
        if collision_line and have_beat:
            title += "   (dotted: spectator resonance, beat $=0$)"
    fig.suptitle(title, fontsize=12)

    if out is None:
        base = source if os.path.isdir(source) else os.path.dirname(source) or "."
        os.makedirs(os.path.join(base, "figs"), exist_ok=True)
        out = os.path.join(base, "figs", f"fidelity_map_{metric}.png")
    else:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _peak_2d(Z: np.ndarray, wb: np.ndarray, spec: np.ndarray):
    """Return ``(F, wb, spec)`` at the maximum of ``Z``, or ``(nan, None, None)``
    if ``Z`` is entirely non-finite.

    Parameters
    ----------
    Z : ndarray, shape (n_spec, n_wb)
        Fidelity grid (rows indexed by ``spec``, columns by ``wb``).
    wb, spec : ndarray
        Grid axes matching ``Z``'s columns and rows respectively.
    """
    if not np.isfinite(Z).any():
        return (float("nan"), None, None)
    i, j = np.unravel_index(int(np.nanargmax(Z)), Z.shape)
    return (float(Z[i, j]), float(wb[j]), float(spec[i]))


def _fidelity_summary(wb: np.ndarray, spec: np.ndarray,
                      z_off: np.ndarray, z_on: np.ndarray) -> Dict[str, object]:
    """Assemble the best-fidelity summary from the two fidelity grids.

    ``best_of_both`` is the element-wise ``nan``-aware max (``np.fmax``): a cell
    with only one condition present keeps that value, so it is the best fidelity
    *available* at each allocation point.
    """
    best = np.fmax(z_off, z_on)
    return {
        "wb": wb, "spec": spec,
        "F_nodrag": z_off, "F_drag": z_on, "best_of_both": best,
        "nodrag": _peak_2d(z_off, wb, spec),
        "drag": _peak_2d(z_on, wb, spec),
        "overall": _peak_2d(best, wb, spec),
    }


def best_fidelities(source: str) -> Dict[str, object]:
    """Extract the best average iSWAP fidelity achieved with and without DRAG.

    Parameters
    ----------
    source : str
        Sweep directory (containing ``summary.csv``) or a direct CSV path.

    Returns
    -------
    dict
        Keys: ``wb``, ``spec`` (grid axes, ndarray); ``F_nodrag``, ``F_drag``
        (2-D fidelity grids, shape ``(n_spec, n_wb)``, ``nan`` where a point is
        missing or absent for that condition); ``best_of_both`` (element-wise
        ``nan``-aware max of the two); and ``nodrag`` / ``drag`` / ``overall``,
        each a ``(F, wb, spec)`` tuple giving the peak fidelity and where it
        occurs (``(nan, None, None)`` if that condition has no data).
    """
    rows = load_summary_rows(source)
    wb, spec, z_off, z_on, _beat = build_fidelity_grids(rows, metric="fidelity")
    return _fidelity_summary(wb, spec, z_off, z_on)


def plot_diff_and_best(
    source: str,
    *,
    collision_line: bool = True,
    diff_pct: float = 95.0,
    out: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[str, Dict[str, object]]:
    """Render the DRAG-minus-no-DRAG fidelity difference alongside the
    best-of-both map, and return the PNG path plus the best-fidelity summary.

    The left panel is :math:`\\Delta F = F_{\\mathrm{DRAG}} - F_{\\mathrm{no\\,DRAG}}`
    on a diverging scale (blue where DRAG improves the gate, red where it degrades
    it, grey where a cell is missing in either condition). The right panel is
    :math:`\\max(F_{\\mathrm{DRAG}}, F_{\\mathrm{no\\,DRAG}})` -- the best fidelity
    obtainable at each allocation point by choosing the better of the two gates.

    Parameters
    ----------
    source : str
        Sweep directory (containing ``summary.csv``) or a direct CSV path.
    collision_line : bool, optional
        Overlay the ``nearest_beat = 0`` spectator-resonance contour on both
        panels. Default ``True``.
    diff_pct : float, optional
        Percentile of ``|ΔF|`` used to set the symmetric colour range of the diff
        panel (default 95). Cells beyond it saturate; this keeps the collision
        bands -- where the DRAG calibration mislocates and ``ΔF`` is large -- from
        washing out the small genuine differences elsewhere.
    out : str or None, optional
        Output PNG path. ``None`` writes ``<dir>/figs/fidelity_diff_best.png``.
    title : str or None, optional
        Figure suptitle.

    Returns
    -------
    (str, dict)
        The written PNG path and the :func:`best_fidelities` summary dict.
    """
    rows = load_summary_rows(source)
    wb, spec, z_off, z_on, beat = build_fidelity_grids(rows, metric="fidelity")
    if wb.size == 0 or spec.size == 0:
        raise ValueError("no (wb_GHz, spec_GHz) grid found in the summary.")
    info = _fidelity_summary(wb, spec, z_off, z_on)

    diff = z_on - z_off                                   # +ve => DRAG improves F
    finite_d = np.abs(diff[np.isfinite(diff)])
    m = float(np.percentile(finite_d, diff_pct)) if finite_d.size else 1.0
    m = max(m, 1e-6)

    wb_mesh, sp_mesh = np.meshgrid(wb, spec)
    have_beat = np.isfinite(beat).any() and (np.nanmin(beat) < 0 < np.nanmax(beat))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                             constrained_layout=True, sharey=True)

    # left: difference (red = DRAG worse, blue = DRAG better)
    cmap_d = plt.get_cmap("RdBu").copy()
    cmap_d.set_bad("0.85")
    im0 = axes[0].pcolormesh(wb, spec, np.ma.masked_invalid(diff), shading="nearest",
                             cmap=cmap_d, norm=Normalize(vmin=-m, vmax=+m))
    axes[0].set_title(r"$\Delta F = F_{\mathrm{DRAG}} - F_{\mathrm{no\,DRAG}}$"
                      "\n(blue: DRAG better, red: worse)", fontsize=10)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.9, pad=0.02)
    cb0.set_label(rf"$\Delta F$  (saturates at $\pm{m:.3f}$)")

    # right: best of both
    fb = info["best_of_both"][np.isfinite(info["best_of_both"])]
    cmap_b = plt.get_cmap("viridis").copy()
    cmap_b.set_bad("0.85")
    norm_b = Normalize(vmin=float(fb.min()), vmax=float(fb.max())) if fb.size else None
    im1 = axes[1].pcolormesh(wb, spec, np.ma.masked_invalid(info["best_of_both"]),
                             shading="nearest", cmap=cmap_b, norm=norm_b)
    axes[1].set_title(r"best of both: $\max(F_{\mathrm{DRAG}},\,F_{\mathrm{no\,DRAG}})$",
                      fontsize=10)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.9, pad=0.02)
    cb1.set_label(r"fidelity $F$")

    for ax in axes:
        if collision_line and have_beat:
            ax.contour(wb_mesh, sp_mesh, beat, levels=[0.0], colors="0.3",
                       linestyles=":", linewidths=1.0, alpha=0.7)
        ax.set_xlabel(r"$\omega_b$ (GHz)")
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel(r"$\omega_c$ spectator (GHz)")

    fig.suptitle(title or "DRAG vs no-DRAG: difference and best-of-both", fontsize=12)

    if out is None:
        base = source if os.path.isdir(source) else os.path.dirname(source) or "."
        os.makedirs(os.path.join(base, "figs"), exist_ok=True)
        out = os.path.join(base, "figs", "fidelity_diff_best.png")
    else:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, info


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", "--csv", dest="source", required=True,
                    help="sweep directory (with summary.csv) or a summary.csv path")
    ap.add_argument("--metric", choices=["fidelity", "infidelity"], default="fidelity",
                    help="colour quantity (default: fidelity)")
    ap.add_argument("--log", dest="log", action="store_true", default=None,
                    help="log colour scale (default: auto -- log for infidelity)")
    ap.add_argument("--no-log", dest="log", action="store_false",
                    help="force a linear colour scale")
    ap.add_argument("--cmap", default=None,
                    help="matplotlib colormap (default: viridis / inferno)")
    ap.add_argument("--no-collision-line", dest="collision_line", action="store_false",
                    help="do not overlay the beat=0 spectator-resonance contour")
    ap.add_argument("--diff", action="store_true",
                    help="render the DRAG-minus-no-DRAG difference and best-of-both "
                         "panels, and print the best fidelity in each condition")
    ap.add_argument("--diff-pct", type=float, default=95.0,
                    help="percentile of |ΔF| for the diff colour range (default 95)")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--title", default=None, help="figure suptitle")
    args = ap.parse_args()

    if args.diff:
        path, info = plot_diff_and_best(args.source, collision_line=args.collision_line,
                                        diff_pct=args.diff_pct, out=args.out,
                                        title=args.title)

        def _fmt(peak) -> str:
            F, w, s = peak
            return ("n/a (no data)" if w is None
                    else f"F = {F:.4f}  (1-F = {1 - F:.3e})  at  "
                         f"omega_b = {w:.3f} GHz, omega_c = {s:.3f} GHz")

        print(f"best without DRAG : {_fmt(info['nodrag'])}")
        print(f"best with DRAG    : {_fmt(info['drag'])}")
        print(f"best of both      : {_fmt(info['overall'])}")
        d = info["F_drag"] - info["F_nodrag"]
        both = np.isfinite(d)
        if both.any():
            print(f"DRAG improves F in {float((d[both] > 0).mean()) * 100:.0f}% of the "
                  f"{int(both.sum())} cells where both conditions ran "
                  f"(median dF = {float(np.median(d[both])):+.4f}).")
        print(f"wrote {path}")
        return

    path = plot_fidelity_map(args.source, metric=args.metric, log=args.log,
                             cmap=args.cmap, collision_line=args.collision_line,
                             out=args.out, title=args.title)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()