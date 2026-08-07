"""2D calibration landscape: pump frequency offset vs pump strength.

Scans the (pump-frequency-offset, pump-amplitude) plane and scores each point by
the leakage-aware iSWAP fidelity or the |01>->|10> transfer probability (the swap
population), so the global optimum is read off directly instead of from
alternating 1D amplitude/frequency scans (which rail when the AC-Stark shift and
the Rabi amplitude feed back on each other).

Two propagation engines, identical grid and output shape:
  * engine='reduced' : the fast rotating-frame reduced model from ``grape.py``
    (scipy expm, no QuTiP); the pump offset shifts precomputed carriers so the
    operator basis is built once. Good for quick looks / weak-to-moderate drive.
  * engine='qutip'   : compiled ``qt.sesolve`` on the FULL Hamiltonian via
    ``ZhouCoupler.propagator_columns`` -- exact, the cluster path. A fresh coupler
    is built per grid point through the usual ``build_coupler`` plumbing
    (``build_system``), so the grid is embarrassingly parallel (``--jobs``) and
    free of pump-mutation hazards, and DRAG / spectator context come along for
    free exactly as in the reduced map.

Results (offsets, amps, Z, leakage, optimum, operating point) are written to
``--save-npz``; the heatmap with the optimum marked goes to ``--out``. At strong
drive, raise the coupler truncation (``--coupler-levels``) and confirm the map is
converged before trusting values in the bright/leaky region.

CLI
---
    python calibration_map.py --device warren_device.json --t-g-ns 92.6 \
        --engine qutip --metric transfer --jobs 32 \
        --wp-span-MHz 40 --amp-lo 0.6 --amp-hi 1.6 \
        --save-npz figs/swap_map.npz --out figs/swap_map.png
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grape

TWO_PI = 2.0 * np.pi


def scan(cpl, a: int, b: int, t_g: float, *,
         wp_span_MHz: float = 40.0, wp_points: int = 41,
         amp_lo: float = 0.6, amp_hi: float = 1.4, amp_points: int = 41,
         cutoff_GHz: float = 1.0, n_ctrl: int = 32, metric: str = "fidelity",
         carrier_resolution: float = 0.3) -> Dict[str, Any]:
    """Scan (pump offset, amplitude) and score each point with the REDUCED model.

    Parameters
    ----------
    cpl : ZhouCoupler
        Coupler with a pump set; ``peak_eta`` sets the amp=1 reference.
    a, b : int
        Target-qubit mode indices.
    t_g : float
        Gate duration (ns).
    wp_span_MHz : float
        Full width of the pump-offset axis (centered on 0), in MHz.
    wp_points, amp_points : int
        Grid resolution.
    amp_lo, amp_hi : float
        Amplitude-scale range (multiples of the nominal peak_eta).
    cutoff_GHz : float
        Rotating-frame carrier cutoff for the reduced model.
    n_ctrl : int
        Piecewise-constant slices approximating the raised cosine.
    metric : {'fidelity', 'transfer'}
        Score per point: leakage-aware iSWAP fidelity, or P(|01>->|10>).

    Returns
    -------
    dict
        offsets_MHz, amps, Z (shape [amp_points, wp_points]), best (dict),
        metric, plus scan metadata.
    """
    from zhou_coupler import ZhouCoupler
    terms, H_anh, idx, max_Omega = grape._prepare(cpl, a, b, cutoff_GHz)
    dt_ctrl = t_g / n_ctrl
    n_sub = max(1, int(np.ceil((max_Omega + abs(wp_span_MHz) * 1e-3 * TWO_PI)
                               * dt_ctrl / carrier_resolution)))

    peak = float(cpl.peak_eta())
    ts = (np.arange(n_ctrl) + 0.5) * dt_ctrl
    base = peak * 0.5 * (1.0 - np.cos(2.0 * np.pi * ts / t_g))    # raised cosine

    offsets = np.linspace(-wp_span_MHz / 2.0, wp_span_MHz / 2.0, wp_points)  # MHz
    amps = np.linspace(amp_lo, amp_hi, amp_points)
    Z = np.full((amp_points, wp_points), np.nan)

    # |01> is column 1 of the propagator basis (|00>,|01>,|10>,|11>); |10> is row 2
    for i, amp in enumerate(amps):
        eta_ctrl = (amp * base).astype(complex)
        for j, off_MHz in enumerate(offsets):
            U = grape._propagate(eta_ctrl, t_g, terms, H_anh, idx, n_sub,
                                 offset_rad=off_MHz * 1e-3 * TWO_PI)
            if metric == "transfer":
                Z[i, j] = abs(U[2, 1]) ** 2                        # |01> -> |10>
            else:
                F, _ = ZhouCoupler._iswap_fidelity_from_U(U, True)
                Z[i, j] = F

    bi, bj = np.unravel_index(np.nanargmax(Z), Z.shape)
    best = dict(amp_scale=float(amps[bi]), wp_offset_MHz=float(offsets[bj]),
                score=float(Z[bi, bj]))
    return dict(offsets_MHz=offsets, amps=amps, Z=Z, best=best, metric=metric,
                n_sub=n_sub, cutoff_GHz=cutoff_GHz, peak_eta=peak, engine="reduced")


def scan_qutip(build_fn: Callable[[float, float], Tuple[Any, float, float]],
               t_g: float, *, wp_span_MHz: float = 40.0, wp_points: int = 41,
               amp_lo: float = 0.6, amp_hi: float = 1.4, amp_points: int = 41,
               metric: str = "transfer", atol: float = 1e-10, rtol: float = 1e-8,
               nsteps: int = 500000, jobs: int = 1) -> Dict[str, Any]:
    """QuTiP-exact analogue of ``scan``: identical grid and return shape, but every
    point is a compiled ``qt.sesolve`` on the FULL Hamiltonian via
    ``ZhouCoupler.propagator_columns`` -- no reduced model, no pruned terms.

    ``build_fn(grid_amp, grid_off_MHz) -> (cpl, w_p, eta_pk)`` returns a FRESH
    coupler with the grid point already folded into the pump (via the usual
    ``build_system`` / ``device_utils.build_coupler`` plumbing), so points are
    independent and safe to evaluate in parallel. Leakage is recorded for free
    from the same 4x4 U. ``metric`` selects the colour ('transfer' = swap
    population |U[2,1]|^2, or leakage-aware 'fidelity').
    """
    from zhou_coupler import ZhouCoupler
    offsets = np.linspace(-wp_span_MHz / 2.0, wp_span_MHz / 2.0, wp_points)  # MHz
    amps = np.linspace(amp_lo, amp_hi, amp_points)
    _, _, eta_pk = build_fn(1.0, 0.0)                      # nominal amp=1 reference

    def one(i: int, j: int, amp: float, off_MHz: float):
        cpl, _w_p, _eta = build_fn(amp, off_MHz)
        U = cpl.propagator_columns(0, 1, t_g, atol=atol, rtol=rtol, nsteps=nsteps)
        fid, leak = ZhouCoupler._iswap_fidelity_from_U(U, True)
        score = abs(U[2, 1]) ** 2 if metric == "transfer" else fid   # swap pop / F
        return i, j, float(score), float(leak)

    grid = [(i, j, a, o) for i, a in enumerate(amps) for j, o in enumerate(offsets)]
    if jobs and jobs > 1:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=jobs, verbose=5)(
            delayed(one)(i, j, a, o) for (i, j, a, o) in grid)
    else:
        out = []
        for k, (i, j, a, o) in enumerate(grid):
            out.append(one(i, j, a, o))
            if (k + 1) % wp_points == 0:
                print(f"  row {k // wp_points + 1}/{amp_points} "
                      f"(amp_scale={a:.3f})")

    Z = np.full((amp_points, wp_points), np.nan)
    L = np.full((amp_points, wp_points), np.nan)
    for i, j, s, l in out:
        Z[i, j], L[i, j] = s, l

    bi, bj = np.unravel_index(np.nanargmax(Z), Z.shape)
    best = dict(amp_scale=float(amps[bi]), wp_offset_MHz=float(offsets[bj]),
                score=float(Z[bi, bj]), leakage=float(L[bi, bj]))
    return dict(offsets_MHz=offsets, amps=amps, Z=Z, leakage=L, best=best,
                metric=metric, peak_eta=float(eta_pk), engine="qutip")


def plot_map(result: Dict[str, Any], out: str = "figs/calibration_map.png",
             title: Optional[str] = None) -> None:
    r"""Render the 2D calibration landscape with the optimum marked.

    The y axis is the PHYSICAL drive, peak :math:`|\eta|`, with ``amp_scale`` on a
    twin axis on the right. The two differ only by the constant
    :math:`\eta_{\rm nom} = 1/(12 g_3 \lambda_a \lambda_b t_g)`, but they answer
    different questions: :math:`|\eta|` says which drive REGIME the point is in (how
    large the eta^2 / eta^3 terms are), and is therefore comparable between maps at
    different ``t_g``, whereas ``amp_scale`` is what ``--save-point`` writes into the
    device and is only meaningful at one ``t_g``. The dotted line marks
    ``amp_scale = 1``, i.e. the amplitude at which the LEADING-ORDER pulse area is
    exactly pi/2 -- so the distance of the optimum from that line is a direct read
    on how far first-order theory is off at this operating point.
    """
    offsets, amps, Z = result["offsets_MHz"], result["amps"], result["Z"]
    metric = result["metric"]
    best = result["best"]
    eta_nom = float(result["peak_eta"])              # peak |eta| at amp_scale = 1
    eta_axis = amps * eta_nom
    eta_best = best["amp_scale"] * eta_nom
    zlabel = ("iSWAP fidelity $F$" if metric == "fidelity"
              else r"transfer $P(|01\rangle\!\to\!|10\rangle)$")

    fig, ax = plt.subplots(figsize=(7.6, 5.3), dpi=200)
    pcm = ax.pcolormesh(offsets, eta_axis, Z, shading="nearest", cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax, pad=0.12)
    cb.set_label(zlabel)
    # the pi/2 (leading-order) amplitude
    if eta_axis.min() <= eta_nom <= eta_axis.max():
        ax.axhline(eta_nom, color="w", ls=":", lw=1.1, alpha=0.85)
        ax.text(offsets[0], eta_nom, r"  $\pi/2$ normalisation (amp scale 1)",
                color="w", va="bottom", ha="left", fontsize=7.5, alpha=0.9)
    # optimum
    ax.plot(best["wp_offset_MHz"], eta_best, marker="*", ms=18,
            mfc="#C0392B", mec="white", mew=1.2, zorder=5)
    lead = zlabel.split("$")[0].strip() or metric
    ax.annotate(f"opt: {best['wp_offset_MHz']:+.1f} MHz\n"
                f"$|\\eta|$ = {eta_best:.3f}  (amp scale {best['amp_scale']:.3f})\n"
                f"{lead} = {best['score']:.4f}",
                xy=(best["wp_offset_MHz"], eta_best),
                xytext=(0.98, 0.02), textcoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5, color="white",
                bbox=dict(boxstyle="round", fc="#00000099", ec="none"))
    ax.set_xlabel(r"pump frequency offset  $\delta\omega_p$  (MHz)")
    ax.set_ylabel(r"pump strength   peak $|\eta|$")
    # twin axis: the amp_scale that gets written to the device
    sec = ax.secondary_yaxis("right",
                             functions=(lambda e: e / eta_nom, lambda a: a * eta_nom))
    sec.set_ylabel(r"amp scale  ($\times\,\eta_{\rm nom}$, written to the device)")
    eng = result.get("engine", "reduced")
    t_g = result.get("t_g_ns")
    sub = f"({metric}, {eng}" + (f", $t_g$={t_g:.1f} ns, "
                                 f"$\\eta_{{\\rm nom}}$={eta_nom:.3f})" if t_g else ")")
    ax.set_title(title or f"iSWAP calibration landscape  {sub}", fontsize=10.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out, "and", out.rsplit(".", 1)[0] + ".pdf")


def save_npz(result: Dict[str, Any], path: str) -> None:
    """Persist the full scan (arrays + optimum + context + operating point) so the
    map can be re-plotted or post-processed without re-solving."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = dict(
        offsets_MHz=result["offsets_MHz"], amps=result["amps"], Z=result["Z"],
        peak_eta_axis=result["amps"] * float(result["peak_eta"]),
        metric=str(result["metric"]), peak_eta=float(result["peak_eta"]),
        engine=str(result.get("engine", "reduced")))
    if "leakage" in result:
        payload["leakage"] = result["leakage"]
    if "w_p_GHz" in result:
        payload["w_p_GHz"] = float(result["w_p_GHz"])
    if "t_g_ns" in result:
        payload["t_g_ns"] = float(result["t_g_ns"])
        payload["best_peak_eta"] = float(result["best"]["amp_scale"]
                                         * result["peak_eta"])
    for k, v in result["best"].items():
        payload[f"best_{k}"] = v
    for group in ("operating_point", "context"):
        if group in result:
            for k, v in result[group].items():
                payload[f"{group[:3]}_{k}"] = (np.nan if v is None else v)
    np.savez_compressed(path, **payload)
    print("saved", path)


def build_system(config: Dict[str, Any], t_g: float, *,
                 wa_GHz: Optional[float] = None, wb_GHz: Optional[float] = None,
                 delta_GHz: Optional[float] = None,
                 spec_abs_GHz: Optional[float] = None,
                 drag_beat_GHz: Optional[float] = None,
                 amp_scale: float = 1.0, wp_offset_GHz: float = 0.0):
    """Build the gate system for ANY sweep context (bare / spectator / target).

    The three sweep families differ only in which frequencies move, so they all
    reduce to a choice of (w_a, w_b, spectator):

    * bare / no-spectator : leave ``spec_abs_GHz`` and ``delta_GHz`` unset.
    * target sweep        : pass the swept ``wb_GHz`` (and ``spec_abs_GHz`` for the
                            absolute spectator placement).
    * spectator sweep     : pass ``delta_GHz`` = w_b - w_spec; the absolute
                            spectator frequency is derived as w_b - delta.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    t_g : float
        Gate duration (ns).
    wa_GHz, wb_GHz : float, optional
        Override the device pair (target sweeps vary w_b).
    delta_GHz : float, optional
        Spectator-sweep detuning w_b - w_spec (GHz). Mutually exclusive with
        ``spec_abs_GHz``.
    spec_abs_GHz : float, optional
        Absolute spectator frequency (GHz).
    drag_beat_GHz : float, optional
        Apply a DRAG quadrature at this beat, so the map is of the DRAG-on gate.
    amp_scale, wp_offset_GHz : float
        Nominal pump scaling/offset the scan grid is applied on top of.

    Returns
    -------
    (ZhouCoupler, float, float, dict)
        Coupler, pump frequency (GHz), peak |eta|, and the resolved context.
    """
    from device_utils import build_coupler

    pair = list(np.asarray(config["qubit_freqs_GHz"], dtype=float))
    if wa_GHz is not None:
        pair[0] = float(wa_GHz)
    if wb_GHz is not None:
        pair[1] = float(wb_GHz)
    if delta_GHz is not None:
        if spec_abs_GHz is not None:
            raise SystemExit("give either --delta-GHz or --spec-abs-GHz, not both")
        spec_abs_GHz = pair[1] - float(delta_GHz)      # spectator-sweep convention

    cfg = dict(config)
    cfg["qubit_freqs_GHz"] = pair
    cpl, w_p, eta_pk = build_coupler(cfg, t_g=t_g, amp_scale=amp_scale,
                                     wp_offset_GHz=wp_offset_GHz,
                                     spec_abs_GHz=spec_abs_GHz,
                                     drag_beat_GHz=drag_beat_GHz)
    context = dict(wa_GHz=pair[0], wb_GHz=pair[1], t_g_ns=t_g,
                   spec_abs_GHz=(None if spec_abs_GHz is None else float(spec_abs_GHz)),
                   drag_beat_GHz=drag_beat_GHz)
    return cpl, w_p, eta_pk, context


def run(config: Dict[str, Any], t_g: float, *, amp_scale: float = 1.0,
        wp_offset_GHz: float = 0.0, spec_abs_GHz: Optional[float] = None,
        wa_GHz: Optional[float] = None, wb_GHz: Optional[float] = None,
        delta_GHz: Optional[float] = None, drag_beat_GHz: Optional[float] = None,
        engine: str = "reduced", coupler_levels: Optional[int] = None,
        atol: float = 1e-10, rtol: float = 1e-8, nsteps: int = 500000,
        jobs: int = 1, save_npz_path: Optional[str] = None,
        out: Optional[str] = "figs/calibration_map.png", title: Optional[str] = None,
        **kw) -> Dict[str, Any]:
    """Build the system for the given sweep context, scan (reduced or QuTiP), plot,
    optionally persist, and return results.

    The returned dict includes ``context`` (where it was calibrated) and
    ``operating_point`` (a record ready for ``operating_points.save_point``).
    ``engine='qutip'`` runs the exact solver; both engines share the grid, the
    ``best``/``operating_point`` bookkeeping, and the plot.
    """
    if coupler_levels is not None:                     # truncation override, reused
        config = {**config, "coupler_levels": int(coupler_levels)}

    cpl, w_p, eta_pk, context = build_system(
        config, t_g, wa_GHz=wa_GHz, wb_GHz=wb_GHz, delta_GHz=delta_GHz,
        spec_abs_GHz=spec_abs_GHz, drag_beat_GHz=drag_beat_GHz,
        amp_scale=amp_scale, wp_offset_GHz=wp_offset_GHz)

    if engine == "qutip":
        # per-point rebuild through the SAME plumbing, grid folded onto the nominal
        def build_fn(grid_amp: float, grid_off_MHz: float):
            return build_system(
                config, t_g, wa_GHz=wa_GHz, wb_GHz=wb_GHz, delta_GHz=delta_GHz,
                spec_abs_GHz=spec_abs_GHz, drag_beat_GHz=drag_beat_GHz,
                amp_scale=amp_scale * grid_amp,
                wp_offset_GHz=wp_offset_GHz + grid_off_MHz * 1e-3)[:3]
        qkeys = ("wp_span_MHz", "wp_points", "amp_lo", "amp_hi", "amp_points", "metric")
        qkw = {k: v for k, v in kw.items() if k in qkeys}
        result = scan_qutip(build_fn, t_g, atol=atol, rtol=rtol, nsteps=nsteps,
                            jobs=jobs, **qkw)
    else:
        result = scan(cpl, 0, 1, t_g, **kw)

    result["w_p_GHz"] = w_p
    result["t_g_ns"] = t_g
    result["context"] = context
    best = result["best"]
    # the scan grid is relative to the nominal (amp_scale, wp_offset) it was built on
    result["operating_point"] = dict(
        amp_scale=amp_scale * best["amp_scale"],
        wp_offset_GHz=wp_offset_GHz + best["wp_offset_MHz"] * 1e-3,
        metric=result["metric"], score=best["score"], **context)
    if out:
        plot_map(result, out=out, title=title)
    if save_npz_path:
        save_npz(result, save_npz_path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", required=True)
    ap.add_argument("--t-g-ns", type=float, required=True)
    ap.add_argument("--wa-GHz", type=float, default=None,
                    help="override w_a (default: device pair)")
    ap.add_argument("--wb-GHz", type=float, default=None,
                    help="override w_b -- the target sweep's swept frequency")
    ap.add_argument("--spec-abs-GHz", type=float, default=None,
                    help="absolute spectator frequency (target-sweep convention)")
    ap.add_argument("--delta-GHz", type=float, default=None,
                    help="spectator-sweep detuning Delta = w_b - w_spec (GHz)")
    ap.add_argument("--drag-beat-GHz", type=float, default=None,
                    help="map the DRAG-on gate, with the quadrature at this beat")
    ap.add_argument("--engine", choices=["reduced", "qutip"], default="reduced",
                    help="reduced rotating-frame model (fast, CPU) or exact QuTiP "
                         "sesolve (pass --engine qutip for the trustworthy map)")
    ap.add_argument("--coupler-levels", type=int, default=None,
                    help="override device coupler truncation; bump at strong drive")
    ap.add_argument("--jobs", type=int, default=1,
                    help="joblib workers over the grid (QuTiP engine)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--save-point", default=None,
                    help="save the optimum into the device JSON under this name")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow --save-point to replace an existing point")
    ap.add_argument("--save-npz", default=None,
                    help="write the full scan (arrays + optimum + context) to this .npz")
    ap.add_argument("--wp-span-MHz", type=float, default=40.0)
    ap.add_argument("--wp-points", type=int, default=41)
    ap.add_argument("--amp-lo", type=float, default=0.6)
    ap.add_argument("--amp-hi", type=float, default=1.4)
    ap.add_argument("--amp-points", type=int, default=41)
    ap.add_argument("--cutoff-GHz", type=float, default=1.0,
                    help="reduced-engine carrier cutoff (ignored by --engine qutip)")
    ap.add_argument("--metric", choices=["fidelity", "transfer"], default="fidelity",
                    help="'transfer' colours by the swap population P(|01>->|10>)")
    ap.add_argument("--out", default="figs/calibration_map.png")
    args = ap.parse_args()

    from paths import resolve_device
    from device_utils import load_device
    device_path = resolve_device(args.device)
    cfg = load_device(device_path)
    result = run(cfg, args.t_g_ns, wa_GHz=args.wa_GHz, wb_GHz=args.wb_GHz,
                 spec_abs_GHz=args.spec_abs_GHz, delta_GHz=args.delta_GHz,
                 drag_beat_GHz=args.drag_beat_GHz, engine=args.engine,
                 coupler_levels=args.coupler_levels, atol=args.atol, rtol=args.rtol,
                 nsteps=args.nsteps, jobs=args.jobs, save_npz_path=args.save_npz,
                 wp_span_MHz=args.wp_span_MHz, wp_points=args.wp_points,
                 amp_lo=args.amp_lo, amp_hi=args.amp_hi, amp_points=args.amp_points,
                 cutoff_GHz=args.cutoff_GHz, metric=args.metric, out=args.out)
    b, ctx = result["best"], result["context"]
    print(f"context: w_a={ctx['wa_GHz']} w_b={ctx['wb_GHz']} t_g={ctx['t_g_ns']} ns "
          f"spec={ctx['spec_abs_GHz']} drag_beat={ctx['drag_beat_GHz']} engine={args.engine}")
    print(f"optimum: wp_offset = {b['wp_offset_MHz']:+.2f} MHz, "
          f"amp_scale = {b['amp_scale']:.4f}, {args.metric} = {b['score']:.5f}"
          + (f", leak = {b['leakage']:.4f}" if "leakage" in b else ""))
    if args.save_point:
        from operating_points import save_point
        rec = save_point(device_path, args.save_point, result["operating_point"],
                         overwrite=args.overwrite)
        print(f"saved operating point {args.save_point!r} to {device_path}: "
              f"amp_scale={rec['amp_scale']:.4f}, wp_offset={rec['wp_offset_GHz']:+.6f} GHz")
    if args.engine == "reduced":
        print("  (reduced model -- validate this (offset, amp) with --engine qutip)")


if __name__ == "__main__":
    main()