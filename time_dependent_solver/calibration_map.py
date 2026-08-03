"""2D calibration landscape: pump frequency offset vs pump strength.

Scans the (pump-frequency-offset, pump-amplitude) plane and evaluates the
leakage-aware iSWAP fidelity (or the |01>->|10> transfer probability) at every
point, so you can read off the global optimum directly instead of relying on
alternating 1D amplitude/frequency scans (which rail when the AC-Stark shift and
the Rabi amplitude feed back on each other). Propagation uses the same fast
rotating-frame reduced model as ``grape.py`` (scipy expm, no QuTiP): the pump
offset shifts the precomputed carriers, so the operator basis is built once.

The output is a heatmap with the optimum marked. Validate the chosen
(offset, amp) in the full QuTiP ``iswap_fidelity`` on the cluster.

CLI
---
    python calibration_map.py --device warren_device.json --t-g-ns 92.6 \
        --wp-span-MHz 40 --amp-lo 0.6 --amp-hi 1.4 [--metric transfer]
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

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
    """Scan (pump offset, amplitude) and score each point.

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
                n_sub=n_sub, cutoff_GHz=cutoff_GHz, peak_eta=peak)


def plot_map(result: Dict[str, Any], out: str = "figs/calibration_map.png",
             title: Optional[str] = None) -> None:
    """Render the 2D calibration landscape with the optimum marked."""
    offsets, amps, Z = result["offsets_MHz"], result["amps"], result["Z"]
    metric = result["metric"]
    best = result["best"]
    zlabel = "iSWAP fidelity $F$" if metric == "fidelity" else r"transfer $P(|01\rangle\!\to\!|10\rangle)$"

    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=200)
    pcm = ax.pcolormesh(offsets, amps, Z, shading="nearest", cmap="viridis")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(zlabel)
    # optimum
    ax.plot(best["wp_offset_MHz"], best["amp_scale"], marker="*", ms=18,
            mfc="#C0392B", mec="white", mew=1.2, zorder=5)
    ax.annotate(f"opt: {best['wp_offset_MHz']:+.1f} MHz, "
                f"amp {best['amp_scale']:.3f}\n{zlabel.split('$')[0].strip()} "
                f"= {best['score']:.4f}",
                xy=(best["wp_offset_MHz"], best["amp_scale"]),
                xytext=(0.98, 0.02), textcoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color="white",
                bbox=dict(boxstyle="round", fc="#00000088", ec="none"))
    ax.set_xlabel(r"pump frequency offset  $\delta\omega_p$  (MHz)")
    ax.set_ylabel(r"pump strength  (amp scale, $\times\,\eta_{\rm nom}$)")
    ax.set_title(title or f"iSWAP calibration landscape  ({metric})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out, "and", out.rsplit(".", 1)[0] + ".pdf")


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
        out: Optional[str] = "figs/calibration_map.png", title: Optional[str] = None,
        **kw) -> Dict[str, Any]:
    """Build the system for the given sweep context, scan, plot, and return results.

    The returned dict includes ``context`` (where it was calibrated) and
    ``operating_point`` (a record ready for ``operating_points.save_point``).
    """
    cpl, w_p, eta_pk, context = build_system(
        config, t_g, wa_GHz=wa_GHz, wb_GHz=wb_GHz, delta_GHz=delta_GHz,
        spec_abs_GHz=spec_abs_GHz, drag_beat_GHz=drag_beat_GHz,
        amp_scale=amp_scale, wp_offset_GHz=wp_offset_GHz)
    result = scan(cpl, 0, 1, t_g, **kw)
    result["w_p_GHz"] = w_p
    result["context"] = context
    best = result["best"]
    # the scan grid is relative to the nominal (amp_scale, wp_offset) it was built on
    result["operating_point"] = dict(
        amp_scale=amp_scale * best["amp_scale"],
        wp_offset_GHz=wp_offset_GHz + best["wp_offset_MHz"] * 1e-3,
        metric=result["metric"], score=best["score"], **context)
    if out:
        plot_map(result, out=out, title=title)
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
    ap.add_argument("--save-point", default=None,
                    help="save the optimum into the device JSON under this name")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow --save-point to replace an existing point")
    ap.add_argument("--wp-span-MHz", type=float, default=40.0)
    ap.add_argument("--wp-points", type=int, default=41)
    ap.add_argument("--amp-lo", type=float, default=0.6)
    ap.add_argument("--amp-hi", type=float, default=1.4)
    ap.add_argument("--amp-points", type=int, default=41)
    ap.add_argument("--cutoff-GHz", type=float, default=1.0)
    ap.add_argument("--metric", choices=["fidelity", "transfer"], default="fidelity")
    ap.add_argument("--out", default="figs/calibration_map.png")
    args = ap.parse_args()

    from paths import resolve_device
    from device_utils import load_device
    device_path = resolve_device(args.device)
    cfg = load_device(device_path)
    result = run(cfg, args.t_g_ns, wa_GHz=args.wa_GHz, wb_GHz=args.wb_GHz,
                 spec_abs_GHz=args.spec_abs_GHz, delta_GHz=args.delta_GHz,
                 drag_beat_GHz=args.drag_beat_GHz,
                 wp_span_MHz=args.wp_span_MHz, wp_points=args.wp_points,
                 amp_lo=args.amp_lo, amp_hi=args.amp_hi, amp_points=args.amp_points,
                 cutoff_GHz=args.cutoff_GHz, metric=args.metric, out=args.out)
    b, ctx = result["best"], result["context"]
    print(f"context: w_a={ctx['wa_GHz']} w_b={ctx['wb_GHz']} t_g={ctx['t_g_ns']} ns "
          f"spec={ctx['spec_abs_GHz']} drag_beat={ctx['drag_beat_GHz']}")
    print(f"optimum: wp_offset = {b['wp_offset_MHz']:+.2f} MHz, "
          f"amp_scale = {b['amp_scale']:.4f}, {args.metric} = {b['score']:.5f}")
    if args.save_point:
        from operating_points import save_point
        rec = save_point(device_path, args.save_point, result["operating_point"],
                         overwrite=args.overwrite)
        print(f"saved operating point {args.save_point!r} to {device_path}: "
              f"amp_scale={rec['amp_scale']:.4f}, wp_offset={rec['wp_offset_GHz']:+.6f} GHz")
    print("  (validate this (offset, amp) in the full QuTiP iswap_fidelity)")


if __name__ == "__main__":
    main()