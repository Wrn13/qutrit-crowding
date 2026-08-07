"""
validate_engines.py
===================

Measure the batched engine (`jax_engine`) against the QuTiP reference, instead of
assuming they agree.

This exists because two choices in the engine are approximations that must be
justified by numbers, not by argument:

1. **cutoff_GHz** -- pruning fast carriers is what makes the engine fast, but it
   is a rotating-wave reduction, not an identity. ``--cutoff-scan`` reports
   fidelity and ``||U_engine - U_qutip||`` across a range of cutoffs so the
   production value is chosen from data and the residual error is a REPORTED
   number.
2. **precision** -- ``f32`` is mixed (float64 phases, complex64 state). Whether
   it is worth using is a property of the node, so ``--precision-scan`` reports
   both the accuracy delta and the measured wall-clock speedup. If the speedup is
   marginal, the honest answer is to stay on f64.

Usage
-----
    python validate_engines.py --device 2Gate4.9SNAIL.json --cutoff-scan
    python validate_engines.py --device 2Gate4.9SNAIL.json --precision-scan
    python validate_engines.py --device 2Gate4.9SNAIL.json --batch-scan 8,32,128

Notes
-----
The QuTiP reference is expensive (four exact ``sesolve`` runs per point), so the
default device is deliberately small unless ``--spec-abs-GHz`` adds the
spectator. ``--quick`` shrinks the mode truncation for a fast smoke test.
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional

import numpy as np

TWO_PI = 2.0 * np.pi


def _build(args, levels: Optional[List[int]] = None):
    """A coupler from a device JSON, with optional truncation override."""
    from device_utils import build_coupler, load_device
    from paths import resolve_device

    cfg = load_device(resolve_device(args.device))
    if levels is not None:
        cfg = dict(cfg)
        cfg["qubit_levels"], cfg["coupler_levels"] = levels[0], levels[1]
    t_g = float(args.t_g_ns if args.t_g_ns is not None else cfg["t_g_ns"])
    chirp = ([float(x) for x in args.chirp_GHz.split(",") if x.strip()]
             if args.chirp_GHz else None)
    cpl, w_p, eta = build_coupler(cfg, t_g, float(cfg["amp_scale"]),
                                  float(cfg["wp_offset_GHz"]),
                                  spec_abs_GHz=args.spec_abs_GHz,
                                  chirp_coeffs_GHz=chirp)
    return cfg, cpl, t_g, w_p, eta


def _reference(cpl, t_g: float, atol: float, rtol: float):
    """The QuTiP truth: exact Hamiltonian, four sesolve columns."""
    t0 = time.time()
    U = cpl.propagator_columns(0, 1, t_g, atol=atol, rtol=rtol, nsteps=500_000)
    return U, time.time() - t0


def _score(U):
    from zhou_coupler import ZhouCoupler
    return ZhouCoupler._iswap_fidelity_from_U(U, True)


def cutoff_scan(args) -> None:
    """Fidelity and ||dU|| vs cutoff_GHz -- the table that picks the production cutoff."""
    import jax_engine as JE

    levels = [2, 3] if args.quick else None
    _cfg, cpl, t_g, w_p, eta = _build(args, levels)
    print(f"device={args.device}  dim={cpl.dim}  t_g={t_g:.3f} ns  w_p={w_p:.6f} GHz  "
          f"|eta|={eta:.4f}")

    U_ref, dt_ref = _reference(cpl, t_g, args.atol, args.rtol)
    F_ref, leak_ref = _score(U_ref)
    print(f"reference (QuTiP sesolve, exact): F={F_ref:.10f}  leak={leak_ref:.3e}  "
          f"[{dt_ref:.1f}s]\n")

    cutoffs = [float(x) for x in args.cutoffs.split(",")] if args.cutoffs else \
        [1.0, 2.0, 3.0, 5.0, 10.0, np.inf]
    print(f"{'cutoff':>8}  {'terms':>6}  {'nnz':>7}  {'steps':>7}  {'F':>12}  "
          f"{'F-F_ref':>10}  {'max|dU|':>10}  {'time':>8}")
    print("-" * 82)
    for cut in cutoffs:
        eng = JE.build_engine(cpl, cutoff_GHz=cut, precision=args.precision)
        n_steps, _o, _s = eng.step_plan(t_g, carrier_resolution=args.carrier_resolution)
        t0 = time.time()
        U = JE.propagator_columns(eng, t_g, carrier_resolution=args.carrier_resolution)
        dt = time.time() - t0
        F, _leak = _score(np.asarray(U))
        print(f"{cut:>8.1f}  {eng.n_terms:>6d}  {eng.val.size:>7d}  {n_steps:>7d}  "
              f"{F:>12.9f}  {F - F_ref:>+10.2e}  {np.max(np.abs(U - U_ref)):>10.2e}  "
              f"{dt:>7.2f}s")
    print(f"\nPick the smallest cutoff whose max|dU| is below the sweep's own ODE "
          f"tolerance (atol={args.atol:g}).")


def precision_scan(args) -> None:
    """f32-mixed vs f64: accuracy delta AND measured speedup, on this node."""
    import jax_engine as JE

    levels = [2, 3] if args.quick else None
    _cfg, cpl, t_g, _w_p, _eta = _build(args, levels)
    cut = float(args.cutoff)
    print(f"device={args.device}  dim={cpl.dim}  cutoff={cut} GHz  t_g={t_g:.3f} ns")

    ref = None
    print(f"\n{'precision':>10}  {'F':>12}  {'max|dU| vs f64':>16}  {'time':>9}  {'speedup':>8}")
    print("-" * 64)
    base_t = None
    for prec in ("f64", "f32"):
        eng = JE.build_engine(cpl, cutoff_GHz=cut, precision=prec)
        t0 = time.time()
        U = np.asarray(JE.propagator_columns(eng, t_g,
                                             carrier_resolution=args.carrier_resolution))
        dt = time.time() - t0
        if ref is None:
            ref, base_t = U, dt
        F, _ = _score(U)
        print(f"{prec:>10}  {F:>12.9f}  {np.max(np.abs(U - ref)):>16.2e}  {dt:>8.2f}s  "
              f"{base_t / dt:>7.2f}x")
    print("\nIf the speedup is below ~1.3x, stay on f64: at this Hilbert-space size the\n"
          "step is a small scatter-add (launch/bandwidth bound), not a dense GEMM, so a\n"
          "narrower float buys much less than the tensor-core figures suggest.")


def batch_scan(args) -> None:
    """Per-point cost vs batch size -- the actual case for the GPU path."""
    import jax_engine as JE

    levels = [2, 3] if args.quick else None
    _cfg, cpl, t_g, w_p, _eta = _build(args, levels)
    eng = JE.build_engine(cpl, cutoff_GHz=float(args.cutoff), precision=args.precision)
    print(f"device={args.device}  dim={cpl.dim}  {eng!r}")

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("jax not available; skipping batch scan")
        return
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    sizes = [int(x) for x in args.batch_scan.split(",")]
    base = np.asarray(eng.omega_vec0)
    print(f"\n{'batch':>7}  {'total':>10}  {'per point':>11}  {'vs serial':>10}")
    print("-" * 44)
    serial = None
    for B in sizes:
        # a pump-offset scan: the pump lives in the same frequency vector, so this
        # is the calibration-map / Stark-chevron workload verbatim
        off = np.linspace(-0.005, 0.005, B)
        batch = np.repeat(base[None, :], B, axis=0)
        batch[:, -1] = base[-1] + off * TWO_PI
        U = JE.propagator_columns_batched(eng, t_g, omega_batch=batch,
                                          carrier_resolution=args.carrier_resolution)
        U.block_until_ready()                       # exclude async dispatch
        t0 = time.time()
        U = JE.propagator_columns_batched(eng, t_g, omega_batch=batch,
                                          carrier_resolution=args.carrier_resolution)
        U.block_until_ready()
        dt = time.time() - t0
        if serial is None:
            serial = dt / B
        print(f"{B:>7d}  {dt:>9.2f}s  {dt / B:>10.4f}s  {serial / (dt / B):>9.2f}x")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="2Gate4.9SNAIL.json")
    ap.add_argument("--t-g-ns", type=float, default=None)
    ap.add_argument("--spec-abs-GHz", type=float, default=None,
                    help="add the spectator as a 4th mode at this absolute frequency")
    ap.add_argument("--chirp-GHz", default=None,
                    help="comma list of Legendre chirp coefficients (GHz)")
    ap.add_argument("--cutoff-scan", action="store_true")
    ap.add_argument("--precision-scan", action="store_true")
    ap.add_argument("--batch-scan", default=None, help="comma list of batch sizes")
    ap.add_argument("--cutoffs", default=None, help="comma list for --cutoff-scan")
    ap.add_argument("--cutoff", type=float, default=float("inf"),
                    help="cutoff used by --precision-scan / --batch-scan")
    ap.add_argument("--precision", choices=["f64", "f32"], default="f64")
    ap.add_argument("--carrier-resolution", type=float, default=0.1,
                    help="max |Omega| * dt for the fixed-step propagator")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--quick", action="store_true",
                    help="truncate to 2-level qubits / 3-level coupler for a fast check")
    args = ap.parse_args()

    if not (args.cutoff_scan or args.precision_scan or args.batch_scan):
        args.cutoff_scan = True
    if args.cutoff_scan:
        cutoff_scan(args)
    if args.precision_scan:
        precision_scan(args)
    if args.batch_scan:
        batch_scan(args)


if __name__ == "__main__":
    main()
