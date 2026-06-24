"""
figure8_data.py
===============

Reproduce the data behind Fig. 8 of McKinney et al., "Spectator-Aware Frequency
Allocation in Tunable-Coupler Quantum Architectures" (arXiv:2409.18262), but with
the EXACT Zhou dressed-mode coupler (zhou_coupler.ZhouCoupler) instead of the
paper's RWA effective Hamiltonian.

  Panel (a)  Population exchange P(|01> -> |10>) vs gate time and pump strength
             |eta| (constant drive), with the analytic full-iSWAP duration
             t_f(eta) = pi / (2 * 6 |eta| g3 lambda_a lambda_b) overlaid.
  Panel (b)  Coherent qubit-qubit-spectator infidelity 1 - F vs pump strength,
             for several spectator detunings delta_Q, computed WITH and WITHOUT
             DRAG. DRAG adds the second pump quadrature eps^y = -d(eps^x)/dt / delta_Q
             (Zhou SI Disc. 1) to cancel leakage into the detuned spectator; it
             requires a shaped pulse, so panel (b) uses a raised-cosine envelope.

Backends:
  dense  -- scipy on ZhouCoupler.hamiltonian_matrix (no QuTiP); good for small
            grids and in-sandbox validation.
  qutip  -- ZhouCoupler.evolve_trajectory / .iswap_fidelity (compiled sparse
            sesolve); for production grids on a compute node.

Parallelism: the (eta) columns of panel (a) and the (drag, delta_Q, eta) points
of panel (b) are independent and are evaluated across a process pool (--jobs N,
defaulting to $SLURM_CPUS_PER_TASK or the CPU count). Keep BLAS single-threaded
per worker (the SLURM script exports OMP_NUM_THREADS=1) to avoid oversubscription.

Usage
-----
    python figure8_data.py --mode both --backend qutip --jobs 16 \
        --out fig8.npz --plot fig8.png
See snail_figure8.slurm for the cluster job.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI: float = 2.0 * np.pi

# Local defaults (2-level qubits = McKinney's two-level swap picture).
FIG8_DEFAULTS: Dict[str, Any] = {
    "qubit_freqs_GHz": [5.00, 4.60],
    "coupler_freq_GHz": 7.00,
    "g3_GHz": 0.10,
    "g4_GHz": 0.0,
    "lam_a": 0.20, "lam_b": 0.20,
    "qubit_levels": 2,
    "coupler_levels": 5,
    "spec_levels": 2,
}


# ---------------------------------------------------------------------------
# Analytic helpers
# ---------------------------------------------------------------------------
def iswap_duration_ns(g3_GHz: float, lam_a: float, lam_b: float, eta: float,
                      n: int = 1) -> float:
    """Full (n=1) iSWAP duration for a CONSTANT pump of peak |eta|
    (McKinney Eq. 11 / Zhou Eq. 73): pi/(2n) = 6 t_f |eta| g3 lambda^2.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.
    eta : float
        Peak pump strength |eta| = sqrt(n_s).
    n : int, default 1
        Root index (1 = full iSWAP).

    Returns
    -------
    float
        Gate duration t_f (ns) for a constant pump.
    """
    g_eff = 6.0 * eta * (g3_GHz * TWO_PI) * lam_a * lam_b      # rad/ns
    return (np.pi / (2 * n)) / g_eff


def raised_cosine_iswap_duration(g3_GHz: float, lam_a: float, lam_b: float,
                                 eta: float) -> float:
    """Full-iSWAP duration for a RAISED-COSINE pump of peak |eta|. The Hann window
    integral is eta*t_g/2, so it needs twice the constant-pump duration.

    Parameters
    ----------
    g3_GHz : float
        Three-wave non-linearity g3 (GHz).
    lam_a, lam_b : float
        Qubit participations.
    eta : float
        Peak pump strength |eta|.

    Returns
    -------
    float
        Gate duration t_g (ns) for a raised-cosine pump.
    """
    return 2.0 * iswap_duration_ns(g3_GHz, lam_a, lam_b, eta)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """FIG8_DEFAULTS merged with an optional device JSON (run_sweep-style keys).

    Parameters
    ----------
    path : str or None
        Device JSON path, or None for the built-in defaults.

    Returns
    -------
    dict
        Merged configuration.
    """
    config = dict(FIG8_DEFAULTS)
    if path:
        with open(path) as f:
            config.update(json.load(f))
    return config


# ---------------------------------------------------------------------------
# Coupler construction
# ---------------------------------------------------------------------------
def build_gate(config: Dict[str, Any], eta: float, t_env: float,
               spectator: Optional[Tuple[float, float]] = None,
               wp_offset_GHz: float = 0.0, envelope: str = "constant",
               drag: bool = False, delta_drag_GHz: Optional[float] = None):
    """Build the coupler with a pump of peak |eta| (no normalization, so
    peak_eta() == eta) on the (a, b) iSWAP. Optionally add one qubit-qubit
    spectator detuned from the pump by delta_Q, choose the envelope, and enable
    DRAG.

    Parameters
    ----------
    config : dict
        Merged configuration.
    eta : float
        Pump strength |eta| to set directly.
    t_env : float
        Pulse duration (ns).
    spectator : tuple(float, float), optional
        (delta_Q_GHz, lam_spec): a 2-level spectator whose exchange with qubit a is
        off-resonant by delta_Q (placed at w_spec = w_b - (w_p + delta_Q)).
    wp_offset_GHz : float, default 0.0
        Offset added to the pump frequency w_b - w_a (GHz).
    envelope : {"constant", "raised_cosine"}, default "constant"
        Pump envelope shape. DRAG requires "raised_cosine" (nonzero derivative).
    drag : bool, default False
        Enable the DRAG second quadrature.
    delta_drag_GHz : float, optional
        DRAG detuning (typically the spectator detuning delta_Q).

    Returns
    -------
    (ZhouCoupler, float)
        The coupler and the analytic iSWAP rate g_eff = 6 g3 lambda^2 |eta| (rad/ns).
    """
    from zhou_coupler import ZhouCoupler, PumpTone, RaisedCosine, ConstantPulse

    wa, wb = (float(config["qubit_freqs_GHz"][0]), float(config["qubit_freqs_GHz"][1]))
    ws = float(config["coupler_freq_GHz"])
    w_p_GHz = abs(wb - wa) + wp_offset_GHz
    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    freqs = [wa, wb, ws]
    participations = {0: float(config["lam_a"]), 1: float(config["lam_b"])}
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"])]
    if spectator is not None:
        delta_Q_GHz, lam_spec = spectator
        freqs.append(wb - (w_p_GHz + delta_Q_GHz))
        participations[3] = float(lam_spec)
        levels.append(int(config["spec_levels"]))

    cpl = ZhouCoupler(mode_freqs_GHz=freqs, coupler_index=2,
                      participations=participations, nonlinearities=nonlin, levels=levels)
    EnvCls = RaisedCosine if envelope == "raised_cosine" else ConstantPulse
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=EnvCls(amp=eta, t_g=t_env),
                          is_eta=True, drag=drag, delta_drag_GHz=delta_drag_GHz),
                 normalize_iswap=None)
    return cpl, cpl.iswap_rate(0, 1)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _trajectory(cpl, init_occ: Sequence[int], times: np.ndarray, backend: str,
                solver: Dict[str, Any], max_step: float) -> np.ndarray:
    """State vector at each time in `times` (shape (len(times), dim))."""
    if backend == "qutip":
        return cpl.evolve_trajectory(init_occ, times, **solver)
    from scipy.integrate import solve_ivp
    y0 = np.zeros(cpl.dim, dtype=complex)
    y0[cpl.fock_index(list(init_occ))] = 1.0
    sol = solve_ivp(lambda t, y: -1j * (cpl.hamiltonian_matrix(t) @ y),
                    (0.0, float(times[-1])), y0, t_eval=times,
                    rtol=solver["rtol"], atol=solver["atol"], max_step=max_step, method="RK45")
    return sol.y.T


def _propagator4(cpl, a: int, b: int, t_g: float, backend: str,
                 solver: Dict[str, Any], max_step: float) -> np.ndarray:
    """4x4 projected propagator on the (a, b) computational subspace."""
    if backend == "qutip":
        return cpl.propagator_columns(a, b, t_g, **solver)
    from scipy.integrate import solve_ivp
    idx = cpl._subspace_indices(a, b)
    U = np.zeros((4, 4), dtype=complex)
    for col, start in enumerate(idx):
        y0 = np.zeros(cpl.dim, dtype=complex); y0[start] = 1.0
        sol = solve_ivp(lambda t, y: -1j * (cpl.hamiltonian_matrix(t) @ y),
                        (0.0, t_g), y0, rtol=solver["rtol"], atol=solver["atol"],
                        max_step=max_step, method="RK45")
        psi = sol.y[:, -1]
        for row, end in enumerate(idx):
            U[row, col] = psi[end]
    return U


# ---------------------------------------------------------------------------
# Parallel map helper
# ---------------------------------------------------------------------------
def _resolve_jobs(n_jobs: Optional[int]) -> int:
    """Number of worker processes: explicit n_jobs, else $SLURM_CPUS_PER_TASK,
    else the CPU count."""
    if n_jobs and n_jobs > 0:
        return int(n_jobs)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def _pmap(func, arg_list: List[Any], n_jobs: Optional[int]) -> List[Any]:
    """Apply `func` to each item of `arg_list`, in parallel over a process pool
    (order preserved). Falls back to a serial loop for n_jobs<=1 or a single item.

    Parameters
    ----------
    func : callable
        A module-level worker taking one (picklable) argument.
    arg_list : list
        Argument tuples, one per task.
    n_jobs : int or None
        Worker count (see _resolve_jobs).

    Returns
    -------
    list
        Results in input order.
    """
    jobs = _resolve_jobs(n_jobs)
    if jobs <= 1 or len(arg_list) <= 1:
        return [func(a) for a in arg_list]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(func, arg_list))


# ---------------------------------------------------------------------------
# Panel (a): population-exchange map
# ---------------------------------------------------------------------------
def _map_column_worker(args: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One eta column of the population map (P10, P01, n_coupler, leak over time)."""
    config, eta, times, backend, solver, max_step = args
    cpl, _ = build_gate(config, eta, t_env=float(times[-1]))
    i00 = cpl.fock_index([0, 0, 0]); i01 = cpl.fock_index([0, 1, 0])
    i10 = cpl.fock_index([1, 0, 0]); i11 = cpl.fock_index([1, 1, 0])
    states = _trajectory(cpl, [0, 1, 0], times, backend, solver, max_step)
    pops = np.abs(states) ** 2
    ncpl = np.array([cpl.mean_occupation(p, 2) for p in pops])
    leak = 1.0 - (pops[:, i00] + pops[:, i01] + pops[:, i10] + pops[:, i11])
    return pops[:, i10], pops[:, i01], ncpl, leak


def population_map(config: Dict[str, Any], eta_grid: np.ndarray, times: np.ndarray,
                   backend: str = "dense", solver: Optional[Dict[str, Any]] = None,
                   max_step: float = 0.02, n_jobs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """P(|01> -> |10>) over the (eta, time) grid (parallel over eta), plus
    diagnostics and the analytic full-iSWAP line t_f(eta).

    Parameters
    ----------
    config : dict
        Merged configuration.
    eta_grid : ndarray
        Pump strengths |eta| to scan (x-axis).
    times : ndarray
        Output times in ns (y-axis); sorted, starting at 0.
    backend : {"dense", "qutip"}
        Solver backend.
    solver : dict, optional
        Tolerances (atol, rtol, nsteps); defaults provided.
    max_step : float, default 0.02
        Dense-backend max ODE step (ns).
    n_jobs : int, optional
        Worker processes (see _resolve_jobs).

    Returns
    -------
    dict
        eta_grid, times, P10 / P01 / n_coupler / leak [n_eta, n_t], and t_f (n_eta).
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    args = [(config, float(e), times, backend, solver, max_step) for e in eta_grid]
    results = _pmap(_map_column_worker, args, n_jobs)
    P10 = np.array([r[0] for r in results]); P01 = np.array([r[1] for r in results])
    ncpl = np.array([r[2] for r in results]); leak = np.array([r[3] for r in results])
    t_f = np.array([iswap_duration_ns(config["g3_GHz"], config["lam_a"],
                                      config["lam_b"], float(e)) for e in eta_grid])
    return {"eta_grid": np.asarray(eta_grid), "times": np.asarray(times),
            "P10": P10, "P01": P01, "n_coupler": ncpl, "leak": leak, "t_f": t_f}


# ---------------------------------------------------------------------------
# Panel (b): spectator coherent infidelity vs eta, with/without DRAG
# ---------------------------------------------------------------------------
def _spectator_point_worker(args: Tuple) -> Tuple[float, float]:
    """One (drag, delta_Q, eta) spectator point: returns (infidelity, leakage)."""
    config, eta, delta_Q, lam_spec, drag, backend, solver, max_step = args
    t_g = raised_cosine_iswap_duration(config["g3_GHz"], config["lam_a"],
                                       config["lam_b"], eta)
    cpl, _ = build_gate(config, eta, t_env=t_g, spectator=(delta_Q, lam_spec),
                        envelope="raised_cosine", drag=drag,
                        delta_drag_GHz=(delta_Q if drag else None))
    U = _propagator4(cpl, 0, 1, t_g, backend, solver, max_step)
    F, lk = cpl._iswap_fidelity_from_U(U, True)
    return 1.0 - F, lk


def spectator_infidelity(config: Dict[str, Any], eta_grid: np.ndarray,
                         delta_Q_GHz: Sequence[float], lam_spec: float = 0.20,
                         drags: Sequence[bool] = (False, True),
                         backend: str = "dense", solver: Optional[Dict[str, Any]] = None,
                         max_step: float = 0.02, n_jobs: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Coherent iSWAP infidelity 1 - F vs pump strength, for each spectator detuning
    and DRAG setting, on a raised-cosine gate at its full-iSWAP duration t_g(eta).
    Parallel over the flattened (drag, delta_Q, eta) grid.

    Parameters
    ----------
    config : dict
        Merged configuration.
    eta_grid : ndarray
        Pump strengths |eta| to scan.
    delta_Q_GHz : sequence of float
        Spectator detunings from the pump (GHz).
    lam_spec : float, default 0.20
        Spectator participation.
    drags : sequence of bool, default (False, True)
        DRAG settings to compare.
    backend : {"dense", "qutip"}
        Solver backend.
    solver : dict, optional
        Tolerances; defaults provided.
    max_step : float, default 0.02
        Dense-backend max ODE step (ns).
    n_jobs : int, optional
        Worker processes (see _resolve_jobs).

    Returns
    -------
    dict
        eta_grid, delta_Q_GHz, drags, and infidelity / leak [n_drag, n_delta, n_eta].
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    deltas = np.asarray(delta_Q_GHz, dtype=float)
    drags = list(drags)
    args = [(config, float(e), float(d), lam_spec, bool(dr), backend, solver, max_step)
            for dr in drags for d in deltas for e in eta_grid]
    results = _pmap(_spectator_point_worker, args, n_jobs)
    infid = np.array([r[0] for r in results]).reshape(len(drags), len(deltas), len(eta_grid))
    leak = np.array([r[1] for r in results]).reshape(len(drags), len(deltas), len(eta_grid))
    return {"eta_grid": np.asarray(eta_grid), "delta_Q_GHz": deltas,
            "drags": np.array(drags, dtype=bool), "infidelity": infid, "leak": leak}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_figure8(npz_path: str, png_path: str) -> None:
    """Render Fig.8-style panels from a saved .npz (map and/or spectator data).
    Panel (b) overlays DRAG-off (solid) and DRAG-on (dashed) per detuning.

    Parameters
    ----------
    npz_path : str
        Path to the .npz produced by main().
    png_path : str
        Output PNG path.

    Returns
    -------
    None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from plot_results import set_literature_style
        set_literature_style()
    except Exception:
        pass

    data = np.load(npz_path)
    has_map = "P10" in data.files
    has_spec = "infidelity" in data.files
    n_panels = int(has_map) + int(has_spec)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(6.2 * max(n_panels, 1), 4.6),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    ax_i = 0

    if has_map:
        ax = axes[ax_i]; ax_i += 1
        eta = data["eta_grid"]; t = data["times"]; P10 = data["P10"]
        mesh = ax.pcolormesh(eta, t, P10.T, shading="auto", cmap="viridis", vmin=0, vmax=1)
        ax.plot(eta, data["t_f"], "r--", lw=1.8, label=r"full iSWAP $t_f(\eta)$")
        ax.set_xlabel(r"pump strength $|\eta|$"); ax.set_ylabel("gate time (ns)")
        ax.set_ylim(t[0], t[-1]); ax.set_title(r"(a) population exchange $P(|01\rangle\to|10\rangle)$")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.colorbar(mesh, ax=ax, label=r"$P(|10\rangle)$")

    if has_spec:
        ax = axes[ax_i]; ax_i += 1
        eta = data["spec_eta_grid"] if "spec_eta_grid" in data.files else data["eta_grid"]
        infid = data["infidelity"]; deltas = data["delta_Q_GHz"]
        drags = data["drags"] if "drags" in data.files else np.array([False])
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
        for jd, dq in enumerate(deltas):
            color = colors[jd % len(colors)] if colors else None
            for kd, dr in enumerate(drags):
                style = "--" if bool(dr) else "-"
                tag = " +DRAG" if bool(dr) else ""
                ax.semilogy(eta, np.clip(infid[kd, jd], 1e-6, None), style, color=color,
                            marker="o", ms=3, label=fr"$\delta_Q={dq*1e3:.0f}$ MHz{tag}")
        ax.set_xlabel(r"pump strength $|\eta|$"); ax.set_ylabel(r"coherent infidelity $1-F$")
        ax.set_title("(b) qubit-qubit spectator infidelity"); ax.legend(framealpha=0.9, fontsize=8)

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate Fig.8 data with the exact Zhou coupler and save to .npz (+ optional PNG)."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help="device JSON (overrides FIG8_DEFAULTS)")
    ap.add_argument("--mode", choices=["map", "spectator", "both"], default="both")
    ap.add_argument("--backend", choices=["dense", "qutip"], default="dense")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = SLURM_CPUS_PER_TASK or CPU count)")
    ap.add_argument("--out", default="fig8.npz", help="output .npz")
    ap.add_argument("--plot", default=None, help="optional output PNG")
    # panel (a) grid
    ap.add_argument("--eta-min", type=float, default=0.1)
    ap.add_argument("--eta-max", type=float, default=2.4)
    ap.add_argument("--eta-points", type=int, default=24)
    ap.add_argument("--t-max", type=float, default=None, help="time axis max (ns); default ~1.3 t_f(eta_min)")
    ap.add_argument("--t-points", type=int, default=120)
    # panel (b) spectator
    ap.add_argument("--deltas-MHz", default="100,200,400", help="comma list of spectator detunings (MHz)")
    ap.add_argument("--lam-spec", type=float, default=0.20)
    ap.add_argument("--no-drag-compare", action="store_true", help="only compute DRAG-off")
    # solver
    ap.add_argument("--max-step", type=float, default=0.02, help="dense-backend max ODE step (ns)")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--nsteps", type=int, default=500000)
    args = ap.parse_args()

    config = load_config(args.device)
    solver = {"atol": args.atol, "rtol": args.rtol, "nsteps": args.nsteps}
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.eta_points)
    out: Dict[str, np.ndarray] = {}
    print(f"jobs={_resolve_jobs(args.jobs)} backend={args.backend}")

    if args.mode in ("map", "both"):
        t_max = args.t_max or 1.3 * iswap_duration_ns(config["g3_GHz"], config["lam_a"],
                                                      config["lam_b"], args.eta_min)
        times = np.linspace(0.0, t_max, args.t_points)
        print(f"[map] eta {args.eta_min}-{args.eta_max} x{args.eta_points}, t 0-{t_max:.1f} ns x{args.t_points}")
        out.update(population_map(config, eta_grid, times, args.backend, solver,
                                  args.max_step, n_jobs=args.jobs))

    if args.mode in ("spectator", "both"):
        deltas = [float(x) / 1000.0 for x in args.deltas_MHz.split(",")]
        drags = (False,) if args.no_drag_compare else (False, True)
        print(f"[spectator] deltas {args.deltas_MHz} MHz, lam_spec={args.lam_spec}, drags={drags}")
        spec = spectator_infidelity(config, eta_grid, deltas, args.lam_spec, drags,
                                    args.backend, solver, args.max_step, n_jobs=args.jobs)
        out.update({"infidelity": spec["infidelity"], "delta_Q_GHz": spec["delta_Q_GHz"],
                    "drags": spec["drags"], "spec_leak": spec["leak"],
                    "eta_grid": spec["eta_grid"], "spec_eta_grid": spec["eta_grid"]})

    np.savez(args.out, **out)
    print(f"written {args.out}")
    if args.plot:
        plot_figure8(args.out, args.plot)
        print(f"plotted {args.plot}")


if __name__ == "__main__":
    main()