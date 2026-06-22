"""
figure8_data.py
===============

Reproduce the data behind Fig. 8 of McKinney et al., "Spectator-Aware Frequency
Allocation in Tunable-Coupler Quantum Architectures" (arXiv:2409.18262), but with
the EXACT Zhou dressed-mode coupler (zhou_coupler.ZhouCoupler) instead of the
paper's RWA effective Hamiltonian.

  Panel (a)  Population exchange P(|01> -> |10>) as a function of gate time and
             pump strength |eta|, with the analytic full-iSWAP duration
             t_f(eta) = pi / (2 * 6 |eta| g3 lambda_a lambda_b)  (McKinney Eq. 11,
             = Zhou Eq. 73) overlaid as the dashed line.
  Panel (b)  Coherent qubit-qubit-spectator infidelity 1 - F vs pump strength,
             for several spectator detunings delta_Q (McKinney Eq. 12 / Fig. 8b),
             using the same spectator convention as run_sweep_zhou.py.

The point of using the exact coupler: in McKinney's RWA model the target is a full
iSWAP at the dashed line for ANY eta (eta only sets speed), so panel (a) is a clean
fan of swaps. The exact integration additionally carries the strong subharmonic /
counter-rotating SNAIL self-terms (his Table I, ~100x the target) that he brackets
out as a breakdown constraint -- so the exact map should track the dashed line at
small eta and DEVIATE (incomplete / frequency-pulled swaps) as eta grows, which is
precisely the effect that motivates per-device pump calibration.

Two backends:
  dense  -- scipy on ZhouCoupler.hamiltonian_matrix (the dense oracle); no QuTiP,
            good for small grids and in-sandbox validation.
  qutip  -- ZhouCoupler.evolve_trajectory / .iswap_fidelity (compiled sparse
            sesolve); use on a compute node for production grids.

Usage
-----
    python figure8_data.py --mode both --backend qutip --out fig8.npz --plot fig8.png
    python figure8_data.py --mode map  --backend dense --eta-points 16 --t-points 80
See snail_figure8.slurm for the cluster job.
"""

from __future__ import annotations

import argparse
import json
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
    """Full (n=1) or fractional (n>1) iSWAP duration for a constant pump of peak
    |eta| (McKinney Eq. 11 / Zhou Eq. 73): pi/(2n) = 6 t_f |eta| g3 lambda^2.

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
        Gate duration t_f (ns).
    """
    g_eff = 6.0 * eta * (g3_GHz * TWO_PI) * lam_a * lam_b      # rad/ns
    return (np.pi / (2 * n)) / g_eff


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
               wp_offset_GHz: float = 0.0):
    """Build the coupler with a CONSTANT pump of peak |eta| (no normalization, so
    peak_eta() == eta) on the (a, b) iSWAP. Optionally add one qubit-qubit
    spectator detuned from the pump by delta_Q.

    Parameters
    ----------
    config : dict
        Merged configuration.
    eta : float
        Pump strength |eta| to set directly.
    t_env : float
        Constant-pulse duration (ns); cover the full evolution window.
    spectator : tuple(float, float), optional
        (delta_Q_GHz, lam_spec): a 2-level spectator whose exchange with qubit a is
        off-resonant by delta_Q (placed at w_spec = w_b - (w_p + delta_Q), matching
        run_sweep_zhou's beat = spec_freq - w_p convention).
    wp_offset_GHz : float, default 0.0
        Offset added to the pump frequency w_b - w_a (GHz).

    Returns
    -------
    (ZhouCoupler, float)
        The coupler and the analytic iSWAP rate g_eff = 6 g3 lambda^2 |eta| (rad/ns).
    """
    from zhou_coupler import ZhouCoupler, PumpTone, ConstantPulse

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
        w_spec = wb - (w_p_GHz + delta_Q_GHz)
        freqs.append(w_spec)
        participations[3] = float(lam_spec)
        levels.append(int(config["spec_levels"]))

    cpl = ZhouCoupler(mode_freqs_GHz=freqs, coupler_index=2,
                      participations=participations, nonlinearities=nonlin, levels=levels)
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=ConstantPulse(amp=eta, t_g=t_env),
                          is_eta=True), normalize_iswap=None)
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
# Panel (a): population-exchange map
# ---------------------------------------------------------------------------
def population_map(config: Dict[str, Any], eta_grid: np.ndarray, times: np.ndarray,
                   backend: str = "dense", solver: Optional[Dict[str, Any]] = None,
                   max_step: float = 0.02) -> Dict[str, np.ndarray]:
    """P(|01> -> |10>) over the (eta, time) grid, plus diagnostics and the analytic
    full-iSWAP line t_f(eta).

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

    Returns
    -------
    dict
        eta_grid, times, P10 / P01 / n_coupler / leak [n_eta, n_t], and t_f (n_eta).
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    n_eta, n_t = len(eta_grid), len(times)
    P10 = np.zeros((n_eta, n_t)); P01 = np.zeros((n_eta, n_t))
    ncpl = np.zeros((n_eta, n_t)); leak = np.zeros((n_eta, n_t))
    i01 = None
    for ie, eta in enumerate(eta_grid):
        cpl, _ = build_gate(config, float(eta), t_env=float(times[-1]))
        if i01 is None:
            i01 = cpl.fock_index([0, 1, 0]); i10 = cpl.fock_index([1, 0, 0])
            i00 = cpl.fock_index([0, 0, 0]); i11 = cpl.fock_index([1, 1, 0])
        states = _trajectory(cpl, [0, 1, 0], times, backend, solver, max_step)
        pops = np.abs(states) ** 2
        P10[ie] = pops[:, i10]; P01[ie] = pops[:, i01]
        ncpl[ie] = [cpl.mean_occupation(p, 2) for p in pops]
        comp = pops[:, i00] + pops[:, i01] + pops[:, i10] + pops[:, i11]
        leak[ie] = 1.0 - comp
    t_f = np.array([iswap_duration_ns(config["g3_GHz"], config["lam_a"],
                                      config["lam_b"], float(e)) for e in eta_grid])
    return {"eta_grid": np.asarray(eta_grid), "times": np.asarray(times),
            "P10": P10, "P01": P01, "n_coupler": ncpl, "leak": leak, "t_f": t_f}


# ---------------------------------------------------------------------------
# Panel (b): spectator coherent infidelity vs eta
# ---------------------------------------------------------------------------
def spectator_infidelity(config: Dict[str, Any], eta_grid: np.ndarray,
                         delta_Q_GHz: Sequence[float], lam_spec: float = 0.20,
                         backend: str = "dense", solver: Optional[Dict[str, Any]] = None,
                         max_step: float = 0.02) -> Dict[str, np.ndarray]:
    """Coherent iSWAP infidelity 1 - F vs pump strength, for each spectator detuning,
    evaluating the gate at its own full-iSWAP duration t_f(eta).

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
    backend : {"dense", "qutip"}
        Solver backend.
    solver : dict, optional
        Tolerances; defaults provided.
    max_step : float, default 0.02
        Dense-backend max ODE step (ns).

    Returns
    -------
    dict
        eta_grid, delta_Q_GHz, and infidelity / leakage [n_delta, n_eta].
    """
    solver = solver or {"atol": 1e-10, "rtol": 1e-8, "nsteps": 500000}
    deltas = np.asarray(delta_Q_GHz, dtype=float)
    infid = np.zeros((len(deltas), len(eta_grid)))
    leak = np.zeros((len(deltas), len(eta_grid)))
    for jd, dq in enumerate(deltas):
        for ie, eta in enumerate(eta_grid):
            t_f = iswap_duration_ns(config["g3_GHz"], config["lam_a"], config["lam_b"], float(eta))
            cpl, _ = build_gate(config, float(eta), t_env=t_f, spectator=(float(dq), lam_spec))
            U = _propagator4(cpl, 0, 1, t_f, backend, solver, max_step)
            F, lk = cpl._iswap_fidelity_from_U(U, True)
            infid[jd, ie] = 1.0 - F
            leak[jd, ie] = lk
    return {"eta_grid": np.asarray(eta_grid), "delta_Q_GHz": deltas,
            "infidelity": infid, "leak": leak}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_figure8(npz_path: str, png_path: str) -> None:
    """Render Fig.8-style panels from a saved .npz (whichever of map / spectator
    data it contains).

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
        ax.set_ylim(t[0], t[-1]); ax.set_title("(a) population exchange  $P(|01\\rangle\\to|10\\rangle)$")
        ax.legend(loc="upper right", framealpha=0.9)
        fig.colorbar(mesh, ax=ax, label=r"$P(|10\rangle)$")

    if has_spec:
        ax = axes[ax_i]; ax_i += 1
        eta = data["spec_eta_grid"] if "spec_eta_grid" in data.files else data["eta_grid"]
        infid = data["infidelity"]; deltas = data["delta_Q_GHz"]
        for jd, dq in enumerate(deltas):
            ax.semilogy(eta, np.clip(infid[jd], 1e-6, None), marker="o", ms=3,
                        label=fr"$\delta_Q={dq*1e3:.0f}$ MHz")
        ax.set_xlabel(r"pump strength $|\eta|$"); ax.set_ylabel(r"coherent infidelity $1-F$")
        ax.set_title("(b) qubit-qubit spectator infidelity"); ax.legend(framealpha=0.9)

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

    if args.mode in ("map", "both"):
        t_max = args.t_max or 1.3 * iswap_duration_ns(config["g3_GHz"], config["lam_a"],
                                                      config["lam_b"], args.eta_min)
        times = np.linspace(0.0, t_max, args.t_points)
        print(f"[map] eta {args.eta_min}-{args.eta_max} x{args.eta_points}, "
              f"t 0-{t_max:.1f} ns x{args.t_points}, backend={args.backend}")
        out.update(population_map(config, eta_grid, times, args.backend, solver, args.max_step))

    if args.mode in ("spectator", "both"):
        deltas = [float(x) / 1000.0 for x in args.deltas_MHz.split(",")]
        print(f"[spectator] deltas {args.deltas_MHz} MHz, lam_spec={args.lam_spec}, backend={args.backend}")
        spec = spectator_infidelity(config, eta_grid, deltas, args.lam_spec,
                                    args.backend, solver, args.max_step)
        out.update({"infidelity": spec["infidelity"], "delta_Q_GHz": spec["delta_Q_GHz"],
                    "spec_leak": spec["leak"], "eta_grid": spec["eta_grid"],
                    "spec_eta_grid": spec["eta_grid"]})

    np.savez(args.out, **out)
    print(f"written {args.out}")
    if args.plot:
        plot_figure8(args.out, args.plot)
        print(f"plotted {args.plot}")


if __name__ == "__main__":
    main()
