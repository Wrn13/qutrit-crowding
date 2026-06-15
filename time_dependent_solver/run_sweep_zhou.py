#!/usr/bin/env python3
r"""
run_sweep_zhou.py
=================

Batch driver for ``zhou_coupler.ZhouCoupler`` -- the dressed-mode (first-
principles) counterpart of ``run_sweep.py``. Where ``run_sweep.py`` drives the
adiabatically-eliminated effective model (``SNAILProcessor``), this driver builds
the Hamiltonian directly from EXPERIMENTAL inputs in Chao Zhou's framework
(thesis Ch. 2): the coupler non-linearity g3 (+ optional g4) and the measured
participations lambda_is = g_is/Delta_is. The central SNAIL is an explicit mode,
not an effective edge.

System (4 modes)
----------------
    index 0 : target qubit a
    index 1 : target qubit b   (anchor for the spectator-frequency sweep)
    index 2 : coupler S (the SNAIL; carries g3/g4; participation 1)
    index 3 : spectator         (the "free" mode that is swept)

The pump sits on the coupler at  w_p = w_b - w_a  so the target iSWAP on (a,b)
is fixed; the pump amplitude is normalized to a full iSWAP at t_g. Only the
spectator frequency moves, so the target gate is untouched as the sweep runs:

    w_spec = w_b - (spec_freq_GHz * 2pi).

Sweep axes  (channel x spec_freq x lam_spec x drag)
---------------------------------------------------
* channel   : "qubit_qubit" (2-level neighbor) or "qubit_sub" (3-level neighbor,
              captures |1>->|2> leakage). In the dressed-mode picture the SNAIL
              is the coupler, so the old "qubit_snail" channel is reported on
              EVERY point as the coupler-occupation column ``n_coupler``.
* spec_freq : spectator transition frequency Delta = w_b - w_spec (GHz), swept
              broadly. For a single-tone, pure-g3 coupler the spectator's only
              collision in the qubit band is the one-pump exchange with the
              anchor, resonant at Delta = w_p (beat = Delta - w_p). Two-mode
              higher-order collisions require g4 or a second pump tone; the FULL
              sim still captures whatever is actually present.
* lam_spec  : spectator participation lambda_spec = g_spec,s/Delta_spec,s. This
              replaces the old phenomenological spectator ``eta``.
* drag      : first-order DRAG quadrature (Motzoi et al., PRL 103, 110501 (2009))
              on the pump, eta(t) -> eta(t) - i d eta/dt / (2 pi * beat), tuned to
              the spectator's beat detuning. Suppresses the OFF-resonant spectator;
              skipped (drag_applied=False) when |beat| < 5e-4 GHz, since an
              on-resonant collision needs frequency allocation, not DRAG.

Two metrics per point
----------------------
* ANALYTIC (always, free -- no integration): nearest collision order m, its beat
  detuning, and the Eq.-62 effective rate g_spec_eff together with the target
  rate g_iswap_eff. This alone is the frequency-allocation collision map.
* FULL (config ``integrate=true``, default): integrate the dressed-mode
  Hamiltonian g_n X(t)^n and report the leakage-aware iSWAP fidelity on (a,b)
  plus the spectator and coupler occupations. This is the non-perturbative check
  and is EXPENSIVE -- the dense, fast-oscillating X^n forces small steps and the
  Hilbert space is d_a*d_b*d_S*d_spec. For broad first passes set
  ``integrate=false`` (or --no-integrate) and read the analytic map; reserve full
  integration for a handful of flagged points, ideally as a SLURM array.

Workflow (identical CLI shape to run_sweep.py)
----------------------------------------------
    python run_sweep_zhou.py prepare --outdir results_zhou/
    RUNNER=run_sweep_zhou.py OUTDIR=results_zhou sbatch --array=0-<M-1> snail_sweep.slurm
    python run_sweep_zhou.py local --outdir results_zhou/ --nproc 8
    python run_sweep_zhou.py collect --outdir results_zhou/

Modes 'prepare' and 'collect' do NOT import the solver; 'point'/'local' import
``zhou_coupler`` lazily.
"""
from __future__ import annotations

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported, so many single-point
# processes (array tasks / pool workers) do not oversubscribe cores.
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

TWO_PI = 2.0 * np.pi
_CHANNELS = ("qubit_qubit", "qubit_sub")
_DELTA_EPS_GHz = 5e-4   # |beat| below this: collision is essentially on-resonance

# ---------------------------------------------------------------------------
# Default device + simulation configuration (override with --device cfg.json).
# Frequencies / nonlinearities in GHz; the runner converts to rad/ns.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # target qubits a, b
    "qubit_freqs_GHz": [5.00, 4.60],
    "qubit_levels":    3,            # >=3 so target-pair leakage is captured
    "lam_a":           0.08,         # participation lambda_as = g_as/Delta_as
    "lam_b":           0.08,         # participation lambda_bs = g_bs/Delta_bs
    # coupler S (the SNAIL)
    "coupler_freq_GHz": 7.00,
    "coupler_levels":   5,
    "g3_GHz":           0.06,        # measured cubic (engine of every process)
    "g4_GHz":           0.0,         # optional quartic (four-wave mixing)
    # spectator mode
    "spec_levels_qubit": 2,          # "qubit_qubit" channel
    "spec_levels_sub":   3,          # "qubit_sub" channel
    "anchor":            1,          # spectator freq measured below qubit b
    # pulse / solver
    "t_g_ns":   60.0,
    "n_tlist":  240,
    "envelope": "raised_cosine",
    "integrate": True,               # set False for the instant analytic map only
    "rtol": 1e-7, "atol": 1e-9, "max_step": 0.05,
}

# Default sweep axes (override with --channels/--specfreqs/--lams).
DEFAULT_CHANNELS = ["qubit_qubit", "qubit_sub"]
# Broad spectator-frequency sweep (GHz). With w_p = 0.40 this crosses the
# fundamental exchange (m=1, 0.40) and the two-pump process (m=2, 0.80).
DEFAULT_SPECFREQS_GHz = [round(0.20 + 0.05 * k, 3) for k in range(15)]  # 0.20..0.90
DEFAULT_LAMS = [0.10, 0.20]
DEFAULT_DRAGS = [False, True]


@dataclass
class Point:
    index: int
    channel: str
    spec_freq_GHz: float    # Delta = w_b - w_spec
    lam_spec: float
    drag: bool


# ---------------------------------------------------------------------------
# Grid construction (deterministic, solver-free)
# ---------------------------------------------------------------------------
def build_grid(channels, specfreqs, lams, drags) -> List[Point]:
    """Cartesian product, stable order: channel -> spec_freq -> lam_spec -> drag."""
    pts, k = [], 0
    for ch in channels:
        if ch not in _CHANNELS:
            raise ValueError(f"unknown channel {ch!r}; choose from {_CHANNELS}")
        for sf in specfreqs:
            for lam in lams:
                for g in drags:
                    pts.append(Point(index=k, channel=ch, spec_freq_GHz=float(sf),
                                     lam_spec=float(lam), drag=bool(g)))
                    k += 1
    return pts


def write_grid(outdir, config, points: List[Point]) -> str:
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    path = os.path.join(outdir, "grid.json")
    with open(path, "w") as f:
        json.dump({"config": config, "points": [asdict(p) for p in points]},
                  f, indent=2)
    return path


def load_grid(outdir):
    with open(os.path.join(outdir, "grid.json")) as f:
        blob = json.load(f)
    return blob["config"], [Point(**p) for p in blob["points"]]


# ---------------------------------------------------------------------------
# Single-point computation (imports zhou_coupler lazily)
# ---------------------------------------------------------------------------
def run_point(pt: Point, config: dict) -> dict:
    from zhou_coupler import ZhouCoupler, PumpTone
    from envelope import ConstantPulse, RaisedCosine

    t0 = time.time()
    a, b, coupler, spec = 0, 1, 2, 3
    anchor = int(config["anchor"])

    # --- frequencies (rad/ns) ---------------------------------------------
    wa, wb = (np.array(config["qubit_freqs_GHz"], dtype=float) * TWO_PI)
    ws = config["coupler_freq_GHz"] * TWO_PI
    w_p = abs(wb - wa)                              # iSWAP pump = |detuning| (fixed)
    w_p_GHz = w_p / TWO_PI
    w_spec = wb - pt.spec_freq_GHz * TWO_PI         # move ONLY the spectator
    freqs_GHz = [wa / TWO_PI, wb / TWO_PI, ws / TWO_PI, w_spec / TWO_PI]

    # --- levels per channel ------------------------------------------------
    if pt.channel == "qubit_qubit":
        spec_levels = int(config["spec_levels_qubit"])
    elif pt.channel == "qubit_sub":
        spec_levels = int(config["spec_levels_sub"])
    else:
        raise ValueError(f"unknown channel {pt.channel!r}")
    levels = [int(config["qubit_levels"]), int(config["qubit_levels"]),
              int(config["coupler_levels"]), spec_levels]

    nonlin = {3: float(config["g3_GHz"])}
    if float(config.get("g4_GHz", 0.0)) != 0.0:
        nonlin[4] = float(config["g4_GHz"])

    cpl = ZhouCoupler(
        mode_freqs_GHz=freqs_GHz,
        coupler_index=coupler,
        participations={a: float(config["lam_a"]),
                        b: float(config["lam_b"]),
                        spec: float(pt.lam_spec)},
        nonlinearities=nonlin,
        levels=levels,
    )

    # DRAG targets the off-resonant spectator<->anchor exchange, detuned from the
    # pump by beat = spec_freq - w_p. It is singular as beat -> 0 (an on-resonant
    # collision needs frequency allocation, not DRAG), so skip it there.
    beat_GHz = pt.spec_freq_GHz - w_p_GHz
    use_drag = bool(pt.drag)
    status_drag = "ok"
    if pt.drag and abs(beat_GHz) < _DELTA_EPS_GHz:
        use_drag = False
        status_drag = "drag_skipped_resonant_spectator"

    # pump at w_b - w_a, amplitude normalized to a full iSWAP on (a,b)
    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else ConstantPulse
    env = EnvCls(amp=1.0, t_g=float(config["t_g_ns"]))
    cpl.set_pump(PumpTone(w_p_GHz=w_p_GHz, envelope=env, is_eta=True,
                          drag=use_drag,
                          delta_drag_GHz=(beat_GHz if use_drag else None)),
                 normalize_iswap=(a, b))

    # --- ANALYTIC collision prediction (free; Eq. 62) ---------------------
    # For a single-tone, pure-g3 (n=3) coupler the spectator's only collision in
    # the qubit band is the ONE-pump exchange with the anchor, resonant at
    # spec_freq = w_p. (Higher-order/two-mode collisions require g4 or a second
    # pump tone; the FULL sim below still captures whatever is actually there.)
    eta_peak = cpl.peak_eta()
    g_iswap = cpl.iswap_rate(a, b)                  # 6 g3 la lb |eta|
    g_spec = cpl.effective_rate([anchor, spec], n=3, C=6)   # 6 g3 l_anchor l_spec |eta|

    out = {
        "index": pt.index,
        "channel": pt.channel,
        "spec_freq_GHz": pt.spec_freq_GHz,
        "lam_spec": pt.lam_spec,
        "drag": bool(pt.drag),
        "drag_applied": bool(use_drag),
        "beat_GHz": round(float(beat_GHz), 6),
        "eta_peak": round(float(eta_peak), 5),
        "g_iswap_eff_MHz": round(float(g_iswap / TWO_PI * 1e3), 4),
        "g_spec_eff_MHz": round(float(g_spec / TWO_PI * 1e3), 4),
        "w_p_GHz": round(float(w_p_GHz), 6),
        "w_spec_GHz": round(float(w_spec / TWO_PI), 6),
        "t_g_ns": float(config["t_g_ns"]),
        "status": status_drag if status_drag != "ok" else "analytic",
        "F_avg": "", "leakage": "", "n_spec": "", "n_coupler": "", "p_transfer": "",
        "U_proj": None,
    }

    if not config.get("integrate", True):
        out["wall_s"] = time.time() - t0
        return out

    # --- FULL non-perturbative integration ---------------------------------
    t_g = float(config["t_g_ns"])
    solver = dict(rtol=float(config["rtol"]), atol=float(config["atol"]),
                  max_step=float(config["max_step"]))

    # (a) spectator diagnostic: excite qubit a, watch where population goes.
    occ0 = [0, 0, 0, 0]; occ0[a] = 1
    tlist = np.linspace(0.0, t_g, int(config["n_tlist"]))
    sol = cpl.evolve(cpl.basis_state(occ0), tlist, **solver)
    pops = np.abs(sol.y[:, -1]) ** 2
    occ_b = [0, 0, 0, 0]; occ_b[b] = 1
    out["n_spec"] = float(cpl.mean_occupation(pops, spec))
    out["n_coupler"] = float(cpl.mean_occupation(pops, coupler))
    out["p_transfer"] = float(pops[cpl.fock_index(occ_b)])

    # (b) leakage-aware iSWAP fidelity on the target pair (4 trajectories).
    F_avg, leakage, U_proj = cpl.iswap_fidelity(a, b, t_g, fit_virtual_z=True,
                                                **solver)
    out["F_avg"] = float(F_avg)
    out["leakage"] = float(leakage)
    out["U_proj"] = U_proj
    out["status"] = status_drag        # "ok", or the drag-skip note if it fired
    out["wall_s"] = time.time() - t0
    return out


def save_point(result: dict, outdir: str):
    path = os.path.join(outdir, "points", f"point_{result['index']:05d}.npz")
    U = result.pop("U_proj", None)
    if U is None:
        U = np.zeros((4, 4), dtype=complex)
    np.savez_compressed(path,
                        U_proj_real=np.real(U), U_proj_imag=np.imag(U),
                        meta=json.dumps(result))
    return path


# ---------------------------------------------------------------------------
# Collection -> summary.csv + combined.npz
# ---------------------------------------------------------------------------
def collect(outdir: str):
    files = sorted(glob.glob(os.path.join(outdir, "points", "point_*.npz")))
    if not files:
        print("No point_*.npz found; nothing to collect.", file=sys.stderr)
        return
    rows, U_stack, idx_stack = [], [], []
    for fpath in files:
        with np.load(fpath, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            U = z["U_proj_real"] + 1j * z["U_proj_imag"]
        rows.append(meta); U_stack.append(U); idx_stack.append(meta["index"])

    rows.sort(key=lambda r: r["index"])
    cols = ["index", "channel", "spec_freq_GHz", "lam_spec", "drag", "drag_applied",
            "beat_GHz", "eta_peak", "g_iswap_eff_MHz", "g_spec_eff_MHz",
            "status", "F_avg", "leakage", "n_spec", "n_coupler", "p_transfer",
            "w_p_GHz", "w_spec_GHz", "t_g_ns", "wall_s"]
    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    order = np.argsort(idx_stack)
    np.savez_compressed(os.path.join(outdir, "combined.npz"),
                        index=np.array(idx_stack)[order],
                        U_proj=np.array(U_stack)[order])
    print(f"Collected {len(rows)} points -> {csv_path} and combined.npz")


# ---------------------------------------------------------------------------
# Modes / CLI
# ---------------------------------------------------------------------------
def _parse_list(s, cast):
    return [cast(x) for x in s.split(",")] if s else None


def _bool_list(s):
    if not s:
        return None
    return [tok.strip().lower() in ("1", "true", "t", "yes", "on")
            for tok in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prepare", "point", "local", "collect"])
    ap.add_argument("--outdir", default="results_zhou")
    ap.add_argument("--device", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--index", type=int, help="point index (mode=point)")
    ap.add_argument("--grid", help="path to grid.json (default: <outdir>/grid.json)")
    ap.add_argument("--nproc", type=int, default=4, help="processes for mode=local")
    ap.add_argument("--channels", help="comma list: qubit_qubit,qubit_sub")
    ap.add_argument("--specfreqs", help="comma list of spectator freqs Delta (GHz)")
    ap.add_argument("--lams", help="comma list of spectator participations")
    ap.add_argument("--drags", help="comma list of bools, e.g. false,true")
    ap.add_argument("--no-integrate", action="store_true",
                    help="analytic collision map only (no time integration)")
    args = ap.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.device:
        with open(args.device) as f:
            config.update(json.load(f))
    if args.no_integrate:
        config["integrate"] = False

    if args.mode == "prepare":
        channels = _parse_list(args.channels, str) or DEFAULT_CHANNELS
        specfreqs = _parse_list(args.specfreqs, float) or DEFAULT_SPECFREQS_GHz
        lams = _parse_list(args.lams, float) or DEFAULT_LAMS
        drags = _bool_list(args.drags) or DEFAULT_DRAGS
        points = build_grid(channels, specfreqs, lams, drags)
        path = write_grid(args.outdir, config, points)
        m = len(points)
        print(f"Wrote {m} points -> {path}")
        print(f"integrate = {config['integrate']}  "
              f"({'FULL sim per point' if config['integrate'] else 'analytic map only'})")
        print(f"Submit with:\n  RUNNER=run_sweep_zhou.py OUTDIR={args.outdir} "
              f"sbatch --array=0-{m-1} snail_sweep.slurm")
        return

    if args.mode == "collect":
        collect(args.outdir)
        return

    grid_path = args.grid or os.path.join(args.outdir, "grid.json")
    if not os.path.exists(grid_path):
        sys.exit(f"grid.json not found at {grid_path}; run `prepare` first.")
    config, points = load_grid(args.outdir if not args.grid
                               else os.path.dirname(args.grid) or ".")

    if args.mode == "point":
        if args.index is None:
            sys.exit("mode=point requires --index")
        if not (0 <= args.index < len(points)):
            sys.exit(f"index {args.index} out of range 0..{len(points)-1}")
        res = run_point(points[args.index], config)
        path = save_point(res, args.outdir)
        f_str = res["F_avg"] if res["F_avg"] != "" else "  --  "
        print(f"[point {args.index}] beat={res['beat_GHz']:+.3f}GHz "
              f"drag={res['drag_applied']} eta={res['eta_peak']:.3f} "
              f"g_spec={res['g_spec_eff_MHz']:.3f}MHz "
              f"F={f_str} ({res['wall_s']:.1f}s) -> {path}")
        return

    if args.mode == "local":
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        os.makedirs(os.path.join(args.outdir, "points"), exist_ok=True)
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.nproc, mp_context=ctx) as ex:
            futs = [ex.submit(run_point, pt, config) for pt in points]
            for done, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                save_point(res, args.outdir)
                print(f"[{done}/{len(points)}] idx={res['index']:>4} "
                      f"drag={res['drag_applied']} beat={res['beat_GHz']:+.3f} "
                      f"g_spec={res['g_spec_eff_MHz']:.3f}MHz F={res['F_avg']}")
        print(f"Local sweep done: {len(points)} points in {time.time()-t0:.1f}s")
        collect(args.outdir)
        return


if __name__ == "__main__":
    main()
