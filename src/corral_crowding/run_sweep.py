#!/usr/bin/env python3
r"""
run_sweep.py
============

Batch driver for snail_parametric_sim.SNAILProcessor. Sweeps the spectator
detuning ``delta_s``, the spectator pump participation ``eta``, and the DRAG
toggle, writing one result file per grid point so the sweep parallelizes
trivially (SLURM array job: one task per point) or runs locally on a pool.

Spectator-detuning parametrization
----------------------------------
The pump is resonant on the target edge (p, q):  w_d = w_p - w_q  (fixed).
A spectator edge (anchor, free) has beat  delta_s = (w_anchor - w_free) - w_d.
To realize a requested ``delta_s`` we move ONLY the free spectator transmon:

    w_free = w_anchor - w_d - delta_s

so w_d (hence the target gate) is untouched as delta_s is swept. With DRAG on,
the DRAG detuning is set to the same delta_s (Eq. 4 of the model).

Workflow
--------
    # 1) build the grid (no QuTiP needed); prints the exact sbatch line
    python run_sweep.py prepare --outdir results/

    # 2a) cluster: submit the array with the printed range
    sbatch --array=0-<M-1> snail_sweep.slurm
    # 2b) workstation: run all points on N processes
    python run_sweep.py local --outdir results/ --nproc 8

    # 3) gather per-point .npz into summary.csv + combined.npz
    python run_sweep.py collect --outdir results/

Per-point output (results/points/point_XXXXX.npz) contains all swept params,
F_avg, leakage, spectator <n>, target transfer, the 4x4 projected propagator,
and wall time. Modes 'prepare' and 'collect' do NOT import QuTiP.
"""
from __future__ import annotations

# Pin BLAS/OpenMP to one thread BEFORE numpy is imported anywhere, so that many
# single-point processes (array tasks or pool workers) do not oversubscribe cores.
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Default device + simulation configuration (override with --device cfg.json).
# Frequencies / anharmonicities in GHz; the runner converts to rad/ns.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "freqs_GHz":   [5.00, 4.60, 4.35],   # base frequencies; index `free` is overwritten per point
    "anh_GHz":     [-0.20, -0.20, -0.20],
    "levels":      3,
    "target_edge": [0, 1],               # (p, q): resonantly pumped iSWAP
    "spectator_edge": [1, 2],            # (anchor, free) on the device graph
    "spectator_anchor": 1,               # fixed-frequency endpoint of spectator edge
    "spectator_free":   2,               # endpoint whose frequency encodes delta_s
    "target_eta":  1.0,                  # pump participation of the target edge
    "static_g_GHz": 0.0,                 # static exchange on both edges (usually 0 here)
    "t_g_ns":      40.0,                 # gate length
    "n_tlist":     200,                  # output sampling (solver is adaptive)
    "envelope":    "raised_cosine",      # or "gaussian_flat"
}

# Default sweep axes (override on the CLI with --deltas/--etas/--drags).
DEFAULT_DELTAS_GHz = [-0.30, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_ETAS = [0.3, 0.6, 0.9]
DEFAULT_DRAGS = [False, True]

TWO_PI = 2.0 * np.pi
_DELTA_EPS_GHz = 1e-6  # below this |delta_s| DRAG is undefined (resonant spectator)


@dataclass
class Point:
    """Parameters for a single run of solver."""
    index: int
    delta_s_GHz: float
    eta: float
    drag: bool


# ---------------------------------------------------------------------------
# Grid construction (deterministic, QuTiP-free)
# ---------------------------------------------------------------------------
def build_grid(deltas, etas, drags) -> List[Point]:
    """Cartesian product with a stable ordering: delta (outer) -> eta -> drag."""
    pts, k = [], 0
    for d in deltas:
        for e in etas:
            for g in drags:
                pts.append(Point(index=k, delta_s_GHz=float(d), eta=float(e), drag=bool(g)))
                k += 1
    return pts


def write_grid(outdir, config, points: List[Point]) -> str:
    os.makedirs(os.path.join(outdir, "points"), exist_ok=True)
    path = os.path.join(outdir, "grid.json")
    with open(path, "w") as f:
        json.dump({"config": config, "points": [asdict(p) for p in points]}, f, indent=2)
    return path


def load_grid(outdir):
    with open(os.path.join(outdir, "grid.json")) as f:
        blob = json.load(f)
    pts = [Point(**p) for p in blob["points"]]
    return blob["config"], pts


# ---------------------------------------------------------------------------
# Single-point computation (imports QuTiP lazily, so prepare/collect stay light)
# ---------------------------------------------------------------------------
def run_point(pt: Point, config: dict) -> dict:
    from corral_crowding.snail_parametric_sim import (  # noqa: WPS433 (deliberate lazy import)
        SNAILProcessor, Edge, PumpSpec, RaisedCosine, GaussianFlat,
    )

    t0 = time.time()
    p, q = config["target_edge"]
    anchor = config["spectator_anchor"]
    free = config["spectator_free"]

    freqs = np.array(config["freqs_GHz"], dtype=float) * TWO_PI
    anh = np.array(config["anh_GHz"], dtype=float) * TWO_PI
    g_static = config["static_g_GHz"] * TWO_PI

    w_d = freqs[p] - freqs[q]                  # fixed by target edge
    delta_s = pt.delta_s_GHz * TWO_PI
    freqs[free] = freqs[anchor] - w_d - delta_s  # realize requested beat

    edges = [
        Edge(p, q, g=g_static, eta=config["target_eta"]),
        Edge(anchor, free, g=g_static, eta=pt.eta),
    ]
    proc = SNAILProcessor(freqs, anh, edges, levels=config["levels"])

    # DRAG against a resonant spectator (delta_s == 0) is singular: skip it.
    use_drag = pt.drag
    status = "ok"
    if pt.drag and abs(pt.delta_s_GHz) < _DELTA_EPS_GHz:
        use_drag = False
        status = "drag_skipped_resonant_spectator"

    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else GaussianFlat
    env = EnvCls(amp=1.0, t_g=config["t_g_ns"])
    proc.set_pump(PumpSpec(
        target_edge=(p, q), envelope=env, w_d=w_d,
        drag=use_drag, delta_drag=(delta_s if use_drag else None),
    ))

    tlist = np.linspace(0.0, config["t_g_ns"], config["n_tlist"])

    # (a) single-trajectory spectator diagnostic: excite the driven transmon p.
    occ0 = [0] * proc.N
    occ0[p] = 1
    res = proc.evolve(proc.fock(occ0), tlist)
    pops = np.abs(res.states[-1].full().ravel()) ** 2
    n_spec = 0.0
    for idx in range(proc.D):
        # decode occupation of `free` from flat index
        rem, occ = idx, []
        for _ in range(proc.N):
            rem, r = divmod(rem, proc.d)
            occ.append(r)
        occ = occ[::-1]
        n_spec += pops[idx] * occ[free]
    occ_t = [0] * proc.N
    occ_t[q] = 1                      # target transfer p->q in single-excitation manifold
    p_transfer = float(pops[proc.fock_index(occ_t)])

    # (b) two-qubit gate metric on the target pair (4 trajectories).
    F_avg, leakage, U_proj = proc.iswap_fidelity(tlist)

    return {
        "index": pt.index,
        "delta_s_GHz": pt.delta_s_GHz,
        "eta": pt.eta,
        "drag": bool(pt.drag),
        "drag_applied": bool(use_drag),
        "status": status,
        "w_d_GHz": w_d / TWO_PI,
        "w_free_GHz": freqs[free] / TWO_PI,
        "F_avg": F_avg,
        "leakage": leakage,
        "n_spec": n_spec,
        "p_transfer": p_transfer,
        "U_proj": U_proj,
        "t_g_ns": config["t_g_ns"],
        "levels": config["levels"],
        "wall_s": time.time() - t0,
    }


def save_point(result: dict, outdir: str):
    path = os.path.join(outdir, "points", f"point_{result['index']:05d}.npz")
    U = result.pop("U_proj")
    np.savez_compressed(
        path,
        U_proj_real=np.real(U),
        U_proj_imag=np.imag(U),
        meta=json.dumps(result),
    )
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
        rows.append(meta)
        U_stack.append(U)
        idx_stack.append(meta["index"])

    rows.sort(key=lambda r: r["index"])
    cols = ["index", "delta_s_GHz", "eta", "drag", "drag_applied", "status",
            "F_avg", "leakage", "n_spec", "p_transfer",
            "w_d_GHz", "w_free_GHz", "t_g_ns", "levels", "wall_s"]
    csv_path = os.path.join(outdir, "summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    order = np.argsort(idx_stack)
    np.savez_compressed(
        os.path.join(outdir, "combined.npz"),
        index=np.array(idx_stack)[order],
        U_proj=np.array(U_stack)[order],
    )
    print(f"Collected {len(rows)} points -> {csv_path} and combined.npz")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _parse_list(s, cast):
    return [cast(x) for x in s.split(",")] if s else None


def _bool_list(s):
    if not s:
        return None
    out = []
    for tok in s.split(","):
        out.append(tok.strip().lower() in ("1", "true", "t", "yes", "on"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["prepare", "point", "local", "collect"])
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--device", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--index", type=int, help="point index (mode=point)")
    ap.add_argument("--grid", help="path to grid.json (default: <outdir>/grid.json)")
    ap.add_argument("--nproc", type=int, default=4, help="processes for mode=local")
    ap.add_argument("--deltas", help="comma list of delta_s in GHz (overrides default)")
    ap.add_argument("--etas", help="comma list of spectator eta (overrides default)")
    ap.add_argument("--drags", help="comma list of bools, e.g. false,true")
    args = ap.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.device:
        with open(args.device) as f:
            config.update(json.load(f))

    if args.mode == "prepare":
        deltas = _parse_list(args.deltas, float) or DEFAULT_DELTAS_GHz
        etas = _parse_list(args.etas, float) or DEFAULT_ETAS
        drags = _bool_list(args.drags) or DEFAULT_DRAGS
        points = build_grid(deltas, etas, drags)
        path = write_grid(args.outdir, config, points)
        m = len(points)
        print(f"Wrote {m} points -> {path}")
        print(f"Submit with:\n  sbatch --array=0-{m-1} snail_sweep.slurm")
        return

    if args.mode == "collect":
        collect(args.outdir)
        return

    grid_path = args.grid or os.path.join(args.outdir, "grid.json")
    if not os.path.exists(grid_path):
        sys.exit(f"grid.json not found at {grid_path}; run `prepare` first.")
    config, points = load_grid(args.outdir if not args.grid else os.path.dirname(args.grid) or ".")

    if args.mode == "point":
        if args.index is None:
            sys.exit("mode=point requires --index")
        if not (0 <= args.index < len(points)):
            sys.exit(f"index {args.index} out of range 0..{len(points)-1}")
        res = run_point(points[args.index], config)
        path = save_point(res, args.outdir)
        print(f"[point {args.index}] F={res['F_avg']:.4f} leak={res['leakage']:.4f} "
              f"n_spec={res['n_spec']:.5f} ({res['wall_s']:.1f}s) -> {path}")
        return

    if args.mode == "local":
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        os.makedirs(os.path.join(args.outdir, "points"), exist_ok=True)
        ctx = mp.get_context("spawn")  # fresh interpreters honor the thread-pinning env
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.nproc, mp_context=ctx) as ex:
            futs = [ex.submit(run_point, pt, config) for pt in points]
            for done, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                save_point(res, args.outdir)
                print(f"[{done}/{len(points)}] idx={res['index']:>4} "
                      f"F={res['F_avg']:.4f} leak={res['leakage']:.4f} "
                      f"n_spec={res['n_spec']:.5f}")
        print(f"Local sweep done: {len(points)} points in {time.time()-t0:.1f}s")
        collect(args.outdir)
        return


if __name__ == "__main__":
    main()
