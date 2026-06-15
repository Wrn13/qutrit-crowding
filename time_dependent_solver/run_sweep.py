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
# Frequencies / anharmonicities / g3 / g4 in GHz; runner converts to rad/ns.
#
# Two spectator CHANNELS (select via the --channels sweep axis):
#   "qubit_qubit" : target pair (0,1) + a third transmon (2) as spectator.
#   "qubit_snail" : target pair (0,1) + an explicit SNAIL mode (index 2) carrying
#                   a cubic g3 (+ optional quartic g4) self-Hamiltonian -- NOT a
#                   transmon Duffing Kerr.
# Spectator edge is (anchor=1, free=2). We sweep the spectator TRANSITION
# FREQUENCY Delta_rs = w_anchor - w_free broadly; wherever it crosses an order
# m*w_d of the pump comb, that spectator lights up. The pump frequency
# w_d = w_0 - w_1 (the target) is held fixed, so only the free mode moves.
#
# General spectators come from the multi-tone pump COMB: a real SNAIL pump is
# not a pure cos(w_d t) -- the cubic/quartic mix it into harmonics (m=2,3 ...)
# and subharmonics (m=1/2 ...). `pump_tones` lists [mult, amp] pairs; the m=1
# tone drives the iSWAP. DRAG (when on) targets whichever order is nearest the
# swept Delta_rs, suppressing that spectator.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # target pair (transmons 0 and 1)
    "pair_freqs_GHz": [5.00, 4.60],
    "pair_anh_GHz":   [-0.20, -0.20],
    "pair_levels":    3,
    # spectator transmon (qubit_qubit channel)
    "spec_transmon_anh_GHz": -0.20,
    "spec_transmon_levels":  3,
    # explicit SNAIL (qubit_snail channel): cubic + quartic, NOT a Duffing Kerr.
    # Defaults to a (cheap) linear spectator; set g3/g4 to include nonlinearity.
    "snail_g3_GHz":   0.0,
    "snail_g4_GHz":   0.0,
    "snail_levels":   4,
    "snail_self_rwa": True,   # keep slow cubic + static g4 Kerr; drop >=2*w_s terms
    # pump comb: [multiplier, relative amplitude]; m=1 drives the iSWAP
    "pump_tones": [[1.0, 1.0], [2.0, 0.5]],
    # topology / drive
    "target_edge": [0, 1],
    "spectator_anchor": 1,
    "target_eta":  1.0,
    "static_g_GHz": 0.0,
    # pulse / solver
    "t_g_ns":      40.0,
    "n_tlist":     200,
    "envelope":    "raised_cosine",
}

# Default sweep axes (override with --channels/--specfreqs/--etas/--drags).
DEFAULT_CHANNELS = ["qubit_qubit", "qubit_snail"]
# Broad spectator-transition-frequency sweep (GHz). With w_d=0.4 this crosses the
# fundamental (m=1, 0.40) and second harmonic (m=2, 0.80) of the comb.
DEFAULT_SPECFREQS_GHz = [round(0.20 + 0.025 * k, 3) for k in range(29)]  # 0.20..0.90
DEFAULT_ETAS = [0.6, 0.9]
DEFAULT_DRAGS = [False, True]

TWO_PI = 2.0 * np.pi
_DELTA_EPS_GHz = 5e-4   # below this beat DRAG is singular (near-resonant spectator)
_CHANNELS = ("qubit_qubit", "qubit_snail")


@dataclass
class Point:
    index: int
    channel: str
    spec_freq_GHz: float   # spectator transition frequency Delta_rs = w_anchor - w_free
    eta: float
    drag: bool


# ---------------------------------------------------------------------------
# Grid construction (deterministic, QuTiP-free)
# ---------------------------------------------------------------------------
def build_grid(channels, specfreqs, etas, drags) -> List[Point]:
    """Cartesian product, stable order: channel -> spec_freq -> eta -> drag."""
    pts, k = [], 0
    for ch in channels:
        if ch not in _CHANNELS:
            raise ValueError(f"unknown channel {ch!r}; choose from {_CHANNELS}")
        for sf in specfreqs:
            for e in etas:
                for g in drags:
                    pts.append(Point(index=k, channel=ch, spec_freq_GHz=float(sf),
                                     eta=float(e), drag=bool(g)))
                    k += 1
    return pts


def _nearest_order(delta_rs_GHz, w_d_GHz, tones):
    """Return (mult*, beat*_GHz) for the comb tone whose m*w_d is closest to
    the spectator transition frequency Delta_rs."""
    best = None
    for (m, _amp) in tones:
        beat = delta_rs_GHz - m * w_d_GHz
        if best is None or abs(beat) < abs(best[1]):
            best = (m, beat)
    return best


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
    from snail_parametric_sim import (  # noqa: WPS433 (deliberate lazy import)
        SNAILProcessor, Edge, PumpSpec, PumpTone
    )
    from envelope import RaisedCosine, GaussianFlat

    t0 = time.time()
    p, q = config["target_edge"]
    anchor = config["spectator_anchor"]      # a target-pair qubit (default q = 1)
    free = 2                                 # third mode is always the spectator/free node

    # --- assemble the three modes for this channel -------------------------
    freqs = list(np.array(config["pair_freqs_GHz"], dtype=float))
    anh = list(np.array(config["pair_anh_GHz"], dtype=float))
    lvls = [config["pair_levels"], config["pair_levels"]]
    snail_modes = None

    if pt.channel == "qubit_qubit":
        anh.append(config["spec_transmon_anh_GHz"])
        lvls.append(config["spec_transmon_levels"])
    elif pt.channel == "qubit_snail":
        anh.append(0.0)                           # SNAIL has NO Duffing Kerr
        lvls.append(config["snail_levels"])
        snail_modes = {free: {"g3": config["snail_g3_GHz"] * TWO_PI,
                              "g4": config["snail_g4_GHz"] * TWO_PI}}
    else:
        raise ValueError(f"unknown channel {pt.channel!r}")
    freqs.append(0.0)  # placeholder; overwritten below to realize Delta_rs

    freqs = np.array(freqs) * TWO_PI
    anh = np.array(anh) * TWO_PI
    g_static = config["static_g_GHz"] * TWO_PI

    w_d = freqs[p] - freqs[q]                       # fixed by target edge
    w_d_GHz = w_d / TWO_PI
    # sweep the spectator TRANSITION FREQUENCY Delta_rs = w_anchor - w_free
    freqs[free] = freqs[anchor] - pt.spec_freq_GHz * TWO_PI

    edges = [
        Edge(p, q, g=g_static, eta=config["target_eta"]),
        Edge(anchor, free, g=g_static, eta=pt.eta),
    ]
    proc = SNAILProcessor(freqs, anh, edges, levels=lvls,
                          snail_modes=snail_modes,
                          snail_self_rwa=config.get("snail_self_rwa", True))

    # --- pump comb + nearest-order DRAG targeting --------------------------
    tones_cfg = config["pump_tones"]                # [[mult, amp], ...]
    mult_star, beat_GHz = _nearest_order(pt.spec_freq_GHz, w_d_GHz, tones_cfg)

    use_drag = pt.drag
    status = "ok"
    if pt.drag and abs(beat_GHz) < _DELTA_EPS_GHz:
        use_drag = False                            # DRAG singular at the resonance
        status = "drag_skipped_resonant_spectator"

    tones = []
    for (m, amp) in tones_cfg:
        on = bool(use_drag and m == mult_star)
        tones.append(PumpTone(mult=float(m), amp=float(amp), drag=on,
                              delta_drag=(beat_GHz * TWO_PI if on else None)))

    EnvCls = RaisedCosine if config["envelope"] == "raised_cosine" else GaussianFlat
    env = EnvCls(amp=1.0, t_g=config["t_g_ns"])
    proc.set_pump(PumpSpec(target_edge=(p, q), envelope=env, w_d=w_d, tones=tones))

    tlist = np.linspace(0.0, config["t_g_ns"], config["n_tlist"])

    # (a) single-trajectory spectator diagnostic: excite driven transmon p,
    #     measure mean occupation of the free/spectator mode (qubit OR SNAIL).
    occ0 = [0] * proc.N
    occ0[p] = 1
    res = proc.evolve(proc.fock(occ0), tlist)
    pops = np.abs(res.states[-1].full().ravel()) ** 2
    n_spec = proc.mean_occupation(pops, free)
    occ_t = [0] * proc.N
    occ_t[q] = 1
    p_transfer = float(pops[proc.fock_index(occ_t)])

    # (b) two-qubit gate metric on the target pair (4 trajectories). Leakage
    #     captures population lost to the spectator/SNAIL out of {|0>,|1>}^2.
    F_avg, leakage, U_proj = proc.iswap_fidelity(tlist)

    return {
        "index": pt.index,
        "channel": pt.channel,
        "spec_freq_GHz": pt.spec_freq_GHz,
        "eta": pt.eta,
        "drag": bool(pt.drag),
        "drag_applied": bool(use_drag),
        "nearest_mult": float(mult_star),
        "beat_GHz": float(beat_GHz),
        "status": status,
        "w_d_GHz": w_d_GHz,
        "w_free_GHz": float(freqs[free] / TWO_PI),
        "F_avg": F_avg,
        "leakage": leakage,
        "n_spec": n_spec,
        "p_transfer": p_transfer,
        "t_g_ns": config["t_g_ns"],
        "U_proj": U_proj,
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
    cols = ["index", "channel", "spec_freq_GHz", "eta", "drag", "drag_applied",
            "nearest_mult", "beat_GHz", "status",
            "F_avg", "leakage", "n_spec", "p_transfer",
            "w_d_GHz", "w_free_GHz", "t_g_ns", "wall_s"]
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
    ap.add_argument("--channels", help="comma list: qubit_qubit,qubit_snail")
    ap.add_argument("--specfreqs", help="comma list of spectator transition freqs Delta_rs in GHz")
    ap.add_argument("--etas", help="comma list of spectator eta (overrides default)")
    ap.add_argument("--drags", help="comma list of bools, e.g. false,true")
    args = ap.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.device:
        with open(args.device) as f:
            config.update(json.load(f))

    if args.mode == "prepare":
        channels = _parse_list(args.channels, str) or DEFAULT_CHANNELS
        specfreqs = _parse_list(args.specfreqs, float) or DEFAULT_SPECFREQS_GHz
        etas = _parse_list(args.etas, float) or DEFAULT_ETAS
        drags = _bool_list(args.drags) or DEFAULT_DRAGS
        points = build_grid(channels, specfreqs, etas, drags)
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