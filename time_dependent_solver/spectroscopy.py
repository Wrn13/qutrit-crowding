"""Power-vs-frequency spectroscopy of a transmon transition, with stored data.

Produces the standard drive-amplitude x drive-frequency map coloured by the
population of a target level (|e> by default, starting from |g>): the direct
resonance appears as a vertical line that bends with power (the AC-Stark pull),
and multiphoton / subharmonic features appear at fractions of the transition
frequency, e.g. ge/2 where two drive photons bridge |g> -> |e>.

The grid is written to an ``.npz`` so figures can be restyled without recomputing
(scans are the expensive part), and ``--replot`` renders straight from a stored
file.

Evolution uses the SAME ``ZhouCoupler`` as the sweeps, integrated with QuTiP
``sesolve`` on the exact time-dependent Hamiltonian (``evolve_state`` for a
fixed-duration probe, ``evolve_trajectory`` when reducing over the probe). There is
no rotating frame and no carrier cutoff here, so every multiphoton process the
model contains appears -- and the map is directly comparable to sweep results,
which use the same solver. QuTiP is therefore REQUIRED.

WHAT THIS MODEL CAN AND CANNOT SHOW
-----------------------------------
The drive enters through the coupler nonlinearity ``sum_n g_n X(t)^n``. A cubic
(g3) truncation gives at most two drive quanta alongside one mode operator, so it
supports the direct line (ge) and the second subharmonic (ge/2) -- but NOT ge/3.
For a third subharmonic the expansion must reach fourth order: put ``g4_GHz`` in
the device (``nonlinearities={4: ...}``), and correspondingly higher orders for
ge/6. If a published map shows ge/3 or eh/6, reproducing those features requires
those higher-order terms; with g3 alone they are absent by construction, not
because the scan missed them.

CLI
---
    python spectroscopy.py --device warren_device.json \\
        --f-lo 1.5 --f-hi 3.8 --f-points 121 --amp-hi 3.0 --amp-points 41 \\
        --probe-ns 200 --reduce max --nproc 8 \\
        --save-data results/ge_map.npz --out figs/ge_map.png

    python spectroscopy.py --replot results/ge_map.npz --out figs/ge_restyled.png
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_PI = 2.0 * np.pi
BLUE, RED = "#2980B9", "#C0392B"


# --------------------------------------------------------------------------- #
# population helpers
# --------------------------------------------------------------------------- #
def marginal_population(probs: np.ndarray, dims: Sequence[int], mode: int,
                        level: int) -> float:
    """Population of ``level`` in ``mode``, marginalised over all other modes.

    Reading a single Fock amplitude would undercount whenever the drive also
    excites another mode (the coupler is a real dynamical mode here), so the
    marginal is the honest observable to compare with a measured |e> population.

    Parameters
    ----------
    probs : ndarray
        Probabilities over the full product basis, ordered as ``dims``.
    dims : sequence of int
        Per-mode truncation levels.
    mode : int
        Mode whose level population is wanted.
    level : int
        Level index (1 = |e>, 2 = |f>, ...).

    Returns
    -------
    float
        Summed population of the requested level.
    """
    t = np.asarray(probs, dtype=float).reshape(tuple(dims))
    if level >= t.shape[mode]:
        return float("nan")
    return float(np.take(t, level, axis=mode).sum())


def _reduce_populations(pops: np.ndarray, reduce: str) -> float:
    """Reduce a population time series to one number (see ``reduce`` in scan_ge)."""
    if reduce == "max":
        return float(np.max(pops))
    if reduce == "mean":
        return float(np.mean(pops))
    return float(pops[-1])


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def transition_lines(config: Dict[str, Any], mode: int = 0,
                     orders: Sequence[int] = (1, 2, 3)) -> Dict[str, float]:
    """Expected drive frequencies of ge / ef features and their subharmonics.

    ``ge`` sits at the mode frequency and ``ef`` one anharmonicity below it; the
    n-th subharmonic of a transition at w is at w / n.
    """
    freqs = list(np.asarray(config["qubit_freqs_GHz"], dtype=float))
    w_ge = float(freqs[mode])
    alpha = float(config.get("anharm_qubit_GHz", 0.0))
    out: Dict[str, float] = {}
    for n in orders:
        out[f"ge/{n}" if n > 1 else "ge"] = w_ge / n
        out[f"ef/{n}" if n > 1 else "ef"] = (w_ge + alpha) / n
    return out


def _scan_column(task: Tuple[int, float, Dict[str, Any], Dict[str, Any]]
                 ) -> Tuple[int, np.ndarray]:
    """Compute one drive-frequency column using the ZhouCoupler / QuTiP solver.

    Top-level (not a closure) so it is picklable for multiprocessing. The mode
    construction comes from ``device_utils.build_coupler`` -- identical to what the
    sweeps use -- and the pump is then replaced by a single tone at this column's
    drive frequency, so ``|eta|_peak`` equals the requested amplitude exactly.

    Evolution is QuTiP ``sesolve`` on the EXACT time-dependent Hamiltonian
    (``ZhouCoupler.evolve_state`` / ``evolve_trajectory``): no rotating frame and no
    carrier cutoff, so every multiphoton process the model contains is present.

    Parameters
    ----------
    task : tuple
        ``(column_index, drive_frequency_GHz, config, params)``.

    Returns
    -------
    (int, ndarray)
        The column index and the population column over the amplitude axis.
    """
    jf, f_d, config, prm = task
    from device_utils import build_coupler
    from zhou_coupler import PumpTone, RaisedCosine, ConstantPulse

    amps = np.asarray(prm["amps"], dtype=float)
    probe_ns = float(prm["probe_ns"])
    reduce = prm["reduce"]
    EnvCls = RaisedCosine if prm["envelope"] == "raised_cosine" else ConstantPulse
    solver = dict(atol=prm["atol"], rtol=prm["rtol"], nsteps=prm["nsteps"])

    # same mode construction as the sweeps (levels, participations, anharmonicities,
    # optional spectator); the pump it sets is replaced below.
    cpl, _wp, _eta = build_coupler(config, t_g=probe_ns, amp_scale=1.0,
                                   wp_offset_GHz=0.0,
                                   spec_abs_GHz=prm["spec_abs_GHz"])
    occ0 = [0] * len(cpl.dims)                                 # all modes in |g>
    times = (np.linspace(0.0, probe_ns, int(prm["n_times"]))
             if reduce in ("max", "mean") else None)

    col = np.empty(len(amps))
    for ia, amp in enumerate(amps):
        if amp == 0.0:
            col[ia] = 0.0                                      # no drive, stays in |g>
            continue
        # amp IS the peak |eta| for this envelope (verified: amp -> peak_eta)
        cpl.set_pump(PumpTone(w_p_GHz=f_d, envelope=EnvCls(amp=float(amp),
                                                          t_g=probe_ns),
                              is_eta=True))
        if times is None:
            psi = cpl.evolve_state(occ0, probe_ns, **solver)
            pops = np.array([marginal_population(np.abs(psi) ** 2, cpl.dims,
                                                 prm["target_mode"], prm["level"])])
        else:
            traj = cpl.evolve_trajectory(occ0, times, **solver)
            pops = np.array([marginal_population(np.abs(psi_t) ** 2, cpl.dims,
                                                 prm["target_mode"], prm["level"])
                             for psi_t in traj])
        col[ia] = _reduce_populations(pops, reduce)
    return jf, col


def scan_ge(config: Dict[str, Any], *, f_lo: float, f_hi: float, f_points: int = 81,
            amp_lo: float = 0.0, amp_hi: float = 2.0, amp_points: int = 31,
            probe_ns: float = 200.0, target_mode: int = 0, level: int = 1,
            envelope: str = "constant", reduce: str = "final",
            n_times: int = 41, rtol: float = 1e-8, atol: float = 1e-10,
            nsteps: int = 500000,
            spec_abs_GHz: Optional[float] = None,
            nproc: int = 1, columns: Optional[Sequence[int]] = None,
            verbose: bool = True) -> Dict[str, Any]:
    """Scan drive frequency x drive amplitude, recording a level population.

    The coupler is rebuilt at every frequency column so each column carries its own
    rotating frame -- correct across a wide sweep, where a single reference frame's
    RWA cutoff would wrongly discard terms that become resonant elsewhere.

    Parameters
    ----------
    config : dict
        Merged device configuration.
    f_lo, f_hi : float
        Drive-frequency range (GHz).
    f_points : int
        Frequency-axis resolution.
    amp_lo, amp_hi : float
        Drive amplitude range in units of |eta| (dimensionless displaced amplitude).
    amp_points : int
        Amplitude-axis resolution.
    probe_ns : float
        Drive duration (ns). Longer probes give sharper lines (resolution ~ 1/T).
    target_mode : int
        Mode whose population is recorded (0 = qubit a).
    level : int
        Level to record (1 = |e>).
    n_times : int
        Output times for ``reduce`` in {'max', 'mean'} (ignored for 'final').
    envelope : {'constant', 'raised_cosine'}
        Probe shape. ``constant`` is the usual spectroscopy probe and gives the
        sharpest lines; ``raised_cosine`` matches the gate pulse instead.
    reduce : {'final', 'max', 'mean'}
        Reduction of the population over the probe. ``final`` reproduces a
        fixed-duration measurement (and its Rabi fringes); ``max`` gives the
        cleanest map of where transitions live, which is usually what a published
        power-vs-frequency figure is showing.
    rtol, atol : float
        QuTiP sesolve tolerances.
    nsteps : int
        Maximum internal solver steps between outputs.
    spec_abs_GHz : float, optional
        Include a spectator at this absolute frequency.
    nproc : int
        Worker processes over frequency columns (1 = serial).
    columns : sequence of int, optional
        Compute only these column indices (used for array sharding); the rest of the
        grid is left NaN and merged later by :func:`merge_shards`.
    verbose : bool
        Print progress per frequency column.

    Returns
    -------
    dict
        freqs_GHz, amps, Z (shape [amp_points, f_points]), lines, and metadata.
    """
    freqs = np.linspace(f_lo, f_hi, f_points)
    amps = np.linspace(amp_lo, amp_hi, amp_points)
    Z = np.full((amp_points, f_points), np.nan)
    todo = list(range(f_points)) if columns is None else sorted(set(columns))

    params = dict(amps=amps, probe_ns=probe_ns, target_mode=target_mode, level=level,
                  envelope=envelope, reduce=reduce, n_times=n_times,
                  rtol=rtol, atol=atol, nsteps=nsteps, spec_abs_GHz=spec_abs_GHz)
    tasks = [(jf, float(freqs[jf]), config, params) for jf in todo]

    if nproc and nproc > 1 and len(tasks) > 1:
        # columns are independent (each rebuilds its own rotating frame), so this is
        # embarrassingly parallel; imap_unordered keeps workers fed when per-column
        # cost varies (n_sub grows with drive frequency).
        import multiprocessing as mp
        # 'fork' where available: 'spawn' re-imports __main__ in every worker, which
        # hangs or fails when the caller is a notebook, a heredoc, or any module
        # without an `if __name__ == "__main__"` guard. Workers here are pure
        # compute, and SLURM pins BLAS to one thread, so fork is safe.
        try:
            ctx = mp.get_context("fork")
        except ValueError:                              # non-POSIX
            ctx = mp.get_context()
        with ctx.Pool(processes=int(nproc)) as pool:
            for k, (jf, col) in enumerate(pool.imap_unordered(_scan_column, tasks)):
                Z[:, jf] = col
                if verbose:
                    print(f"    [{k + 1}/{len(tasks)}] f={freqs[jf]:6.3f} GHz  "
                          f"max pop = {np.nanmax(col):.3f}", flush=True)
    else:
        for k, task in enumerate(tasks):
            jf, col = _scan_column(task)
            Z[:, jf] = col
            if verbose:
                print(f"    [{k + 1}/{len(tasks)}] f={freqs[jf]:6.3f} GHz  "
                      f"max pop = {np.nanmax(col):.3f}", flush=True)

    return dict(freqs_GHz=freqs, amps=amps, Z=Z,
                lines=transition_lines(config, target_mode),
                meta=dict(probe_ns=probe_ns, target_mode=target_mode, level=level,
                          envelope=envelope, reduce=reduce, n_times=n_times,
                          solver="qutip_sesolve_exact", rtol=rtol, atol=atol,
                          spec_abs_GHz=spec_abs_GHz,
                          columns=list(todo), f_points=int(f_points),
                          qubit_freqs_GHz=list(np.asarray(
                              config["qubit_freqs_GHz"], dtype=float)),
                          coupler_freq_GHz=config.get("coupler_freq_GHz"),
                          anharm_qubit_GHz=config.get("anharm_qubit_GHz"),
                          g3_GHz=config.get("g3_GHz"),
                          g4_GHz=config.get("g4_GHz", 0.0)))


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def save_scan(result: Dict[str, Any], path: str) -> str:
    """Write the scan grid + metadata to an ``.npz`` so it can be re-plotted."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, freqs_GHz=result["freqs_GHz"], amps=result["amps"],
                        Z=result["Z"], lines_json=json.dumps(result["lines"]),
                        meta_json=json.dumps(result["meta"]))
    print("wrote", path)
    return path


def load_scan(path: str) -> Dict[str, Any]:
    """Load a scan written by :func:`save_scan`."""
    with np.load(path, allow_pickle=False) as d:
        return dict(freqs_GHz=d["freqs_GHz"], amps=d["amps"], Z=d["Z"],
                    lines=json.loads(str(d["lines_json"])),
                    meta=json.loads(str(d["meta_json"])))


def export_csv(result: Dict[str, Any], path: str) -> str:
    """Write the grid as long-form CSV (freq, amp, population) for external tools."""
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drive_freq_GHz", "amp_eta", "population"])
        for ia, amp in enumerate(result["amps"]):
            for jf, f in enumerate(result["freqs_GHz"]):
                w.writerow([f"{f:.6f}", f"{amp:.6f}", f"{result['Z'][ia, jf]:.6f}"])
    print("wrote", path)
    return path


def merge_shards(prefix: str, out_npz: Optional[str] = None) -> Dict[str, Any]:
    """Merge per-shard ``.npz`` files into one full grid.

    Each shard holds the same axes but only its own frequency columns (the rest
    NaN), so merging is a NaN-aware overlay. Reports any column no shard filled,
    which is how a failed or still-running array task shows up -- better than
    silently plotting a striped map.

    Parameters
    ----------
    prefix : str
        Shard path prefix; files matching ``<prefix>_shard*.npz`` are merged.
    out_npz : str, optional
        Also write the merged grid here.

    Returns
    -------
    dict
        The merged scan (same structure as :func:`scan_ge`).
    """
    import glob
    paths = sorted(glob.glob(f"{prefix}_shard*.npz"))
    if not paths:
        raise SystemExit(f"no shards matching {prefix}_shard*.npz")
    merged: Optional[Dict[str, Any]] = None
    filled: set = set()
    for path in paths:
        part = load_scan(path)
        if merged is None:
            merged = dict(freqs_GHz=part["freqs_GHz"], amps=part["amps"],
                          Z=np.full_like(part["Z"], np.nan),
                          lines=part["lines"], meta=dict(part["meta"]))
        elif (len(part["freqs_GHz"]) != len(merged["freqs_GHz"])
              or len(part["amps"]) != len(merged["amps"])):
            raise SystemExit(f"{path}: axes do not match the other shards")
        cols = part["meta"].get("columns")
        cols = range(part["Z"].shape[1]) if cols is None else cols
        for jf in cols:
            if np.isfinite(part["Z"][:, jf]).any():
                merged["Z"][:, jf] = part["Z"][:, jf]
                filled.add(int(jf))
    n_tot = merged["Z"].shape[1]
    missing = sorted(set(range(n_tot)) - filled)
    merged["meta"]["columns"] = sorted(filled)
    merged["meta"]["shards_merged"] = len(paths)
    print(f"merged {len(paths)} shard(s): {len(filled)}/{n_tot} columns filled")
    if missing:
        print(f"WARNING: {len(missing)} column(s) missing (failed/incomplete tasks): "
              f"{missing[:12]}{' ...' if len(missing) > 12 else ''}")
    if out_npz:
        save_scan(merged, out_npz)
    return merged


# --------------------------------------------------------------------------- #
# plot
# --------------------------------------------------------------------------- #
def plot_spectroscopy(result: Dict[str, Any], out: str = "figs/ge_map.png",
                      title: Optional[str] = None, annotate: bool = True,
                      cmap: str = "viridis") -> None:
    """Amplitude x frequency map coloured by the recorded level population."""
    freqs, amps, Z = result["freqs_GHz"], result["amps"], result["Z"]
    meta = result.get("meta", {})
    lvl = int(meta.get("level", 1))
    lname = {1: r"|e\rangle", 2: r"|f\rangle"}.get(lvl, rf"|{lvl}\rangle")

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200)
    pcm = ax.pcolormesh(freqs, amps, Z, shading="nearest", cmap=cmap,
                        vmin=0.0, vmax=1.0)
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(rf"${lname}$ population")

    if annotate:
        lines = result.get("lines", {})
        lo, hi = float(np.min(freqs)), float(np.max(freqs))
        for label, f0 in sorted(lines.items(), key=lambda kv: kv[1]):
            if not (lo <= f0 <= hi):
                continue
            ax.axvline(f0, color="w", lw=0.7, ls=":", alpha=0.8, zorder=3)
            ax.text(f0, amps[-1], f" {label}", color="w", fontsize=8.5,
                    rotation=90, va="top", ha="left", zorder=4)

    ax.set_xlabel("drive frequency (GHz)")
    ax.set_ylabel(r"drive amplitude  $|\eta|$")
    ax.set_title(title or rf"${lname}$ population vs drive frequency and power",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    fig.savefig(out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out, "and", out.rsplit(".", 1)[0] + ".pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replot", default=None,
                    help="render from a stored .npz instead of scanning")
    ap.add_argument("--device", default=None)
    ap.add_argument("--f-lo", type=float, default=1.5)
    ap.add_argument("--f-hi", type=float, default=3.8)
    ap.add_argument("--f-points", type=int, default=81)
    ap.add_argument("--amp-lo", type=float, default=0.0)
    ap.add_argument("--amp-hi", type=float, default=2.0)
    ap.add_argument("--amp-points", type=int, default=31)
    ap.add_argument("--probe-ns", type=float, default=200.0)
    ap.add_argument("--target-mode", type=int, default=0)
    ap.add_argument("--level", type=int, default=1, help="1 = |e>, 2 = |f>")
    ap.add_argument("--n-times", type=int, default=41,
                    help="output times for --reduce max/mean (ignored for 'final')")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--nsteps", type=int, default=500000)
    ap.add_argument("--envelope", choices=["constant", "raised_cosine"],
                    default="constant")
    ap.add_argument("--reduce", choices=["final", "max", "mean"], default="final",
                    help="population reduction over the probe: 'final' matches a "
                         "fixed-duration measurement (shows Rabi fringes, and "
                         "speckles on a coarse grid); 'max' maps where transitions "
                         "live without Rabi-phase aliasing")
    ap.add_argument("--spec-abs-GHz", type=float, default=None)
    ap.add_argument("--nproc", type=int, default=1,
                    help="worker processes over frequency columns (within one node)")
    ap.add_argument("--shard", type=int, default=None,
                    help="array sharding: this task's index (0-based). Columns are "
                         "taken strided (shard::nshards) so per-column cost, which "
                         "grows with drive frequency, balances across tasks.")
    ap.add_argument("--nshards", type=int, default=None,
                    help="array sharding: total number of shards")
    ap.add_argument("--merge", default=None,
                    help="merge <PREFIX>_shard*.npz into one grid and plot it")
    ap.add_argument("--save-data", default=None, help="write the grid to this .npz")
    ap.add_argument("--save-csv", default=None, help="also write long-form CSV")
    ap.add_argument("--out", default="figs/ge_map.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()

    if args.merge:
        result = merge_shards(args.merge, out_npz=args.save_data)
    elif args.replot:
        result = load_scan(args.replot)
        print(f"loaded {args.replot}: {result['Z'].shape} grid, "
              f"meta = {result['meta']}")
    else:
        if not args.device:
            ap.error("--device is required unless --replot is given")
        try:
            import qutip  # noqa: F401
        except ModuleNotFoundError:
            raise SystemExit(
                "spectroscopy scans integrate the exact Hamiltonian with QuTiP "
                "sesolve (via ZhouCoupler), so qutip must be installed:\n"
                "    uv pip install qutip\n"
                "Plot-only modes (--replot / --merge) do not need it.")
        from paths import resolve_device
        from device_utils import load_device
        cfg = load_device(resolve_device(args.device))
        if float(cfg.get("g4_GHz", 0.0)) == 0.0:
            print("NOTE: g4_GHz = 0, so the expansion is cubic: the ge line and the "
                  "ge/2 subharmonic are reachable, ge/3 is NOT (it needs g4).")
        columns = None
        if args.shard is not None:
            if not args.nshards:
                ap.error("--shard requires --nshards")
            columns = list(range(args.shard, args.f_points, args.nshards))
            if not columns:
                print(f"shard {args.shard} has no columns of {args.f_points} "
                      f"across {args.nshards} shards -- nothing to do")
                return
            print(f"shard {args.shard}/{args.nshards}: {len(columns)} column(s)")
        result = scan_ge(cfg, f_lo=args.f_lo, f_hi=args.f_hi, f_points=args.f_points,
                         amp_lo=args.amp_lo, amp_hi=args.amp_hi,
                         amp_points=args.amp_points, probe_ns=args.probe_ns,
                         target_mode=args.target_mode, level=args.level,
                         envelope=args.envelope, reduce=args.reduce,
                         n_times=args.n_times, rtol=args.rtol, atol=args.atol,
                         nsteps=args.nsteps,
                         spec_abs_GHz=args.spec_abs_GHz,
                         nproc=args.nproc, columns=columns)
        if args.save_data:
            path = args.save_data
            if args.shard is not None:                # one file per array task
                base = path[:-4] if path.endswith(".npz") else path
                path = f"{base}_shard{args.shard:03d}.npz"
            save_scan(result, path)
        if args.save_csv:
            export_csv(result, args.save_csv)
        if args.shard is not None:
            print("shard complete; merge with:  --merge "
                  f"{(args.save_data or 'scan')[:-4] if (args.save_data or '').endswith('.npz') else (args.save_data or 'scan')}")
            return                                    # a partial grid is not plottable

    plot_spectroscopy(result, out=args.out, title=args.title,
                      annotate=not args.no_annotate, cmap=args.cmap)
    print("  expected features:", ", ".join(
        f"{k}={v:.3f}GHz" for k, v in sorted(result["lines"].items(),
                                             key=lambda kv: kv[1])))


if __name__ == "__main__":
    main()