"""
paths.py
========

Repo-relative path resolution so the tools can be called from the
``time_dependent_solver`` folder with bare names:

    time_dependent_solver/          <- REPO_ROOT (this package's folder)
    |-- *.py                        <- code (this module lives here)
    |-- devices/                    <- device JSONs
    |-- results/                    <- sweep / calibration outputs
    +-- slurm/                      <- SLURM scripts

``--device dev.json`` resolves to ``devices/dev.json`` and default outputs land
under ``results/``. The root is discovered by walking up from this file until a
folder containing ``devices``/``results``/``slurm`` is found, so it still works if
the code sits one level below the data folders.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_root(start: Path) -> Path:
    """First ancestor of `start` (inclusive) that holds devices/, results/, or
    slurm/; falls back to `start` itself."""
    for d in (start, *start.parents):
        if (d / "devices").is_dir() or (d / "results").is_dir() or (d / "slurm").is_dir():
            return d
    return start


CODE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = _find_root(CODE_DIR)
DEVICES_DIR: Path = REPO_ROOT / "devices"
RESULTS_DIR: Path = REPO_ROOT / "results"
SLURM_DIR: Path = REPO_ROOT / "slurm"


def resolve_device(name: str) -> str:
    """Resolve a device path: use it if it exists, else look under devices/.

    Parameters
    ----------
    name : str
        A path or a bare filename.

    Returns
    -------
    str
        The resolved path (unchanged if already valid or if not found, so the
        caller's own error surfaces).
    """
    p = Path(name)
    if p.exists():
        return str(p)
    cand = DEVICES_DIR / name
    return str(cand) if cand.exists() else str(p)


def in_results(name: str) -> str:
    """Place a relative output name under results/ (absolute paths pass through).
    Ensures the parent directory exists.

    Parameters
    ----------
    name : str
        Output path or bare name.

    Returns
    -------
    str
        The resolved output path.
    """
    p = Path(name)
    if p.is_absolute():
        out = p
    elif p.parts and p.parts[0] == RESULTS_DIR.name:      # user already wrote results/...
        out = REPO_ROOT / p
    else:
        out = RESULTS_DIR / p
    os.makedirs(out.parent, exist_ok=True)
    return str(out)