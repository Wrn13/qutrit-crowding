"""Durable, tailable progress logging for long solver runs.

Plain ``print()`` is line-buffered when stdout isn't a tty, so under ``srun`` on
SLURM the progress prints from a long calibration/sweep run can sit unflushed
until the job finishes. ``setup_run_logger`` gives callers a logger that writes
timestamped lines to stdout -- which SLURM already captures into that job's own
``slurm-<jobid>.out`` -- and, optionally, to a log file colocated with the run's
outputs. Concurrent jobs (e.g. one per device) that don't pass an explicit,
per-job ``log_path`` therefore land in separate SLURM output files instead of
interleaving into one shared file.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_run_logger(log_path: Optional[str], name: str) -> logging.Logger:
    """Logger that writes timestamped lines to stdout and, if given, ``log_path``.

    Parameters
    ----------
    log_path : str, optional
        File to append to; its parent directory is created if missing. Leave
        unset to log to stdout only -- the right default for concurrent jobs
        that would otherwise share (and interleave writes into) one file.
    name : str
        Logger name. Use a name derived from ``log_path`` so concurrent runs
        writing to different files don't share (and duplicate onto) handlers.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        handlers = [logging.StreamHandler(sys.stdout)]
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            handlers.append(logging.FileHandler(log_path))
        for handler in handlers:
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger
