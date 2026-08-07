"""Durable, tailable progress logging for long solver runs.

Plain ``print()`` is line-buffered when stdout isn't a tty, so under ``srun`` on
SLURM the progress prints from a long calibration/sweep run can sit unflushed
until the job finishes -- and SLURM's own captured ``.out``/``.err`` files live
in the submit directory, disconnected from the run's actual output directory.
``setup_run_logger`` gives callers a logger that writes timestamped lines to
both stdout and a log file colocated with the run's outputs, so progress is
visible with a plain ``tail -f`` while the job is in flight.
"""
from __future__ import annotations

import logging
import os
import sys


def setup_run_logger(log_path: str, name: str) -> logging.Logger:
    """Logger that writes timestamped lines to both stdout and ``log_path``.

    Parameters
    ----------
    log_path : str
        File to append to; its parent directory is created if missing.
    name : str
        Logger name. Use a name derived from ``log_path`` so concurrent runs
        writing to different files don't share (and duplicate onto) handlers.

    Returns
    -------
    logging.Logger
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger
