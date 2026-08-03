"""Named operating points stored in a device JSON.

A calibration scan (``calibration_map.py``) finds a good (pump amplitude, pump
frequency offset) for a given gate context. Rather than re-deriving that tune-up
inside every sweep -- which is what rails when the AC-Stark shift and the Rabi
amplitude feed back on each other -- the point is saved once under a name and
reused across experiments:

    "operating_points": {
      "wb57_bare": {
        "amp_scale": 0.9625, "wp_offset_GHz": 0.002142, "t_g_ns": 92.59,
        "wa_GHz": 3.5, "wb_GHz": 5.7, "spec_abs_GHz": null,
        "drag_beat_GHz": null, "metric": "fidelity", "score": 0.9944,
        "source": "calibration_map", "created": "..."
      }
    }

The context keys (wa/wb/t_g/spectator/DRAG) are recorded so a point can be
checked against the run that is about to use it -- an operating point calibrated
for a different pair or gate length is not valid, and ``check_context`` reports
the mismatch rather than silently applying it.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

#: keys that describe WHERE a point was calibrated (used for validation)
CONTEXT_KEYS = ("wa_GHz", "wb_GHz", "t_g_ns", "spec_abs_GHz", "drag_beat_GHz")
#: keys that the point actually SETS on a run
SETTING_KEYS = ("amp_scale", "wp_offset_GHz", "t_g_ns")


def list_points(config: Dict[str, Any]) -> List[str]:
    """Names of the operating points defined in a loaded device config."""
    return sorted((config.get("operating_points") or {}).keys())


def get_point(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Fetch one operating point by name from a loaded device config.

    Parameters
    ----------
    config : dict
        Merged device configuration (from ``device_utils.load_device``).
    name : str
        Operating-point name.

    Returns
    -------
    dict
        The stored record.

    Raises
    ------
    KeyError
        If no point of that name exists (the message lists what is available).
    """
    points = config.get("operating_points") or {}
    if name not in points:
        raise KeyError(f"no operating point {name!r} in device "
                       f"(available: {sorted(points) or 'none'})")
    return dict(points[name])


def save_point(device_path: str, name: str, record: Dict[str, Any],
               overwrite: bool = False) -> Dict[str, Any]:
    """Write an operating point into a device JSON file, in place.

    Only the device file's own keys are touched (the DEFAULT_CONFIG merge that
    ``load_device`` performs is not written back).

    Parameters
    ----------
    device_path : str
        Path to the device JSON.
    name : str
        Name to store the point under.
    record : dict
        The point (amp_scale / wp_offset_GHz plus context and provenance).
    overwrite : bool, default False
        Allow replacing an existing point of the same name.

    Returns
    -------
    dict
        The record as written (with ``created`` and ``source`` filled in).
    """
    with open(device_path) as fh:
        raw = json.load(fh)
    points = raw.setdefault("operating_points", {})
    if name in points and not overwrite:
        raise SystemExit(f"operating point {name!r} already exists in {device_path}; "
                         f"pass overwrite=True (CLI: --overwrite) to replace it")
    rec = dict(record)
    rec.setdefault("source", "calibration_map")
    rec.setdefault("created", datetime.datetime.now().isoformat(timespec="seconds"))
    points[name] = rec
    tmp = device_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(raw, fh, indent=2)
    os.replace(tmp, device_path)
    return rec


def check_context(point: Dict[str, Any], config: Dict[str, Any],
                  t_g: Optional[float] = None, *,
                  wa_GHz: Optional[float] = None, wb_GHz: Optional[float] = None,
                  spec_abs_GHz: Optional[float] = None,
                  tol_GHz: float = 1e-6, tol_ns: float = 1e-3) -> List[str]:
    """Return a list of human-readable context mismatches (empty == consistent).

    Parameters
    ----------
    point : dict
        A stored operating point.
    config : dict
        The configuration the point is about to be applied to.
    t_g : float, optional
        Gate duration of the run (ns); defaults to ``config['t_g_ns']``.
    wa_GHz, wb_GHz, spec_abs_GHz : float, optional
        Run context overriding the config values (e.g. a target sweep's swept w_b).
    tol_GHz, tol_ns : float
        Comparison tolerances.
    """
    pair = list(config.get("qubit_freqs_GHz", [None, None]))
    run = {
        "wa_GHz": pair[0] if wa_GHz is None else wa_GHz,
        "wb_GHz": pair[1] if wb_GHz is None else wb_GHz,
        "t_g_ns": config.get("t_g_ns") if t_g is None else t_g,
        "spec_abs_GHz": spec_abs_GHz,
    }
    issues: List[str] = []
    for key, value in run.items():
        want = point.get(key)
        if want is None or value is None:
            continue
        tol = tol_ns if key.endswith("_ns") else tol_GHz
        if abs(float(want) - float(value)) > tol:
            issues.append(f"{key}: point={want} vs run={value}")
    return issues


def apply_point(config: Dict[str, Any], point: Dict[str, Any],
                set_t_g: bool = True) -> Dict[str, Any]:
    """Return a copy of ``config`` with the point's settings applied.

    Sets ``amp_scale`` and ``wp_offset_GHz`` (and ``t_g_ns`` unless ``set_t_g``
    is False). Context keys are not applied -- they describe where the point was
    calibrated, and are for ``check_context`` to validate against.
    """
    out = dict(config)
    for key in SETTING_KEYS:
        if key == "t_g_ns" and not set_t_g:
            continue
        if point.get(key) is not None:
            out[key] = point[key]
    return out


def resolve(config: Dict[str, Any], name: str, *, t_g: Optional[float] = None,
            wa_GHz: Optional[float] = None, wb_GHz: Optional[float] = None,
            spec_abs_GHz: Optional[float] = None, strict: bool = False,
            set_t_g: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Look up, validate, and apply an operating point in one call.

    Prints a warning for any context mismatch (or raises when ``strict``).

    Returns
    -------
    (dict, dict)
        The updated config and the point that was applied.
    """
    try:
        point = get_point(config, name)
    except KeyError as exc:
        raise SystemExit(str(exc).strip('"')) from None
    # when the point supplies t_g itself there is nothing to disagree with; only
    # compare gate lengths when the run overrides it (set_t_g False).
    t_g_check = point.get("t_g_ns") if set_t_g else t_g
    issues = check_context(point, config, t_g_check, wa_GHz=wa_GHz, wb_GHz=wb_GHz,
                           spec_abs_GHz=spec_abs_GHz)
    if issues:
        msg = (f"operating point {name!r} was calibrated in a different context: "
               + "; ".join(issues))
        if strict:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")
    return apply_point(config, point, set_t_g=set_t_g), point


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect operating points in a device JSON.")
    ap.add_argument("--device", required=True)
    ap.add_argument("--show", default=None, help="print one point by name")
    args = ap.parse_args()

    from paths import resolve_device
    from device_utils import load_device
    path = resolve_device(args.device)
    config = load_device(path)
    if args.show:
        print(json.dumps(get_point(config, args.show), indent=2))
        return
    names = list_points(config)
    if not names:
        print(f"{path}: no operating points defined")
        return
    print(f"{path}: {len(names)} operating point(s)")
    for name in names:
        p = config["operating_points"][name]
        ctx = " ".join(f"{k}={p[k]}" for k in CONTEXT_KEYS if p.get(k) is not None)
        print(f"  {name:24s} amp={p.get('amp_scale')} "
              f"wp_off={p.get('wp_offset_GHz')} GHz  [{ctx}]"
              + (f"  {p.get('metric')}={p.get('score'):.5f}"
                 if isinstance(p.get("score"), (int, float)) else ""))


if __name__ == "__main__":
    main()