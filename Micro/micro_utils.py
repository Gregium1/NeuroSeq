"""
Shared Utilities for the NeuroSeq Micro Pipeline
================================================

Module
------
Micro.micro_utils

Purpose
-------
Provide reusable utility functions shared across the NeuroSeq Micro pipeline.

Scientific Role
---------------
These utilities support the infrastructure of the Micro analysis without
performing scientific inference themselves. They provide standardized routines
for directory creation, configuration handling, spike-train loading, filename
parsing, JSON serialization, neuron indexing, and canonical file discovery.

Maintaining these operations in a common module ensures consistent behavior
across all stages of the Micro workflow.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _ensure_dir(p: str | Path) -> None:
    """
    Create a directory together with any missing parent directories.

    Parameters
    ----------
    p : str or Path
        Directory to create.

    Notes
    -----
    Existing directories are left unchanged.
    """
    Path(p).mkdir(parents=True, exist_ok=True)


def _tag_float(x: float, ndp: int = 4) -> str:
    """
    Convert a floating-point value into a filesystem-safe tag.

    Parameters
    ----------
    x : float
        Value to encode.
    ndp : int, optional
        Number of decimal places.

    Returns
    -------
    str
        Filesystem-safe string representation.
    """
    s = f"{float(x):.{ndp}f}".rstrip("0").rstrip(".")
    return s.replace(".", "p").replace("-", "m")


def _write_json(path: str | Path, obj: dict[str, Any]) -> None:
    """
    Write a dictionary as formatted UTF-8 JSON.

    Parameters
    ----------
    path : str or Path
        Output JSON file.
    obj : dict
        Dictionary to serialize.
    """
    path = Path(path)

    # Ensure the destination directory exists.
    _ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _cfg_value(cfg: Any, name: str, default: Any = None) -> Any:
    """
    Retrieve a configuration value from either a dictionary or object.

    Parameters
    ----------
    cfg : Any
        Configuration container.
    name : str
        Configuration key.
    default : Any, optional
        Default value.

    Returns
    -------
    Any
        Requested configuration value.
    """
    if isinstance(cfg, dict):
        if name in cfg:
            return cfg[name]
        micro = cfg.get("micro", {})
        if isinstance(micro, dict) and name in micro:
            return micro[name]
        return default
    return getattr(cfg, name, default)


def _load_spike_times(csv_path: str | Path) -> np.ndarray:
    """
    Load, clean, and sort spike times from one neuron.

    Parameters
    ----------
    csv_path : str or Path
        Spike-time CSV.

    Returns
    -------
    ndarray
        Sorted finite spike times.
    """

    # Load spike times regardless of one- or two-column layout.
    df = pd.read_csv(csv_path, header=None)

    col = 1 if df.shape[1] >= 2 else 0

    # Remove invalid numerical entries.
    vals = pd.to_numeric(df.iloc[:, col], errors="coerce").dropna().to_numpy(float)
    vals = vals[np.isfinite(vals)]

    # Return spike times in chronological order.
    return np.sort(vals)


def _su_sort_key(x: str | Path) -> int:
    """
    Extract the numeric identifier from an SU filename.

    Parameters
    ----------
    x : str or Path
        Neuron identifier.

    Returns
    -------
    int
        Numeric neuron index suitable for deterministic sorting.
    """
    m = re.search(r"SU_(\d+)", str(x))
    return int(m.group(1)) if m else 10**12


def _as_su_filter(selected_su_ids: list[str] | None) -> set[str] | None:
    """
    Convert an optional neuron list into a membership set.

    Parameters
    ----------
    selected_su_ids : list[str] or None

    Returns
    -------
    set[str] or None
    """
    if selected_su_ids is None:
        return None
    return set(str(x) for x in selected_su_ids)


# Public aliases retained for the newer naming convention.
ensure_dir = _ensure_dir
tag_float = _tag_float
write_json = _write_json
cfg_value = _cfg_value
load_spike_times = _load_spike_times
su_sort_key = _su_sort_key
as_su_filter = _as_su_filter


def _find_latest_counts_csv(search_root: str | Path, window_size: float) -> str:
    """
    Locate the newest compatible spike-count table.

    Parameters
    ----------
    search_root : str or Path
        Root search directory.
    window_size : float
        Window size used during counting.

    Returns
    -------
    str
        Path to the newest compatible counts table.
    """
    root = Path(search_root)

    # Match canonical NeuroSeq counts filenames.
    pattern = f"spike_counts_allSUs_*_win{float(window_size):.3f}s.csv"

    matches = sorted(
        root.rglob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not matches:
        raise FileNotFoundError(f"No counts CSV matching {pattern!r} under {root}")

    return str(matches[0])


def _parse_counts_bounds(path: str | Path) -> tuple[float | None, float | None, float | None]:
    """
    Recover recording bounds encoded in a canonical counts filename.

    Parameters
    ----------
    path : str or Path
        Counts filename.

    Returns
    -------
    tuple
        Recording start time, end time, and window size. Missing values are
        returned as None.
    """
    name = Path(path).name

    # Parse canonical NeuroSeq filename conventions.
    match = re.search(
        r"spike_counts_allSUs_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_win(\d+(?:\.\d+)?)s\.csv$",
        name,
    )

    if not match:
        return None, None, None

    return tuple(float(x) for x in match.groups())


find_latest_counts_csv = _find_latest_counts_csv
parse_counts_bounds = _parse_counts_bounds