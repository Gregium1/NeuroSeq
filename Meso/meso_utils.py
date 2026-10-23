"""
Shared Meso Utility Functions
=============================

Module
------
Meso.meso_utils

Purpose
-------
Provide small reusable utility functions shared throughout the NeuroSeq Meso
pipeline.

Scientific Role
---------------
Although these routines do not perform scientific analyses directly, they
provide the common infrastructure required by the Meso workflow. They ensure
consistent output-directory creation, deterministic alphabetic family labels,
and robust estimation of characteristic scales used throughout the pipeline.

Outputs
-------
This module returns utility values used by multiple Meso components,
including validated output directories, canonical alphabetic labels, and
stable numerical scale estimates.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """
    Create an output directory if it does not already exist.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path to create.

    Returns
    -------
    pathlib.Path
        Path object corresponding to the created or existing directory.

    Notes
    -----
    Parent directories are created automatically, allowing downstream Meso
    modules to write results without additional filesystem checks.
    """

    # Create the requested directory together with any missing parents.
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    return out


def letter_label(index: int) -> str:
    """
    Convert a non-negative integer into an alphabetic family label.

    Parameters
    ----------
    index : int
        Zero-based family index.

    Returns
    -------
    str
        Canonical alphabetic label (A, B, ..., Z, AA, AB, ...).

    Notes
    -----
    This deterministic mapping provides human-readable family identifiers
    while preserving the underlying integer ordering.
    """

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    k = int(index)

    # Reject invalid negative family indices.
    if k < 0:
        raise ValueError("index must be non-negative")

    # Convert the integer index into an Excel-style alphabetic label.
    while True:
        out = chars[k % 26] + out
        k = k // 26 - 1
        if k < 0:
            return out


def safe_scale(values, fallback: float = 1.0) -> float:
    """
    Estimate a robust characteristic scale from numeric values.

    Parameters
    ----------
    values : array-like
        Numeric observations.
    fallback : float, default=1.0
        Value returned when no reliable scale can be estimated.

    Returns
    -------
    float
        Standard deviation when available, otherwise the median absolute value,
        or the supplied fallback.

    Notes
    -----
    This hierarchy provides stable scale estimates even for small or nearly
    degenerate datasets, preventing downstream numerical instability.
    """

    # Retain only finite numeric observations.
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]

    # Prefer the population standard deviation when informative.
    if vals.size > 1:
        sd = float(np.nanstd(vals, ddof=0))
        if np.isfinite(sd) and sd > 1e-12:
            return sd

    # Otherwise use the median absolute value as a robust scale estimate.
    if vals.size:
        med = float(np.nanmedian(np.abs(vals)))
        if np.isfinite(med) and med > 1e-12:
            return med

    # Fall back to the user-supplied default scale.
    return float(fallback)