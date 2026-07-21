"""
Gaussian Mixture Threshold Estimation
====================================

Module
------
Micro.micro_threshold

Purpose
-------
Estimate the consensus adjacent-window NMI threshold using a two-component
Gaussian Mixture Model (GMM2).

Scientific Role
---------------
NeuroSeq identifies temporal state transitions by thresholding adjacent-window
NMI values. Rather than requiring a manually selected threshold, this module
fits a two-component Gaussian mixture to the NMI distribution and estimates the
decision boundary separating highly similar adjacent windows from likely state
transitions.

The preferred threshold is the intersection of the two Gaussian probability
densities whenever a valid solution exists. Robust fallback strategies are used
for degenerate or poorly conditioned datasets.

Outputs
-------
This module returns in-memory threshold estimates and summary tables used by
later Micro analysis steps.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def _gmm2_intersection(values: np.ndarray, n_init: int = 10, random_state: int = 42) -> float:
    """
    Estimate the decision threshold separating two Gaussian mixture components.

    Parameters
    ----------
    values : ndarray
        Adjacent-window NMI values.
    n_init : int, optional
        Number of GMM initializations.
    random_state : int, optional
        Random seed used during model fitting.

    Returns
    -------
    float
        Estimated intersection threshold between the low-mean and high-mean
        Gaussian components.

    Notes
    -----
    The preferred threshold is the intersection between the two fitted
    Gaussian densities. When a stable intersection cannot be determined,
    robust median or midpoint fallbacks are returned.
    """

    # Remove invalid numerical values.
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    # Degenerate datasets cannot support reliable mixture estimation.
    if x.size == 0:
        return float("nan")
    if x.size < 5:
        return float(np.nanmedian(x))
    if float(np.nanmax(x) - np.nanmin(x)) <= 1e-12:
        return float(np.nanmedian(x))

    vals = x.reshape(-1, 1)

    # Fit a two-component Gaussian mixture.
    try:
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            n_init=int(n_init),
            random_state=int(random_state),
        )
        gmm.fit(vals)
    except Exception:
        return float(np.nanmedian(x))

    pi = gmm.weights_.astype(float)
    mu = gmm.means_.flatten().astype(float)

    # Recover component variances.
    sig2 = np.zeros(2, dtype=float)
    for k in range(2):
        cov = np.asarray(gmm.covariances_[k], dtype=float)
        sig2[k] = float(cov.reshape(-1).mean())
        sig2[k] = max(sig2[k], 1e-12)

    # Identify low- and high-mean Gaussian components.
    lo_comp = int(np.argmin(mu))
    hi_comp = int(np.argmax(mu))

    pi0, pi1 = pi[lo_comp], pi[hi_comp]
    mu0, mu1 = mu[lo_comp], mu[hi_comp]
    s0, s1 = sig2[lo_comp], sig2[hi_comp]

    # Solve analytically for the Gaussian intersection.
    a = 0.5 * ((1.0 / s1) - (1.0 / s0))
    b = (mu0 / s0) - (mu1 / s1)
    c = (
        0.5 * ((mu1 * mu1 / s1) - (mu0 * mu0 / s0))
        + math.log((pi1 * math.sqrt(s0)) / (pi0 * math.sqrt(s1) + 1e-12) + 1e-12)
    )

    # Handle nearly linear solutions.
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return float(0.5 * (mu0 + mu1))
        r = -c / b
        return float(r) if np.isfinite(r) else float(0.5 * (mu0 + mu1))

    disc = b * b - 4.0 * a * c

    # No real-valued intersection exists.
    if disc < 0:
        return float(0.5 * (mu0 + mu1))

    r1 = (-b - math.sqrt(disc)) / (2.0 * a)
    r2 = (-b + math.sqrt(disc)) / (2.0 * a)

    lo, hi = sorted([float(mu0), float(mu1)])

    # Prefer intersections located between the two Gaussian means.
    candidates = [r for r in (r1, r2) if lo <= r <= hi and np.isfinite(r)]
    if candidates:
        return float(candidates[0])

    # Fall back to the midpoint between component means.
    return float(0.5 * (mu0 + mu1))


def _thresholds_from_nmi_gmm2(nmi_values: np.ndarray) -> pd.DataFrame:
    """
    Build the canonical GMM2 threshold summary table.

    Parameters
    ----------
    nmi_values : ndarray
        Adjacent-window NMI values.

    Returns
    -------
    pandas.DataFrame
        Summary table containing the estimated threshold together with
        diagnostic statistics.
    """

    # Retain only finite NMI values.
    vals = np.asarray(nmi_values, dtype=float)
    vals = vals[np.isfinite(vals)]

    # Estimate the preferred threshold.
    if vals.size == 0:
        thr = 0.5
    else:
        thr = _gmm2_intersection(vals)
        if not np.isfinite(thr):
            thr = float(np.nanmedian(vals))

    # Assemble the canonical summary table.
    return pd.DataFrame(
        [
            {
                "method": "gmm2",
                "threshold": float(thr),
                "n_finite_nmi": int(vals.size),
                "count_below_threshold": int(np.sum(vals < thr)) if vals.size else 0,
            }
        ]
    )


def _pick_gmm2_threshold(summary_df: pd.DataFrame) -> float:
    """
    Extract the preferred finite GMM2 threshold.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Threshold summary table.

    Returns
    -------
    float
        Preferred finite threshold used by the Micro pipeline.

    Notes
    -----
    Preference is always given to the GMM2 threshold. If no finite threshold
    exists, the first available finite threshold is used, followed by the
    default value of 0.5.
    """

    # Standardize the summary table.
    df = summary_df.copy()
    df["method"] = df["method"].astype(str).str.lower().str.strip()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df = df[df["threshold"].notna() & np.isfinite(df["threshold"])]

    # Prefer the canonical GMM2 estimate.
    hit = df[df["method"] == "gmm2"]
    if not hit.empty:
        return float(hit["threshold"].iloc[0])

    # Otherwise use the first available finite threshold.
    if not df.empty:
        return float(df["threshold"].iloc[0])

    # Final robust fallback.
    return 0.5