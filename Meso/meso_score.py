# ============================================================
#      meso_score.py — exponential S-score construction     
# ============================================================

"""
Exponential Meso Similarity-Score Construction
==============================================

Module
------
Meso.meso_score

Purpose
-------
Construct the pairwise Meso similarity score from Micro-state firing-rate
distance and normalized mutual information measurements.

Scientific Role
---------------
The Meso analysis combines two complementary descriptions of Micro-state
similarity. Firing-rate distance measures differences in population activity,
whereas normalized mutual information measures similarity between neuronal
community assignments. Each measurement is transformed through a Gaussian
kernel, and the resulting similarities are multiplied to produce the final
S-score affinity.

This module also estimates an automatic S-score threshold from the geometry of
the sorted score distribution and materializes the complete symmetric
state-by-state affinity matrix used by spectral family discovery.

Outputs
-------
This module returns pairwise tables augmented with normalized distance,
distance similarity, NMI similarity, and S-score values. It also returns
threshold diagnostics and writes the canonical symmetric S-score matrix.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from .meso_utils import safe_scale


def _point_line_distances(x, y) -> np.ndarray:
    """
    Compute perpendicular distances from sampled points to the endpoint chord.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the sampled curve.

    Returns
    -------
    numpy.ndarray
        Per-point absolute perpendicular distances to the straight line joining
        the first and last samples.

    Notes
    -----
    These distances quantify how strongly each point departs from the linear
    trend connecting the endpoints. Large departures identify candidate elbow
    locations in the sorted S-score distribution.
    """

    # Convert the sampled coordinates to finite floating-point arrays.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Define the chord connecting the first and last curve samples.
    x1, y1 = float(x[0]), float(y[0])
    x2, y2 = float(x[-1]), float(y[-1])

    # Compute the chord length used to normalize perpendicular distances.
    denom = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    # Return zero distances when the endpoint chord is numerically degenerate.
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.zeros_like(y, dtype=float)

    # Evaluate the absolute point-to-line distance for every sampled score.
    return np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom


def _score_elbow_midpoint_threshold_from_scores(scores) -> dict:
    """
    Estimate the S-score threshold from the two strongest elbow candidates.

    The finite scores are sorted and normalized, perpendicular distances to the
    endpoint chord are computed, and the midpoint between the two strongest
    candidate locations is returned. Median fallbacks preserve behavior for
    small, flat, or invalid inputs.

    Parameters
    ----------
    scores : array-like
        Pairwise S-scores.

    Returns
    -------
    dict
        Threshold value together with elbow and largest-gap diagnostics.

    Notes
    -----
    The primary threshold is derived from the midpoint between the two score
    values associated with the strongest geometric elbow candidates. The
    largest adjacent score gap is retained as a diagnostic but does not replace
    the elbow-based threshold.
    """

    # Retain only finite S-scores for threshold estimation.
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]

    # Report an undefined threshold when no valid scores are available.
    if s.size == 0:
        return {"S_thr": float("nan"), "reason": "no_finite_scores"}

    # Use the median when too few scores exist for stable elbow geometry.
    if s.size < 4:
        return {"S_thr": float(np.nanmedian(s)), "reason": "fallback_median_small_n"}

    # Sort the S-scores to form the empirical score curve.
    s_sorted = np.sort(s)
    s_min, s_max = float(s_sorted[0]), float(s_sorted[-1])
    score_range = s_max - s_min

    # Use the median when the score distribution is effectively flat.
    if not np.isfinite(score_range) or score_range <= 1e-12:
        return {"S_thr": float(np.nanmedian(s_sorted)), "reason": "fallback_median_flat_score_range"}

    # Normalize both curve axes to the unit interval.
    x = np.linspace(0.0, 1.0, len(s_sorted))
    y = (s_sorted - s_min) / score_range

    # Measure the departure of each point from the endpoint chord.
    distances = _point_line_distances(x, y)
    distances[0] = -np.inf
    distances[-1] = -np.inf

    # Rank valid interior samples by perpendicular distance.
    order = [int(i) for i in np.argsort(distances)[::-1] if np.isfinite(distances[int(i)])]

    # Fall back to the median if no valid elbow candidate remains.
    if not order:
        return {"S_thr": float(np.nanmedian(s_sorted)), "reason": "fallback_median_no_elbow_candidate"}

    # Select the two strongest elbow candidates.
    idx1 = order[0]
    idx2 = order[1] if len(order) > 1 else min(idx1 + 1, len(s_sorted) - 1)
    lo, hi = min(idx1, idx2), max(idx1, idx2)

    # Define the threshold as the midpoint of the two candidate score values.
    s_thr = float(0.5 * (float(s_sorted[lo]) + float(s_sorted[hi])))

    # Compute the largest adjacent score gap for diagnostic comparison.
    gaps = np.diff(s_sorted)
    if gaps.size and np.any(np.isfinite(gaps)):
        gap_idx = int(np.nanargmax(gaps))
        largest_gap_thr = float(0.5 * (s_sorted[gap_idx] + s_sorted[gap_idx + 1]))
        largest_gap_size = float(gaps[gap_idx])
    else:
        gap_idx = None
        largest_gap_thr = float("nan")
        largest_gap_size = float("nan")

    # Return the threshold together with complete geometric diagnostics.
    return {
        "S_thr": s_thr,
        "reason": "max_perpendicular_distance_two_point_midpoint",
        "elbow_indices": [int(lo), int(hi)],
        "elbow_values": [float(s_sorted[lo]), float(s_sorted[hi])],
        "largest_gap_index": gap_idx,
        "largest_gap_thr": largest_gap_thr,
        "largest_gap_size": largest_gap_size,
        "S_min": s_min,
        "S_max": s_max,
        "n_scores": int(len(s_sorted)),
    }


def compute_s_scores(pairs: pd.DataFrame, *, n_states: int) -> Tuple[pd.DataFrame, dict]:
    """
    Compute the pairwise exponential S-score and its automatic threshold.

    Distance is normalized by the maximum finite distance. Distance similarity
    and NMI similarity are then computed with Gaussian kernels and multiplied
    to obtain the final affinity score.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Pairwise state table containing ``Dist`` and ``NMI`` columns.
    n_states : int
        Number of Micro states. Retained as part of the function interface.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Pair table augmented with S-score columns and a diagnostic dictionary.

    Notes
    -----
    The final affinity is defined as the product of a firing-rate similarity
    term and a community-structure similarity term. Consequently, a state pair
    receives a large S-score only when both representations indicate strong
    similarity.
    """

    # Preserve the original pairwise table before adding score measurements.
    out = pairs.copy()

    # Extract firing-rate distances and NMI values as numeric arrays.
    dist_vals = out["Dist"].to_numpy(float)
    nmi_vals = out["NMI"].to_numpy(float)

    # Isolate finite observations for robust scale estimation.
    finite_dist = dist_vals[np.isfinite(dist_vals)]
    finite_nmi = nmi_vals[np.isfinite(nmi_vals)]

    # Normalize firing-rate distances by the maximum observed finite distance.
    D_scale = float(np.nanmax(finite_dist)) if finite_dist.size else 1.0
    if not np.isfinite(D_scale) or D_scale <= 1e-12:
        D_scale = 1.0

    D_norm = np.clip(dist_vals / D_scale, 0.0, 1.0)

    # Estimate robust Gaussian-kernel widths for both similarity components.
    sigma_D = safe_scale(D_norm[np.isfinite(D_norm)], fallback=0.25)
    sigma_NMI = safe_scale(1.0 - finite_nmi, fallback=0.25)

    # Convert normalized firing-rate distance into a similarity measure.
    dist_sim = np.exp(-(D_norm ** 2) / (2.0 * sigma_D ** 2))

    # Convert deviation from perfect NMI into a community similarity measure.
    nmi_sim = np.exp(-((1.0 - nmi_vals) ** 2) / (2.0 * sigma_NMI ** 2))

    # Require simultaneous firing-rate and community similarity.
    S = dist_sim * nmi_sim
    S[~np.isfinite(S)] = np.nan

    # Attach all intermediate and final score quantities to the pair table.
    out["D_scale"] = float(D_scale)
    out["D_norm"] = D_norm
    out["DistSim"] = dist_sim
    out["NMISim"] = nmi_sim
    out["S_score"] = S

    # Estimate and constrain the automatic S-score threshold.
    elbow = _score_elbow_midpoint_threshold_from_scores(S)
    S_thr = float(np.clip(float(elbow["S_thr"]), 0.0, 1.0))

    # Preserve the complete score definition and threshold diagnostics.
    diag = {
        "score_formula": "DistSim * NMISim",
        "DistSim_formula": "exp(-D_norm^2/(2*sigma_D_norm^2)); D_norm=Dist/max(D)",
        "NMISim_formula": "exp(-(1-NMI)^2/(2*sigma_NMI^2))",
        "threshold_method": "score_elbow_midpoint",
        "D_scale": float(D_scale),
        "sigma_D_norm": float(sigma_D),
        "sigma_NMI": float(sigma_NMI),
        "S_elbow_thr": float(S_thr),
        "S_elbow": elbow,
    }

    return out, diag


def save_s_score_matrix(*, pairs_score: pd.DataFrame, state_ids: List[str], out_csv: str | Path) -> str:
    """
    Build and save the symmetric state-by-state S-score matrix.

    Parameters
    ----------
    pairs_score : pandas.DataFrame
        Pairwise table containing state indices ``i`` and ``j`` and
        ``S_score``.
    state_ids : list[str]
        State identifiers defining matrix row and column order.
    out_csv : str or pathlib.Path
        Destination CSV path.

    Returns
    -------
    str
        Path to the written matrix.

    Notes
    -----
    Pairwise affinities are mirrored across the matrix diagonal. Self-affinity
    is fixed at one, and a final symmetrization step protects against numerical
    or assignment asymmetry.
    """

    # Initialize the dense state-by-state affinity matrix.
    n = len(state_ids)
    A = np.zeros((n, n), dtype=float)

    # Insert each finite pairwise S-score symmetrically.
    for (i, j), val in zip(pairs_score[["i", "j"]].to_numpy(int), pairs_score["S_score"].to_numpy(float)):
        if np.isfinite(val):
            vv = float(np.clip(val, 0.0, 1.0))
            A[int(i), int(j)] = vv
            A[int(j), int(i)] = vv

    # Assign unit self-affinity and enforce exact matrix symmetry.
    np.fill_diagonal(A, 1.0)
    A = 0.5 * (A + A.T)

    # Save the canonical affinity matrix with state identifiers on both axes.
    pd.DataFrame(A, index=state_ids, columns=state_ids).to_csv(out_csv)

    return str(out_csv)