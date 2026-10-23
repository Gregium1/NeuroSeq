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

    # Preserve the complete score definition and threshold diagnostics.
    diag = {
        "score_formula": "DistSim * NMISim",
        "DistSim_formula": "exp(-D_norm^2/(2*sigma_D_norm^2)); D_norm=Dist/max(D)",
        "NMISim_formula": "exp(-(1-NMI)^2/(2*sigma_NMI^2))",
        "D_scale": float(D_scale),
        "sigma_D_norm": float(sigma_D),
        "sigma_NMI": float(sigma_NMI),
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