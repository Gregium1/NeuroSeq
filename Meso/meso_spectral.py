# ============================================================
#    meso_spectral.py — eigengaps, modularity, clustering   
# ============================================================

"""
Spectral Family Discovery and Candidate-k Evaluation
====================================================

Module
------
Meso.meso_spectral

Purpose
-------
Perform spectral analysis of the Meso affinity matrix, evaluate candidate
numbers of state families, and assign final spectral family labels.

Scientific Role
---------------
The Meso affinity matrix summarizes pairwise relationships between Micro
states. This module transforms that matrix into the symmetric normalized graph
Laplacian, computes its eigenspectrum, evaluates candidate eigengaps, and
generates spectral-clustering partitions.

For each candidate number of families, the resulting partition is also
characterized using weighted modularity and within-versus-between affinity
separation. These diagnostics quantify how strongly the proposed families
organize the state-affinity graph, while the configured selection rule
determines the final value of k.

Outputs
-------
This module returns normalized-Laplacian eigenvalue diagnostics, spectral
family labels, weighted modularity and affinity-separation measurements,
normalized candidate metrics, candidate-k tables, and the selected number of
state families.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

def normalized_laplacian_eigengaps(A: np.ndarray, *, min_k: int, max_k: int | None) -> dict:
    """Compute normalized-Laplacian eigenvalues and candidate eigengaps.
    
    The input affinity matrix is cleaned, symmetrized, and converted to the
    symmetric normalized graph Laplacian. Candidate gaps are evaluated over the
    configured k range.
    
    Parameters
    ----------
    A : numpy.ndarray
        Precomputed state-affinity matrix.
    min_k : int
        Smallest candidate number of clusters.
    max_k : int or None
        Largest candidate number of clusters.
    
    Returns
    -------
    dict
        Eigenvalues, eigengaps, candidate k values, and the largest-gap k."""
    # Convert the supplied affinity matrix to a finite symmetric graph.
    A0 = np.asarray(A, dtype=float).copy()
    A0 = np.nan_to_num(A0, nan=0.0, posinf=1.0, neginf=0.0)
    A0 = np.clip(A0, 0.0, 1.0)
    A0 = 0.5 * (A0 + A0.T)
    np.fill_diagonal(A0, 0.0)

    # Determine the number of Micro states represented in the graph.
    n = int(A0.shape[0])
    if n <= 1:
        return {"selected_k": 1, "candidate_ks": [], "eigenvalues": [], "eigengaps": []}

    # Restrict candidate family counts to valid graph dimensions.
    if max_k is None:
        max_k = min(50, n - 1)
    max_k = int(max(min(max_k, n - 1), 1))
    min_k = int(max(1, min(int(min_k), max_k)))

    # Compare observed within-family weight with the configuration-model expectation.
    deg = np.sum(A0, axis=1)
    deg_safe = np.where(deg > 1e-12, deg, 1.0)
    inv_sqrt = 1.0 / np.sqrt(deg_safe)
    L = np.eye(n) - (inv_sqrt[:, None] * A0 * inv_sqrt[None, :])
    L = 0.5 * (L + L.T)

    # Compute the ordered Laplacian eigenspectrum.
    vals = np.linalg.eigvalsh(L)
    vals = np.clip(np.sort(np.real(vals)), 0.0, None)
    candidate_ks = list(range(min_k, max_k + 1))
    gaps = [float(vals[k] - vals[k - 1]) if k < len(vals) else float("nan") for k in candidate_ks]

    # Identify the candidate associated with the largest finite eigengap.
    if np.any(np.isfinite(gaps)):
        best_i = int(np.nanargmax(np.asarray(gaps, dtype=float)))
        largest_k = int(candidate_ks[best_i])
    else:
        largest_k = int(min_k)

    return {
        "selected_k": largest_k,
        "candidate_ks": [int(x) for x in candidate_ks],
        "eigenvalues": [float(x) for x in vals.tolist()],
        "eigengaps": [float(x) if np.isfinite(x) else float("nan") for x in gaps],
        "gap_after_k": {str(k): float(g) if np.isfinite(g) else float("nan") for k, g in zip(candidate_ks, gaps)},
    }

def spectral_labels(
    A: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    eigen_solver: str = "arpack",
) -> np.ndarray:
    """Assign spectral-clustering labels from a precomputed affinity matrix.
    
    Parameters
    ----------
    A : numpy.ndarray
        Symmetric state-affinity matrix.
    n_clusters : int
        Requested number of spectral clusters.
    random_state : int
        Random seed passed to scikit-learn.
    eigen_solver : str, default="arpack"
        Eigenvalue solver used by spectral clustering.
    
    Returns
    -------
    numpy.ndarray
        Integer cluster label for each state."""
    # Clean and symmetrize the precomputed affinity matrix.
    A2 = np.asarray(A, dtype=float)
    A2 = np.nan_to_num(A2, nan=0.0, posinf=1.0, neginf=0.0)
    A2 = np.clip(A2, 0.0, 1.0)
    A2 = 0.5 * (A2 + A2.T)
    np.fill_diagonal(A2, 1.0)
    # Constrain the requested cluster count to the available states.
    n = int(A2.shape[0])
    k = int(max(1, min(int(n_clusters), n)))
    # Return the deterministic one-family partition when appropriate.
    if k <= 1:
        return np.zeros(n, dtype=int)
    # Cluster the spectral embedding using k-means label assignment.
    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        eigen_solver=str(eigen_solver),
        random_state=int(random_state),
    )
    return np.asarray(model.fit_predict(A2), dtype=int)

def weighted_modularity(A: np.ndarray, labels: np.ndarray) -> float:
    """Compute weighted graph modularity for a candidate partition.
    
    Parameters
    ----------
    A : numpy.ndarray
        Weighted affinity matrix.
    labels : numpy.ndarray
        Candidate cluster labels.
    
    Returns
    -------
    float
        Weighted modularity, or NaN when total graph weight is degenerate."""
    # Convert the supplied affinity matrix to a finite symmetric graph.
    A0 = np.asarray(A, dtype=float).copy()
    A0 = np.nan_to_num(A0, nan=0.0, posinf=1.0, neginf=0.0)
    A0 = np.clip(A0, 0.0, 1.0)
    A0 = 0.5 * (A0 + A0.T)
    np.fill_diagonal(A0, 0.0)
    labels = np.asarray(labels, dtype=int)
    # Compute twice the total graph weight used by modularity.
    m2 = float(np.sum(A0))
    if not np.isfinite(m2) or m2 <= 1e-12:
        return float("nan")
    # Compare observed within-family weight with the configuration-model expectation.
    deg = np.sum(A0, axis=1)
    same = labels[:, None] == labels[None, :]
    expected = np.outer(deg, deg) / m2
    q = np.sum((A0 - expected) * same) / m2
    return float(q)

def affinity_separation(A: np.ndarray, labels: np.ndarray) -> dict:
    """Summarize within-cluster and between-cluster affinities.
    
    Parameters
    ----------
    A : numpy.ndarray
        State-affinity matrix.
    labels : numpy.ndarray
        Candidate cluster labels.
    
    Returns
    -------
    dict
        Mean within affinity, mean between affinity, their difference and ratio,
        and the corresponding pair counts."""
    # Convert the supplied affinity matrix to a finite symmetric graph.
    A0 = np.asarray(A, dtype=float).copy()
    A0 = 0.5 * (A0 + A0.T)
    np.fill_diagonal(A0, np.nan)
    labels = np.asarray(labels, dtype=int)
    n = A0.shape[0]
    # Separate unique state pairs into within-family and between-family groups.
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            val = A0[i, j]
            if not np.isfinite(val):
                continue
            if labels[i] == labels[j]:
                within.append(float(val))
            else:
                between.append(float(val))
    # Summarize affinity concentration within and across candidate families.
    within_mean = float(np.nanmean(within)) if within else float("nan")
    between_mean = float(np.nanmean(between)) if between else float("nan")
    separation = within_mean - between_mean if np.isfinite(within_mean) and np.isfinite(between_mean) else float("nan")
    ratio = within_mean / between_mean if np.isfinite(within_mean) and np.isfinite(between_mean) and between_mean > 1e-12 else float("nan")
    return {
        "within_mean_affinity": within_mean,
        "between_mean_affinity": between_mean,
        "affinity_separation": float(separation),
        "affinity_ratio": float(ratio),
        "n_within_pairs": int(len(within)),
        "n_between_pairs": int(len(between)),
    }

def normalize_metric(vals: np.ndarray) -> np.ndarray:
    """Min-max normalize a diagnostic metric over its finite values.
    
    Degenerate finite inputs map to ones, while entirely non-finite inputs map to
    zeros, matching the original candidate-scoring behavior.
    
    Parameters
    ----------
    vals : numpy.ndarray
        Metric values to normalize.
    
    Returns
    -------
    numpy.ndarray
        Normalized values."""
    # Prepare affinity ratios for logarithmic normalization.
    vals = np.asarray(vals, dtype=float)
    # Normalize only across finite candidate measurements.
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals, dtype=float)
    vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    if not np.isfinite(vmax - vmin) or (vmax - vmin) <= 1e-12:
        return np.ones_like(vals, dtype=float)
    return np.nan_to_num((vals - vmin) / (vmax - vmin), nan=0.0)

def joint_eigengap_modularity_score(
    eigengap_norm: np.ndarray,
    modularity_norm: np.ndarray,
) -> np.ndarray:
    """
    Sum of normalized eigengap and normalized weighted modularity.

    Both criteria contribute equally to the final candidate-k score.
    """
    # Replace invalid normalized eigengap values with neutral zeros.
    g = np.nan_to_num(
        np.asarray(eigengap_norm, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    # Replace invalid normalized modularity values with neutral zeros.
    q = np.nan_to_num(
        np.asarray(modularity_norm, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return g + q

def normalize_affinity_ratio(vals: np.ndarray) -> np.ndarray:
    """
    Normalize the within/between affinity ratio for candidate-k scoring.

    Ratio interpretation:
        ratio = 1 means within-family affinity equals between-family affinity,
        so there is no useful separation. Its normalized value is forced to 0.

        ratio > 1 means within-family affinity is stronger than between-family
        affinity. Larger values indicate stronger separation.

    A log transform is used because ratios can become very large.
    """
    # Prepare affinity ratios for logarithmic normalization.
    vals = np.asarray(vals, dtype=float)
    out = np.zeros_like(vals, dtype=float)

    finite = np.isfinite(vals)
    if not np.any(finite):
        return out

    # Ratios at or below one encode no positive family separation.
    clipped = np.where(finite, np.maximum(vals, 1.0), np.nan)
    log_vals = np.log(clipped)
    finite_log = log_vals[np.isfinite(log_vals)]

    if finite_log.size == 0:
        return out

    max_log = float(np.nanmax(finite_log))

    if not np.isfinite(max_log) or max_log <= 1e-12:
        return out

    out = np.nan_to_num(log_vals / max_log, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)
    return out

def evaluate_candidate_ks(
    A: np.ndarray,
    *,
    min_k: int,
    max_k: int | None,
    top_n: int,
    random_state: int,
    eigen_solver: str = "arpack",
) -> Tuple[int, dict, pd.DataFrame]:
    """
    Select k* using the classical largest-eigengap rule.

        k* = argmax_k g_k

    where

        g_k = lambda_{k+1} - lambda_k

    is the eigengap of the normalized graph Laplacian.

    Weighted modularity, affinity separation, and affinity ratio are still
    computed for diagnostics, but they do not determine k*.
    """
    _ = top_n

    # Compute the complete eigengap profile over the candidate-k range.
    eig = normalized_laplacian_eigengaps(A, min_k=min_k, max_k=max_k)

    candidate_ks_all = [int(x) for x in eig.get("candidate_ks", [])]
    gaps_all = [float(x) for x in eig.get("eigengaps", [])]

    # Provide a deterministic fallback when no candidate range exists.
    if not candidate_ks_all:
        selected_k = 1
        df = pd.DataFrame(
            [
                {
                    "candidate_k": 1,
                    "eigengap": float("nan"),
                    "weighted_modularity": float("nan"),
                    "within_mean_affinity": float("nan"),
                    "between_mean_affinity": float("nan"),
                    "affinity_separation": float("nan"),
                    "affinity_ratio": float("nan"),
                    "n_within_pairs": 0,
                    "n_between_pairs": 0,
                    "eigengap_norm": 1.0,
                    "separation_norm": 1.0,
                    "modularity_norm": 1.0,
                    "affinity_ratio_norm": 0.0,
                    "combined_score": 1.0,
                    "largest_eigengap_candidate": True,
                    "best_modularity_candidate": True,
                    "best_affinity_ratio_candidate": True,
                    "selected": True,
                }
            ]
        )
        diag = {
            "reason": "largest_eigengap_single_state_fallback",
            "selected_k": int(selected_k),
            "largest_eigengap_k": int(selected_k),
            "candidate_ks_tested": [int(selected_k)],
            "score_reason": "largest_eigengap",
            "eigengap_diagnostics_all_k": eig,
        }
        return selected_k, diag, df

    selected_k = int(eig.get("selected_k", candidate_ks_all[0]))

    # Evaluate every candidate partition using graph and affinity diagnostics.
    rows = []
    for k, gap in zip(candidate_ks_all, gaps_all):
        labels_k = spectral_labels(
            A,
            n_clusters=int(k),
            random_state=random_state,
            eigen_solver=eigen_solver,
        )
        sep = affinity_separation(A, labels_k)
        q = weighted_modularity(A, labels_k)

        rows.append(
            {
                "candidate_k": int(k),
                "eigengap": float(gap) if np.isfinite(gap) else float("nan"),
                "weighted_modularity": q,
                **sep,
            }
        )

    df = pd.DataFrame(rows)

    # Normalize candidate diagnostics for direct comparison.
    df["eigengap_norm"] = normalize_metric(df["eigengap"].to_numpy(float))
    df["separation_norm"] = normalize_metric(df["affinity_separation"].to_numpy(float))
    df["modularity_norm"] = normalize_metric(df["weighted_modularity"].to_numpy(float))

    if "affinity_ratio" in df.columns:
        df["affinity_ratio_norm"] = normalize_affinity_ratio(
            df["affinity_ratio"].to_numpy(float)
        )
    else:
        df["affinity_ratio_norm"] = 0.0

    # In largest-eigengap mode, the combined score is just the normalized
    # eigengap. Modularity, separation, and ratio are diagnostics only.
    df["combined_score"] = df["eigengap_norm"]

    # Mark the strongest candidate under each diagnostic criterion.
    df["largest_eigengap_candidate"] = False
    if df["eigengap"].notna().any():
        df.loc[df["eigengap"].astype(float).idxmax(), "largest_eigengap_candidate"] = True

    df["best_modularity_candidate"] = False
    if df["weighted_modularity"].notna().any():
        df.loc[df["weighted_modularity"].astype(float).idxmax(), "best_modularity_candidate"] = True

    df["best_affinity_ratio_candidate"] = False
    if "affinity_ratio" in df.columns and df["affinity_ratio"].notna().any():
        df.loc[df["affinity_ratio"].astype(float).idxmax(), "best_affinity_ratio_candidate"] = True

    # Record the family count selected by the configured eigengap rule.
    df["selected"] = df["candidate_k"].astype(int) == int(selected_k)

    df = df.sort_values(
        ["selected", "eigengap", "weighted_modularity", "affinity_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    # Preserve the complete candidate-selection rationale and eigenspectrum.
    diag = {
        "reason": "largest_eigengap_k_selection",
        "selected_k": int(selected_k),
        "largest_eigengap_k": int(selected_k),
        "candidate_ks_tested": [int(x) for x in candidate_ks_all],
        "score_reason": "k_star_equals_argmax_eigengap",
        "eigengap_diagnostics_all_k": eig,
    }

    return selected_k, diag, df