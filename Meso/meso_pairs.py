#  Meso/meso_pairs.py

"""
Pairwise State Similarity and Firing-Rate Distance
==================================================

Module
------
Meso.meso_pairs

Purpose
-------
Compute pairwise similarity and distance measurements between NeuroSeq Micro
states for use in the Meso-level affinity analysis.

Scientific Role
---------------
Each Proto-state is characterized by two complementary representations: a
community assignment describing its neuronal organization and a firing-rate
vector describing its population activity. This module compares every valid
pair of states using normalized mutual information to quantify similarity
between community structures and an aligned root-mean-square distance to
quantify differences between firing-rate profiles.

Together, these measurements provide the structural and activity-based
information required to construct the Meso similarity score and identify
higher-order families of related neural states.

Outputs
-------
This module returns the ordered Proto-state identifiers and a canonical
pairwise table containing the state indices, normalized mutual information,
and aligned firing-rate distance for every valid state pair.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def _aligned_rms_distance(values_1, neurons_1, values_2, neurons_2) -> float:
    """
    Compute the firing-rate distance between two neuron-aligned state vectors.

    Parameters
    ----------
    values_1 : array-like
        Firing-rate values for the first Proto-state.
    neurons_1 : array-like
        Neuron identifiers corresponding to ``values_1``.
    values_2 : array-like
        Firing-rate values for the second Proto-state.
    neurons_2 : array-like
        Neuron identifiers corresponding to ``values_2``.

    Returns
    -------
    float
        Root-mean-square firing-rate distance evaluated across neurons shared
        by both states. ``NaN`` is returned when no shared neurons or no finite
        aligned values are available.

    Notes
    -----
    The alignment step ensures that firing rates are compared only between
    matching neurons. This prevents differences in neuron ordering or state
    coverage from introducing artificial distance.
    """

    # Associate every neuron identifier with its state-specific firing rate.
    map_1 = dict(zip(map(str, neurons_1), np.asarray(values_1, dtype=float)))
    map_2 = dict(zip(map(str, neurons_2), np.asarray(values_2, dtype=float)))

    # Restrict the comparison to neurons represented in both Proto-states.
    shared = sorted(set(map_1).intersection(map_2))
    if not shared:
        return float("nan")

    # Construct neuron-aligned firing-rate vectors.
    a = np.asarray([map_1[n] for n in shared], dtype=float)
    b = np.asarray([map_2[n] for n in shared], dtype=float)

    # Exclude non-finite measurements from the pairwise distance.
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")

    # Return the root-mean-square difference across shared valid neurons.
    return float(np.sqrt(np.mean((a[finite] - b[finite]) ** 2)))


def compute_state_pairs(
    state_vectors: dict[str, dict[str, Any]],
    state_clusters: dict[str, dict[str, Any]],
) -> tuple[list[str], pd.DataFrame]:
    """
    Compute pairwise community similarity and firing-rate distance.

    Parameters
    ----------
    state_vectors : dict[str, dict[str, Any]]
        Mapping from each Proto-state identifier to its firing-rate vector,
        neuron identifiers, and associated state metadata.
    state_clusters : dict[str, dict[str, Any]]
        Mapping from each Proto-state identifier to its neuron identifiers and
        Louvain community assignments.

    Returns
    -------
    tuple[list[str], pandas.DataFrame]
        Ordered Proto-state identifiers and a pairwise table with columns
        ``i``, ``j``, ``NMI``, and ``Dist``.

    Notes
    -----
    Normalized mutual information measures similarity between the neuronal
    community assignments of two states after matching neurons by identifier.
    The firing-rate distance is calculated independently from the aligned
    population-rate vectors.

    State pairs without shared cluster neurons are omitted. Pairs with an
    undefined firing-rate distance are also excluded from the final table.
    """

    # Preserve the canonical ordering of Proto-states from the vector mapping.
    state_ids = list(state_vectors)
    rows = []

    # Compare every unique unordered pair of Proto-states.
    for i, sid1 in enumerate(state_ids):
        c1 = state_clusters[sid1]

        # Build a unique neuron-to-community table for the first state.
        df1 = pd.DataFrame({
            "Neuron": c1["neurons"],
            "C1": c1["clusters"],
        }).drop_duplicates("Neuron")

        for j in range(i + 1, len(state_ids)):
            sid2 = state_ids[j]
            c2 = state_clusters[sid2]

            # Build the corresponding neuron-to-community table for the
            # second state.
            df2 = pd.DataFrame({
                "Neuron": c2["neurons"],
                "C2": c2["clusters"],
            }).drop_duplicates("Neuron")

            # Align community assignments using neurons shared by both states.
            merged = df1.merge(df2, on="Neuron", how="inner")
            if merged.empty:
                continue

            # Quantify similarity between the two Louvain partitions.
            nmi = normalized_mutual_info_score(merged["C1"], merged["C2"])

            # Quantify the difference between aligned firing-rate profiles.
            v1, v2 = state_vectors[sid1], state_vectors[sid2]
            dist = _aligned_rms_distance(
                v1["vector"],
                v1["neurons"],
                v2["vector"],
                v2["neurons"],
            )

            # Retain only state pairs with a valid firing-rate distance.
            if np.isfinite(dist):
                rows.append((int(i), int(j), float(nmi), float(dist)))

    # Return the canonical pairwise measurements used by the Meso pipeline.
    return state_ids, pd.DataFrame(
        rows,
        columns=["i", "j", "NMI", "Dist"],
    )