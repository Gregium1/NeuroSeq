# Meso/meso_proto_nmi.py

"""
Pairwise State Louvain NMI Analysis
==================================

Module
------
Mesoo.meso_proto_nmi

Purpose
-------
Compute pairwise Normalized Mutual Information (NMI) between the neuronal
Louvain community assignments of every pair of final Micro states.

Following construction of the consensus Proto-States and within-state Louvain
community detection, this module quantifies the similarity of neuronal
community organization across all state pairs.

The resulting pairwise NMI matrices provide one of the principal inputs to the
NeuroSeq Meso analysis.

Scientific Role
---------------
Each final Micro state is characterized by a Louvain partition assigning every
recorded neuron to a neuronal community.

For every pair of states,

    (State_i, State_j),

the Louvain community labels are compared using Normalized Mutual Information
(NMI).

Only neurons present in both states contribute to the comparison.

The resulting NMI values quantify the similarity of neuronal community
organization between states:

- NMI ≈ 1 indicates nearly identical neuronal community structure.
- NMI ≈ 0 indicates substantially different community organization.

The complete pairwise similarity matrix serves as one component of the Meso
state-affinity calculation.

Outputs
-------
This module produces two canonical files:

``proto_state_louvain_nmi.csv``
    Square symmetric matrix whose rows and columns correspond to the final
    Micro states.

``proto_state_louvain_nmi_long.csv``
    Long-form table containing every pairwise state comparison together with
    the number of neurons included in each comparison.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def _state_sort_key(label: str) -> int:
    """
    Convert a NeuroSeq state label into a sortable integer.

    Parameters
    ----------
    label : str
        State label of the form

        ``State_1``, ``State_2``, ...

    Returns
    -------
    int
        Integer suffix extracted from the state label.

    Notes
    -----
    Labels that do not follow the expected naming convention receive a very
    large sorting value so they appear after correctly formatted state labels.
    """

    # Extract the numeric suffix used to order NeuroSeq state labels.
    try:
        return int(str(label).split("_")[-1])

    # Unexpected labels are placed at the end of the sorted state list.
    except Exception:
        return 10**12


def run_state_pair_nmi(
    base_dir: str,
) -> dict[str, str]:
    """
    Compute pairwise Louvain NMI between all final Micro states.

    This function compares the neuronal Louvain partitions associated with
    every pair of consensus Micro states. The resulting similarities are
    written in both square-matrix and long-form representations.

    Parameters
    ----------
    base_dir : str
        Root directory containing the canonical ``Proto-States`` outputs.

    Returns
    -------
    dict[str, str]
        Dictionary containing the paths to

        - ``proto_state_louvain_nmi.csv``
        - ``proto_state_louvain_nmi_long.csv``

    Raises
    ------
    FileNotFoundError
        Raised when no state Louvain assignment table is found.

    ValueError
        Raised when the required columns

        - ``State``
        - ``Neuron_ID``
        - ``Cluster``

        are missing.

    Notes
    -----
    Only neurons appearing in both states contribute to an NMI calculation.

    Pairwise NMI is computed only when at least two common neurons are
    available. Otherwise the comparison is reported as ``NaN``.

    The resulting square matrix is symmetric with ones along the diagonal.
    """

    # Locate the canonical state-level Louvain assignment table produced during
    # the final Micro analysis.
    state_dir = Path(base_dir) / "Proto-States"

    source = state_dir / "proto_state_x_cluster.csv"

    # Preserve compatibility with older NeuroSeq output names.
    if not source.exists():
        source = state_dir / "proto_state_louvain_clusters.csv"

    # Pairwise state comparisons require one of the canonical Louvain
    # assignment tables.
    if not source.exists():
        raise FileNotFoundError(
            "State Louvain assignment CSV not found."
        )

    # Load the state-level neuronal community assignments.
    df = pd.read_csv(source)

    # Verify that the canonical columns required for pairwise comparisons are
    # available.
    required = {
        "State",
        "Neuron_ID",
        "Cluster",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{source} must contain {sorted(required)}"
        )

    # Sort the NeuroSeq states into their natural temporal order.
    states = sorted(
        df["State"].astype(str).unique(),
        key=_state_sort_key,
    )

    # Initialize the symmetric pairwise NMI matrix.
    matrix = np.eye(
        len(states),
        dtype=float,
    )

    # Store every pairwise comparison in long-form representation.
    long_rows = []

    # Compare every state against every other state.
    for i, state_i in enumerate(states):

        # Extract the neuronal community assignments for the first state.
        left = df[
            df["State"].astype(str) == state_i
        ][
            ["Neuron_ID", "Cluster"]
        ]

        for j, state_j in enumerate(states):

            # Extract the neuronal community assignments for the second state.
            right = df[
                df["State"].astype(str) == state_j
            ][
                ["Neuron_ID", "Cluster"]
            ]

            # Compare only neurons that are present in both states.
            merged = left.merge(
                right,
                on="Neuron_ID",
                suffixes=("_i", "_j"),
            )

            # Compute the similarity between the two neuronal community
            # partitions whenever enough shared neurons are available.
            nmi = (
                float(
                    normalized_mutual_info_score(
                        merged["Cluster_i"],
                        merged["Cluster_j"],
                    )
                )
                if len(merged) >= 2
                else np.nan
            )

            # Store the pairwise similarity in the symmetric matrix.
            matrix[i, j] = nmi

            # Preserve the same comparison in long-form format together with
            # the number of neurons contributing to the calculation.
            long_rows.append(
                {
                    "State_1": state_i,
                    "State_2": state_j,
                    "NMI": nmi,
                    "Compared_Neurons": int(len(merged)),
                }
            )

    # Define the canonical output filenames.
    square_path = state_dir / "proto_state_louvain_nmi.csv"
    long_path = state_dir / "proto_state_louvain_nmi_long.csv"

    # Write the square pairwise similarity matrix.
    pd.DataFrame(
        matrix,
        index=states,
        columns=states,
    ).to_csv(square_path)

    # Write the long-form pairwise comparison table.
    pd.DataFrame(
        long_rows,
    ).to_csv(
        long_path,
        index=False,
    )

    # Return the canonical output paths for downstream Micro and Meso modules.
    return {
        "state_louvain_nmi_csv": str(square_path),
        "state_louvain_nmi_long_csv": str(long_path),
    }

def save_pairwise_nmi_from_pairs(pairs, state_ids, state_clusters, state_dir):
    """Save the Proto-state NMI tables using already computed pairwise NMI.

    The NMI values are reused from the Meso pairwise table. No additional
    normalized-mutual-information calculations are performed.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    n = len(state_ids)
    matrix = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    # Only neuron counts are derived here; community NMI is never recomputed.
    neurons = [set(map(str, state_clusters[sid]["neurons"])) for sid in state_ids]
    for row in pairs.itertuples(index=False):
        i, j = int(row.i), int(row.j)
        matrix[i, j] = matrix[j, i] = float(row.NMI)
    long_rows = []
    for i, sid_i in enumerate(state_ids):
        for j, sid_j in enumerate(state_ids):
            long_rows.append({
                "State_1": f"proto_state_{i+1}", "State_2": f"proto_state_{j+1}",
                "NMI": float(matrix[i, j]),
                "Compared_Neurons": len(neurons[i] & neurons[j]),
            })
    square = state_dir / "proto_state_louvain_nmi.csv"
    long = state_dir / "proto_state_louvain_nmi_long.csv"
    pd.DataFrame(matrix, index=[f"proto_state_{i+1}" for i in range(n)], columns=[f"proto_state_{i+1}" for i in range(n)]).to_csv(square)
    pd.DataFrame(long_rows).to_csv(long, index=False)
    return {"state_louvain_nmi_csv": str(square), "state_louvain_nmi_long_csv": str(long)}
