# Micro/micro_state_pair_nmi.py

"""
Pairwise State Louvain NMI Analysis
==================================

Module
------
Micro.micro_state_pair_nmi

Purpose
-------
Compute pairwise Normalized Mutual Information (NMI) between the neuronal
Louvain community assignments of every pair of final Micro states.

Following construction of the consensus State_Windows and within-state Louvain
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

``state_louvain_nmi.csv``
    Square symmetric matrix whose rows and columns correspond to the final
    Micro states.

``state_louvain_nmi_long.csv``
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
        Root directory containing the canonical ``State_Windows`` outputs.

    Returns
    -------
    dict[str, str]
        Dictionary containing the paths to

        - ``state_louvain_nmi.csv``
        - ``state_louvain_nmi_long.csv``

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
    state_dir = Path(base_dir) / "State_Windows"

    source = state_dir / "state__x_cluster.csv"

    # Preserve compatibility with older NeuroSeq output names.
    if not source.exists():
        source = state_dir / "louvain_clusters_all_states.csv"

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
    square_path = state_dir / "state_louvain_nmi.csv"
    long_path = state_dir / "state_louvain_nmi_long.csv"

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