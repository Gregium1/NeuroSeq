"""
Adjacent-Window Normalized Mutual Information
=============================================

Module
------
Micro.micro_adjacent_nmi

Purpose
-------
Compute the similarity between the neuronal community structures of
consecutive temporal windows using Normalized Mutual Information (NMI).

During the Micro analysis, Louvain community detection is performed
independently within every temporal window. This module quantifies how
similar the resulting neuronal partitions remain from one window to the
next. Abrupt decreases in NMI indicate substantial changes in the underlying
population organization and therefore represent candidate neural
state-transition boundaries.

Scientific Role
---------------
For every pair of consecutive temporal windows,

    Window_i  --->  Window_{i+1}

the Louvain community assignments of neurons present in both windows are
compared using Normalized Mutual Information (NMI). The resulting temporal
NMI profile forms the basis for adaptive threshold estimation and critical-
point detection during the automatic Micro gamma sweep.

Only neurons observed in both windows contribute to the NMI calculation,
ensuring that the similarity score reflects shared neuronal activity rather
than differences caused by missing neurons.

Outputs
-------
The returned table contains one row for every adjacent pair of temporal
windows and includes

• Window identifiers
• Normalized Mutual Information (NMI)
• Number of neurons included in the comparison
• Transition time
• Boundary index

These values are subsequently used by the threshold estimation and critical-
point detection modules.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def _compute_adjacent_nmi_in_memory(
    *,
    per_window_partitions: dict[str, pd.DataFrame],
    start_time: float,
    T: float,
) -> pd.DataFrame:
    """
    Compute adjacent-window Normalized Mutual Information.

    This function compares the Louvain community partitions obtained from
    consecutive temporal windows and measures their similarity using
    Normalized Mutual Information (NMI).

    For every neighboring pair of windows, only neurons present in both
    partitions are included in the comparison. High NMI values indicate that
    the neuronal community structure remains stable across the transition,
    whereas low NMI values suggest a substantial network reorganization that
    may correspond to a neural state transition.

    Parameters
    ----------
    per_window_partitions : dict[str, pandas.DataFrame]
        Dictionary whose keys are temporal window names (for example,
        ``Window_0001``) and whose values are Louvain partition tables
        containing at least the columns

        - ``Neuron_ID``
        - ``Cluster``

    start_time : float
        Beginning of the analyzed recording interval, in seconds.

    T : float
        Width of each temporal analysis window, in seconds.

    Returns
    -------
    pandas.DataFrame
        Table containing one row for every adjacent-window comparison.

        Columns include

        - ``Window_1``
        - ``Window_2``
        - ``NMI``
        - ``Compared_Neurons``
        - ``Start_Time_s``
        - ``Boundary_Index``

    Raises
    ------
    RuntimeError
        Raised when no valid adjacent-window comparisons can be computed.

    Notes
    -----
    This function performs no thresholding or critical-point detection.
    It only computes the adjacent-window similarity profile used by later
    stages of the Micro analysis.
    """

    # Sort the temporal windows according to their numerical indices so that
    # consecutive windows are compared in chronological order.
    window_names = sorted(
        per_window_partitions.keys(),
        key=lambda x: int(str(x).split("_")[-1]),
    )

    # Store one summary row for every valid adjacent-window comparison.
    rows = []

    # Compare each temporal window with its immediate successor.
    for idx in range(len(window_names) - 1):

        w1 = window_names[idx]
        w2 = window_names[idx + 1]

        d1 = per_window_partitions[w1]
        d2 = per_window_partitions[w2]

        # Skip comparisons if either partition is missing the required
        # Neuron_ID or Cluster columns.
        if not {"Neuron_ID", "Cluster"}.issubset(d1.columns):
            continue

        if not {"Neuron_ID", "Cluster"}.issubset(d2.columns):
            continue

        # Retain only neurons present in both temporal windows so that the
        # NMI reflects changes in community organization rather than missing
        # observations.
        merged = pd.merge(
            d1[["Neuron_ID", "Cluster"]],
            d2[["Neuron_ID", "Cluster"]],
            on="Neuron_ID",
            suffixes=("_w1", "_w2"),
        )

        # At least two shared neurons are required to compute a meaningful
        # Normalized Mutual Information score.
        if len(merged) < 2:
            continue

        # Measure the similarity between the two Louvain partitions.
        nmi = normalized_mutual_info_score(
            merged["Cluster_w1"],
            merged["Cluster_w2"],
        )

        # Compute the physical recording time corresponding to this temporal
        # boundary.
        real_time = round(
            float(start_time)
            + float(idx + 1) * float(T),
            6,
        )

        # Store the complete adjacent-window similarity measurement.
        rows.append(
            {
                "Window_1": w1,
                "Window_2": w2,
                "NMI": round(float(nmi), 6),
                "Compared_Neurons": int(len(merged)),
                "Start_Time_s": real_time,
                "Boundary_Index": int(idx),
            }
        )

    # The downstream threshold estimation requires at least one valid
    # adjacent-window comparison.
    if not rows:
        raise RuntimeError(
            "No valid adjacent-window NMI comparisons found."
        )

    # Return the complete adjacent-window similarity profile.
    return pd.DataFrame(rows)