# Micro/micro_graphs.py

"""
Weighted Temporal-Window Graph Construction
===========================================

Module
------
Micro.micro_graphs

Purpose
-------
Construct one weighted neuronal similarity graph for every temporal spike-count
window in the NeuroSeq Micro analysis.

Each temporal window is represented as an undirected graph whose nodes
correspond to recorded neurons. Pairwise edge weights quantify the similarity
between the neurons' spike counts within that window.

These graphs provide the network representation used by the temporal Louvain
community-detection stage.

Scientific Role
---------------
For a given temporal window, let

    c_i

denote the spike count of neuron ``i``. For every neuron pair ``(i, j)``, the
absolute spike-count difference is computed as

    delta_ij = |c_i - c_j|.

The corresponding edge weight is

    w_ij = max(0, 100 - delta_ij / T),

where ``T`` controls how rapidly neuronal similarity decreases as the
difference in spike count increases.

Neurons with identical or similar spike counts receive large edge weights,
whereas neurons with strongly different counts receive smaller weights.

Only edges satisfying

    w_ij >= lambda_threshold

are retained in the graph.

Graph Policy
------------
All selected neurons are added to every temporal graph, including neurons that
do not retain any edges after thresholding.

This preserves a consistent node set across temporal windows and ensures that
isolated neurons remain available to the Louvain community-detection stage.

The graph topology can vary across windows because edge weights are recomputed
from the window-specific spike counts.

Main Outputs
------------
For every temporal window, this module produces

- a weighted NetworkX graph,
- a tabular edge list containing source neuron, target neuron, and edge weight,
- and a mapping from internal integer node identifiers to original single-unit
  identifiers.

The graph objects are consumed directly by the in-memory Louvain sweep, whereas
the edge tables are retained for canonical Micro output generation.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import pandas as pd


def _build_window_graphs_from_counts(
    counts_df: pd.DataFrame,
    *,
    T: float,
    lambda_threshold: float,
) -> tuple[dict[str, nx.Graph], dict[str, pd.DataFrame], dict[int, str]]:
    """
    Construct one weighted neuronal graph for every temporal window.

    The input spike-count table is converted into a sequence of undirected
    weighted graphs. Each graph contains one node for every neuron and one edge
    for every neuron pair whose spike-count similarity weight satisfies the
    requested threshold.

    Parameters
    ----------
    counts_df : pandas.DataFrame
        Neuron-by-window spike-count table.

        The table must contain

        - ``SU_ID`` as the neuron identifier column,
        - and one or more temporal columns whose names begin with
          ``Window_``.

    T : float
        Scaling parameter controlling how strongly the edge weight decreases
        with the absolute spike-count difference.

        For two neurons with counts ``c1`` and ``c2``, the weight is

        ``max(0, 100 - abs(c1 - c2) / T)``.

    lambda_threshold : float
        Minimum edge weight required for a neuron pair to be retained in the
        graph.

    Returns
    -------
    graphs : dict[str, networkx.Graph]
        Mapping from each temporal window name to its weighted undirected
        neuronal graph.

    edge_dfs : dict[str, pandas.DataFrame]
        Mapping from each temporal window name to a tabular edge list with the
        columns

        - ``Source``
        - ``Target``
        - ``Weight``

    id_to_neuron : dict[int, str]
        Mapping from internal one-based integer graph-node identifiers to the
        original single-unit identifiers.

    Raises
    ------
    ValueError
        Raised when ``counts_df`` does not contain the required ``SU_ID``
        column.

    Notes
    -----
    All neurons are added to every graph before edge construction. Therefore,
    neurons remain represented even when they have no retained edges in a
    particular temporal window.

    The same neuron-to-integer mapping is used across all windows, allowing
    Louvain partitions to be translated consistently back to the original
    single-unit identifiers.

    This function performs no community detection. It only constructs the
    weighted temporal graphs consumed by the Louvain stage.
    """

    # The single-unit identifier column is required to establish a stable node
    # identity across all temporal graphs.
    if "SU_ID" not in counts_df.columns:
        raise ValueError("counts_df must contain SU_ID.")

    # Sort neurons deterministically and use their identifiers as the table
    # index for window-specific count access.
    df = (
        counts_df
        .sort_values("SU_ID")
        .reset_index(drop=True)
        .set_index("SU_ID")
    )

    # Preserve the ordered single-unit identifiers used throughout graph
    # construction.
    su_ids = list(df.index)

    # NetworkX graphs use compact one-based integer node identifiers. The same
    # mapping is applied to every temporal window.
    neuron_to_id = {
        su: i + 1
        for i, su in enumerate(su_ids)
    }
    id_to_neuron = {
        i + 1: su
        for i, su in enumerate(su_ids)
    }

    # Identify the temporal spike-count columns that define the graph sequence.
    window_cols = [
        c
        for c in df.columns
        if str(c).startswith("Window_")
    ]

    # Store the in-memory graph and edge-table representation of every temporal
    # window.
    graphs: dict[str, nx.Graph] = {}
    edge_dfs: dict[str, pd.DataFrame] = {}

    # Construct each temporal graph independently from its window-specific
    # neuronal spike counts.
    for wname in window_cols:
        counts = df[wname]

        # Initialize an undirected graph and add all neurons before evaluating
        # pairwise similarities. This preserves isolated neurons.
        G = nx.Graph()
        G.add_nodes_from(neuron_to_id.values())

        # Retain the accepted graph edges in tabular form for output and
        # diagnostics.
        rows = []

        # Evaluate every unique unordered pair of neurons in the current
        # temporal window.
        for (u, c1), (v, c2) in combinations(counts.items(), 2):

            # Quantify the difference in spike count between the two neurons.
            delta = abs(float(c1) - float(c2))

            # Convert the spike-count difference into a bounded nonnegative
            # similarity weight.
            weight = max(
                0.0,
                100.0 - (delta / float(T)),
            )

            # Retain only sufficiently strong neuronal similarities.
            if weight >= float(lambda_threshold):
                uid = neuron_to_id[u]
                vid = neuron_to_id[v]

                # Add the weighted edge to the graph consumed by Louvain
                # community detection.
                G.add_edge(
                    uid,
                    vid,
                    weight=float(weight),
                )

                # Preserve the same edge using the original neuron identifiers.
                rows.append(
                    {
                        "Source": str(u),
                        "Target": str(v),
                        "Weight": float(weight),
                    }
                )

        # Store the completed weighted graph for the current temporal window.
        graphs[str(wname)] = G

        # Store a schema-consistent edge table, including when no edges survive
        # the threshold.
        edge_dfs[str(wname)] = pd.DataFrame(
            rows,
            columns=[
                "Source",
                "Target",
                "Weight",
            ],
        )

    # Return the complete temporal graph sequence, edge tables, and stable node
    # identifier mapping.
    return graphs, edge_dfs, id_to_neuron