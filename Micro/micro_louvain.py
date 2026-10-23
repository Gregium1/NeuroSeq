# Micro/micro_louvain.py

"""
Repeated-Seed Louvain Community Detection
=========================================

Module
------
Micro.micro_louvain

Purpose
-------
Perform Louvain community detection independently within every temporal window
for one fixed Louvain resolution (gamma).

The Louvain algorithm is stochastic because its optimization procedure depends
on the initial ordering of node updates. Consequently, different random seeds
may produce different community partitions even when applied to the same
weighted neuronal graph.

To improve robustness, NeuroSeq evaluates multiple random initializations for
every temporal window and retains the partition producing the largest graph
modularity.

Scientific Role
---------------
For a fixed Louvain resolution

    gamma,

each temporal neuronal graph is clustered independently.

Suppose

    G_t

denotes the weighted neuronal graph associated with temporal window ``t``.
Multiple Louvain optimizations are performed,

    P_1, P_2, ..., P_S,

where ``S`` is the number of random seeds.

For every partition

    P_i,

the weighted graph modularity

    Q(P_i)

is computed.

The partition maximizing modularity,

    argmax Q(P_i),

is retained as the representative neuronal community structure for that
temporal window.

This procedure substantially reduces variability caused by the stochastic
nature of Louvain optimization while preserving computational efficiency.

Empty-Graph Policy
------------------
Some temporal windows may contain no retained graph edges after thresholding.

Rather than failing or producing undefined labels, every neuron is assigned to
one common community.

This guarantees that every temporal window possesses a valid partition and
allows adjacent-window NMI to be computed consistently across the entire
recording.

Outputs
-------
For every temporal window, this module returns a partition table containing

- integer graph node identifiers,
- original neuron identifiers,
- Louvain community labels.

These partitions are subsequently compared across neighboring windows by the
adjacent-window NMI module.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain


def _run_single_gamma_louvain_in_memory(
    *,
    graphs: dict[str, nx.Graph],
    id_to_neuron: dict[int, str],
    gamma: float,
    seeds_per_gamma: int,
) -> dict[str, pd.DataFrame]:
    """
    Run repeated-seed Louvain community detection for one gamma value.

    For every temporal neuronal graph, Louvain community detection is executed
    multiple times using different random seeds. The partition producing the
    largest weighted graph modularity is retained as the representative
    clustering for that temporal window.

    Parameters
    ----------
    graphs : dict[str, networkx.Graph]
        Mapping from temporal window names to weighted neuronal graphs.

    id_to_neuron : dict[int, str]
        Mapping from internal graph node identifiers to the original single-unit
        identifiers.

    gamma : float
        Louvain resolution parameter controlling the granularity of detected
        neuronal communities.

    seeds_per_gamma : int
        Number of independent Louvain optimizations performed for every
        temporal window.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary mapping each temporal window name to a partition table
        containing

        - ``Node_ID``
        - ``Neuron_ID``
        - ``Cluster``

    Notes
    -----
    The partition with the largest weighted modularity is retained.

    Temporal windows whose graphs contain no retained edges are assigned a
    single shared community label so that downstream NMI calculations remain
    well defined.

    This function performs clustering independently for each temporal window.
    No temporal information is used during community detection.
    """

    # Construct the sequence of deterministic random seeds used for repeated
    # Louvain optimization.
    seeds = [42 + i for i in range(int(seeds_per_gamma))]

    # Store the representative partition obtained for every temporal window.
    out: dict[str, pd.DataFrame] = {}

    # Process temporal windows in chronological order.
    for wname in sorted(
        graphs.keys(),
        key=lambda x: int(str(x).split("_")[-1]),
    ):

        G = graphs[wname]

        # Graphs containing no retained edges cannot be partitioned
        # meaningfully. Every neuron is therefore assigned to one common
        # community.
        if G.number_of_edges() == 0:
            part = {
                int(n): 0
                for n in G.nodes()
            }

        else:

            # Store every candidate Louvain partition together with its
            # corresponding weighted modularity.
            partitions = []
            modularities = []

            # Repeat Louvain optimization using multiple random initializations.
            for seed in seeds:

                # Compute one stochastic Louvain partition.
                p = community_louvain.best_partition(
                    G,
                    weight="weight",
                    resolution=float(gamma),
                    random_state=int(seed),
                )

                # Normalize node and community identifiers to integers.
                p = {
                    int(k): int(v)
                    for k, v in p.items()
                }

                # Evaluate the quality of the partition using weighted graph
                # modularity.
                try:
                    q = float(
                        community_louvain.modularity(
                            p,
                            G,
                            weight="weight",
                        )
                    )

                # A failed modularity evaluation is assigned negative infinity
                # so it can never be selected as the representative partition.
                except Exception:
                    q = -np.inf

                partitions.append(p)
                modularities.append(q)

            # Select the partition maximizing weighted modularity.
            best_i = int(np.nanargmax(modularities))
            part = partitions[best_i]

        # Convert the representative partition into the canonical NeuroSeq
        # partition table.
        df = pd.DataFrame(
            {
                "Node_ID": list(part.keys()),
                "Neuron_ID": [
                    id_to_neuron[int(n)]
                    for n in part.keys()
                ],
                "Cluster": list(part.values()),
            }
        ).sort_values("Node_ID")

        # Store the representative partition for the current temporal window.
        out[str(wname)] = df

    # Return the complete set of temporal Louvain partitions for this gamma.
    return out