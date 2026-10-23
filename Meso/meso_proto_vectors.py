# Meso/meso_proto_vectors.py

"""
State Firing-Rate Vectors and Pairwise State Matrices
=====================================================

Module
------
Meso.meso_proto_vectors

Purpose
-------
Construct firing-rate representations for every consensus Micro state and
compute pairwise similarity and distance matrices describing the relationships
between those states.

Each final Micro state is represented by a firing-rate vector whose elements
correspond to the mean firing rates of all recorded neurons during that state.

These state-level representations provide one of the principal inputs to the
NeuroSeq Meso analysis.

Scientific Role
---------------
After the final consensus state windows have been determined, every neuron's
activity is summarized within each state interval.

For a state of duration

    Δt,

the firing rate of neuron

    i

is computed as

    r_i = (# spikes in state) / Δt.

Collecting the firing rates of every recorded neuron produces a state vector,

    v = (r_1, r_2, ..., r_N),

where N is the number of neurons.

These vectors summarize the population activity associated with each temporal
state.

The module then computes two complementary pairwise measures:

Mean firing-rate distance
    Normalized Euclidean distance between state vectors.

Cosine similarity
    Angular similarity between state vectors that emphasizes relative firing
    patterns rather than absolute firing-rate magnitude.

Together with the state Louvain NMI matrix, these quantities are used during
the Meso analysis to characterize relationships between Micro states.

Outputs
-------
The canonical outputs written to

    Proto-States/

are

- ``state_vectors.csv``
- ``cosine_similarity.csv``
- ``mean_firing_rate_distance.csv``

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Micro.micro_utils import (
    _ensure_dir,
    _find_latest_counts_csv,
    _load_spike_times,
    _parse_counts_bounds,
)


def _mean_firing_rate_distance(v1, v2):
    """
    Compute the normalized Euclidean distance between two state vectors.

    Parameters
    ----------
    v1, v2 : array-like
        State firing-rate vectors.

    Returns
    -------
    float
        Normalized Euclidean distance

            ||v1 - v2|| / sqrt(N),

        where N is the number of neurons.

        Returns ``NaN`` when the vectors have different dimensions or are
        empty.

    Notes
    -----
    Normalization by ``sqrt(N)`` removes the dependence of the distance scale
    on the number of recorded neurons, allowing comparisons across datasets
    with different population sizes.
    """

    # Convert both vectors into floating-point NumPy arrays.
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    # Distance is undefined when the vectors differ in dimension or contain no
    # neurons.
    if v1.size != v2.size or v1.size == 0:
        return np.nan

    # Compute the normalized Euclidean distance between the two firing-rate
    # vectors.
    return np.linalg.norm(v1 - v2) / np.sqrt(v1.size)


def run_state_vectors_and_matrices(
    base_dir: str,
    data_dir: str,

    nmi_threshold: float,
    T: float,
    use_auto_find_counts_csv: bool,
    counts_csv_search_root: str,
    manual_start_time: float,
    manual_end_time: float,
):
    """
    Construct state firing-rate vectors and pairwise state matrices.

    This function reconstructs the final consensus state intervals, computes
    one firing-rate vector for every state, and derives pairwise cosine
    similarity and normalized firing-rate distance matrices.

    Parameters
    ----------
    base_dir : str
        Directory containing the canonical Micro outputs.

    data_dir : str
        Directory containing the original ``SU_*.csv`` spike-time files.


    nmi_threshold : float
        Threshold defining the consensus temporal state boundaries.

    T : float
        Width of the original temporal windows.

    use_auto_find_counts_csv : bool
        Whether the recording interval should be inferred automatically from
        the most recent counts file.

    counts_csv_search_root : str
        Search directory used when automatic count-file discovery is enabled.

    manual_start_time : float
        Beginning of the analyzed recording interval.

    manual_end_time : float
        End of the analyzed recording interval.

    Returns
    -------
    dict[str, str]
        Dictionary containing the paths to

        - ``state_vectors.csv``
        - ``cosine_similarity.csv``
        - ``mean_firing_rate_distance.csv``

    Raises
    ------
    FileNotFoundError
        Raised when the consensus NMI table or neuron spike files cannot be
        located.

    ValueError
        Raised when the NMI table is missing required columns or contains no
        temporal comparisons.

    RuntimeError
        Raised when no valid state vectors can be constructed.

    Notes
    -----
    State boundaries are reconstructed from the consensus adjacent-window NMI
    representation.

    Firing rates are computed as

        spikes / state duration.

    Pairwise cosine similarity and normalized Euclidean distance are computed
    using the complete firing-rate vectors.
    """

    # Locate the canonical consensus NMI table.
    nmi_csv = os.path.join(
        base_dir,
        f"nmi_adjacent_micro-epoch.csv",
    )

    # Fall back to a recursive search if the expected file is not located
    # directly.
    if not os.path.exists(nmi_csv):
        matches = glob.glob(
            os.path.join(
                base_dir,
                "**",
                f"nmi_adjacent_micro-epoch.csv",
            ),
            recursive=True,
        )

        if not matches:
            raise FileNotFoundError("NMI CSV not found.")

        # Use the most recently written consensus NMI table.
        matches.sort(
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        nmi_csv = matches[0]

    # Load the consensus adjacent-window NMI representation.
    metrics_df = pd.read_csv(nmi_csv)

    # Verify that the temporal information required for state reconstruction is
    # available.
    if not {"Start_Time_s", "NMI"}.issubset(metrics_df.columns):
        raise ValueError(
            "NMI CSV missing ['Start_Time_s', 'NMI']."
        )

    # Initialize the recording interval supplied by the Micro workflow.
    start_time = float(manual_start_time)
    end_time = float(manual_end_time)

    # Optionally recover the recording interval from the most recent canonical
    # counts table.
    if use_auto_find_counts_csv:
        try:
            cpath = _find_latest_counts_csv(
                counts_csv_search_root,
                T,
            )

            s, e, _ = _parse_counts_bounds(cpath)

            if s is not None and e is not None:
                start_time = float(s)
                end_time = float(e)

        except Exception:
            pass

    # Extract the temporal boundary locations and binary consensus NMI values.
    times = metrics_df["Start_Time_s"].to_numpy(float)
    nmi_vals = metrics_df["NMI"].to_numpy(float)

    # State reconstruction requires at least one adjacent-window comparison.
    if times.size == 0:
        raise ValueError(
            "NMI CSV contains no adjacent-window comparisons."
        )

    eps = 1e-9

    # Convert relative window times to absolute recording times whenever
    # necessary.
    if (
        np.nanmin(times) < start_time - eps
        or np.nanmax(times) <= start_time + T + eps
    ):
        times = times + start_time

    # Align adjacent-window times with the actual transition locations.
    if np.isclose(times.min(), start_time, atol=eps):
        times = times + T

    # Recover the consensus state boundaries from the binary NMI trace.
    dividers = sorted(
        np.unique(
            np.round(
                [
                    float(t)
                    for t, n in zip(times, nmi_vals)
                    if (
                        n < nmi_threshold
                        and start_time + eps < t < end_time - eps
                    )
                ],
                6,
            )
        ).tolist()
    )

    # Construct the complete sequence of state boundaries.
    edges = [start_time] + dividers + [end_time]

    # Locate all recorded neurons participating in the analysis.
    su_files = sorted(
        f
        for f in os.listdir(data_dir)
        if f.startswith("SU_") and f.endswith(".csv")
    )

    neuron_ids = [
        os.path.splitext(f)[0]
        for f in su_files
    ]

    if not neuron_ids:
        raise FileNotFoundError(
            f"No SU_*.csv files found in {data_dir}"
        )

    # Load the spike times of every recorded neuron.
    spikes = {}

    for su in neuron_ids:
        try:
            spikes[su] = _load_spike_times(
                os.path.join(
                    data_dir,
                    su + ".csv",
                )
            )

        # Unreadable spike files are represented as neurons with no spikes.
        except Exception:
            spikes[su] = np.array([], dtype=float)

    # Construct one firing-rate vector for every consensus state.
    state_vectors = []
    state_info = []

    for i, (s0, s1) in enumerate(
        zip(edges[:-1], edges[1:]),
        start=1,
    ):

        duration = float(s1 - s0)

        # Ignore degenerate state intervals.
        if duration <= 0:
            continue

        # Compute the firing rate of every neuron within the current state.
        vector = [
            (
                int(
                    np.sum(
                        (spikes[su] >= s0)
                        & (spikes[su] <= s1)
                    )
                )
                / duration
            )
            if spikes[su].size
            else 0.0
            for su in neuron_ids
        ]

        state_vectors.append(vector)

        # Record the temporal boundaries and firing-rate vector associated with
        # the current state.
        row = {
            "State": f"State_{i}",
            "Start": s0,
            "End": s1,
            "Duration": duration,
        }

        row.update(
            dict(zip(neuron_ids, vector))
        )

        state_info.append(row)

    # Create the canonical state-level output directory.
    out_dir = os.path.join(
        base_dir,
        "Proto-States",
    )

    _ensure_dir(out_dir)

    # Save the complete state firing-rate table.
    pd.DataFrame(state_info).to_csv(
        os.path.join(
            out_dir,
            "state_vectors.csv",
        ),
        index=False,
    )

    # At least one valid state vector is required for pairwise analysis.
    if not state_vectors:
        raise RuntimeError(
            "No valid state vectors were produced."
        )

    # Construct the ordered list of state labels.
    labels = [
        f"State_{i + 1}"
        for i in range(len(state_vectors))
    ]

    vectors = np.asarray(
        state_vectors,
        dtype=float,
    )

    # Compute pairwise cosine similarity between every pair of state vectors.
    cosine = cosine_similarity(vectors)

    pd.DataFrame(
        cosine,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            out_dir,
            "cosine_similarity.csv",
        )
    )

    # Compute the normalized firing-rate distance matrix.
    n = len(vectors)

    distance = np.zeros(
        (n, n),
        dtype=float,
    )

    for i in range(n):
        for j in range(n):
            distance[i, j] = _mean_firing_rate_distance(
                vectors[i],
                vectors[j],
            )

    # Save the pairwise firing-rate distance matrix.
    pd.DataFrame(
        distance,
        index=labels,
        columns=labels,
    ).to_csv(
        os.path.join(
            out_dir,
            "mean_firing_rate_distance.csv",
        )
    )

    # Return the canonical state-level output paths.
    return {
        "state_vectors_csv": os.path.join(
            out_dir,
            "state_vectors.csv",
        ),
        "cosine_similarity_csv": os.path.join(
            out_dir,
            "cosine_similarity.csv",
        ),
        "mean_firing_rate_distance_csv": os.path.join(
            out_dir,
            "mean_firing_rate_distance.csv",
        ),
    }