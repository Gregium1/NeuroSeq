# Micro/micro_counts.py

"""
Spike-Count Construction and Raster Output
==========================================

Module
------
Micro.micro_counts

Purpose
-------
Construct the temporal spike-count representation used throughout the
NeuroSeq Micro analysis and generate the canonical spike raster for the
selected recording interval.

The Micro pipeline begins by dividing the recording into consecutive temporal
windows. For each single unit, this module counts the number of spikes occurring
inside every window and stores those values in a neuron-by-window matrix.

The resulting spike-count table provides the temporal activity representation
used to construct neuronal similarity graphs. The interval-filtered spike trains
are also retained for diagnostic plotting and downstream output generation.

Scientific Role
---------------
Let the analyzed interval be divided into temporal windows

    W_1, W_2, ..., W_T.

For every neuron ``SU_i``, the module computes

    C_{i,t} = number of spikes emitted by neuron i during window t.

The complete matrix of counts summarizes how each neuron's activity changes
over time. These counts are later used to define pairwise neuronal similarities
within each temporal window and therefore form the foundation of the Micro
network analysis.

The final temporal window includes its right endpoint. This ensures that a
spike occurring exactly at the requested end of the recording interval is not
discarded.

Outputs
-------
This module can produce

- the neuron-by-window spike-count table,
- interval-filtered spike-time arrays,
- temporal bin edges,
- a canonical spike-count CSV file,
- and a spike raster image for the analyzed interval.

The canonical output paths are recorded in the Micro context file and reused
by later stages of the pipeline.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .micro_utils import _ensure_dir, _load_spike_times, _su_sort_key, _as_su_filter


def _build_spike_counts_df(
    *,
    data_folder: str | Path,
    start_time: float,
    total_duration: float,
    T: float,
    selected_su_ids: list[str] | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """
    Construct the temporal spike-count matrix for one recording interval.

    This function loads all requested single-unit spike trains, restricts them
    to the analyzed recording interval, divides that interval into consecutive
    temporal windows, and counts the number of spikes emitted by each neuron
    inside every window.

    The final temporal window includes its right endpoint so that spikes
    occurring exactly at the end of the requested interval are preserved.

    Parameters
    ----------
    data_folder : str or pathlib.Path
        Directory containing the single-unit spike-time files. Files are
        expected to follow the naming convention ``SU_<index>.csv``.

    start_time : float
        Beginning of the analyzed recording interval, in seconds.

    total_duration : float
        Duration of the analyzed recording interval, in seconds.

    T : float
        Width of each temporal spike-count window, in seconds.

    selected_su_ids : list[str] or None
        Optional subset of single-unit identifiers to include. When ``None``,
        all available ``SU_*.csv`` files are analyzed.

    Returns
    -------
    counts_df : pandas.DataFrame
        Neuron-by-window spike-count table. The first column is ``SU_ID`` and
        the remaining columns are named ``Window_1``, ``Window_2``, and so on.

    spikes : dict[str, numpy.ndarray]
        Dictionary mapping each single-unit identifier to its spike times
        restricted to the analyzed interval.

    edges : numpy.ndarray
        Temporal bin edges defining the spike-count windows.

    Raises
    ------
    FileNotFoundError
        Raised when no matching ``SU_*.csv`` files are found after applying
        the optional neuron selection.

    Notes
    -----
    If an individual spike-time file cannot be read, that neuron is retained
    with an empty spike train rather than terminating the complete analysis.

    The number of temporal windows is computed with ``ceil`` so that the full
    requested duration is represented even when ``total_duration`` is not an
    exact multiple of ``T``.
    """

    # Normalize the data-directory path and determine the absolute end of the
    # requested recording interval.
    data_folder = Path(data_folder)
    end_time = float(start_time) + float(total_duration)

    # Discover all single-unit spike-time files and sort them numerically by
    # neuron identifier.
    su_files = sorted(
        data_folder.glob("SU_*.csv"),
        key=lambda p: _su_sort_key(p.stem),
    )

    # Restrict the analysis to the requested subset of neurons when a filter
    # has been provided.
    wanted = _as_su_filter(selected_su_ids)
    if wanted is not None:
        su_files = [p for p in su_files if p.stem in wanted]

    # The Micro analysis requires at least one usable single-unit file.
    if not su_files:
        raise FileNotFoundError(
            f"No SU_*.csv files found in {data_folder}"
        )

    # Determine the number of temporal windows required to cover the complete
    # recording interval.
    n_windows = int(
        np.ceil(float(total_duration) / float(T))
    )

    # Construct the temporal bin edges. The final edge is explicitly set to
    # the requested end time so that floating-point accumulation or a partial
    # final window does not extend the analysis beyond the selected interval.
    edges = np.arange(
        float(start_time),
        float(start_time) + (n_windows + 1) * float(T),
        float(T),
    )
    edges[-1] = end_time

    # Store one spike-count row per neuron and retain the interval-filtered
    # spike trains for raster generation and downstream outputs.
    rows = []
    spikes: dict[str, np.ndarray] = {}

    # Load and count spikes independently for every selected neuron.
    for p in su_files:
        su = p.stem

        # A malformed or unreadable spike-time file is represented by an empty
        # spike train so that one file does not stop the full population
        # analysis.
        try:
            arr_all = _load_spike_times(p)
        except Exception:
            arr_all = np.array([], dtype=float)

        # Restrict the spike train to the requested interval. Both endpoints
        # are retained at this stage.
        arr = arr_all[
            (arr_all >= float(start_time))
            & (arr_all <= float(end_time))
        ]
        spikes[su] = arr

        # Initialize the count-table row for the current neuron.
        row = {"SU_ID": su}

        # Count spikes within every temporal window.
        for w in range(n_windows):
            t0 = float(edges[w])
            t1 = float(edges[w + 1])

            # Intermediate windows are left-closed and right-open so spikes on
            # shared boundaries are counted exactly once.
            if w < n_windows - 1:
                c = int(
                    np.sum((arr >= t0) & (arr < t1))
                )

            # The final window includes the requested recording endpoint.
            else:
                c = int(
                    np.sum((arr >= t0) & (arr <= t1))
                )

            row[f"Window_{w + 1}"] = c

        rows.append(row)

    # Convert all neuron count rows into the canonical spike-count table.
    counts_df = pd.DataFrame(rows)

    # Enforce a deterministic column order, beginning with the neuron
    # identifier and followed by chronologically ordered temporal windows.
    cols = ["SU_ID"] + [
        f"Window_{i}" for i in range(1, n_windows + 1)
    ]
    counts_df = counts_df.reindex(columns=cols)

    # Return the count matrix, filtered spike trains, and temporal bin edges.
    return counts_df, spikes, edges


def _write_counts_and_raster(
    *,
    run_dir: Path,
    counts_df: pd.DataFrame,
    spikes: dict[str, np.ndarray],
    start_time: float,
    total_duration: float,
    T: float,
) -> dict[str, str | None]:
    """
    Write the canonical spike-count table and spike raster.

    This function saves the temporal spike-count matrix and generates a raster
    visualization of the interval-filtered spike trains. Both outputs are
    written to the canonical ``Micro`` directory for the current analysis run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Root output directory for the current NeuroSeq analysis.

    counts_df : pandas.DataFrame
        Neuron-by-window spike-count table generated by
        :func:`_build_spike_counts_df`.

    spikes : dict[str, numpy.ndarray]
        Dictionary mapping each single-unit identifier to its spike times
        within the analyzed recording interval.

    start_time : float
        Beginning of the analyzed recording interval, in seconds.

    total_duration : float
        Duration of the analyzed recording interval, in seconds.

    T : float
        Width of the temporal spike-count windows, in seconds.

    Returns
    -------
    dict[str, str or None]
        Dictionary containing the canonical paths

        - ``counts_csv_path``
        - ``raster_png``

        These paths are later recorded in ``micro_ctx.json``.

    Notes
    -----
    The raster is generated with a noninteractive Matplotlib backend so the
    function can run safely on remote servers, computing clusters, and other
    environments without a graphical display.
    """

    # Create the canonical directory containing the principal Micro outputs.
    micro_dir = run_dir / "Micro"
    _ensure_dir(micro_dir)

    # Determine the absolute end of the analyzed recording interval.
    end_time = float(start_time) + float(total_duration)

    # Save the complete neuron-by-window spike-count matrix.
    counts_csv = (
        micro_dir
        / (
            f"spike_counts_allSUs_{start_time:.3f}_{end_time:.3f}_"
            f"win{T:.3f}s.csv"
        )
    )
    counts_df.to_csv(counts_csv, index=False)

    # Define the canonical path for the interval spike raster.
    raster_png = (
        micro_dir
        / f"spike_raster_allSUs_{start_time:.3f}_{end_time:.3f}.png"
    )

    # Sort neurons numerically so that raster rows follow the same deterministic
    # ordering used throughout the Micro analysis.
    sorted_items = sorted(
        spikes.items(),
        key=lambda kv: _su_sort_key(kv[0]),
    )

    # Create the population spike raster.
    plt.figure(figsize=(14, 6))

    # Draw one vertical tick for every spike emitted by each neuron.
    for i, (su, arr) in enumerate(sorted_items, start=1):
        if arr.size:
            plt.vlines(
                arr,
                i - 0.4,
                i + 0.4,
                lw=0.8,
            )

    # Label the raster using physical recording time and neuron identifiers.
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron")
    plt.title(
        f"Spike Raster: {start_time:.3f}--{end_time:.3f}s, "
        f"window={T:.3f}s"
    )
    plt.yticks(
        range(1, len(sorted_items) + 1),
        [sid for sid, _ in sorted_items],
    )
    plt.xlim(start_time, end_time)

    # Add light temporal guides without obscuring the spike events.
    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    # Finalize and save the raster at publication-quality resolution.
    plt.tight_layout()
    plt.savefig(raster_png, dpi=300)
    plt.close()

    # Return the canonical output paths recorded by the Micro context.
    return {
        "counts_csv_path": str(counts_csv),
        "raster_png": str(raster_png),
    }