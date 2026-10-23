# Micro/micro_critical_points.py

"""
Critical-Point Processing and Cross-Gamma Consensus
===================================================

Module
------
Micro.micro_critical_points

Purpose
-------
Identify candidate temporal state-transition boundaries from adjacent-window
Normalized Mutual Information, convert those boundaries into temporal state
labels, summarize the resulting state durations, and merge repeated or nearby
critical points detected across the complete Louvain gamma sweep.

This module provides the temporal boundary-processing layer of the NeuroSeq
Micro analysis. It does not perform Louvain clustering, threshold estimation,
or representative-gamma selection. Instead, it receives thresholded NMI
results and transforms them into a consensus temporal segmentation.

Scientific Role
---------------
A critical point is defined as an adjacent-window boundary whose NMI value is
below the selected adaptive threshold. Such a decrease indicates that the
neuronal community organization differs substantially between two consecutive
temporal windows.

Because the same biological transition may be detected at the same or nearby
boundary indices across several gamma values, the complete set of detections is
merged into cross-gamma critical-point bands. Each band is represented by the
rounded median of its contributing boundary indices.

This consensus procedure preserves critical points detected anywhere across
the gamma sweep and does not require selecting a single gamma for temporal
segmentation.

Main Operations
---------------
This module performs five related operations:

1. Extract threshold-crossing boundary indices from an NMI profile.
2. Convert critical-point indices into consecutive temporal state labels.
3. Compute run-length statistics for the resulting state sequence.
4. Merge repeated or nearby critical points across gamma into canonical bands.
5. Encode the final consensus boundaries as a binary NMI trace for the existing
   state-window construction interface.

Boundary Convention
-------------------
A boundary index ``i`` refers to the transition between temporal windows

    Window_{i + 1}  and  Window_{i + 2}

when using zero-based Python indexing.

If there are ``B`` adjacent-window boundaries, then the corresponding temporal
sequence contains ``B + 1`` windows.

Consensus-Band Policy
---------------------
Cross-gamma critical points are sorted by boundary index and grouped into bands
according to the requested tolerance in temporal bins.

Each band is represented by

    rounded median of contributing boundary indices.

The diagnostic table retains the contributing gamma values, raw boundary
indices, frequency of observations, and gamma range.

Synthetic NMI Representation
----------------------------
The final consensus segmentation is converted into a binary NMI trace:

- consensus critical points receive NMI = 0,
- all other boundaries receive NMI = 1,
- the canonical threshold is set to 0.5.

This representation allows the existing state-window builder to consume the
consensus boundaries without changing its public interface.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _extract_critical_indices(
    nmi_df: pd.DataFrame,
    threshold: float,
) -> list[int]:
    """
    Extract temporal boundary indices whose NMI falls below the threshold.

    Candidate critical points are identified directly from the adjacent-window
    NMI profile. A boundary is retained when its NMI value is finite and
    strictly smaller than the selected threshold.

    Parameters
    ----------
    nmi_df : pandas.DataFrame
        Adjacent-window NMI table containing an ``NMI`` column.

    threshold : float
        Adaptive NMI threshold used to distinguish stable temporal transitions
        from candidate changes in neuronal community structure.

    Returns
    -------
    list[int]
        Zero-based indices of adjacent-window boundaries whose NMI values fall
        below the threshold.

    Notes
    -----
    Missing or nonnumeric NMI values are converted to ``NaN`` and excluded from
    the returned critical-point list.

    The comparison is strict:

        NMI < threshold

    Therefore, an NMI value exactly equal to the threshold is not treated as a
    critical point.
    """

    # Convert the NMI column to a numeric array while safely representing
    # malformed or missing values as NaN.
    vals = pd.to_numeric(
        nmi_df["NMI"],
        errors="coerce",
    ).to_numpy(float)

    # Retain only finite NMI values lying strictly below the selected threshold.
    mask = np.isfinite(vals) & (vals < float(threshold))

    # Return the zero-based positions of all detected critical points.
    return np.where(mask)[0].astype(int).tolist()


def _critical_indices_to_window_labels(
    n_boundaries: int,
    critical_indices: list[int],
) -> np.ndarray:
    """
    Convert critical-point boundaries into consecutive temporal state labels.

    The adjacent-window boundary sequence contains one fewer element than the
    temporal window sequence. This function therefore creates
    ``n_boundaries + 1`` window labels and increments the state label after
    every valid critical point.

    Parameters
    ----------
    n_boundaries : int
        Number of adjacent-window boundaries in the temporal recording.

    critical_indices : list[int]
        Zero-based boundary indices marking candidate state transitions.

    Returns
    -------
    numpy.ndarray
        Integer state label assigned to every temporal window.

        Labels begin at zero and increase consecutively after each retained
        critical point.

    Notes
    -----
    Duplicate critical-point indices are removed automatically.

    Critical points outside the valid range

        0 <= index < n_boundaries

    are discarded before the temporal labels are constructed.
    """

    # A sequence with B adjacent-window boundaries contains B + 1 temporal
    # windows.
    n_windows = int(n_boundaries) + 1

    # Initialize every temporal window with the first state label.
    labels = np.zeros(n_windows, dtype=int)

    # Normalize the critical-point list by removing duplicates, sorting the
    # indices, and discarding boundaries outside the valid range.
    cps = sorted(
        set(
            int(i)
            for i in critical_indices
            if 0 <= int(i) < int(n_boundaries)
        )
    )

    # Track the current state label and the first unassigned temporal window.
    cur = 0
    last = 0

    # Assign one consecutive state label to the windows preceding each critical
    # point, then advance to the next state.
    for cp in cps:
        labels[last:cp + 1] = cur
        cur += 1
        last = cp + 1

    # Assign the final state label to every window following the last critical
    # point.
    labels[last:] = cur

    # Return the complete temporal state sequence.
    return labels


def _run_length_stats(
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Summarize the lengths of consecutive temporal state runs.

    This function identifies contiguous runs of identical state labels and
    reports basic statistics describing their lengths in temporal windows.

    Parameters
    ----------
    labels : numpy.ndarray
        One-dimensional temporal sequence of state labels.

    Returns
    -------
    dict[str, float]
        Dictionary containing

        - ``n_segments``
        - ``mean_segment_len``
        - ``median_segment_len``
        - ``min_segment_len``

    Notes
    -----
    Segment lengths are reported in numbers of temporal windows, not seconds.

    For an empty input sequence, all returned statistics are zero.
    """

    # Normalize the supplied label sequence to a NumPy array.
    labels = np.asarray(labels)

    # Return an explicit zero-valued summary when the temporal sequence is
    # empty.
    if labels.size == 0:
        return {
            "n_segments": 0,
            "mean_segment_len": 0.0,
            "median_segment_len": 0.0,
            "min_segment_len": 0.0,
        }

    # Store the lengths of all consecutive state-label runs.
    lengths = []

    # Track the starting index of the current run.
    start = 0

    # Detect every location where the temporal state label changes.
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            lengths.append(i - start)
            start = i

    # Include the final run, which terminates at the end of the sequence.
    lengths.append(len(labels) - start)

    # Convert run lengths to a numeric array for summary-statistic calculation.
    arr = np.asarray(lengths, dtype=float)

    # Return the principal run-length statistics used by the gamma-sweep
    # diagnostics.
    return {
        "n_segments": int(len(arr)),
        "mean_segment_len": float(np.mean(arr)),
        "median_segment_len": float(np.median(arr)),
        "min_segment_len": float(np.min(arr)),
    }


def _merge_all_critical_points_into_bands(
    *,
    critical_sets: dict[float, set[int]],
    tolerance_bins: int,
) -> tuple[list[int], pd.DataFrame]:
    """
    Merge repeated or nearby critical points across the complete gamma sweep.

    Every detected critical point is represented by its gamma value and temporal
    boundary index. The detections are sorted by boundary location and grouped
    into consensus bands according to the requested temporal tolerance.

    Each resulting band is represented by the rounded median of its raw
    boundary indices. The returned diagnostic table preserves the frequency,
    gamma support, and raw locations contributing to every band.

    Parameters
    ----------
    critical_sets : dict[float, set[int]]
        Mapping from each successfully evaluated gamma value to the set of
        critical-point boundary indices detected at that gamma.

    tolerance_bins : int
        Maximum allowed distance, in temporal boundary bins, between a new
        critical point and the current band's median center.

        A value of zero merges only detections at the same boundary index.

    Returns
    -------
    selected : list[int]
        Sorted representative boundary index for each final consensus band.

    band_df : pandas.DataFrame
        Diagnostic table describing all retained critical-point bands.

        Columns include

        - ``band_id``
        - ``boundary_index``
        - ``n_observations``
        - ``n_unique_gammas``
        - ``gamma_min``
        - ``gamma_max``
        - ``gammas``
        - ``raw_boundary_indices``
        - ``source``

    Notes
    -----
    Band membership is evaluated sequentially after sorting all detections by
    boundary index and gamma.

    The center of the current band is recalculated as the rounded median of all
    boundary indices already assigned to that band.

    If multiple bands produce the same rounded representative boundary index,
    only the band with the largest number of observations is retained.
    """

    # Collect every gamma-specific critical-point detection in long-form
    # representation.
    all_points = []

    for gamma, cps in critical_sets.items():
        for cp in sorted(cps):
            all_points.append(
                {
                    "gamma": float(gamma),
                    "boundary_index": int(cp),
                }
            )

    # Return an empty but schema-consistent diagnostic table when no critical
    # points were detected anywhere across the gamma sweep.
    if not all_points:
        return [], pd.DataFrame(
            columns=[
                "band_id",
                "boundary_index",
                "n_observations",
                "n_unique_gammas",
                "gamma_min",
                "gamma_max",
                "gammas",
                "raw_boundary_indices",
                "source",
            ]
        )

    # Sort detections primarily by temporal boundary index and secondarily by
    # gamma so the band-building procedure is deterministic.
    points = sorted(
        all_points,
        key=lambda r: (
            r["boundary_index"],
            r["gamma"],
        ),
    )

    # Negative tolerance values are interpreted as zero.
    tol = max(0, int(tolerance_bins))

    # Initialize the first consensus band with the earliest detected critical
    # point.
    bands: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = [points[0]]

    # Assign each remaining detection to the current band when it lies within
    # the requested distance of that band's rounded median center.
    for row in points[1:]:
        prev_center = int(
            round(
                np.median(
                    [x["boundary_index"] for x in cur]
                )
            )
        )

        if abs(int(row["boundary_index"]) - prev_center) <= tol:
            cur.append(row)
        else:
            bands.append(cur)
            cur = [row]

    # Preserve the final active band.
    bands.append(cur)

    # Convert all consensus bands into diagnostic rows.
    rows = []

    for band_id, band in enumerate(bands):
        # Extract all temporal boundary indices contributing to this band.
        cps = np.asarray(
            [int(x["boundary_index"]) for x in band],
            dtype=int,
        )

        # Identify the unique gamma values supporting this band.
        gammas = sorted(
            set(float(x["gamma"]) for x in band)
        )

        # Represent the band by the rounded median of its raw boundary indices.
        rep = int(
            round(
                float(np.median(cps))
            )
        )

        # Store the full support and provenance of the consensus band.
        rows.append(
            {
                "band_id": int(band_id),
                "boundary_index": int(rep),
                "n_observations": int(len(band)),
                "n_unique_gammas": int(len(gammas)),
                "gamma_min": float(np.min(gammas)),
                "gamma_max": float(np.max(gammas)),
                "gammas": ",".join(
                    f"{float(g):.8g}" for g in gammas
                ),
                "raw_boundary_indices": ",".join(
                    str(int(x))
                    for x in sorted(cps.tolist())
                ),
                "source": "all_unique_cp_bands_no_gamma_selection",
            }
        )

    # Construct the canonical critical-point band table and order the bands by
    # their temporal locations.
    band_df = (
        pd.DataFrame(rows)
        .sort_values("boundary_index")
        .reset_index(drop=True)
    )

    # Rounding may cause multiple bands to share one representative boundary
    # index. Retain the most strongly supported band at each duplicate location.
    band_df = (
        band_df
        .sort_values(
            ["boundary_index", "n_observations"],
            ascending=[True, False],
        )
        .drop_duplicates(
            subset=["boundary_index"],
            keep="first",
        )
        .sort_values("boundary_index")
        .reset_index(drop=True)
    )

    # Reassign consecutive band identifiers after duplicate representatives
    # have been removed.
    band_df["band_id"] = np.arange(
        len(band_df),
        dtype=int,
    )

    # Extract the final sorted consensus boundary indices.
    selected = [
        int(x)
        for x in band_df["boundary_index"].tolist()
    ]

    # Return both the canonical boundary list and the complete diagnostics.
    return selected, band_df


def _make_synthetic_nmi_for_consensus_boundaries(
    *,
    template_nmi_df: pd.DataFrame,
    consensus_critical_indices: list[int],
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Encode consensus critical points as a canonical binary NMI trace.

    This function preserves the structure and timing columns of an existing
    adjacent-window NMI table while replacing its active ``NMI`` values with a
    binary representation of the final consensus segmentation.

    Consensus critical points receive NMI zero, whereas all remaining
    boundaries receive NMI one. With the canonical threshold set to 0.5, the
    existing state-window builder therefore detects exactly the requested
    consensus boundaries.

    Parameters
    ----------
    template_nmi_df : pandas.DataFrame
        Existing adjacent-window NMI table used as the structural and temporal
        template for the synthetic representation.

    consensus_critical_indices : list[int]
        Final zero-based consensus boundary indices.

    threshold : float, default=0.5
        Threshold returned with the synthetic NMI trace.

    Returns
    -------
    out : pandas.DataFrame
        Copy of the template NMI table containing

        - the original NMI values,
        - the binary consensus NMI trace,
        - and a binary ``Consensus_CP`` indicator.

    summary : pandas.DataFrame
        One-row threshold summary describing the consensus-boundary source and
        number of critical points.

    threshold : float
        Canonical threshold associated with the synthetic NMI trace.

    Notes
    -----
    The original representative-gamma NMI values are preserved in

        ``Original_NMI_from_state_louvain_gamma``.

    This function does not alter the consensus boundary locations. It only
    expresses them in the format expected by the existing state-window
    construction module.
    """

    # Copy the template so the original adjacent-window NMI table remains
    # unchanged.
    out = template_nmi_df.copy()

    # Normalize the final consensus critical-point indices into a set for
    # efficient membership testing.
    cps = set(
        int(x)
        for x in consensus_critical_indices
    )

    # Preserve the original NMI profile generated at the representative state
    # Louvain gamma.
    out["Original_NMI_from_state_louvain_gamma"] = pd.to_numeric(
        out["NMI"],
        errors="coerce",
    )

    # Encode consensus critical points as zero and all other boundaries as one.
    out["NMI"] = [
        0.0 if int(i) in cps else 1.0
        for i in range(len(out))
    ]

    # Store an explicit binary indicator for downstream diagnostics.
    out["Consensus_CP"] = [
        int(i in cps)
        for i in range(len(out))
    ]

    # Construct the canonical threshold summary consumed by the existing Micro
    # output and state-window interfaces.
    summary = pd.DataFrame(
        [
            {
                "method": "gmm2",
                "threshold": float(threshold),
                "source": "auto_gamma_consensus_cp_bands",
                "n_consensus_cp": int(len(cps)),
            }
        ]
    )

    # Return the synthetic NMI table, its threshold summary, and the canonical
    # threshold value.
    return out, summary, float(threshold)