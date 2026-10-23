# Micro/micro_gamma_selection.py

"""
Representative State-Louvain Gamma Selection
============================================

Module
------
Micro.micro_gamma_selection

Purpose
-------
Select the representative Louvain resolution used for neuronal community
detection within the final consensus state windows.

The temporal boundaries of the NeuroSeq Micro analysis are determined from
critical points detected across the complete gamma sweep. Consequently, the
representative gamma selected in this module does not determine, remove, or
modify the final temporal segmentation.

Instead, the selected gamma is used exclusively when Louvain community
detection is performed inside the final consensus State_Windows.

Selection Principle
-------------------
For every successfully evaluated gamma value, the Micro gamma sweep produces a
set of candidate critical points. Let

    N_CP(gamma)

denote the number of critical points detected at resolution ``gamma``.

The representative state-Louvain gamma is selected according to the following
rule:

1. Compute ``N_CP(gamma)`` for every successful gamma.
2. Determine the maximum observed critical-point count.
3. Retain all gamma values attaining that maximum.
4. Group those gamma values into contiguous plateaus.
5. Select the plateau containing the largest number of consecutive gamma
   values.
6. If multiple plateaus have the same length, select the plateau with the
   largest gamma endpoint.
7. Select the largest gamma value inside the chosen plateau.

Scientific Rationale
--------------------
The maximum-critical-point plateau identifies a range of Louvain resolutions
at which temporal network reorganizations are detected with consistently high
sensitivity.

Selecting a plateau rather than one isolated gamma reduces sensitivity to a
single resolution value. Selecting the largest gamma within the strongest
plateau favors a finer neuronal community representation while remaining
inside a stable region of the gamma sweep.

This rule intentionally avoids selecting the first low-resolution gamma that
appears stable.

Boundary Independence
---------------------
The representative gamma has no role in defining the final temporal state
boundaries.

Final boundaries are obtained from the all-gamma consensus critical points.
The selected gamma is used only for Louvain clustering within those already
defined state windows.

Main Components
---------------
This module provides

- ``SweepResult``:
  an in-memory container for the results of one successful gamma evaluation;

- ``_contiguous_gamma_plateaus``:
  a utility for grouping neighboring gamma values into contiguous plateaus;

- ``_select_state_louvain_gamma_by_max_cp_plateau``:
  the complete representative-gamma selection procedure.

Outputs
-------
The selection procedure returns

- the selected representative gamma,
- a machine-readable diagnostic dictionary,
- and a per-gamma selection table describing critical-point counts, plateau
  membership, and the final selected gamma.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SweepResult:
    """
    Store the results of one successful gamma evaluation.

    Each instance contains the complete in-memory output generated for one
    Louvain resolution during the automatic Micro gamma sweep. These values
    support cross-gamma critical-point processing, representative-gamma
    selection, and final output generation.

    Attributes
    ----------
    gamma : float
        Louvain resolution evaluated during this sweep iteration.

    threshold : float
        Adaptive adjacent-window NMI threshold selected for this gamma.

    nmi_df : pandas.DataFrame
        Adjacent-window NMI profile computed from the temporal Louvain
        partitions.

    threshold_summary_df : pandas.DataFrame
        Diagnostic summary produced by the NMI threshold-estimation procedure.

    critical_indices : list[int]
        Zero-based temporal boundary indices whose NMI values fall below the
        selected threshold.

    window_labels : numpy.ndarray
        Consecutive temporal state labels generated from the detected critical
        points.

    per_window_partitions : dict[str, pandas.DataFrame]
        Louvain community partitions for every temporal window at this gamma.

    status : str
        Completion status associated with this gamma evaluation.

    error : str, default=""
        Error description associated with the evaluation. Successful results
        normally contain an empty string.
    """

    gamma: float
    threshold: float
    nmi_df: pd.DataFrame
    threshold_summary_df: pd.DataFrame
    critical_indices: list[int]
    window_labels: np.ndarray
    per_window_partitions: dict[str, pd.DataFrame]
    status: str
    error: str = ""


def _contiguous_gamma_plateaus(
    gammas: list[float],
    gamma_step: float | None = None,
) -> list[list[float]]:
    """
    Group neighboring gamma values into contiguous plateaus.

    A plateau is a sequence of sorted gamma values separated by approximately
    one gamma-sweep step. This grouping is used to determine whether the
    maximum critical-point count persists across a stable interval of Louvain
    resolutions rather than appearing at only one isolated gamma.

    Parameters
    ----------
    gammas : list[float]
        Gamma values to organize into contiguous groups.

    gamma_step : float or None, default=None
        Expected increment between consecutive gamma values.

        When a valid positive value is supplied, it defines the expected sweep
        spacing. When ``None``, nonfinite, or nonpositive, the spacing is
        estimated from the median positive difference between the supplied
        gamma values.

    Returns
    -------
    list[list[float]]
        Ordered list of contiguous gamma plateaus. Each inner list contains the
        gamma values assigned to one plateau.

    Notes
    -----
    Gamma values are sorted before plateau construction.

    The continuity tolerance is defined as

        max(1e-10, 0.51 * gamma_step).

    Two consecutive gamma values are placed in the same plateau when their
    separation is no greater than the expected step plus this tolerance.

    This function is used only for representative state-window gamma
    selection. It does not affect the consensus temporal boundaries.
    """

    # Sort and normalize all supplied gamma values before determining
    # contiguous regions.
    vals = sorted(float(g) for g in gammas)

    # No plateaus can be constructed when no gamma values are available.
    if not vals:
        return []

    # A single gamma value forms a one-element plateau.
    if len(vals) == 1:
        return [vals]

    # Estimate the expected sweep increment when no valid gamma step has been
    # supplied explicitly.
    if gamma_step is None or not np.isfinite(float(gamma_step)) or float(gamma_step) <= 0:
        diffs = np.diff(vals)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        gamma_step = float(np.median(diffs)) if diffs.size else 1e-8

    # Define the expected gamma spacing and a numerical tolerance that absorbs
    # small floating-point differences.
    step = float(gamma_step)
    tol = max(1e-10, 0.51 * step)

    # Initialize the plateau collection with the smallest gamma value.
    plateaus: list[list[float]] = []
    cur = [vals[0]]

    # Extend the active plateau while consecutive gamma values remain within
    # approximately one sweep step.
    for g in vals[1:]:
        if abs(float(g) - float(cur[-1])) <= step + tol:
            cur.append(float(g))
        else:
            plateaus.append(cur)
            cur = [float(g)]

    # Preserve the final active plateau.
    plateaus.append(cur)

    # Return all contiguous gamma regions in ascending gamma order.
    return plateaus


def _select_state_louvain_gamma_by_max_cp_plateau(
    *,
    ok_gammas: list[float],
    sweep_results: dict[float, SweepResult],
    gamma_step: float,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """
    Select the representative gamma for final State_Windows Louvain clustering.

    The selection is based exclusively on the number of critical points
    detected at each successfully evaluated gamma. Gamma values attaining the
    maximum critical-point count are grouped into contiguous plateaus, and the
    most persistent high-count plateau is selected.

    The representative gamma is used only for Louvain community detection
    inside the final consensus state windows. It does not determine the final
    temporal boundaries.

    Selection Rule
    --------------
    The procedure applies the following ordered criteria:

    1. Compute ``N_CP(gamma)``, the number of critical points detected at each
       successful gamma.
    2. Determine the maximum observed critical-point count.
    3. Retain all gamma values satisfying

           N_CP(gamma) = max_N_CP.

    4. Group the retained gamma values into contiguous plateaus.
    5. Select the longest plateau.
    6. If multiple plateaus have equal length, select the plateau with the
       largest gamma endpoint.
    7. Select the largest gamma inside the chosen plateau.

    Parameters
    ----------
    ok_gammas : list[float]
        Gamma values whose complete Micro evaluations finished successfully.

    sweep_results : dict[float, SweepResult]
        Mapping from stored gamma values to their complete in-memory sweep
        results.

    gamma_step : float
        Expected spacing between consecutive gamma values in the sweep.

    Returns
    -------
    selected_gamma : float
        Representative Louvain gamma used for community detection inside the
        final consensus State_Windows.

    diag : dict[str, Any]
        Machine-readable summary of the selection procedure, including the
        maximum critical-point count, selected plateau, selected gamma, and
        tie-breaking rules.

    table : pandas.DataFrame
        Per-gamma diagnostic table containing

        - ``gamma``
        - ``N_cp``
        - ``is_max_N_cp``
        - ``plateau_id``
        - ``selected``

    Raises
    ------
    RuntimeError
        Raised when no successful gamma values are available.

    Notes
    -----
    The nearest stored gamma key is used when reading ``sweep_results`` to
    protect against small floating-point representation differences.

    A fallback ranking by decreasing critical-point count and decreasing gamma
    is retained in case no plateau can be constructed.

    The representative gamma does not modify the all-gamma consensus critical
    points used for temporal segmentation.
    """

    # Sort all successfully evaluated gamma values in ascending order.
    gammas = sorted(float(g) for g in ok_gammas)

    # Representative-gamma selection requires at least one successful sweep
    # result.
    if not gammas:
        raise RuntimeError("No valid gamma values available for state-window Louvain.")

    # Store one diagnostic row for every successful gamma value.
    rows = []

    # Determine the number of detected critical points at each gamma.
    for gamma in gammas:
        # Use nearest stored key to avoid floating precision issues.
        key = min(sweep_results.keys(), key=lambda x: abs(float(x) - float(gamma)))
        ncp = int(len(sweep_results[key].critical_indices))

        # Initialize plateau and selection fields. These values are populated
        # after the maximum-critical-point plateaus have been identified.
        rows.append(
            {
                "gamma": float(key),
                "N_cp": int(ncp),
                "is_max_N_cp": False,
                "plateau_id": np.nan,
                "selected": False,
            }
        )

    # Build the canonical per-gamma diagnostic table, remove any duplicate
    # stored gamma keys, and restore ascending gamma order.
    table = pd.DataFrame(rows).drop_duplicates(subset=["gamma"]).sort_values("gamma").reset_index(drop=True)

    # Identify the maximum number of critical points observed anywhere in the
    # successful gamma sweep.
    max_ncp = int(table["N_cp"].max())

    # Mark every gamma value attaining the maximum critical-point count.
    table["is_max_N_cp"] = table["N_cp"].astype(int) == int(max_ncp)

    # Extract the gamma values belonging to the maximum-critical-point set.
    max_gammas = table.loc[table["is_max_N_cp"], "gamma"].astype(float).tolist()

    # Group neighboring maximum-critical-point gamma values into contiguous
    # plateaus.
    plateaus = _contiguous_gamma_plateaus(max_gammas, gamma_step=float(gamma_step))

    # Retain a deterministic fallback in case no plateau can be formed.
    if not plateaus:
        selected_gamma = float(table.sort_values(["N_cp", "gamma"], ascending=[False, False]).iloc[0]["gamma"])
        selected_plateau = [selected_gamma]
        selected_plateau_id = 0

    else:
        # Summarize the length and gamma range of every maximum-critical-point
        # plateau.
        plateau_rows = []

        for pid, plateau in enumerate(plateaus):
            plateau_rows.append(
                {
                    "plateau_id": int(pid),
                    "length": int(len(plateau)),
                    "gamma_min": float(min(plateau)),
                    "gamma_max": float(max(plateau)),
                }
            )

        # Convert the plateau summaries into a table for deterministic ranking.
        plateau_df = pd.DataFrame(plateau_rows)

        # Select the longest plateau. When plateau lengths are tied, prefer the
        # plateau ending at the largest gamma value.
        best = plateau_df.sort_values(
            ["length", "gamma_max"],
            ascending=[False, False],
        ).iloc[0]

        # Recover the selected plateau and choose its largest gamma value.
        selected_plateau_id = int(best["plateau_id"])
        selected_plateau = plateaus[selected_plateau_id]
        selected_gamma = float(max(selected_plateau))

    # Assign a plateau identifier to every gamma belonging to a
    # maximum-critical-point plateau.
    for pid, plateau in enumerate(plateaus):
        mask = table["gamma"].isin([float(g) for g in plateau])
        table.loc[mask, "plateau_id"] = int(pid)

    # Mark the final representative gamma in the per-gamma diagnostic table.
    table["selected"] = np.isclose(table["gamma"].astype(float), float(selected_gamma))

    # Assemble the machine-readable summary of the representative-gamma
    # selection and all tie-breaking decisions.
    diag = {
        "policy": "max_cp_plateau_largest_gamma",
        "max_N_cp": int(max_ncp),
        "n_max_N_cp_gammas": int(len(max_gammas)),
        "selected_gamma": float(selected_gamma),
        "selected_plateau_id": int(selected_plateau_id),
        "selected_plateau_gamma_min": float(min(selected_plateau)),
        "selected_plateau_gamma_max": float(max(selected_plateau)),
        "selected_plateau_length": int(len(selected_plateau)),
        "tie_breaks": [
            "longest contiguous max-N_CP plateau",
            "largest plateau gamma endpoint",
            "largest gamma inside selected plateau",
        ],
    }

    # Return the representative gamma, compact diagnostics, and complete
    # per-gamma selection table.
    return float(selected_gamma), diag, table