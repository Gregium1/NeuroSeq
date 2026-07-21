# Micro/micro_plots.py

"""
Diagnostic Visualization for the Micro Gamma Sweep
==================================================

Module
------
Micro.micro_plots

Purpose
-------
Generate diagnostic figures that summarize the behavior of the NeuroSeq Micro
gamma sweep.

These visualizations provide an overview of how candidate temporal
state-transition boundaries evolve across the Louvain resolution parameter
(gamma). They are intended for interpretation, quality control, and
methodological validation rather than for downstream computation.

Scientific Role
---------------
During the Micro gamma sweep, each Louvain resolution produces a set of
candidate critical points obtained from adjacent-window NMI.

The resulting critical-point persistence map visualizes

- the Louvain resolution (gamma) on the horizontal axis,
- the temporal boundary index on the vertical axis.

Each point therefore represents one detected critical point.

Regions where many neighboring gamma values detect the same boundary indicate
persistent temporal transitions that are robust to changes in Louvain
resolution. Conversely, isolated detections correspond to transitions that are
less stable across the gamma sweep.

When available, the representative state-Louvain gamma is superimposed on the
figure. This reference is provided for visualization only and does not define
the final temporal segmentation.

Outputs
-------
This module produces publication-quality PNG figures documenting the automatic
gamma sweep.

Current Figures
---------------
- Critical-point persistence map

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


def _plot_cp_map(
    out_png: Path,
    crit_df: pd.DataFrame,
    selected_gamma: float | None = None,
) -> None:
    """
    Plot the critical-point persistence map across the Micro gamma sweep.

    Each point represents one critical point detected during the automatic
    gamma sweep. The horizontal coordinate corresponds to the Louvain
    resolution (gamma), whereas the vertical coordinate corresponds to the
    temporal boundary index where the transition was detected.

    When supplied, the representative state-Louvain gamma is displayed as a
    vertical reference line.

    Parameters
    ----------
    out_png : pathlib.Path
        Output filename for the generated PNG figure.

    crit_df : pandas.DataFrame
        Table containing the detected critical points.

        Expected columns include

        - ``gamma``
        - ``boundary_index``

        Each row represents one critical-point observation obtained during the
        gamma sweep.

    selected_gamma : float or None, default=None
        Representative Louvain resolution selected for the final State_Windows
        Louvain analysis.

        When provided, this value is displayed as a vertical dashed line.

    Returns
    -------
    None

    Notes
    -----
    This figure is purely diagnostic.

    The representative gamma shown on the plot does not determine the final
    temporal boundaries. Final boundaries are obtained from the all-gamma
    consensus critical points.

    The persistence map is primarily used to visualize the stability of
    detected temporal transitions across the Louvain resolution sweep.
    """

    # Create the publication-quality figure used to summarize the gamma sweep.
    plt.figure(figsize=(9.0, 5.2))

    # Plot every detected critical point as one observation in gamma-boundary
    # space.
    if crit_df is not None and not crit_df.empty:
        plt.scatter(
            crit_df["gamma"],
            crit_df["boundary_index"],
            s=18,
            alpha=0.8,
        )

    # Display the representative state-Louvain gamma for visual reference.
    if selected_gamma is not None and np.isfinite(selected_gamma):
        plt.axvline(
            selected_gamma,
            linestyle="--",
            linewidth=2,
            label=f"state Louvain gamma = {selected_gamma:.4f}",
        )
        plt.legend(frameon=False)

    # Label the axes using the quantities represented by the gamma sweep.
    plt.xlabel("Micro Louvain resolution gamma")
    plt.ylabel("Critical-point boundary index")
    plt.title("Critical-point persistence map")

    # Add a light grid to improve visual interpretation of persistent
    # transition bands.
    plt.grid(alpha=0.25)

    # Optimize the layout before exporting the figure.
    plt.tight_layout()

    # Save the diagnostic figure at publication resolution.
    plt.savefig(
        out_png,
        dpi=300,
    )

    # Release the figure resources.
    plt.close()