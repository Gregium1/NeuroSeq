# Micro/micro_outputs.py

"""
Canonical Micro Output Writing and Final State-Level Analysis
=============================================================

Module
------
Micro.micro_outputs

Purpose
-------
Write the canonical outputs produced by the NeuroSeq Micro analysis and launch
the final state-level computations required by the Meso pipeline.

This module acts as the output and handoff layer of the Micro workflow. It
receives the completed automatic gamma-sweep results, writes the authoritative
temporal-window files, constructs the final consensus State_Windows, computes
state-level vectors and pairwise matrices, computes pairwise state Louvain NMI,
and records all canonical paths in ``micro_ctx.json``.

Scientific Role
---------------
The automatic Micro gamma sweep separates two distinct operations:

1. Temporal boundary detection
   Final state boundaries are determined from critical points detected across
   the complete gamma sweep.

2. Within-state neuronal clustering
   The selected representative gamma is used only for Louvain community
   detection inside the final consensus state windows.

This module preserves that separation when generating the final outputs.
Consensus critical points define the temporal segmentation, whereas the
representative state-Louvain gamma controls only the neuronal communities
identified within each state interval.

Canonical Output Structure
--------------------------
The principal outputs are written under

    Micro/

and

    Micro/Windows/

The temporal window directories contain only edge tables because window-level
Louvain partitions depend on gamma and are therefore not considered canonical.

The final state-level outputs are written under

    Micro/Windows/State_Windows/

These outputs include

- final state-window definitions,
- within-state Louvain partitions,
- state firing-rate vectors,
- pairwise state-distance matrices,
- pairwise state-similarity matrices,
- pairwise state Louvain NMI,
- and the state-level files consumed by the Meso analysis.

Consensus NMI Representation
----------------------------
The final consensus boundaries are supplied through a binary adjacent-window
NMI representation:

- consensus critical points have NMI = 0,
- all other boundaries have NMI = 1,
- the canonical threshold is 0.5.

This preserves compatibility with the existing state-window construction
interface while ensuring that the state segmentation is determined entirely
by the all-gamma consensus critical points.

Configuration and Context
-------------------------
Two authoritative metadata files are written:

``micro_step_config.json``
    Records the final Micro configuration used for state-level analysis.

``Micro/micro_ctx.json``
    Provides the canonical interface between Micro and downstream NeuroSeq
    stages. It records output paths, threshold information, boundary provenance,
    representative-gamma information, and state-level analysis outputs.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .micro_counts import _write_counts_and_raster
from .micro_utils import _cfg_value, _ensure_dir, _write_json


def _write_clean_auto_gamma_outputs(
    *,
    run_dir: Path,
    cfg: Any,
    counts_df: pd.DataFrame,
    spikes: dict[str, np.ndarray],
    edge_dfs: dict[str, pd.DataFrame],
    selected_nmi_df: pd.DataFrame,
    selected_threshold_summary_df: pd.DataFrame,
    selected_threshold: float,
    consensus_critical_indices: list[int],
    state_louvain_gamma: float,
    gamma_selection_diag: dict[str, Any] | None,
    start_time: float,
    total_duration: float,
    T: float,
    selected_su_ids: list[str] | None,
) -> dict[str, Any]:
    """
    Write canonical Micro outputs and run the final state-level analyses.

    This function writes the authoritative Micro files produced after the
    automatic gamma sweep. It stores the spike-count and raster outputs,
    temporal edge tables, consensus NMI representation, final State_Windows,
    state vectors and matrices, pairwise state Louvain NMI, the Micro
    configuration, and the canonical ``micro_ctx.json`` interface.

    The temporal boundaries and representative gamma remain independent:

    - consensus critical points define the final state intervals;
    - ``state_louvain_gamma`` is used only for Louvain clustering inside those
      intervals.

    Parameters
    ----------
    run_dir : pathlib.Path
        Root output directory for the current NeuroSeq analysis.

    cfg : Any
        Configuration object or dictionary containing the Micro parameters
        required for final output generation.

        Expected values include

        ``T``
            Spike-count similarity scaling parameter.


        ``data_folder``
            Directory containing the original ``SU_*.csv`` spike-time files.

        ``seed_for_states``
            Louvain random seed used inside the final consensus state windows.

    counts_df : pandas.DataFrame
        Canonical neuron-by-window spike-count table.

    spikes : dict[str, numpy.ndarray]
        Mapping from each included neuron identifier to its interval-filtered
        spike times.

    edge_dfs : dict[str, pandas.DataFrame]
        Mapping from each temporal window name to its canonical weighted edge
        table.

    selected_nmi_df : pandas.DataFrame
        Consensus adjacent-window NMI representation consumed by the
        state-window construction interface.

    selected_threshold_summary_df : pandas.DataFrame
        Summary table describing the threshold associated with the consensus
        NMI representation.

    selected_threshold : float
        Threshold used to identify consensus boundaries in
        ``selected_nmi_df``.

    consensus_critical_indices : list[int]
        Final zero-based temporal boundary indices obtained from all-gamma
        critical-point consensus.

    state_louvain_gamma : float
        Representative Louvain resolution used only for neuronal community
        detection inside the final consensus State_Windows.

    gamma_selection_diag : dict[str, Any] or None
        Diagnostic information describing how the representative state-Louvain
        gamma was selected.

    start_time : float
        Beginning of the analyzed recording interval, in seconds.

    total_duration : float
        Duration of the analyzed recording interval, in seconds.

    T : float
        Width of the temporal Micro windows, in seconds.

    selected_su_ids : list[str] or None
        Optional subset of single-unit identifiers included in the analysis.
        When ``None``, all available neurons were used.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the principal canonical output paths, threshold
        information, state-level analysis outputs, representative gamma, and
        the complete Micro context.

    Notes
    -----
    Temporal window directories intentionally contain only edge files.
    Window-level Louvain assignments vary with gamma and are therefore not
    canonical outputs.

    This function launches three downstream state-level procedures:

    1. final State_Windows construction and Louvain clustering;
    2. state-vector and pairwise-matrix construction;
    3. pairwise state Louvain NMI calculation.
    """

    # Normalize the root path and define the canonical Micro output hierarchy.
    run_dir = Path(run_dir)
    micro_dir = run_dir / "Micro"
    windows_base = micro_dir / "Micro-Epochs"

    # Ensure the principal Micro output directories exist before writing files.
    _ensure_dir(micro_dir)
    _ensure_dir(windows_base)

    # Read the final state-level parameters from the supplied Micro
    # configuration.
    T = float(_cfg_value(cfg, "T", 0.5))
    
    data_folder = str(_cfg_value(cfg, "data_folder", ""))
    seed_for_states = int(_cfg_value(cfg, "seed_for_states", 42))

    # Write the canonical spike-count table and interval spike raster.
    paths = _write_counts_and_raster(
        run_dir=run_dir,
        counts_df=counts_df,
        spikes=spikes,
        start_time=float(start_time),
        total_duration=float(total_duration),
        T=float(T),
    )

    # Preserve the neuron ordering used by the canonical count table.
    included_sus = counts_df["SU_ID"].astype(str).tolist()

    # Construct the stable one-based node mapping shared by the temporal graph
    # outputs.
    neuron_to_id = {
        su: i + 1
        for i, su in enumerate(included_sus)
    }

    # Save the authoritative mapping between original neuron identifiers and
    # internal graph-node identifiers.
    pd.DataFrame(
        {
            "Neuron_ID": included_sus,
            "Node_ID": [neuron_to_id[su] for su in included_sus],
        }
    ).to_csv(
        windows_base / "neuron_id_mapping.csv",
        index=False,
    )

    # Window folders intentionally contain only edge files. Window-level
    # Louvain assignments vary with gamma and are therefore not canonical.
    for wname in sorted(
        edge_dfs.keys(),
        key=lambda x: int(str(x).split("_")[-1]),
    ):
        # Create one canonical output directory per temporal window.
        wdir = windows_base / wname.replace("Window_", "Micro-Epoch_")
        _ensure_dir(wdir)

        # Save the weighted neuronal edge table for the current window.
        edge_dfs[wname].to_csv(
            wdir / f"{wname.replace('Window_', 'Micro-Epoch_')}_edges.csv",
            index=False,
        )

    # Store the binary consensus NMI representation consumed by the existing
    # state-window construction interface.
    nmi_csv = windows_base / "nmi_adjacent_micro-epoch.csv"
    selected_nmi_df.to_csv(
        nmi_csv,
        index=False,
    )

    # Generate the canonical diagnostic plot of the consensus adjacent-window
    # NMI representation.
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Plot the binary NMI trace against physical recording time.
    ax.plot(
        selected_nmi_df["Start_Time_s"],
        selected_nmi_df["NMI"],
        marker="o",
        linewidth=1.8,
        markersize=4,
    )

    # Mark the threshold separating consensus boundaries from non-boundaries.
    ax.axhline(
        selected_threshold,
        linestyle="--",
        linewidth=1.5,
        label=f"Consensus threshold = {selected_threshold:.4g}",
    )

    # Label and format the consensus NMI diagnostic.
    ax.set_xlabel("Time from start (s)")
    ax.set_ylabel(
        f"NMI (T={T}s)"
    )
    ax.set_title("Consensus adjacent-window NMI")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    # Save the consensus NMI figure in the canonical Windows directory.
    nmi_png = windows_base / "nmi_adjacent_micro-epoch.png"
    fig.savefig(
        nmi_png,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # State-level computations are performed by Meso, not by Micro.
    # The selected representative gamma is passed through the Micro context.
    manual_end_time = float(start_time) + float(total_duration)

    # Record the authoritative configuration used by the final Micro
    # state-level analysis.
    micro_cfg = {
        "data_folder": data_folder,
        "start_time": float(start_time),
        "total_duration": float(total_duration),
        "T": float(T),
        "use_all_su_ids": selected_su_ids is None,
        "selected_su_ids": selected_su_ids,
        "T": float(T),
        "use_auto_find_counts_csv": False,
        "counts_csv_search_root": "",
        "manual_start_time_for_nmi": float(start_time),
        "manual_end_time": float(manual_end_time),
        "nmi_threshold_method": "gmm2",
        "chosen_nmi_threshold": float(selected_threshold),
        "state_louvain_gamma": float(state_louvain_gamma),
        "louvain_resolution_for_states": float(state_louvain_gamma),
        "seed_for_states": int(seed_for_states),
        "BUILD_STATE_WINDOWS_FOR_ALL_ADJ_METHODS": False,
    }

    # Write the final Micro configuration at the run root.
    config_path = run_dir / "micro_step_config.json"
    _write_json(
        config_path,
        {"micro": micro_cfg},
    )

    # Assemble the authoritative Micro context consumed by downstream NeuroSeq
    # stages.
    ctx = {
        "counts_csv_path": paths["counts_csv_path"],
        "raster_png": paths["raster_png"],
        "windows_base": str(windows_base),
        "nmi_csv_path": str(nmi_csv),
        "nmi_png_path": str(nmi_png),
        "nmi_threshold_method": "gmm2",
        "chosen_nmi_threshold": float(selected_threshold),
        "chosen_nmi_threshold_source": "auto_gamma_consensus_cp_bands",
        "boundary_source": "all_gamma_consensus_critical_points",
        "boundary_gamma": None,
        "state_louvain_gamma": float(state_louvain_gamma),
        "state_louvain_gamma_policy": "max_cp_plateau_largest_gamma",
        "state_louvain_gamma_role": (
            "used only for Louvain clustering inside final consensus "
            "State_Windows"
        ),
        "state_louvain_gamma_selection": gamma_selection_diag or {},
        "selected_consensus_critical_indices": [
            int(x)
            for x in consensus_critical_indices
        ],
        "selected_su_ids": selected_su_ids,
        "start_time": float(start_time),
        "total_duration": float(total_duration),
        "T": float(T),
        "data_folder": data_folder,
    }

    # Write the canonical Micro-to-Meso interface.
    _write_json(
        micro_dir / "micro_ctx.json",
        ctx,
    )

    # Return the principal output paths and state-level analysis metadata to the
    # automatic gamma-sweep orchestrator.
    return {
        "config_path": str(config_path),
        "ctx": ctx,
        "counts_csv": paths["counts_csv_path"],
        "raster_png": paths["raster_png"],
        "windows_base": str(windows_base),
        "nmi_csv": str(nmi_csv),
        "adjacent_nmi_csv": str(nmi_csv),
        "recommended_nmi_threshold": float(selected_threshold),
        "nmi_threshold_method_used": "gmm2",
        "state_louvain_gamma": float(state_louvain_gamma),
    }