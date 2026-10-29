# Micro/auto_select_micro_gamma.py

"""Automatic Micro-Epoch analysis and consensus critical-point selection.

This module performs the **Micro** stage only. It counts spikes in temporal
Micro-Epochs, builds neuronal similarity graphs, evaluates Louvain partitions
across a gamma sweep, computes adjacent-window NMI, identifies candidate
critical points, and consolidates them into consensus temporal boundaries.

The representative gamma is selected using the maximum-critical-point plateau
policy and is passed to Meso for within-Proto-state Louvain clustering.

Important architecture boundary
-------------------------------
Proto-state construction, within-Proto-state neuronal partitions, firing-rate
vectors, and pairwise Proto-state comparisons are performed in ``Meso/``.
Micro does not repeat those calculations. Micro outputs a context JSON that
records the consensus segmentation and parameters required by Meso.

Spike-count arrays and Micro-Epoch graphs are prepared once before the gamma
sweep. Progress hooks used by the GUI report existing computation; they do
not create additional scientific passes.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .micro_adjacent_nmi import _compute_adjacent_nmi_in_memory
from .micro_counts import _build_spike_counts_df
from .micro_critical_points import (
    _critical_indices_to_window_labels,
    _extract_critical_indices,
    _make_synthetic_nmi_for_consensus_boundaries,
    _merge_all_critical_points_into_bands,
    _run_length_stats,
)
from .micro_gamma_selection import (
    SweepResult,
    _select_state_louvain_gamma_by_max_cp_plateau,
)
from .micro_graphs import _build_window_graphs_from_counts
from .micro_louvain import _run_single_gamma_louvain_in_memory
from .micro_outputs import _write_clean_auto_gamma_outputs
from .micro_plots import _plot_cp_map
from .micro_threshold import _pick_gmm2_threshold, _thresholds_from_nmi_gmm2
from .micro_utils import _cfg_value, _ensure_dir, _write_json


def auto_select_micro_gamma(
    *,
    cfg: Any,
    root: Path,
    start_time: float,
    total_duration: float,
    selected_su_ids: list[str] | None,
    nmi_threshold_method: str = "gmm2",
    gamma_min: float = 1.05,
    gamma_max: float = 1.20,
    gamma_step: float = 0.001,
    run_micro_AE_func: Any | None = None,
    persistence_tolerance_bins: int = 1,
    nmi_valley_flank: int = 2,
    min_consensus_frequency: float = 0.50,
    min_consensus_score_quantile: float | None = None,
    keep_all_consensus_boundaries: bool = True,
    delete_nonselected_gamma_outputs: bool = False,
    make_representative_gamma_plots: bool = True,
    consensus_cp_band_tolerance_bins: int | None = 0,
    **_compatibility,
) -> dict[str, Any]:
    """
    Run the complete automatic NeuroSeq Micro gamma sweep.

    This function coordinates the full Micro analysis for one recording
    interval. It constructs the temporal spike-count representation, evaluates
    Louvain community structure across the requested gamma range, detects
    candidate transition boundaries using adjacent-window NMI, merges those
    boundaries across gamma, selects the representative state-level Louvain
    gamma, and writes all canonical Micro outputs.

    Temporal state boundaries and the representative gamma are determined
    independently. Consensus critical points across the full gamma sweep define
    the temporal segmentation. The selected representative gamma is used only
    for Louvain clustering within the final state windows.

    Parameters
    ----------
    cfg : Any
        Configuration object or dictionary containing Micro parameters.

        Expected values include:

        ``data_folder``
            Directory containing the ``SU_*.csv`` spike-time files.

        ``T``
            Width of each temporal analysis window, in seconds.

        ``T``
            Scaling parameter used when converting pairwise spike-count
            differences into graph-edge weights.


        ``seeds_per_gamma``
            Number of Louvain random seeds evaluated for each gamma and
            temporal window.

        ``seed_for_states``
            Louvain random seed used within the final consensus state windows.

    root : pathlib.Path
        Root output directory for the current NeuroSeq analysis interval.

    start_time : float
        Beginning of the analyzed recording interval, in seconds.

    total_duration : float
        Duration of the analyzed recording interval, in seconds.

    selected_su_ids : list[str] or None
        Optional subset of single-unit identifiers to analyze. When ``None``,
        every available ``SU_*.csv`` file is included.

    nmi_threshold_method : str, default="gmm2"
        Compatibility argument identifying the adjacent-window NMI threshold
        method. The active implementation uses a two-component Gaussian
        mixture model.

    gamma_min : float, default=1.05
        Minimum Louvain resolution included in the automatic gamma sweep.

    gamma_max : float, default=1.20
        Maximum Louvain resolution included in the automatic gamma sweep.

    gamma_step : float, default=0.001
        Increment between consecutive gamma values.

    run_micro_AE_func : Any or None, default=None
        Compatibility argument retained for older callers. The current
        implementation performs the Micro analysis directly in memory.

    persistence_tolerance_bins : int, default=1
        Compatibility argument retained for previous neighboring-gamma
        stability calculations.

    nmi_valley_flank : int, default=2
        Compatibility argument retained for previous NMI-valley processing.

    min_consensus_frequency : float, default=0.50
        Compatibility argument retained for previous consensus-frequency
        filtering.

    min_consensus_score_quantile : float or None, default=None
        Compatibility argument retained for previous consensus-score
        filtering.

    keep_all_consensus_boundaries : bool, default=True
        Compatibility argument retained for previous consensus-boundary
        selection behavior.

    delete_nonselected_gamma_outputs : bool, default=False
        Compatibility argument retained for earlier implementations that wrote
        separate gamma-specific output directories.

    make_representative_gamma_plots : bool, default=True
        Compatibility argument retained for older representative-gamma
        plotting behavior.

    consensus_cp_band_tolerance_bins : int or None, default=0
        Maximum boundary-index separation allowed when merging critical points
        detected across gamma into one consensus critical-point band.

        A value of zero merges only critical points occurring at exactly the
        same boundary index. Positive values permit nearby boundary detections
        to be represented by a shared consensus location.

    **_compatibility : Any
        Additional compatibility arguments accepted so older pipeline calls
        continue to execute without modification.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:

        - completion status,
        - automatic gamma-sweep output directory,
        - consensus critical-point indices,
        - consensus segmentation statistics,
        - selected representative state-level Louvain gamma,
        - gamma-selection diagnostics,
        - critical-point persistence outputs,
        - Micro-Epoch output paths,
        - consensus critical-point paths,
        - and the canonical Micro context metadata.

    Raises
    ------
    FileNotFoundError
        If no usable ``SU_*.csv`` files are found in the configured data
        directory.

    RuntimeError
        If every gamma value in the requested sweep fails.

    Notes
    -----
    All temporal graphs are constructed once before the gamma sweep. Only the
    Louvain resolution changes across gamma, which avoids rebuilding identical
    spike-count and graph representations repeatedly.

    The representative gamma is selected using the maximum-critical-point
    plateau rule. It does not choose, remove, or alter the final consensus
    temporal boundaries.
    """

    # These arguments are retained so earlier NeuroSeq callers continue to
    # execute without modification. They are not used by the current modular,
    # in-memory Micro workflow.
    _ = (
        run_micro_AE_func,
        persistence_tolerance_bins,
        nmi_threshold_method,
        nmi_valley_flank,
        min_consensus_frequency,
        min_consensus_score_quantile,
        keep_all_consensus_boundaries,
        delete_nonselected_gamma_outputs,
        make_representative_gamma_plots,
        _compatibility,
    )

    # Create the directory containing automatic gamma-sweep diagnostics,
    # consensus critical-point outputs, and representative-gamma selection
    # results.
    root = Path(root)
    auto_root = root / "Auto_Micro_Gamma"
    _ensure_dir(auto_root)

    # Read the Micro parameters required by the gamma-sweep workflow.
    data_folder = str(_cfg_value(cfg, "data_folder"))
    T = float(_cfg_value(cfg, "T", 0.5))
    T = float(_cfg_value(cfg, "T", 0.5))
    
    seeds_per_gamma = int(_cfg_value(cfg, "seeds_per_gamma", 3))

    # Construct the inclusive sequence of Louvain resolutions evaluated during
    # the automatic Micro sweep. Rounding limits floating-point drift in the
    # gamma dictionary keys and diagnostic files.
    gammas = np.round(
        np.arange(gamma_min, gamma_max + 0.5 * gamma_step, gamma_step),
        8,
    )

    # Construct the temporal spike-count matrix once for the complete recording
    # interval. The same counts are reused at every gamma value.
    counts_df, spikes, _ = _build_spike_counts_df(
        data_folder=data_folder,
        start_time=start_time,
        total_duration=total_duration,
        T=T,
        selected_su_ids=selected_su_ids,
    )

    # Build one weighted neuronal similarity graph for each temporal window.
    # These graphs remain fixed throughout the gamma sweep; only the Louvain
    # resolution changes.
    graphs, edge_dfs, id_to_neuron = _build_window_graphs_from_counts(
        counts_df,
        T=T,
    )

    # Accumulate per-gamma summary information, raw critical-point detections,
    # complete critical-point sets, and all successful in-memory sweep results.
    rows = []
    critical_rows = []
    critical_sets: dict[float, set[int]] = {}
    sweep_results: dict[float, SweepResult] = {}

    # Evaluate the temporal community structure independently at every gamma.
    for gamma in gammas:
        gamma = float(gamma)

        try:
            # Run repeated-seed Louvain clustering for every temporal window at
            # the current gamma. The highest-modularity partition is retained
            # for each window.
            partitions = _run_single_gamma_louvain_in_memory(
                graphs=graphs,
                id_to_neuron=id_to_neuron,
                gamma=gamma,
                seeds_per_gamma=seeds_per_gamma,
            )

            # Quantify the similarity between the Louvain partitions of each
            # pair of consecutive temporal windows.
            nmi_df = _compute_adjacent_nmi_in_memory(
                per_window_partitions=partitions,
                start_time=start_time,
                T=T,
            )

            # Fit the active two-component Gaussian mixture model to the
            # adjacent-window NMI values and extract the adaptive threshold.
            threshold_summary = _thresholds_from_nmi_gmm2(
                nmi_df["NMI"].to_numpy(float)
            )
            threshold = _pick_gmm2_threshold(threshold_summary)

            # Mark every adjacent-window comparison below the GMM2 threshold as
            # a candidate temporal critical point.
            critical_indices = _extract_critical_indices(nmi_df, threshold)

            # Convert the detected boundary indices into consecutive temporal
            # state labels and summarize the resulting segment lengths.
            labels = _critical_indices_to_window_labels(
                len(nmi_df),
                critical_indices,
            )
            stats = _run_length_stats(labels)

            # Retain the complete critical-point set for cross-gamma consensus
            # construction.
            critical_sets[gamma] = set(critical_indices)

            # Store one long-form diagnostic row for every critical point
            # detected at the current gamma.
            for cp in critical_indices:
                critical_rows.append({
                    "gamma": gamma,
                    "boundary_index": int(cp),
                    "transition_time_s": start_time + (cp + 1) * T,
                    "threshold": threshold,
                })

            # Record the structural summary for the current successful gamma.
            rows.append({
                "gamma": gamma,
                "status": "ok",
                "threshold": threshold,
                "N_cp": len(critical_indices),
                **stats,
                "error": "",
            })

            # Preserve the full successful result in memory so it can later be
            # used for representative-gamma selection and final output writing.
            sweep_results[gamma] = SweepResult(
                gamma=gamma,
                threshold=threshold,
                nmi_df=nmi_df,
                threshold_summary_df=threshold_summary,
                critical_indices=critical_indices,
                window_labels=labels,
                per_window_partitions=partitions,
                status="ok",
            )

        except Exception as exc:
            # A failure at one gamma does not terminate the complete sweep.
            # Store the error and continue evaluating the remaining gamma
            # values.
            rows.append({
                "gamma": gamma,
                "status": "failed",
                "threshold": np.nan,
                "N_cp": np.nan,
                "n_segments": np.nan,
                "mean_segment_len": np.nan,
                "median_segment_len": np.nan,
                "min_segment_len": np.nan,
                "error": str(exc),
            })

    # Convert the accumulated sweep summaries and critical-point detections into
    # diagnostic tables.
    structure_df = pd.DataFrame(rows)
    critical_df = pd.DataFrame(critical_rows)

    # Only successfully completed gamma values are eligible for consensus
    # boundary construction and representative-gamma selection.
    ok_gammas = sorted(sweep_results)

    # Preserve the failure diagnostics before stopping when no gamma completed
    # successfully.
    if not ok_gammas:
        structure_df.to_csv(
            auto_root / "gamma_structure_selection.csv",
            index=False,
        )
        raise RuntimeError(
            "Automatic Micro gamma sweep failed for all gamma values."
        )

    # Normalize the user-specified consensus-band tolerance. A value of zero
    # merges only identical critical-point indices, whereas positive values
    # allow nearby detections to form one band.
    band_tolerance = (
        int(consensus_cp_band_tolerance_bins)
        if consensus_cp_band_tolerance_bins is not None
        else 0
    )

    # Merge repeated or nearby critical points detected across all successful
    # gamma values into canonical consensus critical-point bands.
    consensus_indices, band_df = _merge_all_critical_points_into_bands(
        critical_sets=critical_sets,
        tolerance_bins=band_tolerance,
    )

    # Select the representative gamma used only for Louvain clustering within
    # the final consensus state windows.
    state_gamma, selection_diag, selection_table = (
        _select_state_louvain_gamma_by_max_cp_plateau(
            ok_gammas=ok_gammas,
            sweep_results=sweep_results,
            gamma_step=gamma_step,
        )
    )

    # Resolve the selected floating-point gamma to the nearest exact dictionary
    # key retained during the sweep.
    representative_key = min(
        sweep_results,
        key=lambda g: abs(g - state_gamma),
    )
    representative = sweep_results[representative_key]
    state_gamma = float(representative_key)

    # Remove any consensus critical-point index falling outside the valid
    # adjacent-window boundary range of the representative result.
    n_boundaries = len(representative.nmi_df)
    consensus_indices = sorted({
        int(cp)
        for cp in consensus_indices
        if 0 <= int(cp) < n_boundaries
    })

    # Convert the final consensus boundaries into temporal state labels and
    # summarize the resulting state-window lengths.
    consensus_labels = _critical_indices_to_window_labels(
        n_boundaries,
        consensus_indices,
    )
    consensus_stats = _run_length_stats(consensus_labels)

    # Encode the consensus segmentation as a canonical binary NMI trace. The
    # existing state-window interface interprets NMI values below 0.5 as
    # temporal boundaries.
    synthetic_nmi, synthetic_summary, synthetic_threshold = (
        _make_synthetic_nmi_for_consensus_boundaries(
            template_nmi_df=representative.nmi_df,
            consensus_critical_indices=consensus_indices,
            threshold=0.5,
        )
    )

    # Write the canonical Micro outputs and launch the final state-window,
    # state-vector, and state-pair analyses.
    written = _write_clean_auto_gamma_outputs(
        run_dir=root,
        cfg=cfg,
        counts_df=counts_df,
        spikes=spikes,
        edge_dfs=edge_dfs,
        selected_nmi_df=synthetic_nmi,
        selected_threshold_summary_df=synthetic_summary,
        selected_threshold=synthetic_threshold,
        consensus_critical_indices=consensus_indices,
        state_louvain_gamma=state_gamma,
        gamma_selection_diag=selection_diag,
        start_time=start_time,
        total_duration=total_duration,
        T=T,
        selected_su_ids=selected_su_ids,
    )

    # Save the complete per-gamma structural summary.
    structure_df.to_csv(
        auto_root / "gamma_structure_selection.csv",
        index=False,
    )

    # Save the long-form collection of every critical point detected throughout
    # the gamma sweep.
    critical_df.to_csv(
        auto_root / "critical_point_persistence_map.csv",
        index=False,
    )

    # Save the merged consensus critical-point band diagnostics.
    band_df.to_csv(
        auto_root / "consensus_critical_point_bands.csv",
        index=False,
    )

    # Save the complete representative-gamma plateau-selection table.
    selection_table.to_csv(
        auto_root / "proto_state_louvain_gamma_selection.csv",
        index=False,
    )

    # Save the final canonical consensus critical points and their corresponding
    # transition times.
    consensus_df = pd.DataFrame([
        {
            "boundary_index": cp,
            "transition_time_s": start_time + (cp + 1) * T,
        }
        for cp in consensus_indices
    ])
    consensus_df.to_csv(
        auto_root / "consensus_critical_points.csv",
        index=False,
    )

    # Generate the critical-point persistence map and mark the representative
    # state-level Louvain gamma.
    _plot_cp_map(
        auto_root / "critical_point_persistence_map.png",
        critical_df,
        state_gamma,
    )

    # Assemble a compact machine-readable summary of the automatic gamma sweep,
    # final consensus segmentation, and principal canonical output paths.
    diagnostics = {
        "boundary_source": "all_gamma_consensus_critical_points",
        "boundary_gamma": None,
        "state_louvain_gamma": state_gamma,
        "state_louvain_gamma_policy": "max_cp_plateau_largest_gamma",
        "state_louvain_gamma_role": (
            "used only for Louvain clustering inside final consensus State_Windows"
        ),
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
        "gamma_step": gamma_step,
        "n_ok_gammas": len(ok_gammas),
        "n_consensus_critical_points": len(consensus_indices),
        "consensus_critical_indices": consensus_indices,
        "consensus_stats": consensus_stats,
        "auto_root": str(auto_root),
        "windows_base": written["windows_base"],
    }

    # Write the authoritative automatic Micro diagnostic summary.
    _write_json(
        auto_root / "diagnostics.json",
        diagnostics,
    )

    # Return the gamma-sweep diagnostics together with the canonical state-level
    # outputs produced by the downstream Micro modules.
    return {
        "status": "OK",
        "auto_root": str(auto_root),
        "boundary_source": "all_gamma_consensus_critical_points",
        "boundary_gamma": None,
        "state_louvain_gamma": state_gamma,
        "state_louvain_gamma_policy": "max_cp_plateau_largest_gamma",
        "state_louvain_gamma_role": (
            "used only for Louvain clustering inside final consensus State_Windows"
        ),
        "consensus_critical_indices": consensus_indices,
        "consensus_stats": consensus_stats,
        "gamma_structure_selection_csv": str(
            auto_root / "gamma_structure_selection.csv"
        ),
        "critical_point_persistence_map_csv": str(
            auto_root / "critical_point_persistence_map.csv"
        ),
        "critical_point_persistence_map_png": str(
            auto_root / "critical_point_persistence_map.png"
        ),
        "consensus_critical_point_bands_csv": str(
            auto_root / "consensus_critical_point_bands.csv"
        ),
        "consensus_critical_points_csv": str(
            auto_root / "consensus_critical_points.csv"
        ),
        "proto_state_louvain_gamma_selection_csv": str(
            auto_root / "proto_state_louvain_gamma_selection.csv"
        ),
        "diagnostics_json": str(
            auto_root / "diagnostics.json"
        ),
        **written,
    }