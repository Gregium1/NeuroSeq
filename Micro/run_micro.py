# Micro/run_micro.py

"""
Public interface for the NeuroSeq Micro analysis pipeline.

This module provides the main user-facing function for running the complete
Micro analysis. It collects the analysis parameters, constructs a compact
configuration object, and delegates execution to the automatic gamma-sweep
orchestrator.

The Micro pipeline identifies candidate neural-state boundaries from changes
in temporal community structure across a range of Louvain resolutions. Final
temporal boundaries are obtained from critical points detected across the full
gamma sweep, while the selected representative gamma is used only for Louvain
clustering within the resulting state windows.

This module intentionally contains no scientific computations. Its role is to
provide a stable and convenient public entry point while keeping the internal
pipeline modular.

Main workflow
-------------
1. Validate and normalize the gamma-sweep specification.
2. Assemble the Micro configuration parameters.
3. Determine whether all neurons or a selected subset should be analyzed.
4. Call the automatic Micro gamma-selection workflow.
5. Return the complete set of generated paths and analysis metadata.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .auto_select_micro_gamma import auto_select_micro_gamma


def run_micro(
    *,
    data_folder: str,
    output_folder: str,
    start_time: float,
    total_duration: float,
    window_size: float,
    use_all_su_ids: bool = True,
    selected_su_ids: list[str] | None = None,
    T: float = 0.5,
    lambda_threshold: float = 0.0,
    gamma_min: float = 1.05,
    gamma_max: float = 1.20,
    gamma_step: float = 0.001,
    gamma_steps: int | None = None,
    seeds_per_gamma: int = 3,
    nmi_threshold_method: str = "gmm2",
    louvain_resolution_for_states: float = 1.05,
    seed_for_states: int = 42,
    consensus_cp_band_tolerance_bins: int | None = 0,
    **extra: Any,
) -> dict[str, Any]:
    """
    Run the complete NeuroSeq Micro workflow.

    This function is the public entry point for Micro analysis. It prepares the
    configuration required by the internal gamma-sweep orchestrator and launches
    the full analysis for the requested recording interval.

    The temporal segmentation is inferred from critical points detected across
    the complete Louvain-resolution sweep. The representative gamma selected by
    the pipeline is used only for community detection within the final state
    windows and does not determine the temporal boundaries themselves.

    Parameters
    ----------
    data_folder : str
        Directory containing the single-unit spike-time files. Files are
        expected to follow the naming convention ``SU_<index>.csv``.

    output_folder : str
        Root directory in which all Micro outputs will be written.

    start_time : float
        Beginning of the analyzed recording interval, in seconds.

    total_duration : float
        Duration of the analyzed recording interval, in seconds.

    window_size : float
        Width of each temporal spike-count window, in seconds.

    use_all_su_ids : bool, default=True
        If ``True``, include every available ``SU_*.csv`` file. If ``False``,
        analyze only the neurons listed in ``selected_su_ids``.

    selected_su_ids : list[str] or None, default=None
        Optional list of single-unit identifiers to include when
        ``use_all_su_ids`` is ``False``.

    T : float, default=0.5
        Scaling parameter used when converting pairwise spike-count differences
        into graph-edge weights.

    lambda_threshold : float, default=0.0
        Minimum graph-edge weight retained during temporal graph construction.

    gamma_min : float, default=1.05
        Minimum Louvain resolution evaluated during the Micro gamma sweep.

    gamma_max : float, default=1.20
        Maximum Louvain resolution evaluated during the Micro gamma sweep.

    gamma_step : float, default=0.001
        Increment between consecutive Louvain resolutions.

    gamma_steps : int or None, default=None
        Optional number of evenly spaced gamma values between ``gamma_min`` and
        ``gamma_max``. When provided and greater than one, ``gamma_step`` is
        recalculated from this value.

    seeds_per_gamma : int, default=3
        Number of Louvain random seeds evaluated for each gamma and temporal
        window. The partition with the highest modularity is retained.

    nmi_threshold_method : str, default="gmm2"
        Method used to determine the adjacent-window NMI threshold. The active
        NeuroSeq Micro implementation uses a two-component Gaussian mixture
        model.

    louvain_resolution_for_states : float, default=1.05
        Compatibility configuration value for state-level Louvain clustering.
        The automatic workflow ultimately selects the representative state
        Louvain gamma using the maximum-critical-point plateau rule.

    seed_for_states : int, default=42
        Random seed used for Louvain clustering within the final state windows.

    consensus_cp_band_tolerance_bins : int or None, default=0
        Maximum boundary-index separation allowed when merging nearby critical
        points detected across gamma values into one consensus band.

    **extra : Any
        Additional configuration values passed to the internal Micro workflow.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the analysis status, selected representative
        gamma, consensus critical points, diagnostic-file paths, and canonical
        Micro output paths.

    Notes
    -----
    This function performs no direct spike-train analysis. All scientific
    computations are delegated to :func:`auto_select_micro_gamma`.
    """
    # When the user specifies the number of gamma values instead of the
    # increment, calculate the corresponding evenly spaced gamma step.
    if gamma_steps is not None and int(gamma_steps) > 1:
        gamma_step = (float(gamma_max) - float(gamma_min)) / (int(gamma_steps) - 1)

    # Assemble the configuration object consumed by the internal Micro modules.
    cfg = SimpleNamespace(
        data_folder=str(data_folder),
        window_size=float(window_size),
        T=float(T),
        lambda_threshold=float(lambda_threshold),
        seeds_per_gamma=int(seeds_per_gamma),
        louvain_resolution_for_states=float(louvain_resolution_for_states),
        seed_for_states=int(seed_for_states),
        **extra,
    )

    # Launch the complete automatic gamma sweep and return its canonical outputs.
    return auto_select_micro_gamma(
        cfg=cfg,
        root=Path(output_folder),
        start_time=float(start_time),
        total_duration=float(total_duration),
        selected_su_ids=None if use_all_su_ids else selected_su_ids,
        nmi_threshold_method=str(nmi_threshold_method),
        gamma_min=float(gamma_min),
        gamma_max=float(gamma_max),
        gamma_step=float(gamma_step),
        consensus_cp_band_tolerance_bins=consensus_cp_band_tolerance_bins,
    )