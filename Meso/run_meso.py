"""
NeuroSeq Meso Workflow Entry Point
==================================

Module
------
run_meso

Purpose
-------
Provide the public entry point for executing the NeuroSeq Meso analysis from
previously generated Micro outputs.

Scientific Role
---------------
This module validates the required Micro outputs, prepares the configuration
payload, launches the Meso spectral workflow, and records the complete
execution summary. It serves as the interface between the Micro pipeline and
the Meso family-discovery analysis while delegating all scientific
computations to the dedicated Meso modules.

Outputs
-------
This module returns a dictionary summarizing the Meso analysis and writes a
JSON record containing the execution configuration, workflow outputs, and
references to the originating Micro results.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Meso.meso_part_1 import run_meso_part_1


def run_meso(
    *,
    micro_outputs: dict[str, Any],
    output_dir: str,
    meso_cluster_resolutions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.3, 1.8, 2.0),
    meso_label_plot_resolution: float = 1.0,
    seed: int = 42,
    data_folder: str | None = None,
    run_mode: str = "full_recording",
    delivery_label: str = "",
    delivery_index: int | None = None,
    delivery_time: float | None = None,
    do_graph: bool = True,
    meso_threshold_mode: str = "spectral_eigengap_persistence",
) -> dict[str, Any]:
    """
    Execute the NeuroSeq Meso workflow from validated Micro outputs.

    Parameters
    ----------
    micro_outputs : dict
        Dictionary produced by the Micro pipeline.
    output_dir : str
        Directory where Meso outputs will be written.
    meso_cluster_resolutions : tuple of float
        Candidate resolutions retained for compatibility with the pipeline.
    meso_label_plot_resolution : float
        Resolution used for label plotting.
    seed : int
        Random seed for reproducible Meso analysis.
    data_folder : str, optional
        Folder containing spike-time files for raster generation.
    run_mode : str
        Execution mode.
    delivery_label, delivery_index, delivery_time
        Delivery metadata propagated through the workflow.
    do_graph : bool
        Whether graphical outputs should be generated.
    meso_threshold_mode : str
        Requested Meso threshold-selection mode.

    Returns
    -------
    dict
        Complete execution summary together with the outputs produced by the
        Meso workflow.

    Notes
    -----
    This routine performs validation and orchestration only. Scientific
    computations are delegated to ``run_meso_part_1``.
    """

    # Ensure the requested output directory exists.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify that the required Micro state-vector table is available.
    state_vectors_csv = micro_outputs.get("state_vectors_csv")
    if state_vectors_csv is None or not Path(state_vectors_csv).exists():
        out = {"status": "SKIPPED", "reason": "state_vectors_csv missing", "state_vectors_csv": state_vectors_csv}
        (output_dir / "meso_outputs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    # Validate that the state-vector table can be loaded successfully.
    try:
        df_states = pd.read_csv(state_vectors_csv)
    except Exception as exc:
        out = {"status": "SKIPPED", "reason": f"Could not read state_vectors_csv: {exc}", "state_vectors_csv": state_vectors_csv}
        (output_dir / "meso_outputs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    # Assemble the configuration payload passed to the Meso workflow.
    base_dir = output_dir.parent
    payload = {
        "BASE_DIR": str(base_dir),
        "SEQUENCE_SUBFOLDER": output_dir.name,
        "DATA_FOLDER": str(data_folder) if data_folder else "",
        "RUN_MODE": str(run_mode),
        "MESO_THRESHOLD_MODE": str(meso_threshold_mode),
        "DELIVERY_LABEL": str(delivery_label),
        "DELIVERY_INDEX": delivery_index,
        "DELIVERY_TIME": delivery_time,
        "DO_GRAPH": bool(do_graph),
        "MESO_CLUSTER_RESOLUTIONS": list(meso_cluster_resolutions),
        "MESO_LABEL_PLOT_RESOLUTION": float(meso_label_plot_resolution),
        "MESO_SEED": int(seed),
    }

    # Execute the complete Meso spectral workflow.
    out = run_meso_part_1(**payload)

    # Collect the final execution summary.
    outputs = {
        "status": "OK",
        "n_states_from_state_vectors_csv": int(len(df_states)),
        "output_dir": str(output_dir),
        "base_dir": str(base_dir),
        "payload": payload,
        "meso_part1_outputs": out,
        "micro_outputs": micro_outputs,
    }

    # Save a complete JSON record of the workflow execution.
    (output_dir / "meso_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")

    return outputs