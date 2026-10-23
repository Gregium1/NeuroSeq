"""Run the Meso stage of NeuroSeq.

Proto-states are constructed from consensus Micro critical points.
Firing-rate vectors and neuronal partitions are calculated within Meso.
Metastable States are then identified by spectral clustering.

Author
------
Greg Owanga
Gregium Theory
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .meso_proto_windows import run_state_windows_louvain
from .meso_proto_vectors import run_state_vectors_and_matrices
from .meso_workflow import run_meso_workflow


def run_meso(*, micro_outputs: dict[str, Any], output_dir: str,
             data_folder: str, seed: int = 42, seed_for_states: int = 42,
             do_graph: bool = True) -> dict[str, Any]:
    """Construct Proto-states and identify their Metastable States.

    Parameters
    ----------
    micro_outputs : dict
        The validated Micro context is supplied.
    output_dir : str
        The Meso output directory is specified.
    data_folder : str
        The SU spike-time directory is specified.
    seed : int
        The spectral random seed is specified.
    seed_for_states : int
        The representative Louvain seed is specified.
    do_graph : bool
        Figure generation is enabled when True.

    Returns
    -------
    dict
        Output paths and spectral results are returned.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    micro_epoch_dir = Path(micro_outputs["windows_base"])
    # The consensus NMI table is copied to Meso for Proto-state construction.
    import shutil
    shutil.copy2(micro_outputs["nmi_csv_path"], out / "nmi_adjacent_micro-epoch.csv")
    common = dict(base_dir=str(out), data_dir=data_folder,
                  nmi_threshold=float(micro_outputs["chosen_nmi_threshold"]),
                  T=float(micro_outputs["T"]),
                  use_auto_find_counts_csv=False, counts_csv_search_root="",
                  manual_start_time=float(micro_outputs["start_time"]),
                  manual_end_time=float(micro_outputs["start_time"] + micro_outputs["total_duration"]))
    run_state_windows_louvain(**common,
                              louvain_resolution=float(micro_outputs["state_louvain_gamma"]),
                              seed=int(seed_for_states))
    run_state_vectors_and_matrices(**{k:v for k,v in common.items() if k not in ("louvain_resolution","seed")})
    # Pairwise NMI is computed once in the Meso workflow and reused for all outputs.
    result = run_meso_workflow(BASE_DIR=str(out.parent),
                               SEQUENCE_SUBFOLDER=out.name,
                               DATA_FOLDER=data_folder,
                               MESO_SEED=int(seed), DO_GRAPH=bool(do_graph))
    (out / "meso_outputs.json").write_text(json.dumps(result, indent=2, default=str),encoding="utf-8")
    return result
