#####  Meso/meso_collect.py    

"""
Canonical Micro-State Collection for Meso Analysis
=================================================

Module
------
Meso.meso_collect

Purpose
-------
Collect the canonical Micro-state representations required by the NeuroSeq
Meso pipeline.

Scientific Role
---------------
This module provides the interface between the Micro and Meso analyses.
It loads the canonical state vectors and their corresponding Louvain
community assignments produced during the Micro analysis, validates that
they are consistent, and assembles them into standardized in-memory
representations used throughout the Meso workflow.

Each collected Proto-state contains:

* Neuronal firing-rate vector.
* Ordered neuron identifiers.
* Louvain community labels.
* Temporal information (start, end, duration).
* Provenance information identifying the recording and source files.

Outputs
-------
Returns two dictionaries:

1. state_vectors
    Canonical firing-rate vectors and temporal metadata.

2. state_clusters
    Canonical neuron-to-community assignments.

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


def _normalize_state(value: object) -> str:
    """
    Normalize state identifiers into a common representation.

    Parameters
    ----------
    value : object
        Original state label.

    Returns
    -------
    str
        Canonical state identifier.
    """
    return str(value).replace("State_", "").replace("State ", "").strip()


def _file_suffix(path: Path, prefix: str) -> str:
    """
    Extract the suffix following a canonical filename prefix.

    Parameters
    ----------
    path : Path
        Input filename.
    prefix : str
        Expected filename prefix.

    Returns
    -------
    str
        Normalized suffix.
    """
    stem = path.stem
    return stem[len(prefix):].lower() if stem.lower().startswith(prefix.lower()) else ""


def _match_cluster_file(vector_path: Path, cluster_paths: list[Path]) -> Path:
    """
    Match a state-vector table with its corresponding Louvain cluster table.

    Parameters
    ----------
    vector_path : Path
        State-vector CSV.
    cluster_paths : list[Path]
        Candidate Louvain CSV files.

    Returns
    -------
    Path
        Matching Louvain cluster table.

    Raises
    ------
    RuntimeError
        If no compatible cluster table can be identified.
    """

    # Match files using their canonical filename suffixes.
    suffix = _file_suffix(vector_path, "state_vectors")
    exact = [p for p in cluster_paths
             if _file_suffix(p, "louvain_clusters_all_states") == suffix]
    if exact:
        return exact[0]

    # Accept the only available cluster table.
    if len(cluster_paths) == 1:
        return cluster_paths[0]

    # Prefer the canonical filename when several tables exist.
    plain = [p for p in cluster_paths
             if p.stem.lower() == "louvain_clusters_all_states"]
    if plain:
        return plain[0]

    raise RuntimeError(f"No Louvain cluster file matches {vector_path.name}")


def collect_proto_states(
    base_dir: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Collect canonical Micro-state vectors and Louvain assignments.

    Parameters
    ----------
    base_dir : str or Path
        NeuroSeq recording directory.

    Returns
    -------
    tuple
        Two dictionaries containing state vectors and Louvain assignments.

    Raises
    ------
    RuntimeError
        If the canonical Micro outputs cannot be located.
    ValueError
        If required columns are missing.

    Notes
    -----
    Only the canonical

        Meso/Proto-States

    directory is used. Alternative folders are intentionally ignored to
    guarantee reproducible downstream Meso analyses.
    """

    base_path = Path(base_dir)

    # Use only the canonical Micro output directory.
    state_dir = base_path / "Meso" / "Proto-States"

    if not state_dir.exists():
        found = sorted(base_path.rglob("Proto-States"))
        message = (
            f"No canonical Proto-States folder found.\nExpected:\n  {state_dir}"
        )
        if found:
            message += (
                "\n\nOther folders were found but intentionally not used:\n"
                + "\n".join(f"  {p}" for p in found)
            )
        raise RuntimeError(message)

    # Locate the canonical state-vector and Louvain outputs.
    vector_paths = sorted(state_dir.glob("state_vectors*.csv"))
    cluster_paths = sorted(state_dir.glob("louvain_clusters_all_states*.csv"))

    if not vector_paths:
        raise RuntimeError(f"No state_vectors*.csv found in {state_dir}")
    if not cluster_paths:
        raise RuntimeError(f"No louvain_clusters_all_states*.csv found in {state_dir}")

    state_vectors: dict[str, dict[str, Any]] = {}
    state_clusters: dict[str, dict[str, Any]] = {}

    # Process every canonical state-vector table.
    for vector_path in vector_paths:

        vectors = pd.read_csv(vector_path)

        if "State" not in vectors.columns:
            raise ValueError(f"{vector_path} is missing State")

        # Match the corresponding Louvain assignments.
        cluster_path = _match_cluster_file(vector_path, cluster_paths)

        clusters = pd.read_csv(cluster_path)

        required = {"Neuron_ID", "Cluster", "State"}
        missing = required - set(clusters.columns)

        if missing:
            raise ValueError(f"{cluster_path} is missing: {sorted(missing)}")

        # Identify the neuronal firing-rate dimensions.
        neuron_cols = [
            c for c in vectors.columns
            if c not in {"State", "Start", "End", "Duration"}
        ]

        if not neuron_cols:
            raise ValueError(f"No neuron columns found in {vector_path}")

        # Standardize neuron identifiers and state labels.
        clusters = clusters.copy()
        clusters["Neuron_ID"] = clusters["Neuron_ID"].astype(str).str.strip()
        clusters["_state"] = clusters["State"].map(_normalize_state)

        # Preserve temporal ordering.
        if "Start" in vectors.columns:
            vectors = vectors.copy()
            vectors["Start"] = pd.to_numeric(vectors["Start"], errors="coerce")
            vectors = vectors.sort_values(
                ["Start", "State"],
                kind="mergesort"
            ).reset_index(drop=True)

        # Assemble one canonical representation per Proto-state.
        for _, row in vectors.iterrows():

            if pd.isna(row.get("State")):
                continue

            readable = str(row["State"])
            normalized = _normalize_state(readable)

            matching = clusters.loc[
                clusters["_state"] == normalized,
                ["Neuron_ID", "Cluster"],
            ]

            if matching.empty:
                raise RuntimeError(
                    f"No Louvain labels found for {readable} in {cluster_path.name}"
                )

            # Maintain deterministic neuron ordering.
            matching = matching.drop_duplicates().sort_values(
                "Neuron_ID",
                kind="mergesort"
            )

            start = float(row["Start"]) if "Start" in vectors.columns and pd.notna(row["Start"]) else 0.0
            end = float(row["End"]) if "End" in vectors.columns and pd.notna(row["End"]) else float("nan")
            duration = float(row["Duration"]) if "Duration" in vectors.columns and pd.notna(row["Duration"]) else float("nan")

            # Recover missing temporal quantities whenever possible.
            if not np.isfinite(duration) and np.isfinite(start) and np.isfinite(end):
                duration = end - start

            if not np.isfinite(end) and np.isfinite(start) and np.isfinite(duration):
                end = start + duration

            # Build a unique identifier for this Proto-state.
            uid = f"Meso/Proto-States|{vector_path.name}|{normalized}"

            state_vectors[uid] = {
                "vector": row[neuron_cols].to_numpy(dtype=float),
                "neurons": [str(c) for c in neuron_cols],
                "file": vector_path.name,
                "state": readable,
                "start": start,
                "end": end,
                "duration": duration,
                "delivery": "NeuroSeq",
            }

            state_clusters[uid] = {
                "neurons": matching["Neuron_ID"].tolist(),
                "clusters": matching["Cluster"].tolist(),
            }

    # Ensure that at least one canonical Proto-state was collected.
    if not state_vectors:
        raise RuntimeError(f"No states collected from {state_dir}")

    return state_vectors, state_clusters