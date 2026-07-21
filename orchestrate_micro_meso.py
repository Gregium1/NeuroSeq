"""
NeuroSeq Micro-to-Meso Workflow Orchestrator
============================================

Module
------
orchestrate_micro_meso

Purpose
-------
Coordinate complete NeuroSeq analyses for either full recordings or
event-centered delivery intervals.

Scientific Role
---------------
This module links the Micro and Meso stages of NeuroSeq into a single
reproducible workflow. For each requested interval, it executes the automatic
Micro gamma sweep, reads the canonical Micro outputs, launches spectral Meso
analysis, records end-to-end runtime, and saves interval-level and aggregate
metadata.

Micro temporal boundaries are determined from consensus critical points
detected across the complete gamma sweep. The selected state-window Louvain
gamma is used only to organize neurons within the final state windows and does
not determine temporal segmentation. Meso consumes only the canonical outputs
referenced by ``Micro/micro_ctx.json``, preventing stale files from entering
the analysis.

Outputs
-------
This module writes interval-specific Micro and Meso outputs, runtime tables,
run summaries, aggregate cross-interval summaries, and failure records. It
also returns the complete list of successful interval summaries.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

import json
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Micro.auto_select_micro_gamma import auto_select_micro_gamma
from Meso.run_meso import run_meso


# ============================================================
# Runtime and file-system utilities
# ============================================================

def _set_thread_env(n: int = 1) -> None:
    """Limit numerical-library thread counts for the current process."""
    n = int(max(1, n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def _ensure_dir(path: str | Path) -> None:
    """Create a directory and any missing parent directories."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Write a dictionary as indented UTF-8 JSON."""
    path = Path(path)
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read and return a UTF-8 JSON object."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _su_sort_key(name: str) -> int:
    """Return the numeric index embedded in a ``SU_<index>`` identifier."""
    try:
        return int(str(name).split("_")[1])
    except Exception:
        return 10**12


def list_available_su_ids(data_folder: str | Path) -> list[str]:
    """List available ``SU_*.csv`` identifiers in numerical order."""
    data_folder = Path(data_folder)
    su_ids = [p.stem for p in data_folder.glob("SU_*.csv")]
    return sorted(su_ids, key=_su_sort_key)


def choose_subset(all_sus: list[str], k: int, seed: int) -> list[str]:
    """Select a reproducible, numerically sorted subset of single units."""
    rng = random.Random(int(seed))
    if int(k) >= len(all_sus):
        return all_sus[:]
    return sorted(rng.sample(all_sus, k=int(k)), key=_su_sort_key)


def _safe_label(x: str) -> str:
    """Convert an interval label into a filesystem-safe name."""
    s = str(x).strip()
    if not s:
        return "Interval"
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "Interval"


def _load_event_times(cts_file: str | Path) -> np.ndarray:
    """
    Load finite, unique event times from a CTS or generic numeric CSV file.

    Non-numeric entries are ignored and the returned times are sorted.
    """
    path = Path(cts_file)

    if not path.exists():
        raise FileNotFoundError(f"CTS/event file not found: {path}")

    raw = pd.read_csv(path, header=None)

    vals = []
    for col in raw.columns:
        arr = pd.to_numeric(raw[col], errors="coerce").dropna().to_numpy(float)
        if arr.size:
            vals.extend(arr.tolist())

    times = np.asarray(vals, dtype=float)
    times = times[np.isfinite(times)]
    times = np.unique(np.round(times, 9))
    times.sort()

    if times.size == 0:
        raise ValueError(f"No valid numeric event times found in: {path}")

    return times


# ============================================================
# Pipeline configuration
# ============================================================

@dataclass(frozen=True)
class RunConfig:
    """
    Immutable configuration for one NeuroSeq pipeline invocation.

    Fields define the input/output locations, run mode, temporal interval,
    automatic Micro gamma sweep, optional neuron sampling, Meso settings, and
    runtime metadata.
    """

    data_folder: str
    output_root: str

    # "full_recording" or "delivery"
    run_mode: str = "full_recording"

    # Full-recording interval.
    start_time: float = 0.0
    total_duration: float = 60.0

    # Delivery mode.
    cts_file: str | None = None
    pre_time: float = 1.0
    post_time: float = 3.0
    delivery_prefix: str = "Delivery"

    # Micro parameters.
    window_size: float = 0.5
    T: float = 0.5
    lambda_threshold: float = 0.0

    auto_gamma_min: float = 1.07
    auto_gamma_max: float = 1.20
    auto_gamma_step: float = 0.001

    seeds_per_gamma: int = 3
    seed_for_states: int = 42

    persistence_tolerance_bins: int = 1
    consensus_cp_band_tolerance_bins: int = 0

    # Optional neuron sampling.
    use_sampling: bool = False
    sample_k: int = 15
    repeats: int = 1
    base_seed: int = 12345

    # Meso parameters.
    meso_seed: int = 42
    do_graph: bool = True

    # Runtime / metadata.
    threads_per_process: int = 1
    true_n_states: int | None = None


# ============================================================
# Canonical Micro output interface
# ============================================================

def _micro_ctx_path(run_dir: Path) -> Path:
    """Return the canonical Micro context path for one run."""
    return run_dir / "Micro" / "micro_ctx.json"


def _micro_outputs_from_ctx(run_dir: Path) -> dict[str, Any]:
    """
    Build and validate the canonical Micro-to-Meso output interface.

    ``micro_ctx.json`` is treated as the authoritative record of the current
    Micro run, preventing Meso from discovering stale outputs recursively.
    """
    ctx_path = _micro_ctx_path(run_dir)
    if not ctx_path.exists():
        raise FileNotFoundError(f"Missing Micro context: {ctx_path}")

    ctx = _read_json(ctx_path)

    windows_base = Path(ctx["windows_base"])
    state_dir = windows_base / "State_Windows"

    outputs = {
        "counts_csv": ctx.get("counts_csv_path"),
        "raster_png": ctx.get("raster_png"),
        "windows_base": str(windows_base),
        "adjacent_nmi_csv": ctx.get("nmi_csv_path"),
        "nmi_csv": ctx.get("nmi_csv_path"),
        "nmi_threshold_method_used": ctx.get("nmi_threshold_method", "gmm2"),
        "micro_nmi_threshold": float(ctx.get("chosen_nmi_threshold", np.nan)),
        "gmm2_threshold_summary_csv": ctx.get("gmm2_threshold_summary_csv"),
        "boundary_source": ctx.get("boundary_source"),
        "boundary_gamma": ctx.get("boundary_gamma"),
        "state_louvain_gamma": ctx.get("state_louvain_gamma"),
        "state_louvain_gamma_role": ctx.get("state_louvain_gamma_role"),
        "state_windows_dir": str(state_dir),
        "state_vectors_csv": str(state_dir / "state_vectors.csv"),
        "mean_firing_rate_distance_csv": str(state_dir / "mean_firing_rate_distance.csv"),
        "cosine_similarity_csv": str(state_dir / "cosine_similarity.csv"),
        "state_clusters_csv": str(state_dir / "louvain_clusters_all_states.csv"),
        "state_louvain_nmi_csv": str(state_dir / "state_louvain_nmi.csv"),
        "state_louvain_nmi_long_csv": str(state_dir / "state_louvain_nmi_long.csv"),
        "micro_ctx_json": str(ctx_path),
    }

    required = [
        "state_vectors_csv",
        "state_clusters_csv",
        "state_louvain_nmi_csv",
        "mean_firing_rate_distance_csv",
    ]

    missing = [key for key in required if not Path(outputs[key]).exists()]
    if missing:
        missing_paths = {key: outputs[key] for key in missing}
        raise FileNotFoundError(f"Missing expected Micro outputs: {missing_paths}")

    return outputs


# ============================================================
# Single-interval execution
# ============================================================

def run_interval_once(
    *,
    cfg: RunConfig,
    selected_su_ids: list[str] | None,
    run_dir: Path,
    run_name: str,
    interval_label: str,
    interval_start: float,
    interval_end: float,
    run_mode: str,
    delivery_index: int | None = None,
    delivery_time: float | None = None,
) -> dict[str, Any]:
    """
    Run Micro and Meso for one temporal interval.

    Returns a complete summary containing runtime, interval metadata, and the
    output dictionaries produced by both stages.
    """

    _ensure_dir(run_dir)

    total_duration = float(interval_end) - float(interval_start)
    if total_duration <= 0:
        raise ValueError(
            f"Invalid interval for {interval_label}: "
            f"start={interval_start}, end={interval_end}"
        )

    # Benchmark the complete Micro + Meso workflow.
    runtime_start = time.perf_counter()

    print("=" * 80)
    print(f"NEUROSEQ | mode={run_mode} | label={interval_label} | run={run_name}")
    print("=" * 80)
    print("Running clean Auto-Micro gamma sweep...")
    print("Boundary source: all-gamma consensus critical points")
    print("State-window Louvain gamma: max critical-point-count plateau")
    print(f"Interval: {float(interval_start):.6f} to {float(interval_end):.6f} s")
    print("Output folder:", run_dir)

    # Consensus critical points determine boundaries. The selected state
    # Louvain gamma is used only to cluster neurons within those states.
    micro_summary = auto_select_micro_gamma(
        cfg={
            "micro": {
                "data_folder": str(cfg.data_folder),
                "window_size": float(cfg.window_size),
                "T": float(cfg.T),
                "lambda_threshold": float(cfg.lambda_threshold),
                "seeds_per_gamma": int(cfg.seeds_per_gamma),
                "seed_for_states": int(cfg.seed_for_states),
            }
        },
        root=run_dir,
        start_time=float(interval_start),
        total_duration=float(total_duration),
        selected_su_ids=selected_su_ids,
        nmi_threshold_method="gmm2",
        gamma_min=float(cfg.auto_gamma_min),
        gamma_max=float(cfg.auto_gamma_max),
        gamma_step=float(cfg.auto_gamma_step),
        persistence_tolerance_bins=int(cfg.persistence_tolerance_bins),
        consensus_cp_band_tolerance_bins=int(cfg.consensus_cp_band_tolerance_bins),
    )

    _write_json(run_dir / "auto_micro_summary.json", micro_summary)

    # Read the canonical Micro interface rather than searching output trees.
    micro_outputs = _micro_outputs_from_ctx(run_dir)

    print("=" * 80)
    print("Running Meso...")
    print("=" * 80)

    meso_outputs = run_meso(
        micro_outputs=micro_outputs,
        output_dir=str(run_dir / "Meso"),
        seed=int(cfg.meso_seed),
        data_folder=str(cfg.data_folder),
        run_mode=str(run_mode),
        delivery_label=str(interval_label),
        delivery_index=delivery_index,
        delivery_time=delivery_time,
        do_graph=bool(cfg.do_graph),
    )

    runtime_sec = time.perf_counter() - runtime_start

    n_neurons_used = (
        len(selected_su_ids)
        if selected_su_ids is not None
        else len(list_available_su_ids(cfg.data_folder))
    )

    print(f"\nNeuroSeq runtime (Micro + Meso): {runtime_sec:.3f} seconds")

    runtime_row = {
        "method": "NeuroSeq",
        "runtime_sec": float(runtime_sec),
        "n_neurons": int(n_neurons_used),
        "n_true_states": cfg.true_n_states,
        "run_mode": str(run_mode),
        "run_name": str(run_name),
        "interval_label": str(interval_label),
        "interval_start": float(interval_start),
        "interval_end": float(interval_end),
        "interval_duration": float(total_duration),
        "delivery_index": delivery_index,
        "delivery_time": delivery_time,
        "data_folder": str(cfg.data_folder),
        "output_dir": str(run_dir),
    }

    pd.DataFrame([runtime_row]).to_csv(run_dir / "runtime.csv", index=False)

    summary = {
        "run_mode": str(run_mode),
        "run_name": str(run_name),
        "run_dir": str(run_dir),
        "interval_label": str(interval_label),
        "interval_start": float(interval_start),
        "interval_end": float(interval_end),
        "interval_duration": float(total_duration),
        "delivery_index": delivery_index,
        "delivery_time": delivery_time,
        "window_size": float(cfg.window_size),
        "runtime_sec": float(runtime_sec),
        "n_neurons": int(n_neurons_used),
        "n_true_states": cfg.true_n_states,
        "micro_threshold_method": "gmm2",
        "micro_threshold_used": micro_outputs.get("micro_nmi_threshold"),
        "boundary_source": micro_outputs.get("boundary_source"),
        "boundary_gamma": micro_outputs.get("boundary_gamma"),
        "state_louvain_gamma": micro_outputs.get("state_louvain_gamma"),
        "micro_outputs": micro_outputs,
        "auto_micro_summary": micro_summary,
        "meso_outputs": meso_outputs,
    }

    _write_json(run_dir / "run_summary.json", summary)

    return summary


# ============================================================
# Run-mode helpers
# ============================================================

def _run_full_recording_mode(
    *,
    cfg: RunConfig,
    selected_su_ids: list[str] | None,
    run_name: str,
) -> list[dict[str, Any]]:
    """Analyze one full-recording interval for the requested neuron set."""

    output_root = Path(cfg.output_root)

    if run_name == "all_neurons":
        run_dir = output_root / "Full_Recording"
    else:
        run_dir = output_root / "Full_Recording" / run_name

    interval_start = float(cfg.start_time)
    interval_end = float(cfg.start_time) + float(cfg.total_duration)

    summary = run_interval_once(
        cfg=cfg,
        selected_su_ids=selected_su_ids,
        run_dir=run_dir,
        run_name=run_name,
        interval_label="Full_Recording",
        interval_start=interval_start,
        interval_end=interval_end,
        run_mode="full_recording",
        delivery_index=None,
        delivery_time=None,
    )

    return [summary]


def _run_delivery_mode(
    *,
    cfg: RunConfig,
    selected_su_ids: list[str] | None,
    run_name: str,
) -> list[dict[str, Any]]:
    """Analyze one pre/post interval around every configured delivery event."""

    if cfg.cts_file is None or str(cfg.cts_file).strip() == "":
        raise ValueError("cfg.cts_file is required when run_mode='delivery'.")

    output_root = Path(cfg.output_root)
    event_times = _load_event_times(cfg.cts_file)

    summaries: list[dict[str, Any]] = []

    for idx, event_time in enumerate(event_times, start=1):
        interval_start = float(event_time) - float(cfg.pre_time)
        interval_end = float(event_time) + float(cfg.post_time)

        if interval_end <= interval_start:
            print(f"SKIP Delivery_{idx:03d}: invalid interval")
            continue

        delivery_label = f"{_safe_label(cfg.delivery_prefix)}_{idx:03d}"

        if run_name == "all_neurons":
            run_dir = output_root / delivery_label
        else:
            run_dir = output_root / delivery_label / run_name

        summary = run_interval_once(
            cfg=cfg,
            selected_su_ids=selected_su_ids,
            run_dir=run_dir,
            run_name=run_name,
            interval_label=delivery_label,
            interval_start=interval_start,
            interval_end=interval_end,
            run_mode="delivery",
            delivery_index=int(idx),
            delivery_time=float(event_time),
        )

        summaries.append(summary)

    return summaries


# ============================================================
# Public pipeline entry point
# ============================================================

def run_pipeline(cfg: RunConfig) -> list[dict[str, Any]]:
    """
    Execute the complete NeuroSeq pipeline.

    The function dispatches full-recording or delivery analyses, optionally
    repeats the workflow for reproducible neuron subsets, and writes aggregate
    summary, runtime, and failure tables. Aggregate files are written from the
    ``finally`` block so completed results remain available after a later
    failure.
    """
    # Limit numerical-library parallelism for reproducible runtime measurements.
    _set_thread_env(cfg.threads_per_process)

    output_root = Path(cfg.output_root)
    _ensure_dir(output_root)

    data_folder = Path(cfg.data_folder)
    # Discover the complete set of available single units.
    all_sus = list_available_su_ids(data_folder)

    if not all_sus:
        raise FileNotFoundError(f"No SU_*.csv files found in {data_folder}")

    mode = str(cfg.run_mode).strip().lower()
    if mode not in {"full_recording", "delivery"}:
        raise ValueError("cfg.run_mode must be either 'full_recording' or 'delivery'.")

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    # Define reproducible neuron-sampling repetitions when requested.
    reps = list(range(int(cfg.repeats))) if cfg.use_sampling else [0]

    try:
        # Execute one complete workflow for each neuron selection.
        for rep in reps:
            if cfg.use_sampling:
                seed = int(cfg.base_seed) + int(rep)
                selected_su_ids = choose_subset(all_sus, int(cfg.sample_k), seed)
                run_name = f"sample_k{cfg.sample_k}_rep{rep:02d}_seed{seed}"
            else:
                selected_su_ids = None
                run_name = "all_neurons"

            if mode == "full_recording":
                summaries.extend(
                    _run_full_recording_mode(
                        cfg=cfg,
                        selected_su_ids=selected_su_ids,
                        run_name=run_name,
                    )
                )

            elif mode == "delivery":
                summaries.extend(
                    _run_delivery_mode(
                        cfg=cfg,
                        selected_su_ids=selected_su_ids,
                        run_name=run_name,
                    )
                )

    except Exception as exc:
        failures.append(
            {
                "stage": "run_pipeline",
                "run_mode": mode,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise

    finally:
        rows = []
        runtime_rows = []

        # Convert completed interval summaries into aggregate tables.
        for s in summaries:
            row = {
                "run_mode": s.get("run_mode", ""),
                "run_name": s.get("run_name", ""),
                "run_dir": s.get("run_dir", ""),
                "interval_label": s.get("interval_label", ""),
                "interval_start": s.get("interval_start", np.nan),
                "interval_end": s.get("interval_end", np.nan),
                "interval_duration": s.get("interval_duration", np.nan),
                "delivery_index": s.get("delivery_index", None),
                "delivery_time": s.get("delivery_time", None),
                "window_size": s.get("window_size", np.nan),
                "runtime_sec": s.get("runtime_sec", np.nan),
                "n_neurons": s.get("n_neurons", np.nan),
                "n_true_states": s.get("n_true_states", np.nan),
                "micro_threshold_method": s.get("micro_threshold_method", "gmm2"),
                "micro_threshold_used": s.get("micro_threshold_used", np.nan),
                "boundary_source": s.get("boundary_source", ""),
                "boundary_gamma": s.get("boundary_gamma", None),
                "state_louvain_gamma": s.get("state_louvain_gamma", np.nan),
            }
            rows.append(row)

            runtime_rows.append(
                {
                    "method": "NeuroSeq",
                    "runtime_sec": s.get("runtime_sec", np.nan),
                    "n_neurons": s.get("n_neurons", np.nan),
                    "n_true_states": s.get("n_true_states", np.nan),
                    "run_mode": s.get("run_mode", ""),
                    "run_name": s.get("run_name", ""),
                    "interval_label": s.get("interval_label", ""),
                    "interval_start": s.get("interval_start", np.nan),
                    "interval_end": s.get("interval_end", np.nan),
                    "delivery_index": s.get("delivery_index", None),
                    "delivery_time": s.get("delivery_time", None),
                    "output_dir": s.get("run_dir", ""),
                    "data_folder": str(cfg.data_folder),
                }
            )

        pd.DataFrame(rows).to_csv(
            output_root / "summary_micro_meso_clean.csv",
            index=False,
        )

        pd.DataFrame(runtime_rows).to_csv(
            output_root / "runtime_all_intervals.csv",
            index=False,
        )

        pd.DataFrame(failures).to_csv(
            output_root / "failures_micro_meso_clean.csv",
            index=False,
        )

        print("DONE.")
        print("Summary:", output_root / "summary_micro_meso_clean.csv")
        print("Runtime:", output_root / "runtime_all_intervals.csv")
        print("Failures:", output_root / "failures_micro_meso_clean.csv")

    return summaries


# ============================================================
# Direct-execution example
# ============================================================

def main() -> None:
    """
    Execute NeuroSeq with the local example configuration.

    The GUI should import ``RunConfig`` and ``run_pipeline`` instead of calling
    this example entry point.
    """
    # Define the local direct-execution configuration.
    cfg = RunConfig(
        data_folder=str(
            Path("C:/Users/owang/OneDrive/Bureau/Vincis_Lab_ID/Animal_ID")
        ),
        output_root=str(
            Path("C:/Users/owang/OneDrive/Bureau/Paper/Animal_ID")
        ),

        # Choose "full_recording" or "delivery".
        run_mode="full_recording",

        # Full-recording interval.
        start_time=0.0,
        total_duration=60.0,

        # Delivery mode settings. Used only when run_mode="delivery".
        cts_file="C:/Users/owang/OneDrive/Bureau/Vincis_Lab_ID/Animal_ID/NaCl_CTS.csv",
        pre_time=1.0,
        post_time=3.0,
        delivery_prefix="delivery",

        # Micro.
        window_size=0.5,
        T=0.5,
        lambda_threshold=0.0,
        auto_gamma_min=1.07,
        auto_gamma_max=1.15,
        auto_gamma_step=0.001,
        seeds_per_gamma=3,
        seed_for_states=42,
        persistence_tolerance_bins=1,
        consensus_cp_band_tolerance_bins=0,

        # Optional sampling.
        use_sampling=False,
        sample_k=15,
        repeats=1,
        base_seed=12345,

        # Meso.
        meso_seed=42,
        do_graph=True,

        # Runtime metadata.
        threads_per_process=1,
        true_n_states=30,
    )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()