"""Coordinate one full-population NeuroSeq analysis.

Micro identifies consensus critical points and the representative gamma.
Meso uses those outputs to construct Proto-states and identify Metastable
States. All SU_*.csv files in the input directory are analyzed; neuron
subsampling and event/stimulus processing are intentionally absent.

The pipeline is invoked once per call. Its output_root receives a ``NeuroSeq``
subdirectory containing stage outputs and reproducibility summaries.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from Micro.auto_select_micro_gamma import auto_select_micro_gamma
from Meso.run_meso import run_meso

@dataclass(frozen=True)
class RunConfig:
    """Configuration shared by the command-line launcher and GUI.

    Gamma settings are intentionally configured in Python, not exposed as
    ordinary GUI controls. ``T`` is the Micro-Epoch duration in seconds.
    """
    data_folder: str
    output_root: str
    start_time: float = 0.0
    total_duration: float = 60.0
    T: float = 0.5
    auto_gamma_min: float = 1.07
    auto_gamma_max: float = 1.15
    auto_gamma_step: float = 0.001
    seeds_per_gamma: int = 3
    seed_for_states: int = 42
    persistence_tolerance_bins: int = 1
    consensus_cp_band_tolerance_bins: int = 0
    meso_seed: int = 42
    do_graph: bool = True


def run_pipeline(cfg: RunConfig) -> list[dict]:
    """Run Micro then Meso exactly once on the full recorded population.

    A single context JSON connects the stages, preventing independent
    reconstruction of Micro segmentation in the GUI. Scientific routines
    are delegated to the existing Micro and Meso packages unchanged.
    """
    data = Path(cfg.data_folder)
    neurons = sorted((p.stem for p in data.glob("SU_*.csv")),
                     key=lambda x: int(x.split("_")[-1]))
    if not neurons:
        raise FileNotFoundError(f"No SU_*.csv files were found in {data}")
    if cfg.T <= 0 or cfg.total_duration <= 0:
        raise ValueError("T and total_duration must be positive")
    output = Path(cfg.output_root) / "NeuroSeq"
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output
    start = time.perf_counter()
    micro = auto_select_micro_gamma(
        cfg={"micro": {"data_folder": str(data), "T": cfg.T,
                       "seeds_per_gamma": cfg.seeds_per_gamma}},
        root=run_dir, start_time=cfg.start_time,
        total_duration=cfg.total_duration, selected_su_ids=None,
        nmi_threshold_method="gmm2", gamma_min=cfg.auto_gamma_min,
        gamma_max=cfg.auto_gamma_max, gamma_step=cfg.auto_gamma_step,
        persistence_tolerance_bins=cfg.persistence_tolerance_bins,
        consensus_cp_band_tolerance_bins=cfg.consensus_cp_band_tolerance_bins)
    ctx_path = run_dir / "Micro" / "micro_ctx.json"
    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    meso = run_meso(micro_outputs=ctx, output_dir=str(run_dir / "Meso"),
                    data_folder=str(data), seed=cfg.meso_seed,
                    seed_for_states=cfg.seed_for_states, do_graph=cfg.do_graph)
    runtime = time.perf_counter() - start
    summary = {"runtime_sec": runtime, "n_neurons": len(neurons),
               "T": cfg.T, "start_time": cfg.start_time,
               "total_duration": cfg.total_duration,
               "representative_gamma": ctx["state_louvain_gamma"],
               "micro_outputs": micro, "meso_outputs": meso}
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame([{"method": "NeuroSeq", "runtime_sec": runtime,
                   "n_neurons": summary["n_neurons"],
                   "duration_sec": cfg.total_duration}]).to_csv(run_dir / "runtime.csv", index=False)
    return [summary]
