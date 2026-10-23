"""GUI-to-pipeline adapter with a separate worker process.

The GUI delegates computation to the same ``run_pipeline`` function used by
the command-line launcher. Queue messages report progress and completion;
they do not independently run Micro or Meso. Numerical thread environment
variables are inherited rather than forced to one thread.

Author
------
Greg Owanga
Gregium Theory
"""
# GUI/neuroseq_backend.py
from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

# Match the standalone launcher's numerical-library environment.
# Do not force OMP/BLAS/MKL to one thread: that can make the GUI slower.
# Respect any settings explicitly supplied by the user or shell.

from orchestrate_micro_meso import RunConfig, run_pipeline

MessageCallback = Callable[[str], None]
ProgressCallback = Callable[[str, float, str], None]
FinishedCallback = Callable[[dict[str, Any]], None]
ErrorCallback = Callable[[str, str], None]
CancelledCallback = Callable[[], None]

# Configuration transferred from the GUI to the worker. Gamma settings stay
# here for reproducibility, even though they are intentionally hidden in the UI.
@dataclass
class NeuroSeqGUIConfig:
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

    def _pipeline_output_parent(self) -> Path:
        """The orchestrator itself appends /NeuroSeq exactly once."""
        path = Path(self.output_root).expanduser()
        return path.parent if path.name.lower() == "neuroseq" else path

    def to_run_config(self) -> RunConfig:
        """Convert GUI settings into the full-recording pipeline configuration."""
        return RunConfig(
            data_folder=str(Path(self.data_folder).expanduser()),
            output_root=str(self._pipeline_output_parent()),
            start_time=float(self.start_time),
            total_duration=float(self.total_duration),
            T=float(self.T),
            auto_gamma_min=float(self.auto_gamma_min),
            auto_gamma_max=float(self.auto_gamma_max),
            auto_gamma_step=float(self.auto_gamma_step),
            seeds_per_gamma=int(self.seeds_per_gamma),
            seed_for_states=int(self.seed_for_states),
            persistence_tolerance_bins=int(self.persistence_tolerance_bins),
            consensus_cp_band_tolerance_bins=int(self.consensus_cp_band_tolerance_bins),
            meso_seed=int(self.meso_seed),
            do_graph=bool(self.do_graph),
        )

class _QueueWriter(TextIO):
    def __init__(self, q): self.q=q; self._buffer=""
    def write(self,text):
        if not text: return 0
        self._buffer += str(text)
        while "\n" in self._buffer:
            line,self._buffer=self._buffer.split("\n",1)
            if line.rstrip("\r"): self.q.put(("message",line.rstrip("\r")))
        return len(text)
    def flush(self):
        if self._buffer: self.q.put(("message",self._buffer.rstrip("\r"))); self._buffer=""
    def writable(self): return True

def _family_count(obj):
    keys=("n_family_labels","n_families","selected_k","k_selected","best_k")
    if isinstance(obj,dict):
        for k in keys:
            v=obj.get(k)
            if isinstance(v,(int,float)) and float(v).is_integer(): return int(v)
        for v in obj.values():
            f=_family_count(v)
            if f is not None: return f
    elif isinstance(obj,list):
        for v in obj:
            f=_family_count(v)
            if f is not None: return f
    return None

def _build_result(cfg, summaries, elapsed):
    """Collect completion metrics without recomputing scientific quantities."""
    counts = []
    metastates = []
    run_dirs = []
    for summary in summaries:
        micro = summary.get("micro_outputs", {}) or {}
        meso = summary.get("meso_outputs", {}) or {}
        ctx = micro.get("ctx", {}) or {}
        cp = ctx.get("selected_consensus_critical_indices")
        if cp is None:
            cp = micro.get("consensus_critical_indices")
        if isinstance(cp, list):
            # The critical-point list may contain an initial boundary index.
            # Prefer the authoritative number of segments when available.
            stats = micro.get("consensus_stats", {}) or {}
            n_segments = stats.get("n_segments")
            counts.append(int(n_segments) if n_segments is not None else len(cp) + 1)
        k = _family_count(meso)
        if k is not None:
            metastates.append(k)
        root = meso.get("root_sequence_folder")
        if root:
            run_dirs.append(str(Path(root).parent))
    return {
        "status": "OK", "elapsed_sec": float(elapsed),
        "n_intervals": len(summaries),
        "n_micro_states_total": sum(counts),
        "n_families_by_interval": metastates,
        "output_root": str(cfg._pipeline_output_parent() / "NeuroSeq"),
        "run_dirs": run_dirs,
        # The complete scientific summary is already written to run_summary.json.
        # Do not serialize large nested matrices/diagnostics through the GUI queue.
        "summary_files": [str(Path(d) / "run_summary.json") for d in run_dirs],
    }

def _pipeline_process_main(config_dict, q):
    """Run exactly one orchestrator invocation in a separate process.

    The GUI receives stage notifications and compact results only. Scientific
    computations are delegated exclusively to the existing orchestrator.
    """
    started = time.perf_counter()
    writer = _QueueWriter(q)
    try:
        cfg = NeuroSeqGUIConfig(**config_dict)
        run_cfg = cfg.to_run_config()
        q.put(("started", os.getpid()))
        q.put(("message", "Numerical thread settings (inherited from environment): " +
               ", ".join(f"{key}={os.environ.get(key, 'unset')}" for key in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))))
        q.put(("message", f"Input: {run_cfg.data_folder}"))
        q.put(("message", f"Output: {run_cfg.output_root}"))
        q.put(("message", f"T={run_cfg.T}; gamma={run_cfg.auto_gamma_min}.."
                          f"{run_cfg.auto_gamma_max} step {run_cfg.auto_gamma_step}; "
                          f"seeds/gamma={run_cfg.seeds_per_gamma}"))

        # Wrap stage entry points solely for progress reporting and timings.
        # Each original function is invoked exactly once per orchestrator call.
        import orchestrate_micro_meso as pipeline
        original_micro = pipeline.auto_select_micro_gamma
        original_meso = pipeline.run_meso

        def tracked_micro(*args, **kwargs):
            """Observe existing Micro calls without recomputing anything."""
            import Micro.auto_select_micro_gamma as micro_module
            # Python's `import package.module as name` can resolve an exported
            # function instead of the module; importlib avoids that ambiguity.
            import importlib
            micro_module = importlib.import_module("Micro.auto_select_micro_gamma")
            originals = {
                name: getattr(micro_module, name)
                for name in ("_build_spike_counts_df", "_build_window_graphs_from_counts",
                             "_run_single_gamma_louvain_in_memory")
            }
            completed = 0
            gamma_min = float(run_cfg.auto_gamma_min)
            gamma_max = float(run_cfg.auto_gamma_max)
            gamma_step = float(run_cfg.auto_gamma_step)
            gamma_total = max(1, int(round((gamma_max - gamma_min) / gamma_step)) + 1)

            def tracked_counts(*a, **kw):
                q.put(("progress", ("prepare", .08, "Preparing spike counts")))
                result = originals["_build_spike_counts_df"](*a, **kw)
                q.put(("message", "Spike counts prepared once for all gamma values."))
                return result

            def tracked_graphs(*a, **kw):
                q.put(("progress", ("networks", .18, "Building Micro-Epoch networks")))
                result = originals["_build_window_graphs_from_counts"](*a, **kw)
                q.put(("message", "Micro-Epoch graphs built once for all gamma values."))
                return result

            def tracked_gamma(*a, **kw):
                nonlocal completed
                gamma = kw.get("gamma", None)
                q.put(("progress", ("gamma", .18 + .39 * completed / gamma_total,
                                     f"Louvain gamma sweep: {completed}/{gamma_total} completed")))
                result = originals["_run_single_gamma_louvain_in_memory"](*a, **kw)
                completed += 1
                # Keep messages sparse; GUI progress still advances each gamma.
                if completed == gamma_total or completed % 10 == 0:
                    q.put(("message", f"Micro gamma sweep: {completed}/{gamma_total} completed"
                                       + (f" (gamma={gamma})" if gamma is not None else "")))
                q.put(("progress", ("gamma", .18 + .39 * completed / gamma_total,
                                     f"Louvain gamma sweep: {completed}/{gamma_total} completed")))
                return result

            micro_module._build_spike_counts_df = tracked_counts
            micro_module._build_window_graphs_from_counts = tracked_graphs
            micro_module._run_single_gamma_louvain_in_memory = tracked_gamma
            t = time.perf_counter()
            try:
                result = original_micro(*args, **kwargs)
            finally:
                for name, fn in originals.items():
                    setattr(micro_module, name, fn)
            q.put(("message", f"MICRO COMPLETED in {time.perf_counter()-t:.3f} s"))
            q.put(("progress", ("critical_points", .62,
                                 "Micro critical-point analysis completed")))
            return result

        def tracked_meso(*args, **kwargs):
            q.put(("progress", ("micro_states", .72, "Constructing Proto-states")))
            t = time.perf_counter()
            result = original_meso(*args, **kwargs)
            q.put(("message", f"MESO COMPLETED in {time.perf_counter()-t:.3f} s"))
            q.put(("progress", ("outputs", .97, "Meso outputs generated")))
            return result

        pipeline.auto_select_micro_gamma = tracked_micro
        pipeline.run_meso = tracked_meso
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                summaries = pipeline.run_pipeline(run_cfg)
        finally:
            pipeline.auto_select_micro_gamma = original_micro
            pipeline.run_meso = original_meso
        writer.flush()
        elapsed = time.perf_counter() - started
        q.put(("message", f"GUI worker completed in {elapsed:.3f} s"))
        q.put(("finished", _build_result(cfg, summaries, elapsed)))
    except BaseException as exc:
        writer.flush()
        q.put(("error", (str(exc), traceback.format_exc())))

class NeuroSeqBackend:
    STAGES = (
        ("validation", "Validating input", .03),
        ("prepare", "Preparing spike data", .08),
        ("networks", "Building Micro-Epoch networks", .18),
        ("gamma", "Sweeping Louvain resolution", .48),
        ("critical_points", "Detecting consensus critical points", .62),
        ("micro_states", "Constructing Proto-states", .72),
        ("similarities", "Computing Proto-state similarities", .79),
        ("meso", "Discovering Metastable States", .90),
        ("outputs", "Generating outputs", .97),
        ("complete", "Completed", 1.0),
    )
    def __init__(self,*,on_message=None,on_progress=None,on_finished=None,on_error=None,on_cancelled=None):
        self.on_message=on_message or (lambda m:None); self.on_progress=on_progress or (lambda s,f,m:None)
        self.on_finished=on_finished or (lambda r:None); self.on_error=on_error or (lambda m,d:None); self.on_cancelled=on_cancelled or (lambda:None)
        self._process=None; self._queue=None; self._running=False; self._cancel=False; self._lock=threading.Lock(); self._stage=-1
    @property
    def is_running(self):
        with self._lock: return self._running
    def start(self,cfg):
        with self._lock:
            if self._running: raise RuntimeError("NeuroSeq is already running.")
            self._running=True; self._cancel=False
        self._stage=-1; self._set_stage("validation")
        errors=self.validate(cfg)
        if errors:
            with self._lock: self._running=False
            raise ValueError("\n".join(errors))
        Path(cfg.output_root).expanduser().mkdir(parents=True,exist_ok=True)
        ctx=mp.get_context("spawn"); self._queue=ctx.Queue(); self._process=ctx.Process(target=_pipeline_process_main,args=(asdict(cfg),self._queue),name="NeuroSeqProcess", daemon=False)
        self._process.start(); threading.Thread(target=self._monitor,daemon=True,name="NeuroSeqMonitor").start()
    def stop(self):
        with self._lock:
            if not self._running: return False
            self._cancel=True; p=self._process
        self.on_message("Stop requested. Terminating NeuroSeq…")
        if p is not None and p.is_alive(): p.terminate()
        return True
    def validate(self,cfg):
        e = []
        data = Path(str(cfg.data_folder).strip()).expanduser()
        out = Path(str(cfg.output_root).strip()).expanduser()
        if not str(cfg.data_folder).strip():
            e.append("Choose a recording folder.")
        elif not data.is_dir():
            e.append(f"Recording folder does not exist: {data}")
        elif not list(data.glob("SU_*.csv")):
            e.append(f"No SU_*.csv files were found in: {data}")
        if not str(cfg.output_root).strip():
            e.append("Choose an output folder.")
        elif out.exists() and not out.is_dir():
            e.append(f"Output path is not a folder: {out}")
        if cfg.T <= 0: e.append("Micro-Epoch T must be greater than zero.")
        if cfg.total_duration <= 0: e.append("Duration must be greater than zero.")
        if cfg.auto_gamma_step <= 0: e.append("Gamma step must be greater than zero.")
        if cfg.auto_gamma_max < cfg.auto_gamma_min: e.append("Gamma maximum must be at least gamma minimum.")
        if cfg.seeds_per_gamma < 1: e.append("Seeds per gamma must be at least 1.")
        if cfg.persistence_tolerance_bins < 0 or cfg.consensus_cp_band_tolerance_bins < 0:
            e.append("Critical-point tolerances cannot be negative.")
        return e
    @staticmethod
    def open_path(path):
        t=Path(path).expanduser().resolve()
        if not t.exists(): raise FileNotFoundError(f"Path does not exist: {t}")
        if sys.platform.startswith("win"): os.startfile(str(t))
        elif sys.platform=="darwin": subprocess.Popen(["open",str(t)])
        else: subprocess.Popen(["xdg-open",str(t)])
    def open_output_folder(self,path): self.open_path(path)
    def _monitor(self):
        terminal=False
        try:
            while True:
                try: kind,payload=self._queue.get(timeout=.15)
                except queue.Empty: kind=None; payload=None
                if kind=="started":
                    self.on_message(f"NeuroSeq worker process started (PID {payload}).")
                elif kind=="message":
                    self._detect(str(payload)); self.on_message(str(payload))
                elif kind=="progress": s,f,m=payload; self._set_stage(s,m)
                elif kind=="finished": terminal=True; self._set_stage("complete"); self.on_finished(payload); break
                elif kind=="error": terminal=True; self.on_error(*payload); break
                if self._process is not None and not self._process.is_alive():
                    self._process.join(timeout=.2)
                    # Queue feeder may still be delivering the final result.
                    if not terminal:
                        try:
                            kind2, payload2 = self._queue.get(timeout=2.0)
                            if kind2 == "finished":
                                terminal = True
                                self._set_stage("complete")
                                self.on_finished(payload2)
                            elif kind2 == "error":
                                terminal = True
                                self.on_error(*payload2)
                            elif kind2 == "message":
                                self.on_message(str(payload2))
                        except queue.Empty:
                            pass
                    if not terminal:
                        if self._cancel: self.on_cancelled()
                        elif self._process.exitcode not in (0,None):
                            self.on_error(
                                f"NeuroSeq process exited with code {self._process.exitcode}.",
                                "The worker process ended before it could report a Python exception.\n"
                                "This is usually a Windows multiprocessing startup problem.\n"
                                "The updated GUI uses an import-light GUI/__init__.py and a non-daemon worker.",
                            )
                    break
        finally:
            with self._lock: self._running=False
    def _set_stage(self,key,message=None):
        for i,(k,d,f) in enumerate(self.STAGES):
            if k==key:
                if i<self._stage: return
                self._stage=i; self.on_progress(k,f,message or d); return
    def _detect(self,line):
        low=line.lower(); rules=(("running clean auto-micro gamma sweep","gamma"),("consensus critical","critical_points"),("boundary source:","critical_points"),("proto-state","micro_states"),("state-pair nmi","similarities"),("saved state vectors and matrices","similarities"),("running meso","meso"), ("joint eigengap","meso"), ("joint selected k","meso"),("affinity graph","outputs"),("state timeline","outputs"))
        for token,stage in rules:
            if token in low: self._set_stage(stage); return
        m=re.search(r"\bgamma\s*[=:]\s*([0-9.]+)",low)
        if m: self._set_stage("gamma",f"Sweeping Louvain resolution — γ {m.group(1)}")
