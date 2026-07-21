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

from orchestrate_micro_meso import RunConfig, run_pipeline

MessageCallback = Callable[[str], None]
ProgressCallback = Callable[[str, float, str], None]
FinishedCallback = Callable[[dict[str, Any]], None]
ErrorCallback = Callable[[str, str], None]
CancelledCallback = Callable[[], None]

@dataclass
class NeuroSeqGUIConfig:
    data_folder: str
    output_root: str
    run_mode: str = "full_recording"
    start_time: float = 0.0
    total_duration: float = 60.0
    cts_file: str = ""
    pre_time: float = 1.0
    post_time: float = 3.0
    delivery_prefix: str = "Delivery"
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
    use_sampling: bool = False
    sample_k: int = 15
    repeats: int = 1
    base_seed: int = 12345
    meso_seed: int = 42
    do_graph: bool = True
    threads_per_process: int = 1
    true_n_states: int | None = None

    def to_run_config(self) -> RunConfig:
        return RunConfig(
            data_folder=str(Path(self.data_folder).expanduser()), output_root=str(Path(self.output_root).expanduser()),
            run_mode=str(self.run_mode).strip().lower(), start_time=float(self.start_time), total_duration=float(self.total_duration),
            cts_file=self.cts_file.strip() or None, pre_time=float(self.pre_time), post_time=float(self.post_time),
            delivery_prefix=str(self.delivery_prefix).strip() or "Delivery", window_size=float(self.window_size), T=float(self.T),
            lambda_threshold=float(self.lambda_threshold), auto_gamma_min=float(self.auto_gamma_min), auto_gamma_max=float(self.auto_gamma_max),
            auto_gamma_step=float(self.auto_gamma_step), seeds_per_gamma=int(self.seeds_per_gamma), seed_for_states=int(self.seed_for_states),
            persistence_tolerance_bins=int(self.persistence_tolerance_bins), consensus_cp_band_tolerance_bins=int(self.consensus_cp_band_tolerance_bins),
            use_sampling=bool(self.use_sampling), sample_k=int(self.sample_k), repeats=int(self.repeats), base_seed=int(self.base_seed),
            meso_seed=int(self.meso_seed), do_graph=bool(self.do_graph), threads_per_process=int(self.threads_per_process),
            true_n_states=None if self.true_n_states in (None, "") else int(self.true_n_states),
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

def _build_result(cfg,summaries,elapsed):
    total=0; fam=[]; run_dirs=[]
    for s in summaries:
        rd=str(s.get("run_dir", "")); run_dirs += [rd] if rd else []
        p=(s.get("micro_outputs",{}) or {}).get("state_vectors_csv")
        if p and Path(p).exists():
            try:
                import pandas as pd
                total += len(pd.read_csv(p))
            except Exception: pass
        f=_family_count(s.get("meso_outputs",{}) or {})
        if f is not None: fam.append(f)
    return {"status":"OK","elapsed_sec":float(elapsed),"n_intervals":len(summaries),"n_micro_states_total":int(total),
            "n_families_by_interval":fam,"output_root":str(Path(cfg.output_root).expanduser()),"run_dirs":run_dirs,"summaries":summaries}

def _pipeline_process_main(config_dict,q):
    started=time.perf_counter(); writer=_QueueWriter(q)
    try:
        cfg=NeuroSeqGUIConfig(**config_dict)
        q.put(("started", os.getpid()))
        q.put(("progress",("prepare",0.08,"Preparing spike data")))
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer): summaries=run_pipeline(cfg.to_run_config())
        writer.flush(); q.put(("finished",_build_result(cfg,summaries,time.perf_counter()-started)))
    except BaseException as exc:
        writer.flush(); q.put(("error",(str(exc),traceback.format_exc())))

class NeuroSeqBackend:
    STAGES=(("validation","Validating input",.03),("prepare","Preparing spike data",.08),("networks","Building window networks",.18),
            ("gamma","Sweeping Louvain resolution",.48),("critical_points","Detecting critical points",.62),("micro_states","Constructing Micro states",.72),
            ("similarities","Computing state similarities",.79),("meso","Discovering Meso families",.90),("outputs","Generating outputs",.97),("complete","Completed",1.0))
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
        e=[]; data=Path(str(cfg.data_folder).strip()).expanduser(); out=Path(str(cfg.output_root).strip()).expanduser(); mode=str(cfg.run_mode).strip().lower()
        if not str(cfg.data_folder).strip(): e.append("Choose a recording folder.")
        elif not data.is_dir(): e.append(f"Recording folder does not exist: {data}")
        elif not list(data.glob("SU_*.csv")): e.append(f"No SU_*.csv files were found in: {data}")
        if not str(cfg.output_root).strip(): e.append("Choose an output folder.")
        elif out.exists() and not out.is_dir(): e.append(f"Output path is not a folder: {out}")
        if mode not in {"full_recording","delivery"}: e.append("Run mode must be Full Recording or Delivery.")
        if cfg.window_size<=0: e.append("Window size must be greater than zero.")
        if cfg.T<=0: e.append("T must be greater than zero.")
        if cfg.auto_gamma_step<=0: e.append("Gamma step must be greater than zero.")
        if cfg.auto_gamma_max<cfg.auto_gamma_min: e.append("Gamma maximum must be at least gamma minimum.")
        if cfg.seeds_per_gamma<1: e.append("Seeds per gamma must be at least 1.")
        if mode=="full_recording" and cfg.total_duration<=0: e.append("Duration must be greater than zero.")
        if mode=="delivery":
            c=Path(str(cfg.cts_file).strip()).expanduser()
            if not str(cfg.cts_file).strip(): e.append("Choose a CTS/event file.")
            elif not c.is_file(): e.append(f"CTS/event file does not exist: {c}")
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
        low=line.lower(); rules=(("running clean auto-micro gamma sweep","gamma"),("consensus critical","critical_points"),("boundary source:","critical_points"),("state_windows","micro_states"),("state-pair nmi","similarities"),("saved state vectors and matrices","similarities"),("running meso","meso"),("affinity graph","outputs"),("state timeline","outputs"))
        for token,stage in rules:
            if token in low: self._set_stage(stage); return
        m=re.search(r"\bgamma\s*[=:]\s*([0-9.]+)",low)
        if m: self._set_stage("gamma",f"Sweeping Louvain resolution — γ {m.group(1)}")
