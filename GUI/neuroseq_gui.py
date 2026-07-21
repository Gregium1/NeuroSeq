# GUI/neuroseq_gui.py
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .gui_settings import DEFAULTS, format_seconds, load_settings, save_settings
from .gui_theme import get_palette
from .gui_widgets import (
    CollapsibleSection,
    ConsoleBox,
    LabeledEntry,
    MetricsPanel,
    PathPicker,
    ProgressChecklist,
    OutputFiles,
)
from .neuroseq_backend import NeuroSeqBackend, NeuroSeqGUIConfig


class NeuroSeqGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NeuroSeq")
        self.resources_dir = Path(__file__).resolve().parent / "resources"
        self.geometry("1480x920")
        self.minsize(1180, 780)
        self.protocol("WM_DELETE_WINDOW", self._request_exit)
        self._set_windows_app_id()
        self._set_window_icon()

        self._result: dict[str, Any] | None = None
        self.theme_mode = tk.StringVar(value="light")
        self._create_variables()
        self._build_style()
        self._build_layout()

        self.backend = NeuroSeqBackend(
            on_message=lambda message: self.after(0, self._append_console, message),
            on_progress=lambda stage, fraction, message: self.after(
                0, self._update_progress, stage, fraction, message
            ),
            on_finished=lambda result: self.after(0, self._run_finished, result),
            on_error=lambda message, details: self.after(0, self._run_error, message, details),
            on_cancelled=lambda: self.after(0, self._run_cancelled),
        )

        self._update_mode_fields()
        self._set_idle_state()
        self._apply_theme()

    def _create_variables(self) -> None:
        d = DEFAULTS
        values = {
            "data_folder": d.data_folder,
            "output_root": d.output_root,
            "run_mode": d.run_mode,
            "start_time": d.start_time,
            "total_duration": d.total_duration,
            "cts_file": d.cts_file,
            "pre_time": d.pre_time,
            "post_time": d.post_time,
            "delivery_prefix": d.delivery_prefix,
            "window_size": d.window_size,
            "T": d.T,
            "lambda_threshold": d.lambda_threshold,
            "auto_gamma_min": d.auto_gamma_min,
            "auto_gamma_max": d.auto_gamma_max,
            "auto_gamma_step": d.auto_gamma_step,
            "seeds_per_gamma": d.seeds_per_gamma,
            "seed_for_states": d.seed_for_states,
            "persistence_tolerance_bins": d.persistence_tolerance_bins,
            "consensus_cp_band_tolerance_bins": d.consensus_cp_band_tolerance_bins,
            "sample_k": d.sample_k,
            "repeats": d.repeats,
            "base_seed": d.base_seed,
            "meso_seed": d.meso_seed,
            "threads_per_process": d.threads_per_process,
            "true_n_states": "",
        }
        self.vars: dict[str, tk.Variable] = {
            key: tk.StringVar(value=str(value)) for key, value in values.items()
        }
        self.vars["use_sampling"] = tk.BooleanVar(value=d.use_sampling)
        self.vars["do_graph"] = tk.BooleanVar(value=d.do_graph)

    def _build_style(self) -> None:
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

    def _apply_theme(self) -> None:
        p = get_palette(self.theme_mode.get())
        self.palette = p
        self.configure(background=p.background)

        s = self.style
        s.configure(".", font=("Segoe UI", 10))
        s.configure("TFrame", background=p.background)
        s.configure("Card.TFrame", background=p.card)
        s.configure("PurpleCard.TFrame", background=p.card_alt)
        s.configure("Viewer.TFrame", background=p.console_bg)
        s.configure("Header.TFrame", background=p.navy)
        s.configure("Header.TLabel", background=p.navy, foreground="#FFFFFF")
        s.configure("Title.Header.TLabel", font=("Segoe UI", 22, "bold"))
        s.configure("Subtitle.Header.TLabel", font=("Segoe UI", 10))
        s.configure("TLabel", background=p.background, foreground=p.text)
        s.configure("Card.TLabel", background=p.card, foreground=p.text)
        s.configure("PurpleTitle.TLabel", background=p.card_alt, foreground=p.purple, font=("Segoe UI", 10, "bold"))
        s.configure("Checklist.TLabel", background=p.card_alt, foreground=p.text)
        s.configure("TLabelframe", background=p.card, foreground=p.text, bordercolor=p.border)
        s.configure("TLabelframe.Label", background=p.card, foreground=p.purple, font=("Segoe UI", 10, "bold"))
        s.configure("TEntry", fieldbackground=p.input_bg, foreground=p.text, insertcolor=p.text)
        s.configure("TCheckbutton", background=p.card, foreground=p.text)
        s.configure("TRadiobutton", background=p.card, foreground=p.text)
        s.configure("Section.TCheckbutton", background=p.purple_soft, foreground=p.text, font=("Segoe UI", 10, "bold"), padding=6)
        s.configure("TButton", padding=(10, 6))
        s.configure("Primary.TButton", background=p.blue, foreground="#FFFFFF", font=("Segoe UI", 11, "bold"), padding=(18, 11))
        s.map("Primary.TButton", background=[("active", p.blue_hover), ("disabled", p.border)])
        s.configure("Accent.TButton", background=p.purple, foreground="#FFFFFF")
        s.map("Accent.TButton", background=[("active", p.blue_hover)])
        s.configure("TNotebook", background=p.card, borderwidth=0)
        s.configure("TNotebook.Tab", background=p.purple_soft, foreground=p.text, padding=(16, 8))
        s.map("TNotebook.Tab", background=[("selected", p.purple)], foreground=[("selected", "#FFFFFF")])
        s.configure("Horizontal.TProgressbar", background=p.teal, troughcolor=p.purple_soft)
        s.configure("Metric.TFrame", background=p.card_alt, relief="flat")
        s.configure("MetricName.TLabel", background=p.card_alt, foreground=p.muted, font=("Segoe UI", 9, "bold"))
        s.configure("MetricValue.TLabel", background=p.card_alt, foreground=p.purple, font=("Segoe UI", 17, "bold"))

        self.settings_canvas.configure(background=p.card)
        self.console.apply_palette(p)

    def _build_layout(self) -> None:
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        header = ttk.Frame(self, style="Header.TFrame", padding=(24, 15))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        self.header_logo = self._load_logo_image(62)
        if self.header_logo is not None:
            ttk.Label(header, image=self.header_logo, style="Header.TLabel").grid(row=0, column=0, rowspan=2, padx=(0, 14))

        ttk.Label(header, text="NeuroSeq", style="Title.Header.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(
            header,
            text="Automatic Micro-state segmentation and Meso-family discovery",
            style="Subtitle.Header.TLabel",
        ).grid(row=1, column=1, sticky="w")

        theme_frame = ttk.Frame(header, style="Header.TFrame")
        theme_frame.grid(row=0, column=2, rowspan=2, sticky="e")
        ttk.Label(theme_frame, text="Theme", style="Header.TLabel").pack(side="left", padx=(0, 7))
        ttk.Radiobutton(theme_frame, text="Light", value="light", variable=self.theme_mode, command=self._apply_theme).pack(side="left")
        ttk.Radiobutton(theme_frame, text="Dark", value="dark", variable=self.theme_mode, command=self._apply_theme).pack(side="left", padx=(4, 0))

        content = ttk.Frame(self, padding=14)
        content.grid(row=1, column=0, sticky="nsew")
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, minsize=410)
        content.columnconfigure(1, weight=1)

        self._build_settings_panel(content)
        self._build_run_panel(content)

    def _build_settings_panel(self, parent) -> None:
        outer = ttk.Frame(parent, style="Card.TFrame", padding=14)
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        outer.rowconfigure(0, weight=1)
        outer.rowconfigure(1, weight=0)
        outer.columnconfigure(0, weight=1)

        self.settings_canvas = tk.Canvas(outer, highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=self.settings_canvas.yview)
        self.settings_canvas.configure(yscrollcommand=scroll.set)
        self.settings_canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        body = ttk.Frame(self.settings_canvas, style="Card.TFrame", padding=(2, 2, 8, 2))
        body.columnconfigure(0, weight=1)
        window = self.settings_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _e: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_canvas.bind("<Configure>", lambda e: self.settings_canvas.itemconfigure(window, width=e.width))

        ttk.Label(body, text="Analysis settings", style="Card.TLabel", font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))
        PathPicker(body, label="Recording folder", variable=self.vars["data_folder"]).grid(row=1, column=0, sticky="ew", pady=4)
        PathPicker(body, label="Output folder", variable=self.vars["output_root"]).grid(row=2, column=0, sticky="ew", pady=4)

        mode = ttk.LabelFrame(body, text="Run mode", padding=9)
        mode.grid(row=3, column=0, sticky="ew", pady=(10, 6))
        ttk.Radiobutton(mode, text="Full Recording", value="full_recording", variable=self.vars["run_mode"], command=self._update_mode_fields).grid(row=0, column=0, padx=(0, 14))
        ttk.Radiobutton(mode, text="Delivery", value="delivery", variable=self.vars["run_mode"], command=self._update_mode_fields).grid(row=0, column=1)

        self.full_frame = ttk.LabelFrame(body, text="Full Recording", padding=9)
        self.full_frame.grid(row=4, column=0, sticky="ew", pady=6)
        LabeledEntry(self.full_frame, label="Start time (s)", variable=self.vars["start_time"]).grid(row=0, column=0, sticky="ew", pady=3)
        LabeledEntry(self.full_frame, label="Duration (s)", variable=self.vars["total_duration"]).grid(row=1, column=0, sticky="ew", pady=3)
        self.full_frame.columnconfigure(0, weight=1)

        self.delivery_frame = ttk.LabelFrame(body, text="Delivery", padding=9)
        self.delivery_frame.grid(row=5, column=0, sticky="ew", pady=6)
        PathPicker(self.delivery_frame, label="CTS/event file", variable=self.vars["cts_file"], mode="file", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]).grid(row=0, column=0, sticky="ew")
        for row, (label, key) in enumerate((("Pre time (s)", "pre_time"), ("Post time (s)", "post_time"), ("Delivery prefix", "delivery_prefix")), 1):
            LabeledEntry(self.delivery_frame, label=label, variable=self.vars[key]).grid(row=row, column=0, sticky="ew", pady=3)
        self.delivery_frame.columnconfigure(0, weight=1)

        core = CollapsibleSection(body, title="Core parameters", expanded=False)
        core.grid(row=6, column=0, sticky="ew", pady=6)
        core_fields = (
            ("Window size (s)", "window_size"),
            ("T", "T"),
            ("Lambda threshold", "lambda_threshold"),
            ("Gamma minimum", "auto_gamma_min"),
            ("Gamma maximum", "auto_gamma_max"),
            ("Gamma step", "auto_gamma_step"),
            ("Seeds per gamma", "seeds_per_gamma"),
        )
        for row, (label, key) in enumerate(core_fields):
            LabeledEntry(core.body, label=label, variable=self.vars[key]).grid(
                row=row,
                column=0,
                sticky="ew",
                pady=3,
            )

        advanced = CollapsibleSection(body, title="Advanced settings", expanded=False)
        advanced.grid(row=7, column=0, sticky="ew", pady=8)
        advanced_fields = (
            ("State Louvain seed", "seed_for_states"),
            ("Persistence tolerance (bins)", "persistence_tolerance_bins"),
            ("Consensus CP tolerance (bins)", "consensus_cp_band_tolerance_bins"),
            ("Meso seed", "meso_seed"),
            ("Threads", "threads_per_process"),
            ("True state count (optional)", "true_n_states"),
        )
        for row, (label, key) in enumerate(advanced_fields):
            LabeledEntry(advanced.body, label=label, variable=self.vars[key]).grid(row=row, column=0, sticky="ew", pady=3)
        ttk.Checkbutton(advanced.body, text="Generate plots", variable=self.vars["do_graph"]).grid(row=len(advanced_fields), column=0, sticky="w", pady=3)
        ttk.Checkbutton(advanced.body, text="Use neuron sampling", variable=self.vars["use_sampling"]).grid(row=len(advanced_fields)+1, column=0, sticky="w", pady=3)

        actions = ttk.Frame(body, style="Card.TFrame")
        actions.grid(row=8, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(actions, text="Load settings", command=self._load_settings_dialog).pack(side="left")
        ttk.Button(actions, text="Save settings", command=self._save_settings_dialog).pack(side="left", padx=6)
        ttk.Button(actions, text="Reset", command=self._reset_settings).pack(side="right")

        progress_card = ttk.Frame(outer, style="PurpleCard.TFrame", padding=10)
        progress_card.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 0),
        )
        progress_card.columnconfigure(0, weight=1)

        ttk.Label(
            progress_card,
            text="Run progress",
            style="PurpleTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.progress_label = ttk.Label(
            progress_card,
            text="Ready",
            style="Checklist.TLabel",
        )
        self.progress_label.grid(row=1, column=0, sticky="w", pady=(5, 2))

        self.progress = ttk.Progressbar(progress_card, maximum=100)
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 6))

        self.checklist = ProgressChecklist(progress_card)
        self.checklist.grid(row=3, column=0, sticky="ew")

    def _build_run_panel(self, parent) -> None:
        right = ttk.Frame(parent, style="Card.TFrame", padding=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(1, weight=2)
        right.rowconfigure(2, weight=3)
        right.columnconfigure(0, weight=1)

        top = ttk.Frame(right, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)

        ttk.Label(
            top,
            text="NeuroSeq run",
            style="Card.TLabel",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")

        self.run_button = ttk.Button(
            top,
            text="Run NeuroSeq",
            style="Primary.TButton",
            command=self._start_run,
        )
        self.run_button.grid(row=1, column=0, sticky="ew", pady=(10, 10))

        console_card = ttk.Frame(right, style="Card.TFrame", padding=10)
        console_card.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        console_card.rowconfigure(1, weight=1)
        console_card.columnconfigure(0, weight=1)

        console_bar = ttk.Frame(console_card, style="Card.TFrame")
        console_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(
            console_bar,
            text="Live console",
            style="Card.TLabel",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

        ttk.Button(
            console_bar,
            text="Clear",
            command=self.console_clear_later,
        ).pack(side="right")

        self.console = ConsoleBox(console_card)
        self.console.grid(row=1, column=0, sticky="nsew")

        files_card = ttk.Frame(right, style="Card.TFrame", padding=10)
        files_card.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        files_card.rowconfigure(1, weight=1)
        files_card.columnconfigure(0, weight=1)

        files_bar = ttk.Frame(files_card, style="Card.TFrame")
        files_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(
            files_bar,
            text="Output files",
            style="Card.TLabel",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

        ttk.Label(
            files_bar,
            text="Double-click a file to open it",
            style="Card.TLabel",
        ).pack(side="left", padx=(12, 0))

        ttk.Button(
            files_bar,
            text="Refresh files",
            style="Accent.TButton",
            command=self._refresh_output_files,
        ).pack(side="right")

        self.output_files = OutputFiles(
            files_card,
            open_path=NeuroSeqBackend.open_path,
        )
        self.output_files.grid(row=1, column=0, sticky="nsew")

        actions = ttk.Frame(right, style="Card.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        self.open_output_button = ttk.Button(
            actions,
            text="Open Output Folder",
            command=self._open_output,
        )
        self.open_output_button.pack(side="left")

        self.save_result_button = ttk.Button(
            actions,
            text="Save Run Summary",
            command=self._save_result_summary,
        )
        self.save_result_button.pack(side="left", padx=6)

        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self._stop_run,
        )
        self.stop_button.pack(side="right", padx=(6, 0))

        self.exit_button = ttk.Button(
            actions,
            text="Exit",
            command=self._request_exit,
        )
        self.exit_button.pack(side="right")

        self.metrics = MetricsPanel(right)
        self.metrics.grid(row=4, column=0, sticky="ew")
        self.metrics.set_metrics([
            ("Status", "Ready"),
            ("Runtime", "—"),
            ("Micro states", "—"),
            ("Meso families", "—"),
        ])

    def console_clear_later(self) -> None:
        self.console.clear()

    def _set_windows_app_id(self) -> None:
        """
        Give NeuroSeq its own Windows taskbar identity.

        This helps Windows use the NeuroSeq window icon rather than grouping
        the application under the generic Python executable.
        """
        if not str(self.tk.call("tk", "windowingsystem")).lower() == "win32":
            return
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "NeuroSeq.Desktop.1"
            )
        except Exception:
            pass

    def _set_window_icon(self) -> None:
        try:
            ico=self.resources_dir / "NeuroSeq_logo.ico"
            if ico.exists(): self.iconbitmap(default=str(ico)); return
        except Exception: pass
        try:
            png=self.resources_dir / "NeuroSeq_logo_256.png"
            self._window_icon_photo=tk.PhotoImage(file=str(png)); self.iconphoto(True,self._window_icon_photo)
        except Exception: self._window_icon_photo=None

    def _load_logo_image(self, size: int):
        try:
            from PIL import Image, ImageTk
            image=Image.open(self.resources_dir / "NeuroSeq_logo_256.png").convert("RGBA"); image.thumbnail((size,size))
            return ImageTk.PhotoImage(image)
        except Exception: return None

    def _stop_run(self) -> None:
        if self.backend.is_running and messagebox.askyesno("Stop NeuroSeq", "Stop the current analysis?\n\nPartial output files may remain."):
            self.stop_button.configure(state="disabled", text="Stopping…"); self.backend.stop()

    def _run_cancelled(self) -> None:
        self._append_console("NeuroSeq analysis was stopped by the user.")
        self.progress_label.configure(text="Stopped")
        self.metrics.set_metrics([("Status","Stopped"),("Runtime","—"),("Micro states","—"),("Meso families","—")])
        self._set_idle_state()

    def _request_exit(self) -> None:
        if self.backend.is_running:
            if not messagebox.askyesno("Exit NeuroSeq", "NeuroSeq is still running. Stop it and exit?"): return
            self.backend.stop()
        self.after(150, self.destroy)

    def _refresh_output_files(self) -> None:
        folder=self._result.get("output_root") if self._result else str(self.vars["output_root"].get()).strip()
        if folder: self.output_files.load_from_folder(folder)

    def _number(self, key: str, cast):
        value = str(self.vars[key].get()).strip()
        if value == "" and key == "true_n_states":
            return None
        try:
            return cast(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {key}: {value!r}") from exc

    def _collect_config(self) -> NeuroSeqGUIConfig:
        return NeuroSeqGUIConfig(
            data_folder=str(self.vars["data_folder"].get()).strip(),
            output_root=str(self.vars["output_root"].get()).strip(),
            run_mode=str(self.vars["run_mode"].get()).strip(),
            start_time=self._number("start_time", float),
            total_duration=self._number("total_duration", float),
            cts_file=str(self.vars["cts_file"].get()).strip(),
            pre_time=self._number("pre_time", float),
            post_time=self._number("post_time", float),
            delivery_prefix=str(self.vars["delivery_prefix"].get()).strip(),
            window_size=self._number("window_size", float),
            T=self._number("T", float),
            lambda_threshold=self._number("lambda_threshold", float),
            auto_gamma_min=self._number("auto_gamma_min", float),
            auto_gamma_max=self._number("auto_gamma_max", float),
            auto_gamma_step=self._number("auto_gamma_step", float),
            seeds_per_gamma=self._number("seeds_per_gamma", int),
            seed_for_states=self._number("seed_for_states", int),
            persistence_tolerance_bins=self._number("persistence_tolerance_bins", int),
            consensus_cp_band_tolerance_bins=self._number("consensus_cp_band_tolerance_bins", int),
            use_sampling=bool(self.vars["use_sampling"].get()),
            sample_k=self._number("sample_k", int),
            repeats=self._number("repeats", int),
            base_seed=self._number("base_seed", int),
            meso_seed=self._number("meso_seed", int),
            do_graph=bool(self.vars["do_graph"].get()),
            threads_per_process=self._number("threads_per_process", int),
            true_n_states=self._number("true_n_states", int),
        )

    def _apply_config(self, cfg: NeuroSeqGUIConfig) -> None:
        for key, variable in self.vars.items():
            value = getattr(cfg, key)
            variable.set("" if value is None else value)
        self._update_mode_fields()

    def _start_run(self) -> None:
        try:
            cfg = self._collect_config()
            errors = self.backend.validate(cfg)
            if errors:
                messagebox.showerror("Invalid settings", "\n".join(errors))
                return
            self._result = None
            self.console.clear()
            self._append_console("Starting NeuroSeq...")
            self._set_running_state()
            self.backend.start(cfg)
        except Exception as exc:
            self._set_idle_state()
            messagebox.showerror("Could not start", str(exc))

    def _update_mode_fields(self) -> None:
        if str(self.vars["run_mode"].get()) == "delivery":
            self.full_frame.grid_remove()
            self.delivery_frame.grid()
        else:
            self.delivery_frame.grid_remove()
            self.full_frame.grid()

    def _save_settings_dialog(self) -> None:
        try:
            cfg = self._collect_config()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON settings", "*.json")], initialfile="neuroseq_settings.json")
        if path:
            save_settings(cfg, path)

    def _load_settings_dialog(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON settings", "*.json"), ("All files", "*.*")])
        if path:
            try:
                self._apply_config(load_settings(path))
            except Exception as exc:
                messagebox.showerror("Could not load settings", str(exc))

    def _reset_settings(self) -> None:
        self._apply_config(NeuroSeqGUIConfig(data_folder="", output_root=""))

    def _open_output(self) -> None:
        path = self._result.get("output_root") if self._result else str(self.vars["output_root"].get()).strip()
        try:
            self.backend.open_output_folder(path)
        except Exception as exc:
            messagebox.showerror("Could not open folder", str(exc))

    def _save_result_summary(self) -> None:
        if not self._result:
            messagebox.showinfo("No result", "Run NeuroSeq before saving a summary.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON summary", "*.json")], initialfile="neuroseq_run_summary.json")
        if path:
            Path(path).write_text(json.dumps(self._result, indent=2, default=str), encoding="utf-8")

    def _append_console(self, message: str) -> None:
        self.console.append(message)

    def _update_progress(self, stage: str, fraction: float, message: str) -> None:
        self.progress["value"] = max(0.0, min(100.0, float(fraction) * 100.0))
        self.progress_label.configure(text=message)
        self.checklist.update_stage(stage)

    def _run_finished(self, result: dict[str, Any]) -> None:
        self._result = result
        families = result.get("n_families_by_interval") or []
        family_text = ", ".join(map(str, families)) if families else "—"
        self.metrics.set_metrics([
            ("Status", "Completed"),
            ("Runtime", format_seconds(result.get("elapsed_sec"))),
            ("Micro states", str(result.get("n_micro_states_total", "—"))),
            ("Meso families", family_text),
        ])
        self._append_console("NeuroSeq completed successfully.")
        self.output_files.load_from_folder(result.get("output_root", ""))
        self._set_idle_state()

    def _run_error(self, message: str, details: str) -> None:
        self._append_console(details)
        self.metrics.set_metrics([
            ("Status", "Failed"),
            ("Runtime", "—"),
            ("Micro states", "—"),
            ("Meso families", "—"),
        ])
        self._set_idle_state()
        messagebox.showerror("NeuroSeq failed", message)

    def _set_running_state(self) -> None:
        self.run_button.configure(state="disabled", text="NeuroSeq is running…")
        self.open_output_button.configure(state="disabled")
        self.save_result_button.configure(state="disabled")
        self.stop_button.configure(state="normal", text="Stop")
        self.progress["value"] = 0
        self.progress_label.configure(text="Starting…")
        self.checklist.reset()
        self.metrics.set_metrics([
            ("Status", "Running"),
            ("Runtime", "—"),
            ("Micro states", "—"),
            ("Meso families", "—"),
        ])

    def _set_idle_state(self) -> None:
        self.run_button.configure(state="normal", text="Run NeuroSeq")
        self.open_output_button.configure(state="normal")
        self.save_result_button.configure(state="normal" if self._result else "disabled")
        self.stop_button.configure(state="disabled", text="Stop")
