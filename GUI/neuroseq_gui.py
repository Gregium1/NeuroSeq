"""Tkinter desktop interface for full-population NeuroSeq analysis.

The interface exposes the Micro-Epoch duration ``T``; the gamma sweep and
seeds remain configurable in Python. Computational work is delegated to
the backend so the Tkinter event loop remains responsive. 

Author
------
Greg Owanga
Gregium Theory
"""
# GUI/neuroseq_gui.py
from __future__ import annotations

import json
import os
import queue
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


# This class owns presentation only. Numerical calculations run in a separate
# backend worker so the Tkinter event loop can continue repainting the UI.
# In particular, the progress checklist never launches scientific calculations.
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

        # The backend monitor runs on a Python thread. Never call Tk from it.
        # Tk's main thread polls this thread-safe queue on every platform.
        self._ui_events = queue.Queue()
        self.backend = NeuroSeqBackend(
            on_message=lambda message: self._ui_events.put(("message", (message,))),
            on_progress=lambda stage, fraction, message: self._ui_events.put(
                ("progress", (stage, fraction, message))
            ),
            on_finished=lambda result: self._ui_events.put(("finished", (result,))),
            on_error=lambda message, details: self._ui_events.put(("error", (message, details))),
            on_cancelled=lambda: self._ui_events.put(("cancelled", ())),
        )
        self.after(50, self._poll_ui_events)

        self._set_idle_state()
        self._apply_theme()

    def _poll_ui_events(self) -> None:
        """Apply worker notifications only on the Tkinter main thread."""
        handlers = {
            "message": self._append_console,
            "progress": self._update_progress,
            "finished": self._run_finished,
            "error": self._run_error,
            "cancelled": self._run_cancelled,
        }
        # Limit work per tick so a large log does not freeze the interface.
        for _ in range(200):
            try:
                kind, args = self._ui_events.get_nowait()
            except queue.Empty:
                break
            handlers[kind](*args)
        self.after(50, self._poll_ui_events)

    def _create_variables(self) -> None:
        d = DEFAULTS
        values = {
            "data_folder": d.data_folder,
            "output_root": d.output_root,
            "start_time": d.start_time,
            "total_duration": d.total_duration,
            "T": d.T,
            "auto_gamma_min": d.auto_gamma_min,
            "auto_gamma_max": d.auto_gamma_max,
            "auto_gamma_step": d.auto_gamma_step,
            "seeds_per_gamma": d.seeds_per_gamma,
            "seed_for_states": d.seed_for_states,
            "persistence_tolerance_bins": d.persistence_tolerance_bins,
            "consensus_cp_band_tolerance_bins": d.consensus_cp_band_tolerance_bins,
            "meso_seed": d.meso_seed,
        }
        self.vars: dict[str, tk.Variable] = {
            key: tk.StringVar(value=str(value)) for key, value in values.items()
        }
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
            text="Automatic Proto-states segmentation and States discovery",
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

        self.full_frame = ttk.LabelFrame(body, text="Full recording", padding=9)
        self.full_frame.grid(row=3, column=0, sticky="ew", pady=6)
        LabeledEntry(self.full_frame, label="Start time (s)", variable=self.vars["start_time"]).grid(row=0, column=0, sticky="ew", pady=3)
        LabeledEntry(self.full_frame, label="Duration (s)", variable=self.vars["total_duration"]).grid(row=1, column=0, sticky="ew", pady=3)
        self.full_frame.columnconfigure(0, weight=1)

        core = CollapsibleSection(body, title="Core parameters", expanded=False)
        core.grid(row=4, column=0, sticky="ew", pady=6)
        core_fields = (("Micro-Epoch T (s)", "T"), ("Gamma minimum", "auto_gamma_min"),
                       ("Gamma maximum", "auto_gamma_max"), ("Gamma step", "auto_gamma_step"))
        for row, (label, key) in enumerate(core_fields):
            LabeledEntry(core.body, label=label, variable=self.vars[key]).grid(
                row=row,
                column=0,
                sticky="ew",
                pady=3,
            )

        actions = ttk.Frame(body, style="Card.TFrame")
        actions.grid(row=5, column=0, sticky="ew", pady=(12, 0))
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
        right.rowconfigure(1, weight=1)
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

        # Keep diagnostic logging without consuming visible space.
        self.console = ConsoleBox(right)
        self.parameter_summary = None

        files_card = ttk.Frame(right, style="Card.TFrame", padding=10)
        files_card.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
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
        self.preview_label = ttk.Label(files_card, text="State sequence preview (first 60 s for long recordings)", style="Card.TLabel")
        self.preview_label.grid(row=2, column=0, sticky="w", pady=(8, 2))
        self.preview_image = ttk.Label(files_card, text="Available after analysis", style="Card.TLabel")
        self.preview_image.grid(row=3, column=0, sticky="ew")
        self._preview_photo = None

        actions = ttk.Frame(right, style="Card.TFrame")
        actions.grid(row=2, column=0, sticky="ew", pady=(0, 10))

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

        self.help_button = ttk.Button(actions, text="Help", command=self._show_help)
        self.help_button.pack(side="right", padx=(0, 6))

        self.metrics = MetricsPanel(right)
        self.metrics.grid(row=3, column=0, sticky="ew")
        self.metrics.set_metrics([
            ("Status", "Ready"),
            ("Runtime", "—"),
            ("Proto-states", "—"),
            ("Metastable States", "—"),
        ])

    def _show_help(self) -> None:
        """Show a readable, scrollable help window without blocking the GUI."""
        dialog = tk.Toplevel(self)
        dialog.title("NeuroSeq | Help")
        dialog.geometry("680x580")
        dialog.minsize(540, 420)
        dialog.transient(self)
        dialog.configure(background=self.palette.background)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(1, weight=1)

        header = ttk.Frame(dialog, style="Header.TFrame", padding=(22, 16))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="NeuroSeq Help", style="Title.Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="A guide to the analysis controls and outputs",
                  style="Subtitle.Header.TLabel").pack(anchor="w", pady=(3, 0))

        area = ttk.Frame(dialog, padding=(20, 14))
        area.grid(row=1, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)
        canvas = tk.Canvas(area, highlightthickness=0, background=self.palette.background)
        scroll = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        body = ttk.Frame(canvas, padding=(4, 2, 14, 8))
        window_id = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))

        sections = [
            ("1. Recording and output folders", [
                ("Recording folder", "Choose the folder containing SU_*.csv spike-train files."),
                ("Output folder", "Choose where NeuroSeq will save its results."),
                ("Browse", "Opens a folder picker starting in the directory where NeuroSeq was launched."),
                ("Load settings", "Loads previously saved analysis settings from a JSON file and fills in the GUI fields. Review the values before running."),
                ("Save settings", "Saves the current analysis settings to a JSON file so they can be reused later. This does not start an analysis."),
                ("Reset", "Restores the default analysis settings and clears the recording and output folder selections."),
            ]),
            ("2. Analysis interval and parameters", [
                ("Start time", "Beginning of the analysis interval, in seconds."),
                ("Duration", "Requested interval length. If it extends past the last observed spike, NeuroSeq shortens the interval."),
                ("Micro-Epoch T", "Duration of each Micro-Epoch window, in seconds."),
                ("Advanced parameters", "Minimum, maximum, and step size for the Louvain gamma sweep."),
            ]),
            ("3. Run controls", [
                ("Run NeuroSeq", "Starts one full-population Micro and Meso analysis."),
                ("Run progress", "Shows the current analysis stage and completed steps."),
                ("Stop", "Requests cancellation of the current run."),
                ("Exit", "Closes the NeuroSeq application."),
            ]),
            ("4. Results", [
                ("Output files", "Double-click a listed file to open it."),
                ("Refresh files", "Updates the list of principal output figures."),
                ("State sequence preview", "Displays the first 60 seconds for long recordings; the complete sequence is saved separately."),
                ("Open Output Folder", "Opens all results, including intermediate scientific files."),
                ("Save Run Summary", "Saves the GUI's summary of the completed analysis."),
            ]),
        ]
        for title, items in sections:
            section = ttk.LabelFrame(body, text=title, padding=(12, 10))
            section.pack(fill="x", pady=(0, 12))
            for label, description in items:
                ttk.Label(section, text=label, font=("Segoe UI", 10, "bold"),
                          wraplength=560).pack(anchor="w", pady=(5, 1))
                ttk.Label(section, text=description, wraplength=560,
                          justify="left").pack(anchor="w", pady=(0, 7))

        footer = ttk.Frame(dialog, padding=(18, 8, 18, 16))
        footer.grid(row=2, column=0, sticky="ew")
        ttk.Button(footer, text="Close", command=dialog.destroy).pack(side="right")
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.focus_set()

    def _update_preview(self, root: str) -> None:
        from PIL import Image, ImageTk
        folder = Path(root)
        # Prefer a readable 60-second preview for long recordings.
        images = list(folder.rglob("state_sequence_first_60s.png")) if folder.exists() else []
        if not images and folder.exists():
            images = list(folder.rglob("state_sequence.png"))
        if not images:
            self.preview_image.configure(image="", text="No state sequence yet")
            self._preview_photo = None
            return
        try:
            with Image.open(images[0]) as original:
                pic = original.copy()
            pic.thumbnail((700, 190))
            self._preview_photo = ImageTk.PhotoImage(pic)
            self.preview_image.configure(image=self._preview_photo, text="")
        except Exception as exc:
            self.preview_image.configure(image="", text=f"Preview unavailable: {exc}")

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
        self.metrics.set_metrics([("Status","Stopped"),("Runtime","—"),("Proto-states","—"),("Metastable States","—")])
        self._set_idle_state()

    def _request_exit(self) -> None:
        if self.backend.is_running:
            if not messagebox.askyesno("Exit NeuroSeq", "NeuroSeq is still running. Stop it and exit?"): return
            self.backend.stop()
        self.after(150, self.destroy)

    def _refresh_output_files(self) -> None:
        folder=self._result.get("output_root") if self._result else str(self.vars["output_root"].get()).strip()
        if folder:
            self.output_files.load_from_folder(folder)
            self._update_preview(folder)

    def _number(self, key: str, cast):
        value = str(self.vars[key].get()).strip()
        try:
            return cast(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {key}: {value!r}") from exc

    def _collect_config(self) -> NeuroSeqGUIConfig:
        return NeuroSeqGUIConfig(
            data_folder=str(self.vars["data_folder"].get()).strip(),
            output_root=str(self.vars["output_root"].get()).strip(),
            start_time=self._number("start_time", float),
            total_duration=self._number("total_duration", float),
            T=self._number("T", float),
            auto_gamma_min=self._number("auto_gamma_min", float),
            auto_gamma_max=self._number("auto_gamma_max", float),
            auto_gamma_step=self._number("auto_gamma_step", float),
            seeds_per_gamma=self._number("seeds_per_gamma", int),
            seed_for_states=self._number("seed_for_states", int),
            persistence_tolerance_bins=self._number("persistence_tolerance_bins", int),
            consensus_cp_band_tolerance_bins=self._number("consensus_cp_band_tolerance_bins", int),
            meso_seed=self._number("meso_seed", int),
            do_graph=bool(self.vars["do_graph"].get()),
        )

    def _apply_config(self, cfg: NeuroSeqGUIConfig) -> None:
        for key, variable in self.vars.items():
            value = getattr(cfg, key)
            variable.set("" if value is None else value)

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
            ("Proto-states", str(result.get("n_micro_states_total", "—"))),
            ("Metastable States", family_text),
        ])
        self._append_console("NeuroSeq completed successfully.")
        self.output_files.load_from_folder(result.get("output_root", ""))
        self._update_preview(result.get("output_root", ""))
        self._set_idle_state()

    def _run_error(self, message: str, details: str) -> None:
        self._append_console(details)
        self.metrics.set_metrics([
            ("Status", "Failed"),
            ("Runtime", "—"),
            ("Proto-states", "—"),
            ("Metastable States", "—"),
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
            ("Proto-states", "—"),
            ("Metastable States", "—"),
        ])

    def _set_idle_state(self) -> None:
        self.run_button.configure(state="normal", text="Run NeuroSeq")
        self.open_output_button.configure(state="normal")
        self.save_result_button.configure(state="normal" if self._result else "disabled")
        self.stop_button.configure(state="disabled", text="Stop")
