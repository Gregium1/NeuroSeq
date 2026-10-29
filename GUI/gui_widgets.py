"""Provide reusable Tkinter widgets for settings, progress and output browsing.

Author
------
Greg Owanga
Gregium Theory
"""

# GUI/gui_widgets.py
from __future__ import annotations

import tkinter as tk
import os
from pathlib import Path
from tkinter import filedialog, ttk

from .gui_theme import Palette


class PathPicker(ttk.Frame):
    def __init__(
        self,
        master,
        *,
        label: str,
        variable: tk.StringVar,
        mode: str = "folder",
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(master)
        self.variable = variable
        self.mode = mode
        self.filetypes = filetypes or [("All files", "*.*")]
        self.columnconfigure(0, weight=1)
        ttk.Label(self, text=label).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Entry(self, textvariable=variable).grid(row=1, column=0, sticky="ew", pady=(3, 0))
        ttk.Button(self, text="Browse…", command=self._browse).grid(row=1, column=1, padx=(7, 0), pady=(3, 0))

    def _browse(self) -> None:
        current = str(self.variable.get()).strip()
        if self.mode == "folder":
            value = filedialog.askdirectory(initialdir=current if current and Path(current).is_dir() else os.getcwd())
        else:
            value = filedialog.askopenfilename(initialdir=current if current and Path(current).is_dir() else os.getcwd(), filetypes=self.filetypes)
        if value:
            self.variable.set(value)


class LabeledEntry(ttk.Frame):
    def __init__(self, master, *, label: str, variable: tk.Variable) -> None:
        super().__init__(master)
        self.columnconfigure(1, weight=1)
        ttk.Label(self, text=label).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(self, textvariable=variable).grid(row=0, column=1, sticky="ew")


class CollapsibleSection(ttk.Frame):
    def __init__(self, master, *, title: str, expanded: bool = False) -> None:
        super().__init__(master)
        self.expanded = tk.BooleanVar(value=expanded)
        self.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            self,
            text=title,
            variable=self.expanded,
            command=self._toggle,
            style="Section.TCheckbutton",
        ).grid(row=0, column=0, sticky="ew")
        self.body = ttk.Frame(self, style="PurpleCard.TFrame", padding=10)
        self.body.columnconfigure(0, weight=1)
        if expanded:
            self.body.grid(row=1, column=0, sticky="ew")

    def _toggle(self) -> None:
        if self.expanded.get():
            self.body.grid(row=1, column=0, sticky="ew")
        else:
            self.body.grid_remove()


class ConsoleBox(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.text = tk.Text(
            self,
            wrap="word",
            height=12,
            font=("Consolas", 9),
            state="disabled",
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
        )
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def apply_palette(self, palette: Palette) -> None:
        self.text.configure(
            background=palette.console_bg,
            foreground=palette.console_fg,
            insertbackground=palette.console_fg,
            selectbackground=palette.purple,
        )

    def append(self, message: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", message.rstrip() + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class ProgressChecklist(ttk.Frame):
    STAGES = (
        ("validation", "Validate input"),
        ("prepare", "Prepare spike data"),
        ("networks", "Build Micro-Epoch networks"),
        ("gamma", "Sweep γ Louvain resolution"),
        ("critical_points", "Detect critical points"),
        ("micro_states", "Construct Proto-states"),
        ("similarities", "Compute Proto-state similarities"),
        ("meso", "Discover States"),
        ("outputs", "Generate outputs"),
    )

    def __init__(self, master) -> None:
        super().__init__(master, style="PurpleCard.TFrame", padding=10)
        self.labels: dict[str, ttk.Label] = {}
        for row, (key, text) in enumerate(self.STAGES):
            label = ttk.Label(self, text=f"○  {text}", style="Checklist.TLabel")
            label.grid(row=row, column=0, sticky="w", pady=1)
            self.labels[key] = label

    def reset(self) -> None:
        for _key, label in self.labels.items():
            label.configure(text=label.cget("text").replace("✓", "○").replace("▶", "○"))

    def update_stage(self, active_key: str) -> None:
        keys = [key for key, _ in self.STAGES]
        try:
            active_index = keys.index(active_key)
        except ValueError:
            active_index = len(keys)

        for index, (key, text) in enumerate(self.STAGES):
            if index < active_index or active_key == "complete":
                marker = "✓"
            elif index == active_index:
                marker = "▶"
            else:
                marker = "○"
            self.labels[key].configure(text=f"{marker}  {text}")


class MetricsPanel(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.values: dict[str, ttk.Label] = {}
        self.columnconfigure((0, 1, 2, 3), weight=1)

    def set_metrics(self, metrics: list[tuple[str, str]]) -> None:
        for child in self.winfo_children():
            child.destroy()
        self.values.clear()

        for col, (name, value) in enumerate(metrics):
            card = ttk.Frame(self, style="Metric.TFrame", padding=(12, 10))
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 4, 0))
            ttk.Label(card, text=name, style="MetricName.TLabel").pack(anchor="w")
            label = ttk.Label(card, text=value, style="MetricValue.TLabel")
            label.pack(anchor="w", pady=(4, 0))
            self.values[name] = label


class OutputFiles(ttk.Frame):
    def __init__(self, master, *, open_path) -> None:
        super().__init__(master)
        self.open_path = open_path
        self.rowconfigure(0, weight=1); self.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(self, columns=("type", "size"), show="tree headings")
        self.tree.heading("#0", text="File"); self.tree.heading("type", text="Type"); self.tree.heading("size", text="Size")
        self.tree.column("#0", width=520); self.tree.column("type", width=90, anchor="center"); self.tree.column("size", width=100, anchor="e")
        scroll=ttk.Scrollbar(self,orient="vertical",command=self.tree.yview); self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0,column=0,sticky="nsew"); scroll.grid(row=0,column=1,sticky="ns")
        self.tree.bind("<Double-1>", self._open_selected)
    def load_from_folder(self, folder):
        root=Path(folder).expanduser()
        for item in self.tree.get_children(): self.tree.delete(item)
        if not root.exists(): return 0
        allowed={".csv",".json",".txt",".png",".jpg",".jpeg",".svg",".gexf",".xlsx"}
        visible = {"state_sequence", "raster_with_states", "spike_raster", "similarity_graph_kmeans"}
        paths=sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed
                     and (p.stem in visible or p.stem.startswith("spike_raster_allSUs_") or (p.parent.name == "Raster_60s" and p.stem.startswith("raster_with_states_") and p.stem.rsplit("_",1)[-1].isdigit())))
        for p in paths:
            size=p.stat().st_size; size_text=f"{size/1024:.1f} KB" if size<1024**2 else f"{size/1024**2:.1f} MB"
            self.tree.insert("","end",iid=str(p),text=str(p.relative_to(root)),values=(p.suffix[1:].upper(),size_text))
        return len(paths)
    def _open_selected(self, _event=None):
        selected=self.tree.selection()
        if selected: self.open_path(Path(selected[0]))
