"""Persist GUI configuration as JSON and format runtime displays.

Author
------
Greg Owanga
Gregium Theory
"""

# GUI/gui_settings.py
from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .neuroseq_backend import NeuroSeqGUIConfig


# The same defaults are used when the user resets the GUI settings.
DEFAULTS = NeuroSeqGUIConfig(data_folder="", output_root="")


def save_settings(settings: NeuroSeqGUIConfig, path: str | Path) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    return str(out)


def load_settings(path: str | Path) -> NeuroSeqGUIConfig:
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(NeuroSeqGUIConfig)}
    clean = {key: value for key, value in raw.items() if key in allowed}
    return NeuroSeqGUIConfig(**clean)


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "—"

    minutes, sec = divmod(value, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {sec:04.1f}s"
    if minutes:
        return f"{minutes:d}m {sec:04.1f}s"
    return f"{sec:.1f}s"
