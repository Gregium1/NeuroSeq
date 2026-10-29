# 1. Getting started

## What NeuroSeq does

NeuroSeq analyzes spike timestamps from all `SU_*.csv` files in a recording folder. **Micro** detects consensus critical points in evolving neuronal functional networks. **Meso** uses these boundaries to construct Proto-states and group related Proto-states into recurring **Metastable States**.

## Install and launch

From the repository root, in Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m GUI.launcher
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` for the first two steps, then the same `python -m ...` commands. Some Linux systems require the OS package `python3-tk`. Use the Python interpreter in the activated environment.

## Included example

`Tutorial/Example_Data/` contains 30 files, `SU_1.csv`–`SU_30.csv`. Each contains one spike timestamp per row in seconds, without a header. The loader accepts the supported one- or two-column spike formats, sorts numeric times, and ignores invalid values. `ground_truth_states.csv` is included only for external validation or illustration.

Select `Tutorial/Example_Data/` as the **Recording folder**. Select a separate, writable **Output folder** (for example `NeuroSeq_Example_Results/`); the application creates `NeuroSeq/` inside it. Do not select the recording folder as the output destination.

**Recording duration:** The GUI accepts a start time and requested duration. If the requested endpoint exceeds the latest observed spike timestamp, the analysis duration is capped to that timestamp. This is a practical fallback, **not a measurement of the true acquisition endpoint**

[Next: GUI walkthrough](02_GUI_Walkthrough.md).
