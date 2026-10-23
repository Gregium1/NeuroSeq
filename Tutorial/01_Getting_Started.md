# 1. Getting started

## What NeuroSeq analyzes

NeuroSeq takes spike timestamps for simultaneously recorded neurons, detects temporal change points (Micro), then constructs Proto-states and groups them into Metastable States (Meso). The current release analyzes **all** matching neurons; it does not use behavioral event files, stimulus labels.

## Install

Run these commands from the repository root. On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m GUI.launcher
```

On macOS/Linux, replace the first two commands with `python3 -m venv .venv` and `source .venv/bin/activate`, then run the remaining `python` commands. On Linux, Tkinter may require an OS package such as `python3-tk`.

## Example data

Use the included `Tutorial/Example_Data/` directory. It contains `SU_1.csv` through `SU_30.csv`, each with **one spike timestamp per row, in seconds, without a header**. The loader accepts a one-column file or a two-column file (reading the second column), converts numeric entries, drops invalid values, and sorts times. All neurons share a common 0–60-second clock.

The file `ground_truth_states.csv` records the synthetic generating regimes. 

## Prepare the run

Create a separate empty output parent folder, for example `NeuroSeq_Example_Results`. Do **not** select `Example_Data` as the output folder. Continue to [Chapter 2](02_GUI_Walkthrough.md).
