# NeuroSeq — Tutorial

This tutorial explains how to run the full-population NeuroSeq GUI, interpret its outputs, and inspect the scientific diagnostics. Follow the six chapters in order:

1. [Getting started](01_Getting_Started.md) — installation and the supplied 30-neuron example.
2. [GUI walkthrough](02_GUI_Walkthrough.md) — folders, duration, parameters, settings, progress, Help, and results.
3. [Understanding Micro](03_Understanding_Micro.md) — Micro-Epochs, gamma sweep, NMI, and consensus critical points.
4. [Understanding Meso](04_Understanding_Meso.md) — Proto-state vectors, the exponential S-score, and Metastable States.
5. [Interpreting outputs](05_Interpreting_Outputs.md) — the output directory, figures, CSVs, and 60-second raster views.
6. [Reproducibility](06_Reproducibility.md) — repeatable runs and troubleshooting.

The bundled [`Example_Data/`](Example_Data/) contains **30 synthetic spike trains spanning a nominal 60-second recording**, named `SU_1.csv` through `SU_30.csv`. `ground_truth_states.csv` contains the simulation's generating state labels and **is not used by NeuroSeq for inference**.

## About the screenshots

The screenshots in [`Images/`](Images/) illustrate a **completed analysis**. Screenshots reflect the updated GUI, including the filtered output list, compact state-sequence preview, and light/dark themes. Micro critical-point diagnostics are still generated and can be opened with **Open Output Folder**, even though they are not listed in the simplified GUI output browser.

[Return to the main README](../README.md).
