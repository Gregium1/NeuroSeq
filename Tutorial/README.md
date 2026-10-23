# NeuroSeq — Written Tutorial

This tutorial accompanies the **full-population** NeuroSeq Python GUI. It uses a small synthetic recording and explains the scientific meaning of the outputs. No Brain Music, stimulus annotations, or neuron sampling are required.

**Follow these chapters in order:**

1. [Getting started](01_Getting_Started.md) — installation, files, and a reproducible example.
2. [GUI walkthrough](02_GUI_Walkthrough.md) — configure and run a full recording.
3. [Micro explained](03_Understanding_Micro.md) — temporal networks, gamma sweep, and consensus boundaries.
4. [Meso explained](04_Understanding_Meso.md) — Proto-states, firing rates, affinity, and Metastable States.
5. [Output interpretation](05_Interpreting_Outputs.md) — where outputs live and what they mean.
6. [Reproducibility and reviewer checklist](06_Reproducibility.md) — verify your run.

The bundled [`Example_Data/`](Example_Data/) contains **30 synthetic neurons recorded for 60 seconds**, with five 12-second generator regimes. `ground_truth_states.csv` documents the generator and **must not be placed in the spike-train input folder when using a different input loader or copied as an `SU_*.csv`**. NeuroSeq reads only the `SU_*.csv` files. The synthetic example is instructional, not a validation claim: inferred boundaries and state labels are not guaranteed to equal the generator labels.

## Screenshots and video

The `Images/` directory contains **screenshots supplied from a real NeuroSeq GUI run**, illustrating the interface, Micro diagnostics, and Meso results. The screenshots are **not** claimed to be generated from the bundled `Example_Data/` recording. A video walkthrough may be linked later, but all steps are documented in text.

[Return to the main README](../README.md).
