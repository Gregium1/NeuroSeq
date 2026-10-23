# NeuroSeq GUI — single-run performance update

This GUI is a presentation and control layer for the existing optimized NeuroSeq pipeline. It **does not** calculate spike counts, Louvain partitions, NMI, Proto-state vectors, or spectral clustering itself.

## Installation

Back up the existing `GUI` directory, then replace it with this folder. Keep the optimized `Micro`, `Meso`, and `orchestrate_micro_meso.py` files already installed. Launch NeuroSeq using the project's existing GUI launcher (typically `python -m GUI.launcher` from the project root).

## Changes

- Exactly one `run_pipeline(cfg)` call is made per Run button press.
- The Windows worker process is spawned once; Micro and Meso run in that worker through the existing orchestrator.
- Numerical-library thread counts are set to 1 before the pipeline is imported in the spawned worker. This helps prevent BLAS/OpenMP oversubscription and makes the GUI's execution environment more comparable to a single-thread benchmark. If the standalone script uses a different thread configuration, performance can still differ.
- The live console prints the input path, parameters, Micro runtime, Meso runtime, and overall GUI worker runtime.
- Stage completion is reported at the actual Micro/Meso function boundaries. Some intermediate stages may remain unchanged until the pipeline prints recognizable progress lines; this does **not** mean the corresponding stage is stalled.
- The completion message sends only compact result metadata to the GUI; the full scientific results remain on disk in `run_summary.json`.
- Scientific algorithms in Micro/Meso are unchanged.

## Benchmarking

Compare the same input directory, `T`, gamma range/step, seeds per gamma, and `do_graph` option. Check that the output is not being concurrently synchronized or scanned by another application. The GUI's elapsed time includes process startup, whereas `runtime.csv` measures the pipeline's internal execution.

If the GUI remains substantially slower than the standalone script, send the live console's Micro and Meso timings and the matching `runtime.csv` to identify the bottleneck.
