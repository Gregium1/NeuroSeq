# 6. Reproducibility and reviewer checklist

## Before running

- [ ] Use a supported Python installation and create a clean `.venv`.
- [ ] Run `python -m pip install -r requirements.txt` and `python -m pip check`.
- [ ] Verify that `Tutorial/Example_Data/` contains 20 `SU_*.csv` files and that the selected interval is 0–60 seconds.
- [ ] Use an empty output parent directory for each run.
- [ ] Record the repository revision and dependency versions (`python -m pip freeze > installed_versions.txt`).

## After running

- [ ] Confirm that `NeuroSeq/run_summary.json` and `NeuroSeq/runtime.csv` exist.
- [ ] Confirm `run_summary.json` reports `n_neurons: 20`, `T: 0.5`, `start_time: 0.0`, `total_duration: 60.0`.
- [ ] Inspect `Micro/micro_ctx.json` for the representative gamma and Micro outputs.
- [ ] Locate consensus critical points and inspect the temporal segmentation.
- [ ] Inspect Meso `state_vectors.csv` and final state labels. Do not assume that five generator regimes must yield five inferred Metastable States.
- [ ] Save the complete output directory with the software revision used.

## GUI versus standalone comparison

To compare execution paths, use identical input data, `T`, duration, gamma sweep, seeds, and environment. Set `data_folder` and `output_root` in `run_neuroseq.py` and run `python run_neuroseq.py`. Compare **numerical outputs** (critical-point times, representative gamma, Proto-state vectors, similarity matrices, final labels). Runtime may vary with thread settings and hardware even when the mathematics agrees.

## Common problems

- `ModuleNotFoundError`: install the requirements inside the activated environment, not into a different Python interpreter.
- `No SU_*.csv files`: select the actual folder containing the spike timestamp CSVs.
- Unexpected nested output path: the orchestrator appends `/NeuroSeq` to the chosen output parent.
- Slow gamma sweep: 81 gamma values and three Louvain seeds are evaluated by default; wait for the completion message and inspect the console.
- `tight_layout()` warning: a Matplotlib layout warning may appear; inspect the exported figure, but it is not evidence that state discovery failed.

## Release quality

This bundled example is generated from a documented synthetic process, but no reference output files are claimed here because an end-to-end run has not been completed in this packaging environment. Before publishing to reviewers, run the example and archive its genuine results; optionally add screenshots captured from that run to `Tutorial/Images/`.

[Return to tutorial home](README.md).
