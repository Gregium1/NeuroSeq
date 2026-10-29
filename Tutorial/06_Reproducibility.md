# 6. Reproducibility and reviewer checklist

## Before running

- [ ] Create and activate a clean Python virtual environment; install `requirements.txt` and run `python -m pip check`.
- [ ] Confirm `Tutorial/Example_Data/` contains **30** `SU_*.csv` files, not 20.
- [ ] Select the spike-train folder as **Recording folder** and a separate, writable **Output folder**.
- [ ] For the example, use start `0`, requested duration `60` s, `T=0.5` s, gamma minimum `1.07`, maximum `1.15`, and step `0.001`.
- [ ] Save your GUI settings to JSON if you want to repeat them later. Record the Git revision and environment (`python -m pip freeze > installed_versions.txt`).
- [ ] Remember that the duration may be capped at the **last observed spike**, which is not guaranteed to be the acquisition end time.

## After running

- [ ] Confirm the GUI shows **Completed** and reports runtime, Proto-state count and Metastable State count.
- [ ] Confirm `<output parent>/NeuroSeq/run_summary.json` and `runtime.csv` exist.
- [ ] Confirm the neuron count is **30**, not 20, and inspect the recorded **effective** analysis duration rather than assuming it is exactly 60.0 seconds.
- [ ] Open `Micro/micro_ctx.json`, review the representative gamma, and inspect the consensus critical-point CSVs through **Open Output Folder**.
- [ ] Inspect `proto_state_vectors.csv`, `proto_state_louvain_nmi.csv`, `spectral_similarity_matrix.csv`, and final labels where generated.
- [ ] Inspect `state_sequence.png`, `raster_with_states.png`, and the numbered `Raster_60s/` figures.
- [ ] Save the complete output directory with the software revision. Do not treat the screenshots' **109 Proto-states / 10 Metastable States** as the expected result of the bundled example.

## GUI versus standalone execution

The GUI and standalone launcher should invoke the same scientific pipeline. To compare them, use the same data, effective interval, `T`, gamma range and step, seeds, dependency versions, and other configuration. If your project includes `run_neuroseq.py`, set its `data_folder` and `output_root` before invoking `python run_neuroseq.py`. Compare numerical outputs (consensus boundaries, gamma, Proto-state vectors, similarity matrices and final labels), not wall-clock runtime.

## Common questions

**Why are critical-point files missing from the GUI list?** They are not deleted; the GUI intentionally displays only primary results. Use **Open Output Folder** to inspect Micro diagnostics.

**What does Refresh files do?** It rereads the current output folder and updates the visible file list. It does **not** clear or delete existing files.

**Why did the effective duration change?** A request past the last observed spike is shortened. Without acquisition metadata, the last spike is only a proxy for the recording endpoint.

**Why do I see multiple numbered raster figures?** Long intervals are split into consecutive 60-second *views* for readability, not into separate scientific runs.

**What if the GUI seems stuck?** Allow time for the gamma sweep and output generation; check whether summary files are written. If an exception appears in the launching terminal, save the traceback for troubleshooting. A completed numerical output file alone does not prove the worker returned successfully.

**What if a dependency is missing?** Activate the intended virtual environment, install `requirements.txt`, and check that `python` refers to that environment.

[Return to tutorial home](README.md).
