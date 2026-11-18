
## Authors and Contributors

- **[Greg Owanga](https://github.com/Gregium1)** — Method development, algorithm design, implementation, and software development.
- **[Richard Bertram](https://github.com/rbertram)** — Scientific supervision, mathematical guidance, and research collaborator.
- **[Roberto Vincis](https://github.com/VincisLab)** — Experimental neuroscience, data collection, and research collaborator

# NeuroSeq

**A multi-scale functional-network toolbox for Proto-state segmentation and Metastable State discovery from neuronal spike trains**

NeuroSeq identifies changes in neuronal population organization across time and groups related temporal patterns into **Metastable States**. It uses a two-stage **Micro–Meso** workflow and provides both a Python desktop GUI and a standalone Python launcher. The input is spike timing data; stimulus annotations, licking events, behavioral labels, and a user-specified number of states are **not required**.

**Scope:** NeuroSeq analyzes the full recorded neuronal population through its Micro–Meso pipeline. It does not require stimulus annotations or a predefined number of Metastable States.

## Contents

- [Installation](#installation)
- [Input format](#input-format)
- [Running the GUI](#running-the-gui)
- [Running from Python](#running-from-python)
- [Scientific workflow](#scientific-workflow)
- [Parameters](#parameters)
- [Outputs](#outputs)
- [Reproducibility and verification](#reproducibility-and-verification)
- [Troubleshooting](#troubleshooting)
- [Architecture and source guide](#architecture-and-source-guide)
- [License and attribution](#license-and-attribution)

## Step-by-step tutorial

A complete written guide, GUI walkthrough, synthetic example recording, and explanation of Micro/Meso output files are provided in **[Tutorial/README.md](Tutorial/README.md)**. The tutorial does not require a video.

## Installation

**Prerequisites:** A supported Python 3 installation, `pip`, and Tkinter for the GUI. Python **3.12** is used in the documented Windows environment, but this release has **not** been tested on all Python versions or operating systems. Python 3.12 is a sensible starting point for reproducing that environment. The exact package versions used in prior runs may differ from a fresh installation.

Download or clone the repository and open a terminal in its root directory (the directory containing `requirements.txt`).

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m GUI.launcher
```

If the PowerShell execution policy blocks activation, use `.venv\Scripts\python.exe` instead of activating the environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m GUI.launcher
```

On some Linux distributions, Tkinter must be installed separately through the system package manager (often `python3-tk`). The GUI requires a graphical desktop session; headless servers can use the Python launcher instead. macOS may require a Python distribution with working Tk support.

**Important:** The import `community` is supplied by the `python-louvain` package. Do not replace it with an unrelated package named `community`.

To check core imports before running:

```bash
python -c "import numpy, pandas, scipy, sklearn, networkx, community, matplotlib, PIL; print('Scientific dependencies imported successfully')"
```

To preserve the exact dependency versions of a successful run:

```bash
python -m pip freeze > environment_used_for_review.txt
```

That file is a **local environment snapshot**, not a cross-platform guarantee. Keep it alongside the reviewer run record if exact replication is needed.

## Input format

The selected input folder must contain one CSV per neuron, with filenames matching `SU_<integer>.csv`:

```text
recording/
├── SU_1.csv
├── SU_2.csv
├── SU_3.csv
└── SU_4.csv
```

Each file must contain numeric spike times **in seconds**, measured on a common recording timeline. The exact column/header conventions are governed by the loader implementations in `Micro/micro_counts.py` and `Meso/meso_proto_windows.py`; inspect or test a small example before converting large datasets. Spike times should lie within the chosen analysis interval. **Do not** provide firing rates or binned counts in place of spike timestamps.

NeuroSeq analyzes **all** matching neuron files in the selected folder. The present release does not provide random neuron sampling. No ground-truth state sequence or stimulus-delivery file is needed for normal analysis.

## Running the GUI

From the repository root:

```bash
python -m GUI.launcher
```

The **Recording folder** and **Output folder** fields are blank when the GUI opens. Select each using **Browse**; the file chooser starts in the directory from which NeuroSeq was launched. Choose a recording folder containing `SU_*.csv` files and a separate parent folder for the results.

Set **Start time (s)** and **Duration (s)** under *Full recording*. Expand **Core parameters** to set the Micro-Epoch duration **T**, **Gamma minimum**, **Gamma maximum**, and **Gamma step**; collapse it to save space. **Load settings** imports previously saved settings, **Save settings** exports them, and **Reset** restores defaults and clears folder selections. The **Help** button explains these controls.

Click **Run NeuroSeq** and follow the stages in **Run progress**. After completion, the bottom status cards show runtime, number of Proto-states, and number of Metastable States. The **Output files** panel lists the main result files; double-click a file to open it. **Refresh files** reloads that list from disk—it does not clear or delete existing results. **Open Output Folder** provides access to all detailed outputs, including critical-point diagnostics. **Save Run Summary** exports a run summary. **Stop** requests cancellation; **Exit** closes the GUI. The **Light/Dark** control changes the interface theme.

The GUI also shows a **State sequence preview**. For recordings longer than 60 seconds, the preview displays the first 60 seconds of the *analyzed interval*, using the global state labels and colors. The full-length figure is still saved to disk. The GUI does not include a Live Console or a separate Run Parameters panel.

The GUI launches the scientific pipeline in a worker process so that its interface can remain responsive. **The worker calls the same orchestrator as the standalone launcher**, and progress messages do not rerun Micro or Meso. Runtime depends on the number of neurons, recording duration, number of Micro-Epochs, gamma values, seeds, and hardware.

### Recording duration and visualization

If the requested end time exceeds the last **observed spike timestamp**, NeuroSeq limits the analysis to the available observed interval; shorter requested intervals remain unchanged. For example, a request for 70 seconds starting at 0 seconds is reduced to 60 seconds if the last observed spike is at 60 seconds. A start time beyond the observed data endpoint is invalid. **Important:** Spike-only CSVs cannot establish the true acquisition stop time: a silent trailing period is indistinguishable from unrecorded time. The requested and effective durations are recorded in `run_summary.json`.

The full-recording `raster_with_states.png` is accompanied by consecutive 60-second raster views in `Meso/Meso_Sequence/Primary_Output/Raster_60s/`, named `raster_with_states_1.png`, `raster_with_states_2.png`, and so on (the final segment may be shorter). These are display-only slices of the **same** state assignments, using absolute recording times and consistent state colors; no independent state discovery occurs.

Principal outputs include `state_sequence.png`, `raster_with_states.png`, `similarity_graph_kmeans.png`, and the Micro spike raster. Figures retain SVG versions. The previous `Materialized` directory is now `Primary_Output`.

## Running from Python

Edit `run_neuroseq.py`, particularly:

```python
cfg = RunConfig(
    data_folder="/path/to/folder/with/SU_csv_files",
    output_root="/path/to/output_parent",
    start_time=0.0,
    total_duration=60.0,
    T=0.5,

    # Default Values
    auto_gamma_min=1.07,
    auto_gamma_max=1.15,
    auto_gamma_step=0.001,
    seeds_per_gamma=3,
    persistence_tolerance_bins=1,
    consensus_cp_band_tolerance_bins=0,
    seed_for_states=42,
    meso_seed=42,
    do_graph=True,
)
```

Then run from the project root:

```bash
python run_neuroseq.py
```

The code above is a **configuration example**. Check the paths in your local `run_neuroseq.py` and edit them for your machine before running.

## Scientific workflow

### Micro: identify temporal critical points and Proto-state boundaries

1. The selected recording interval is divided into Micro-Epochs of duration `T` seconds.
2. Spike counts are computed for each neuron in each Micro-Epoch.
3. A weighted neuronal network is constructed for each Micro-Epoch. For counts `c_i` and `c_j`, the implemented linear affinity is `w_ij = max(0, 100 − |c_i − c_j| / T)` (subject to the implementation's graph-building and threshold policies).
4. Louvain partitions are computed for a configured sweep of resolution parameter `gamma`, with multiple seeds per value.
5. Adjacent Micro-Epoch partitions are compared using normalized mutual information (NMI).
6. Candidate critical points are identified using the implemented NMI thresholding and persistence/consensus procedures.
7. Consensus temporal boundaries are collected across gamma values; a representative gamma is selected using the maximum-critical-point plateau rule.
8. The Micro stage writes its outputs and `Micro/micro_ctx.json` for Meso.

**Micro does not construct final Proto-state firing-rate vectors or pairwise Proto-state matrices.** Those operations belong to Meso in this architecture. The representative gamma controls within-Proto-state neuronal community detection; it does not independently redefine the consensus temporal boundaries.

### Meso: construct Proto-state representations and identify Metastable States

1. Consensus Micro critical points delimit the Proto-state intervals.
2. Within each Proto-state, neuronal activity is summarized and a Louvain partition is computed using the representative gamma.
3. Each Proto-state receives a population firing-rate vector; a component is the neuron's spike count in the interval divided by the interval duration.
4. Pairwise comparisons are computed using neuronal partition similarity (NMI) and firing-rate-vector distance.
5. Pairwise Meso affinity is computed using the **multiplicative exponential Gaussian-kernel S-score**, combining normalized firing-rate distance and neuronal-partition NMI; see the formula below and `Meso/meso_score.py`. The pairwise scores form a symmetric Proto-state affinity matrix with diagonal entries fixed at 1.

\[
S_{ij}=\exp\!\left(-\frac{\widetilde D_{ij}^{\,2}}{2\sigma_D^2}\right)
       \exp\!\left(-\frac{(1-\mathrm{NMI}_{ij})^2}{2\sigma_{\mathrm{NMI}}^2}\right),
\qquad \widetilde D_{ij}=\frac{D_{ij}}{D_{\max}}.
\]

Here, `D_ij` is the population firing-rate-vector distance, `D_max` is the maximum finite pairwise distance, and `NMI_ij` compares neuronal Louvain partitions. Kernel widths `sigma_D` and `sigma_NMI` are computed from the observed pairwise values by `safe_scale` in `Meso/meso_utils.py` (fallback `0.25`). The implementation clips normalized distances to `[0, 1]` and handles nonfinite values. Each pairwise S-score is placed into a **symmetric Proto-state affinity matrix** with diagonal entries fixed at 1. This matrix is used in subsequent spectral candidate selection and Metastable State discovery.

6. Spectral candidate selection combines eigengap and modularity diagnostics to select a number of Metastable States without a user-specified `K`.
7. Temporal labels, tables, plots, and diagnostics are generated.

The scientific method and parameter definitions are controlled by the **source code**. This README explains the implemented workflow but does not replace the manuscript's formal methods or validation experiments.

## Parameters

| Parameter | Meaning | Current default |
|---|---|---:|
| `start_time` | First analyzed time in seconds | 0.0 |
| `total_duration` | Analyzed interval duration in seconds | 60.0 |
| `T` | Micro-Epoch duration in seconds | 0.5 |
| `auto_gamma_min` | First gamma in sweep | 1.07 |
| `auto_gamma_max` | Last gamma in sweep | 1.15 |
| `auto_gamma_step` | Gamma increment | 0.001 |
| `seeds_per_gamma` | Louvain seeds evaluated per gamma | 3 |
| `persistence_tolerance_bins` | Critical-point persistence tolerance | 1 |
| `consensus_cp_band_tolerance_bins` | Consensus boundary band tolerance | 0 |
| `seed_for_states` | Seed for within-Proto-state Louvain | 42 |
| `meso_seed` | Meso spectral random seed | 42 |
| `do_graph` | Whether Meso figures are generated | True |

The default sweep from 1.07 to 1.15 in increments of 0.001 contains **81 gamma values** (including both endpoints). The gamma settings are available in the GUI under the expandable **Core parameters** section. This version uses a fixed `T`; it should not be described as adaptive-T.

## Outputs

The orchestrator creates a `NeuroSeq/` directory **inside** the chosen output parent:

```text
output_parent/
└── NeuroSeq/
    ├── Micro/
    │   └── micro_ctx.json
    ├── Auto_Micro_Gamma/
    │   └── proto_state_louvain_gamma_selection.csv
    ├── Meso/
    │   └── Meso_Sequence/
    │       ├── Primary_Output/
    │       │   ├── state_sequence.png
    │       │   ├── raster_with_states.png
    │       │   └── Raster_60s/
    │       │       ├── raster_with_states_1.png
    │       │       └── ...
    │       └── Similarity_Graphs/
    │           └── similarity_graph_kmeans.png
    ├── run_summary.json
    └── runtime.csv
```

Additional nested output folders are generated by the Micro and Meso modules. The **Proto-States** outputs include `proto_state_vectors.csv`, `proto_state_louvain_clusters.csv`, `proto_state_x_cluster.csv`, `proto_state_louvain_nmi.csv`, and `proto_state_louvain_nmi_long.csv`. Their Proto-state identifiers use names such as `proto_state_1`. The **Similarity_Graphs** folder includes `similarity_graph_edges_for_plot.csv` and `similarity_graph_kmeans.png`/`.svg`. The Meso sequence diagnostics include `spectral_similarity_matrix.csv` and `spectral_state_labels.csv`; the latter uses the `proto_state_index` column. Main figures are also available in SVG where produced.

 Output file availability can depend on the dataset, selected settings, and plotting paths. `run_summary.json` records the pipeline's returned outputs, neuron count, representative gamma, and runtime; `runtime.csv` provides a compact runtime summary. Do not interpret output file presence alone as a validation of state recovery.

**Output-folder caution:** Select an output parent rather than the input spike folder. Existing files in a reused output directory may be overwritten; use a fresh directory for independent reviewer runs.

## Reproducibility and verification

- Record the exact NeuroSeq source revision, Python version, installed package versions, platform, and parameter values.
- Preserve the original input spike-train CSVs and the full output directory for each experiment.
- Compare GUI and command-line runs using **the same input, settings, seeds, and environment**. Wall-clock runtimes can differ even if scientific outputs agree.
- Compare consensus critical-point times, representative gamma, Proto-state boundaries, firing-rate vectors, Meso affinity matrices, and final Metastable State labels—not just the total number of states.
- Cluster labels are arbitrary identifiers; a label permutation alone does not imply a different partition.
- A successful run and valid figures do not by themselves establish numerical equivalence or cross-platform reliability. Perform a scientific regression test before publication.

## Troubleshooting

| Symptom | Suggested check |
|---|---|
| `ModuleNotFoundError: community` | Install `python-louvain` in the active Python environment. |
| Tkinter cannot be imported | Install a Python/Tk distribution appropriate for the operating system. |
| No `SU_*.csv` files found | Check the selected folder and exact filename pattern. |
| GUI seems stalled | Check the Run Progress stage, whether `run_summary.json` and `runtime.csv` were written, and any Python traceback in the launching terminal. Micro can dominate runtime. |
| GUI slower than command line | Compare numerical-thread environment variables, Python executable, dataset, gamma settings, and seeds. |
| Unexpected output nesting | Remember that the orchestrator appends `NeuroSeq/` to `output_root`. |
| `tight_layout` Matplotlib warning | Inspect the affected figure; the warning does not necessarily indicate failed scientific computation. |
| Output differs between machines | Compare package versions, seeds, input file contents, and numeric backends before concluding there is a methodological difference. |

## Architecture and source guide

| File or package | Responsibility |
|---|---|
| `run_neuroseq.py` | Standalone example configuration and launcher |
| `orchestrate_micro_meso.py` | One full-population Micro→Meso execution and summary files |
| `GUI/launcher.py` | Tkinter entry point and multiprocessing setup |
| `GUI/neuroseq_gui.py` | User interface, parameters, and status display |
| `GUI/neuroseq_backend.py` | Worker process, progress, cancellation, and result handling |
| `Micro/micro_counts.py` | Spike counts per Micro-Epoch |
| `Micro/micro_graphs.py` | Micro-Epoch neuronal similarity networks |
| `Micro/micro_louvain.py` | Louvain partitions |
| `Micro/micro_adjacent_nmi.py` | Adjacent-window partition similarity |
| `Micro/micro_critical_points.py` | Critical-point detection |
| `Micro/micro_gamma_selection.py` | Representative gamma selection |
| `Micro/auto_select_micro_gamma.py` | Automatic Micro orchestration and consensus context |
| `Meso/meso_proto_windows.py` | Proto-state interval construction and Louvain |
| `Meso/meso_proto_vectors.py` | Proto-state firing-rate representations |
| `Meso/meso_pairs.py` | Pairwise Proto-state comparison |
| `Meso/meso_score.py` | Multiplicative exponential Gaussian-kernel S-score and symmetric affinity matrix |
| `Meso/meso_spectral.py` | Spectral model selection and clustering |
| `Meso/run_meso.py` | Meso orchestration |


## License and attribution

See [LICENSE](LICENSE). Please cite the associated NeuroSeq manuscript when available. No manuscript DOI or publication citation is asserted in this README.


