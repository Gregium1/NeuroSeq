# NeuroSeq

**NeuroSeq** is a computational framework for identifying neural population state dynamics directly from extracellular spike train recordings. Rather than requiring the number of latent states to be specified beforehand, NeuroSeq automatically discovers neural state transitions and organizes them into hierarchical state families and subtypes.

The framework combines graph-theoretic community detection, information theory, and spectral graph analysis to characterize neural population activity across multiple temporal scales.

---

## Features

- Automatic detection of neural state transitions from spike trains
- Micro-scale analysis using graph-based community detection
- Meso-scale organization of neural states into state families
- Automatic estimation of the number of state families
- Hierarchical state representation (Families → Subtypes)
- Publication-quality visualizations
- Designed for both simulated and experimental neural recordings

---

## Repository Structure

```
NeuroSeq/
│
├── Micro/                     # Micro-scale state discovery
├── Meso/                      # Meso-scale family analysis
├── GUI/                       # Graphical user interface
├── BrainMusic/                # BrainMusic utilities
│
├── orchestrate_micro_meso.py  # Main NeuroSeq pipeline
├── build_neuroseq_windows.ps1 # Windows build script
├── requirements.txt
└── README.md
```

---

## Workflow

```
Spike Train Recording
          │
          ▼
    Micro Analysis
    ──────────────
    • Windowing
    • Community Detection
    • State Windows
    • State Similarity
          │
          ▼
     Meso Analysis
    ──────────────
    • Spectral Clustering
    • Family Discovery
    • Temporal Refinement
    • State Subtypes
          │
          ▼
 Final Neural State Hierarchy
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Gregium1/NeuroSeq.git
cd NeuroSeq
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```powershell
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running NeuroSeq

The complete pipeline can be executed with

```bash
python orchestrate_micro_meso.py
```

---

## Scientific Overview

NeuroSeq performs analysis at two complementary levels.

### Micro Analysis

The Micro module identifies candidate neural state intervals by analyzing changes in network community structure across time. Similarity between states is quantified using both community-based normalized mutual information and firing-rate similarity.

### Meso Analysis

The Meso module organizes Micro states into higher-order state families using spectral graph analysis. Temporal refinement produces robust family labels while preserving biologically meaningful state sequences.

---

## Outputs

NeuroSeq produces

- Neural state windows
- Community assignments
- Similarity matrices
- Spectral affinity matrices
- State families
- State subtypes
- Publication-quality figures

---

## Requirements

Python 3.11+

Major packages include

- NumPy
- SciPy
- pandas
- matplotlib
- networkx
- scikit-learn
- python-louvain

Additional dependencies are listed in `requirements.txt`.

---

## Citation

If you use NeuroSeq in your research, please cite the accompanying publication.

> Citation information will be added following publication.

---

## License

License information will be added prior to the public release of NeuroSeq.