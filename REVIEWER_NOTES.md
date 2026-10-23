# Reviewer reproducibility notes

This package is a source-only snapshot of the supplied NeuroSeq project, cleaned of development environment and Git internals. The scientific execution statements were not modified as part of documentation preparation.

## Intended scope

- Full-population spike-train analysis with Micro and Meso.
- No Brain Music or gustatory-cortex-specific stimulus/licking logic.
- No random neuron sampling or behavioral annotation requirements.
- GUI and standalone entry points share `orchestrate_micro_meso.run_pipeline`.

## Validation performed during preparation

- Static compilation of included Python modules.
- Audit of third-party import names against `requirements.txt`.
- ZIP integrity test.

## Not established by these checks

- Full end-to-end execution on the original experimental datasets.
- Numerical equality to prior NeuroSeq revisions.
- Reproducibility on Windows, macOS, and Linux.
- Whether the dependency versions chosen by a future installation reproduce the manuscript figures.

Before submission, provide at least one shareable spike-train example and its expected outputs, and record a tested dependency snapshot. Avoid shipping local absolute paths as runnable defaults: edit `run_neuroseq.py` for the reviewer dataset.
