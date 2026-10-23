# Development and contribution notes

NeuroSeq separates temporal boundary detection (Micro) from Proto-state construction and clustering (Meso). Keep this boundary intact when modifying the code.

## Scientific changes

Any change to spike counts, network weights, Louvain selection, NMI thresholding, critical-point consensus, Proto-state vectors, affinity, or spectral clustering is a **scientific algorithm change**. Document the reason, changed equations, parameter defaults, and validation against a fixed reference dataset. Do not introduce additional scientific passes for GUI progress updates.

## Code quality

Document each public function's inputs, units, returned objects, edge cases, and randomness. Explain why non-obvious transformations are performed. Prefer explicit type hints, stable output schemas, and deterministic seeds. Keep file paths portable and never commit personal datasets, generated outputs, or environments.

## Suggested tests

1. Input loader: empty, malformed, and valid `SU_*.csv` data.
2. Micro: known fixed-duration synthetic transition example, gamma sweep, and consensus CPs.
3. Meso: Proto-state vectors and pairwise matrices with known neuron count.
4. Pipeline: exactly one Micro and one Meso call per execution.
5. GUI: same scientific results as standalone for identical configuration.
6. Repeated run: compare scientific outputs while allowing runtime variation.
