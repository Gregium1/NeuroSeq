# 4. Understanding Meso

Meso uses the consensus critical points and representative gamma supplied by Micro. It constructs **Proto-states** and groups related Proto-states into **Metastable States**, without requiring the user to specify the number of states.

## Constructing Proto-states

Consensus critical points divide the analyzed interval into consecutive Proto-state intervals. Within each interval, Meso computes a Louvain partition of neurons and a population firing-rate vector. For neuron `i` in Proto-state `p`,

\[
r_{p,i}=\frac{N_{p,i}}{t_p^{\mathrm{end}}-t_p^{\mathrm{start}}},
\]

where `N_{p,i}` is its spike count within the Proto-state. Rates are measured in Hz. Each Proto-state vector has a component for every recorded neuron.

## Comparing Proto-states

Meso compares Proto-states using (1) **NMI** between their Louvain neuronal partitions and (2) a distance `D_ij` between their population firing-rate vectors. The implemented affinity is the **multiplicative exponential Gaussian-kernel S-score**:

\[
S_{ij}=\exp\!\left(-\frac{\widetilde D_{ij}^{\,2}}{2\sigma_D^2}\right)
       \exp\!\left(-\frac{(1-\mathrm{NMI}_{ij})^2}{2\sigma_{\mathrm{NMI}}^2}\right),
\qquad \widetilde D_{ij}=\frac{D_{ij}}{D_{\max}}.
\]

`D_max` is the maximum finite pairwise distance. The widths `sigma_D` and `sigma_NMI` are estimated from the observed pairwise values with the code's `safe_scale` procedure (fallback `0.25`). The implementation clips normalized distances to `[0,1]`, handles nonfinite entries, and builds a symmetric matrix with diagonal entries equal to 1. **This is a product of two exponential factors, not an arithmetic average of NMI and distance similarity.**

The source modules `Meso/meso_score.py` and `Meso/meso_utils.py` define the numerical edge-case behavior. Spectral grouping and candidate-number selection use the resulting affinity, including eigengap and modularity information. The number of Metastable States is inferred rather than entered by the user.

## Understanding state labels

- A **Proto-state** is a contiguous time interval delimited by consensus boundaries.
- A **Metastable State** groups Proto-states with similar neuronal organization and firing-rate signatures. It may recur at different times.
- Letters such as A, B, and C identify inferred Metastable States. They are categorical identifiers, not a ranking or proof of biological equivalence.

The supplied `ground_truth_states.csv` records the synthetic generator's regimes; NeuroSeq does **not** read these labels during inference. Exact recovery of generating states is not guaranteed.

## Reading the state-labeled raster

![Raster with inferred Metastable States](Images/gui_raster_with_states.png)

The horizontal axis is time in seconds; the vertical axis is **Neuron**, labeled by neuron number rather than the `SU_` prefix. Black marks are spikes. Colored bands and letters indicate inferred Metastable States. The same letter at separated times means the algorithm assigned those intervals to the same inferred state. This is a visualization of labels over the spike train, not a plot of the S-score matrix.

[Next: Interpreting outputs](05_Interpreting_Outputs.md).
