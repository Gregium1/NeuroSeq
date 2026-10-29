# 3. Understanding Micro

Micro detects **when neuronal population organization changes**. It does not assign the final recurring Metastable State labels.

## Workflow

1. Restrict spikes to the effective analysis interval and count spikes in consecutive **Micro-Epochs** of duration `T`.
2. Construct a weighted functional network for each Micro-Epoch from population spike activity.
3. Apply Louvain community detection over the selected gamma resolution sweep, using multiple random seeds.
4. Compare neuronal partitions in adjacent Micro-Epochs with **normalized mutual information (NMI)**. Low adjacent NMI signals a larger change in partition structure, but does not by itself prove a biological transition.
5. Use a two-component Gaussian mixture (`gmm2`) in the critical-point detection procedure. Combine candidate boundaries across gamma values according to the implemented persistence and consensus tolerances.
6. Select a representative Louvain gamma and pass the consensus critical-point boundaries to Meso. These boundaries delimit **Proto-state intervals**.

For `T=0.5` s, a 60-second interval contains 120 Micro-Epochs. The default gamma sweep is `1.07`–`1.15` in increments of `0.001`, or 81 values. These values are adjustable in **Core parameters**; changing them can affect the segmentation.

## Where to find Micro outputs

Click **Open Output Folder**; the GUI's simplified **Output files** table does **not** list all diagnostic files. Useful saved files include:

| File | Meaning |
| --- | --- |
| `Micro/micro_ctx.json` | Machine-readable handoff from Micro to Meso; includes representative gamma and output references. |
| `Auto_Micro_Gamma/gamma_structure_selection.csv` | Gamma-sweep diagnostics. |
| `Auto_Micro_Gamma/critical_point_persistence_map.csv` | Candidate critical points across gamma values. |
| `Auto_Micro_Gamma/critical_point_persistence_map.png` | Diagnostic visualization of persistence, when generated. |
| `Auto_Micro_Gamma/consensus_critical_point_bands.csv` | Consolidated critical-point bands. |
| `Auto_Micro_Gamma/consensus_critical_points.csv` | Selected consensus boundaries. |
| `Auto_Micro_Gamma/proto_state_louvain_gamma_selection.csv` | Representative-gamma selection diagnostics. |

These files remain important **even though they are not shown in the main GUI output list**. Inspect the generated CSV headers for the precise field definitions. Do not infer exact critical-point times by reading pixels from a plot.

[Next: Understanding Meso](04_Understanding_Meso.md).
