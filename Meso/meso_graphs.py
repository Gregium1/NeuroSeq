#### Meso/meso_graphs.py

"""
Affinity-Graph Visualization for NeuroSeq Meso Analysis
=======================================================

Module
------
Meso.meso_graphs

Purpose
-------
Visualize the Micro-state affinity structure used by the NeuroSeq Meso
analysis.

Scientific Role
---------------
The Meso pipeline represents pairwise relationships between Micro states
through a complete affinity matrix. This module converts that matrix into a
readable graph visualization while preserving the full affinity information
separately.

Because a dense affinity graph can contain many thousands of edges, only the
strongest edges are selected for plotting. This sparsification affects the
visualization only and does not alter the affinity matrix used for spectral
clustering or downstream inference.

Two complementary graph figures are generated:

1. A raw affinity graph showing the geometric organization of Micro states.
2. The same graph colored and labeled by the spectral family assignments.

The graph layout is computed once and reused across both figures so that the
visual comparison reflects only the addition of family labels rather than a
change in node geometry.

Outputs
-------
``affinity_graph_edges_for_plot.csv``
    Edge list containing the strongest affinities selected for visualization.

``affinity_graph_raw.png`` and ``affinity_graph_raw.svg``
    Unlabeled affinity-graph visualization.

``affinity_graph_spectral_kmeans_labels.png`` and
``affinity_graph_spectral_kmeans_labels.svg``
    Affinity graph colored and labeled by spectral family assignments.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from .meso_utils import ensure_dir, letter_label

# Preserve editable text in SVG exports and maintain a consistent visual style.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12
plt.rcParams["image.composite_image"] = False


def _savefig_png_svg(path_png: str | Path, *, dpi: int = 300) -> None:
    """
    Save the active Matplotlib figure in both PNG and SVG formats.

    Parameters
    ----------
    path_png : str or pathlib.Path
        Output path for the PNG file.
    dpi : int, default=300
        Raster resolution used for the PNG output.

    Notes
    -----
    The corresponding SVG file is written beside the PNG using the same
    filename stem. PNG supports direct viewing, whereas SVG preserves vector
    geometry and editable text for publication workflows.
    """

    path_png = Path(path_png)

    # Ensure the destination directory exists before writing either figure.
    ensure_dir(path_png.parent)

    # Save both raster and vector versions of the same active figure.
    plt.savefig(path_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(path_png.with_suffix(".svg"), bbox_inches="tight")


def edge_rows_for_affinity_plot(
    A: np.ndarray,
    *,
    state_ids: List[str],
    edge_quantile: float = 0.90,
    max_edges: int = 5000,
) -> pd.DataFrame:
    """
    Select the strongest affinity edges for graph visualization.

    Parameters
    ----------
    A : numpy.ndarray
        Square Micro-state affinity matrix.
    state_ids : list[str]
        Ordered Micro-state identifiers corresponding to the matrix rows and
        columns.
    edge_quantile : float, default=0.90
        Quantile used to retain only the strongest positive affinities.
    max_edges : int, default=5000
        Maximum number of edges retained after quantile filtering.

    Returns
    -------
    pandas.DataFrame
        Edge table with matrix indices, state identifiers, and S-scores.

    Notes
    -----
    The complete affinity matrix remains unchanged and is saved elsewhere.
    This function sparsifies only the graph drawn for visualization.

    The matrix is numerically cleaned, clipped to the valid affinity interval,
    symmetrized, and stripped of self-connections before edge selection.
    """

    # Standardize the affinity matrix for visualization without modifying the
    # original array supplied by the caller.
    A0 = np.asarray(A, dtype=float).copy()
    A0 = np.nan_to_num(A0, nan=0.0, posinf=1.0, neginf=0.0)
    A0 = np.clip(A0, 0.0, 1.0)
    A0 = 0.5 * (A0 + A0.T)
    np.fill_diagonal(A0, 0.0)

    rows = []
    n = A0.shape[0]
    vals = []

    # Collect all finite positive upper-triangular affinities so that the
    # visualization threshold is estimated from unique state pairs only.
    for i in range(n):
        for j in range(i + 1, n):
            w = float(A0[i, j])
            if np.isfinite(w) and w > 0:
                vals.append(w)

    # Return a structurally valid empty edge table when no positive affinities
    # are available for plotting.
    if not vals:
        return pd.DataFrame(columns=["i", "j", "State_1", "State_2", "S_score"])

    # Convert the requested quantile into a valid range and determine the
    # minimum affinity displayed in the graph.
    q = float(np.clip(edge_quantile, 0.0, 1.0))
    thr = float(np.nanquantile(np.asarray(vals, dtype=float), q))

    # Retain all positive state pairs whose affinity meets the visualization
    # threshold.
    for i in range(n):
        for j in range(i + 1, n):
            w = float(A0[i, j])
            if np.isfinite(w) and w >= thr and w > 0:
                rows.append({
                    "i": int(i),
                    "j": int(j),
                    "State_1": state_ids[i],
                    "State_2": state_ids[j],
                    "S_score": w,
                })

    # Rank edges by affinity so that any imposed edge limit preserves the
    # strongest relationships.
    rows = sorted(rows, key=lambda r: r["S_score"], reverse=True)
    if max_edges is not None and int(max_edges) > 0:
        rows = rows[:int(max_edges)]

    return pd.DataFrame(rows)


def plot_affinity_graphs(
    *,
    A: np.ndarray,
    state_ids: List[str],
    labels: np.ndarray,
    out_dir: str | Path,
    seed: int = 42,
    edge_quantile: float = 0.90,
    max_edges: int = 5000,
) -> dict:
    """
    Plot raw and spectrally labeled Micro-state affinity graphs.

    Parameters
    ----------
    A : numpy.ndarray
        Complete Micro-state affinity matrix.
    state_ids : list[str]
        Ordered Micro-state identifiers.
    labels : numpy.ndarray
        Spectral family labels aligned with ``state_ids``.
    out_dir : str or pathlib.Path
        Directory in which graph outputs are written.
    seed : int, default=42
        Random seed used by the spring-layout algorithm.
    edge_quantile : float, default=0.90
        Quantile used to retain the strongest affinities for visualization.
    max_edges : int, default=5000
        Maximum number of edges drawn.

    Returns
    -------
    dict
        Graph-generation status, plotting parameters, graph dimensions, and
        canonical output paths.

    Notes
    -----
    This function performs visualization only. It does not alter the affinity
    matrix, discover spectral families, or perform subtype analysis.

    The same node layout is reused for the raw and labeled graphs, allowing the
    family labels to be interpreted directly against the original affinity
    geometry.
    """

    out_dir = Path(out_dir)

    # Create the canonical graph-output directory.
    ensure_dir(out_dir)

    # Select a readable subset of the strongest affinity edges.
    edge_df = edge_rows_for_affinity_plot(
        A,
        state_ids=state_ids,
        edge_quantile=edge_quantile,
        max_edges=max_edges,
    )

    # Save the exact edge subset used to construct the displayed graph.
    edge_csv = out_dir / "affinity_graph_edges_for_plot.csv"
    edge_df.to_csv(edge_csv, index=False)

    # Initialize one graph node for every Micro state, including isolated
    # states that may have no retained visualization edges.
    G = nx.Graph()
    for i, sid in enumerate(state_ids):
        lab = letter_label(int(labels[i]))
        G.add_node(int(i), Unique_ID=str(sid), Component_Label=lab)

    # Add the selected affinity edges with their S-scores as graph weights.
    for _, row in edge_df.iterrows():
        G.add_edge(int(row["i"]), int(row["j"]), weight=float(row["S_score"]))

    # Skip figure generation only when no Micro states are available.
    if G.number_of_nodes() == 0:
        return {"status": "SKIPPED", "reason": "empty_graph"}

    # Use a weighted force-directed layout and fall back to Kamada-Kawai when
    # the primary layout cannot be computed.
    try:
        pos = nx.spring_layout(G, seed=int(seed), weight="weight", iterations=300)
    except Exception:
        pos = nx.kamada_kawai_layout(G, weight="weight")

    # Recover edge weights in graph order for visual width scaling.
    weights = np.asarray(
        [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)],
        dtype=float,
    )

    # Map affinity strength to edge width while preserving a visible minimum.
    if weights.size:
        w_min = float(np.nanmin(weights))
        w_max = float(np.nanmax(weights))
        denom = w_max - w_min if w_max > w_min else 1.0
        edge_widths = [0.25 + 2.75 * ((float(w) - w_min) / denom) for w in weights]
    else:
        edge_widths = []

    # Plot the raw affinity graph without family coloring.
    raw_png = out_dir / "affinity_graph_raw.png"
    fig, ax = plt.subplots(figsize=(16, 12))

    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=edge_widths,
            alpha=0.18,
            edge_color="black",
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color="lightgray",
        node_size=130,
        edgecolors="black",
        linewidths=0.4,
    )

    ax.set_title("Micro-state affinity graph", fontsize=18, pad=18)
    ax.set_axis_off()
    fig.tight_layout()
    _savefig_png_svg(raw_png)
    plt.close(fig)

    # Prepare deterministic family colors and letter labels for the second
    # graph while preserving the same node positions.
    clustered_png = out_dir / "affinity_graph_spectral_kmeans_labels.png"
    unique_labels = list(dict.fromkeys(int(x) for x in labels.tolist()))
    label_to_color_index = {lab: idx for idx, lab in enumerate(unique_labels)}
    cmap = plt.get_cmap("tab20")
    node_colors = [
        cmap(label_to_color_index[int(labels[i])] % 20)
        for i in range(len(state_ids))
    ]
    node_text = {
        i: letter_label(int(labels[i]))
        for i in range(len(state_ids))
    }

    # Plot the affinity graph with spectral family assignments.
    fig, ax = plt.subplots(figsize=(16, 12))

    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=edge_widths,
            alpha=0.16,
            edge_color="black",
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=180,
        edgecolors="black",
        linewidths=0.5,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_text,
        ax=ax,
        font_size=7,
        font_weight="bold",
    )

    ax.set_title(
        "Micro-state affinity graph with spectral labels",
        fontsize=18,
        pad=18,
    )
    ax.set_axis_off()
    fig.tight_layout()
    _savefig_png_svg(clustered_png)
    plt.close(fig)

    # Return the graph summary and canonical output locations.
    return {
        "status": "OK",
        "graph_mode": "affinity_matrix_visualization_only",
        "edge_quantile": float(edge_quantile),
        "max_edges": int(max_edges),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges_drawn": int(G.number_of_edges()),
        "edge_csv": str(edge_csv),
        "raw_graph_png": str(raw_png),
        "raw_graph_svg": str(raw_png.with_suffix(".svg")),
        "spectral_label_graph_png": str(clustered_png),
        "spectral_label_graph_svg": str(clustered_png.with_suffix(".svg")),
    }
