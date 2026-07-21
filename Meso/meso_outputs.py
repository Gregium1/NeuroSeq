# Meso/meso_outputs.py

"""
Canonical Family Outputs and Visualizations
===========================================

Module
------
Meso.meso_outputs

Purpose
-------
Materialize the canonical family-level tables and figures produced by the
NeuroSeq Meso pipeline.

Scientific Role
---------------
After spectral clustering assigns each Micro state to a Meso family, this
module converts those assignments into temporally organized output tables.
It reconstructs the family sequence across the recording, merges adjacent
intervals belonging to the same family, and generates family-labeled timeline
and spike-raster visualizations.

These outputs provide the final interpretable representation of recurring
neural population configurations identified by the Meso analysis. The CSV
files preserve the exact temporal boundaries and family identities, while the
figures display how those families evolve across the recording and how they
relate to the underlying single-unit spiking activity.

Outputs
-------
This module writes canonical family timeline CSV files, merged family timeline
CSV files, PNG and SVG timeline figures, and PNG and SVG family-labeled spike
rasters.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib
import re
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .meso_utils import ensure_dir


def savefig_png_svg(path_png: str | Path, *, dpi: int = 300, bbox_inches: str = "tight") -> None:
    """
    Save the active Matplotlib figure in both PNG and SVG formats.

    Parameters
    ----------
    path_png : str or pathlib.Path
        Destination path for the PNG figure. The SVG figure is written to the
        same location using the ``.svg`` extension.
    dpi : int, default=300
        Resolution used when saving the rasterized PNG figure.
    bbox_inches : str, default="tight"
        Matplotlib bounding-box option applied to both output formats.

    Returns
    -------
    None
        The function writes the figure files directly to disk.

    Notes
    -----
    Saving both formats provides a high-resolution raster image for routine
    inspection and a vector image suitable for publication.
    """

    # Normalize the output path and create its parent directory when needed.
    path_png = Path(path_png)
    path_png.parent.mkdir(parents=True, exist_ok=True)

    # Save publication-ready raster and vector versions of the active figure.
    plt.savefig(path_png, dpi=dpi, bbox_inches=bbox_inches)
    plt.savefig(path_png.with_suffix(".svg"), bbox_inches=bbox_inches)


def _extract_su_number_from_name(name: str) -> int:
    """
    Extract the numerical single-unit identifier from an SU file name.

    Parameters
    ----------
    name : str
        File name or stem expected to contain an identifier of the form
        ``SU_<number>``.

    Returns
    -------
    int
        Extracted single-unit number. Names without a valid identifier receive
        a large fallback value so that they sort after valid SU names.
    """

    # Locate the numerical identifier embedded in the standard SU name.
    match = re.search(r"SU_(\d+)", str(name))
    return int(match.group(1)) if match else 10**12


def list_su_files(data_folder: str | Path) -> List[Path]:
    """
    Return single-unit spike files in numerical SU order.

    Parameters
    ----------
    data_folder : str or pathlib.Path
        Directory containing single-unit CSV files named ``SU_<number>.csv``.

    Returns
    -------
    list[pathlib.Path]
        Existing single-unit CSV files sorted by their numerical SU identifier.
    """

    # Identify all single-unit files in the supplied recording directory.
    root = Path(data_folder)

    # Sort numerically so raster rows follow SU_1, SU_2, and so forth.
    return sorted(
        [p for p in root.glob("SU_*.csv") if p.is_file()],
        key=lambda p: _extract_su_number_from_name(p.stem),
    )


def load_spike_times_for_window(csv_path: str | Path, start_t: float, end_t: float) -> np.ndarray:
    """
    Load finite spike times restricted to a requested temporal interval.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to a single-unit spike-time CSV file.
    start_t : float
        Inclusive lower temporal boundary in seconds.
    end_t : float
        Inclusive upper temporal boundary in seconds.

    Returns
    -------
    numpy.ndarray
        Sorted one-dimensional array of finite spike times within the requested
        interval. An empty array is returned when the file cannot be read or
        contains no valid spikes.

    Notes
    -----
    The loader supports both headerless and header-containing CSV files and
    combines numeric values found across all columns.
    """

    # Attempt the headerless format commonly used for raw spike-time files.
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception:
        # Fall back to a conventional header-containing CSV representation.
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return np.array([], dtype=float)

    # Collect every numeric spike-time value across the available columns.
    vals: List[float] = []
    for col in df.columns:
        arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
        if arr.size:
            vals.extend(arr.tolist())

    if not vals:
        return np.array([], dtype=float)

    # Remove invalid values and retain only spikes inside the plotted window.
    spikes = np.asarray(vals, dtype=float)
    spikes = spikes[np.isfinite(spikes)]
    spikes = spikes[(spikes >= float(start_t)) & (spikes <= float(end_t))]
    spikes.sort()
    return spikes


def timeline_color_map(labels: List[str]) -> Dict[str, Any]:
    """
    Assign deterministic plotting colors in order of first label appearance.

    Parameters
    ----------
    labels : list[str]
        Temporally ordered family or state labels.

    Returns
    -------
    dict[str, Any]
        Mapping from each unique label to a color from Matplotlib's ``tab20``
        colormap.

    Notes
    -----
    First-appearance ordering keeps color assignment aligned with the temporal
    organization of the recording.
    """

    # Preserve the first temporal appearance of every distinct label.
    unique = []
    seen = set()
    for lab in labels:
        lab = str(lab)
        if lab not in seen:
            unique.append(lab)
            seen.add(lab)

    # Cycle through the categorical palette when more than 20 labels occur.
    cmap = plt.cm.tab20
    return {lab: cmap(i % 20) for i, lab in enumerate(unique)}


def merge_consecutive_timeline_df(
    df: pd.DataFrame,
    *,
    label_col: str = "Label",
    start_col: str = "Start",
    end_col: str = "End",
    group_cols: List[str] | None = None,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Merge adjacent or overlapping timeline intervals with identical labels.

    Parameters
    ----------
    df : pandas.DataFrame
        Timeline table containing label, start-time, and end-time columns.
    label_col : str, default="Label"
        Column containing the family or state label to compare.
    start_col : str, default="Start"
        Column containing interval start times.
    end_col : str, default="End"
        Column containing interval end times.
    group_cols : list[str] or None, default=None
        Columns defining independent timelines. When omitted, available
        ``Delivery`` and ``File`` columns are used.
    tol : float, default=1e-6
        Numerical tolerance used when determining whether intervals touch.

    Returns
    -------
    pandas.DataFrame
        Timeline table in which consecutive intervals carrying the same label
        have been combined.

    Notes
    -----
    Merging removes artificial visual boundaries between contiguous Micro
    states assigned to the same Meso family without changing the total temporal
    coverage of that family.
    """

    # Return immediately when no timeline records are available.
    if df is None or df.empty:
        return pd.DataFrame(columns=[] if df is None else df.columns)

    work = df.copy()

    # Treat deliveries and source files as independent timelines when present.
    if group_cols is None:
        group_cols = [c for c in ["Delivery", "File"] if c in work.columns]

    # Convert temporal boundaries to numeric values before interval processing.
    for col in [start_col, end_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Retain only finite, positive-duration timeline intervals.
    work = work[
        work[start_col].notna()
        & work[end_col].notna()
        & (work[end_col] > work[start_col])
    ].copy()
    if work.empty:
        return work

    # Establish deterministic temporal ordering within every recording group.
    sort_cols = list(group_cols) + [start_col, end_col]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    merged_rows = []
    grouped = work.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, work)]

    # Process each recording timeline independently.
    for _, gdf in grouped:
        gdf = gdf.sort_values([start_col, end_col]).reset_index(drop=True)
        if gdf.empty:
            continue

        # Initialize the active interval with the first temporal record.
        cur = gdf.iloc[0].to_dict()
        for idx in range(1, len(gdf)):
            row = gdf.iloc[idx].to_dict()
            same_label = str(row.get(label_col, "")) == str(cur.get(label_col, ""))
            cur_end = float(cur[end_col])
            next_start = float(row[start_col])
            next_end = float(row[end_col])

            # Extend the active interval when the same family continues.
            if same_label and next_start <= cur_end + float(tol):
                cur[end_col] = max(cur_end, next_end)
                if "Duration" in cur:
                    cur["Duration"] = float(cur[end_col]) - float(cur[start_col])
            else:
                # Finalize the active interval before beginning a new family run.
                if "Duration" in cur:
                    cur["Duration"] = float(cur[end_col]) - float(cur[start_col])
                merged_rows.append(cur)
                cur = row

        # Finalize the last active interval in the current timeline group.
        if "Duration" in cur:
            cur["Duration"] = float(cur[end_col]) - float(cur[start_col])
        merged_rows.append(cur)

    # Restore deterministic group-wise temporal ordering in the merged table.
    out = pd.DataFrame(merged_rows)
    if not out.empty:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def plot_timeline(
    *,
    timeline_df: pd.DataFrame,
    out_png: str | Path,
    title: str,
    face_label_col: str = "Label",
    text_label_col: str = "Label",
) -> None:
    """
    Plot a horizontal timeline of Meso family assignments.

    Parameters
    ----------
    timeline_df : pandas.DataFrame
        Timeline table containing ``Start`` and ``End`` boundaries together
        with columns used for segment colors and text labels.
    out_png : str or pathlib.Path
        Destination path for the PNG figure. A matching SVG is also written.
    title : str
        Figure title.
    face_label_col : str, default="Label"
        Column controlling the segment fill color.
    text_label_col : str, default="Label"
        Column displayed as text inside each temporal segment.

    Returns
    -------
    None
        The function writes the timeline figure directly to disk.
    """

    # Skip plotting when no family timeline is available.
    if timeline_df is None or timeline_df.empty:
        return

    # Validate temporal boundaries and remove non-positive intervals.
    df = timeline_df.copy()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"] = pd.to_numeric(df["End"], errors="coerce")
    df = df[np.isfinite(df["Start"]) & np.isfinite(df["End"]) & (df["End"] > df["Start"])].copy()
    if df.empty:
        return

    # Order intervals temporally and assign family colors by first appearance.
    df.sort_values(["Start", "End"], inplace=True)
    labels_for_color = df[face_label_col].astype(str).tolist()
    color_map = timeline_color_map(labels_for_color)

    # Scale figure width to the duration of the analyzed recording.
    xmin = float(df["Start"].min())
    xmax = float(df["End"].max())
    fig_w = max(12.0, min(22.0, (xmax - xmin) / 4.0 + 8.0))

    plt.figure(figsize=(fig_w, 3.2))
    ax = plt.gca()

    # Draw each family interval as a labeled horizontal temporal block.
    for _, row in df.iterrows():
        start = float(row["Start"])
        end = float(row["End"])
        width = end - start
        face_lab = str(row[face_label_col])
        text_lab = str(row[text_label_col])

        ax.barh(
            y=0,
            width=width,
            left=start,
            height=0.75,
            color=color_map.get(face_lab, (0.7, 0.7, 0.7, 1.0)),
            edgecolor="black",
            linewidth=0.7,
        )

        # Adapt label size and orientation to the available segment width.
        if width >= 0.8:
            fs, rot, color = 13, 0, "black"
        elif width >= 0.25:
            fs, rot, color = 10, 0, "black"
        else:
            fs, rot, color = 8, 90, "white"

        ax.text(
            start + width / 2.0,
            0,
            text_lab,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=color,
            rotation=rot,
            clip_on=True,
        )

    # Apply publication-oriented timeline axes and styling.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(title, fontsize=16, pad=12)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    savefig_png_svg(out_png)
    plt.close()


def plot_labeled_raster(
    *,
    data_folder: str | Path,
    timeline_df: pd.DataFrame,
    out_png: str | Path,
    title: str,
    face_label_col: str = "Label",
    text_label_col: str = "Label",
    legend_title: str = "Labels",
) -> None:
    """
    Plot single-unit spike activity over Meso family-labeled intervals.

    Parameters
    ----------
    data_folder : str or pathlib.Path
        Directory containing the ``SU_<number>.csv`` spike-time files.
    timeline_df : pandas.DataFrame
        Timeline table containing temporal boundaries and family labels.
    out_png : str or pathlib.Path
        Destination path for the PNG figure. A matching SVG is also written.
    title : str
        Figure title.
    face_label_col : str, default="Label"
        Column controlling family background colors.
    text_label_col : str, default="Label"
        Column displayed above each family interval.
    legend_title : str, default="Labels"
        Title shown above the family legend.

    Returns
    -------
    None
        The function writes the labeled raster directly to disk.

    Scientific Interpretation
    -------------------------
    The shaded temporal regions connect the inferred Meso family sequence to
    the underlying single-unit spike trains, allowing recurring population
    configurations and their transition structure to be inspected directly.
    """

    # Skip plotting when no family timeline is available.
    if timeline_df is None or timeline_df.empty:
        return

    # Validate the temporal intervals used to define the plotted recording span.
    df = timeline_df.copy()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["End"] = pd.to_numeric(df["End"], errors="coerce")
    df = df[np.isfinite(df["Start"]) & np.isfinite(df["End"]) & (df["End"] > df["Start"])].copy()
    if df.empty:
        return

    df.sort_values(["Start", "End"], inplace=True)
    xmin = float(df["Start"].min())
    xmax = float(df["End"].max())

    # Discover and numerically order the single-unit spike files.
    su_files = list_su_files(data_folder)
    if not su_files:
        return

    neuron_names = [p.stem for p in su_files]
    color_map = timeline_color_map(df[face_label_col].astype(str).tolist())

    # Reserve a dedicated panel for the family-label legend.
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4.8, 1.6], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    # Shade and annotate each family interval across all raster rows.
    for _, row in df.iterrows():
        start = float(row["Start"])
        end = float(row["End"])
        width = end - start
        face_lab = str(row[face_label_col])
        text_lab = str(row[text_label_col])

        ax.axvspan(start, end, color=color_map.get(face_lab, (0.8, 0.8, 0.8, 1.0)), alpha=0.35, zorder=0)
        fs = 14 if width >= 0.8 else 10 if width >= 0.25 else 8
        rot = 90 if width < 0.22 else 0
        ax.text(
            start + width / 2.0,
            len(neuron_names) + 0.25,
            text_lab,
            ha="center",
            va="bottom",
            fontsize=fs,
            fontweight="bold",
            color="black",
            rotation=rot,
            clip_on=False,
        )

    # Draw spike events for every single unit within the analyzed time range.
    for y, su_path in enumerate(su_files):
        spikes = load_spike_times_for_window(su_path, xmin, xmax)
        if spikes.size > 0:
            ax.vlines(spikes, y - 0.38, y + 0.38, color="black", linewidth=0.65, zorder=3)

    # Configure raster axes and single-unit row labels.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1, len(neuron_names) + 1.7)
    ax.set_yticks(np.arange(len(neuron_names)))
    ax.set_yticklabels(neuron_names, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron (SU_ID)")
    ax.set_title(title, fontsize=18, pad=18)

    # Construct one legend entry for each distinct family color.
    handles = []
    seen = set()
    for _, row in df.iterrows():
        face_lab = str(row[face_label_col])
        if face_lab in seen:
            continue
        same_text = df.loc[df[face_label_col].astype(str) == face_lab, text_label_col].astype(str).tolist()
        same_text = list(dict.fromkeys(same_text))
        label = f"{face_lab}: " + ", ".join(same_text[:8])
        if len(same_text) > 8:
            label += ", ..."
        handles.append(mpatches.Patch(facecolor=color_map.get(face_lab), edgecolor="gray", label=label, alpha=0.7))
        seen.add(face_lab)

    if handles:
        ax_leg.legend(handles=handles, title=legend_title, loc="upper left", frameon=False, fontsize=10, title_fontsize=13)

    plt.tight_layout()
    savefig_png_svg(out_png)
    plt.close()


def _rows_from_state_labels(*, state_vectors, state_ids, label_lookup) -> pd.DataFrame:
    """
    Convert state metadata and family assignments into a timeline table.

    Parameters
    ----------
    state_vectors
        Mapping from each Micro-state identifier to its temporal metadata.
    state_ids
        Ordered Micro-state identifiers included in the Meso analysis.
    label_lookup
        Mapping from Micro-state identifiers to canonical family labels.

    Returns
    -------
    pandas.DataFrame
        Canonical state-level family timeline sorted by delivery, source file,
        start time, and end time.

    Notes
    -----
    Missing durations or end times are reconstructed when sufficient temporal
    information is available. Invalid or non-positive intervals are excluded.
    """

    rows = []

    # Build one temporally valid family record for every analyzed Micro state.
    for uid in state_ids:
        info = state_vectors[uid]
        start = float(info.get("start", np.nan)); end = float(info.get("end", np.nan)); duration = float(info.get("duration", np.nan))
        if not np.isfinite(duration) and np.isfinite(start) and np.isfinite(end): duration = end - start
        if not np.isfinite(end) and np.isfinite(start) and np.isfinite(duration): end = start + duration
        if not (np.isfinite(start) and np.isfinite(end) and end > start): continue
        label = str(label_lookup.get(uid, "?"))
        rows.append({"Unique_ID":uid,"Delivery":str(info.get("delivery","Full_Recording")),"File":info.get("file","state_vectors.csv"),"State":info.get("state",""),"Start":start,"End":end,"Duration":duration,"Component_Label":label,"Label":label,"label_map":label.lower()})

    # Return the canonical state sequence in deterministic temporal order.
    df = pd.DataFrame(rows)
    if not df.empty: df = df.sort_values(["Delivery","File","Start","End"]).reset_index(drop=True)
    return df


def materialize_family_outputs(*, out_dir: str, state_vectors, state_ids, family_label_csv: str, data_folder: str, do_plots: bool=True) -> dict:
    """
    Generate the canonical family-level CSV files and visualizations.

    Parameters
    ----------
    out_dir : str
        Destination directory for all family-level outputs.
    state_vectors
        Mapping containing temporal metadata for each Micro state.
    state_ids
        Ordered Micro-state identifiers included in the Meso analysis.
    family_label_csv : str
        Path to the canonical spectral family-label table containing
        ``Unique_ID`` and ``Component_Label`` columns.
    data_folder : str
        Directory containing single-unit spike-time CSV files.
    do_plots : bool, default=True
        Whether to generate family timeline and labeled-raster figures.

    Returns
    -------
    dict
        Paths to generated CSV and figure files together with the number of
        distinct family labels represented in the timeline.

    Scientific Role
    ---------------
    This function is the final family-output materialization step of the Meso
    pipeline. It joins spectral family assignments with Micro-state timing,
    consolidates consecutive family intervals, and exports the tables and
    figures used for interpretation and publication.
    """

    # Create the canonical Meso output directory.
    out_dir = ensure_dir(out_dir)

    # Load and validate the spectral family assignments.
    labels = pd.read_csv(family_label_csv)
    required = {"Unique_ID", "Component_Label"}
    if not required.issubset(labels.columns): raise ValueError(f"Label file must contain {sorted(required)}")
    lookup = dict(zip(labels["Unique_ID"].astype(str), labels["Component_Label"].astype(str)))

    # Reconstruct and save the complete state-level family timeline.
    timeline = _rows_from_state_labels(state_vectors=state_vectors, state_ids=state_ids, label_lookup=lookup)
    timeline_csv = out_dir / "state_timeline_family_labels.csv"; timeline.to_csv(timeline_csv,index=False)

    # Merge consecutive states assigned to the same family and save the result.
    merged = merge_consecutive_timeline_df(timeline, label_col="Label")
    merged_csv = out_dir / "state_timeline_family_labels_merged.csv"; merged.to_csv(merged_csv,index=False)

    # Record the canonical output paths and observed family count.
    result={"timeline_csv":str(timeline_csv),"merged_timeline_csv":str(merged_csv),"n_family_labels":int(timeline["Component_Label"].nunique()) if not timeline.empty else 0}

    # Generate publication-ready family timeline and raster figures when requested.
    if do_plots and str(data_folder).strip():
        timeline_png=out_dir/"state_timeline_family_labels_merged.png"
        plot_timeline(timeline_df=merged,out_png=timeline_png,title="Meso family timeline (spectral labels)",face_label_col="Label",text_label_col="Label")
        raster_png=out_dir/"state_raster_family_labels_merged.png"
        plot_labeled_raster(data_folder=data_folder,timeline_df=merged,out_png=raster_png,title="Meso family-labeled raster",face_label_col="Label",text_label_col="Label",legend_title="Family labels")
        result.update({"timeline_png":str(timeline_png),"timeline_svg":str(timeline_png.with_suffix('.svg')),"raster_png":str(raster_png),"raster_svg":str(raster_png.with_suffix('.svg'))})
    return result