### Meso/meso_labels.py

"""
Final Spectral Family Labels
============================

Module
------
Meso.meso_labels

Purpose
-------
Generate the canonical family-label tables used throughout the NeuroSeq
Meso pipeline.

Scientific Role
---------------
After spectral clustering identifies groups of related Micro states, the
resulting integer labels are converted into a deterministic and interpretable
representation. Clusters are reordered according to their first temporal
appearance, ensuring that Family A corresponds to the earliest discovered
family, Family B to the next, and so forth.

This temporal relabeling produces stable family identifiers that are easier
to interpret and compare across analyses without changing the underlying
spectral clustering result.

Outputs
-------
This module returns canonical family labels together with tables that map
every Micro state to its spectral family.

Author
------
Greg Owanga
Gregium Theory
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from .meso_utils import letter_label


def compress_labels_temporal_order(labels: np.ndarray) -> np.ndarray:
    """
    Relabel spectral clusters according to their first temporal appearance.

    Parameters
    ----------
    labels : numpy.ndarray
        Original integer labels returned by spectral clustering.

    Returns
    -------
    numpy.ndarray
        Consecutive integer labels beginning at zero and ordered by first
        temporal appearance.

    Notes
    -----
    This operation changes only the label numbering. Cluster membership
    remains completely unchanged.
    """

    # Record the first occurrence of each spectral family.
    mapping = {}
    out = np.zeros_like(labels, dtype=int)
    next_id = 0

    # Assign new labels in temporal order.
    for idx, lab in enumerate(np.asarray(labels, dtype=int)):
        lab = int(lab)

        if lab not in mapping:
            mapping[lab] = next_id
            next_id += 1

        out[idx] = mapping[lab]

    return out


def labels_to_family_df(
    state_ids: List[str],
    labels: np.ndarray,
    labels_raw: np.ndarray,
) -> pd.DataFrame:
    """
    Build the canonical spectral family-label table.

    Parameters
    ----------
    state_ids : list[str]
        Ordered Micro-state identifiers.
    labels : numpy.ndarray
        Temporally compressed spectral labels.
    labels_raw : numpy.ndarray
        Original labels returned directly by spectral clustering.

    Returns
    -------
    pandas.DataFrame
        Canonical table linking every Micro state to its raw spectral label,
        temporally compressed label, and alphabetic family identifier.

    Notes
    -----
    Both the raw and temporally compressed labels are retained to preserve
    complete traceability between the spectral clustering output and the
    final NeuroSeq family representation.
    """

    rows = []

    # Assemble one canonical record for every Micro state.
    for idx, uid in enumerate(state_ids):

        # Convert integer family labels into human-readable letters.
        lab = letter_label(int(labels[idx]))

        rows.append({
            "proto_state_index": int(idx),
            "Unique_ID": uid,
            "Spectral_Label_Raw": int(labels_raw[idx]),
            "Spectral_Label": int(labels[idx]),
            "Component_Label": lab,
            "Cluster_Label": lab,
        })

    return pd.DataFrame(rows)