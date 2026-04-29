"""Helpers for XTOMO interior zero-chord masking."""

from __future__ import annotations

import numpy as np


def build_array_zero_chord_mask(
    signals: np.ndarray,
    *,
    threshold: float = 0.02,
    edge_keep: int = 2,
) -> np.ndarray:
    """
    Build a per-array chord mask from time-series signals.

    Parameters
    ----------
    signals
        Array of shape (ntimes, nchords) containing brightness signals.
    threshold
        Fraction of the array-wide maximum peak amplitude used as cutoff.
    edge_keep
        Number of chords to always keep at each edge.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (nchords,), with interior low-amplitude chords
        removed and edge chords retained.
    """
    if signals.ndim != 2:
        raise ValueError(f"signals must be 2-D (ntimes, nchords), got shape={signals.shape}")

    nchords = signals.shape[1]
    if nchords == 0:
        raise ValueError("signals has zero chords.")

    amp = np.nan_to_num(np.nanmax(np.abs(signals), axis=0), nan=0.0)
    max_amp = max(float(np.max(amp)), 1e-30)

    mask = np.ones(nchords, dtype=bool)
    i0 = max(int(edge_keep), 0)
    i1 = nchords - i0
    if i1 > i0:
        mask[i0:i1] = amp[i0:i1] >= threshold * max_amp
    return mask


def build_inversion_chord_mask(
    signals_array1: np.ndarray,
    signals_array3: np.ndarray,
    *,
    threshold: float = 0.02,
    edge_keep: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build array-1, array-3, and concatenated inversion masks.
    """
    mask1 = build_array_zero_chord_mask(signals_array1, threshold=threshold, edge_keep=edge_keep)
    mask3 = build_array_zero_chord_mask(signals_array3, threshold=threshold, edge_keep=edge_keep)
    return mask1, mask3, np.concatenate([mask1, mask3])
