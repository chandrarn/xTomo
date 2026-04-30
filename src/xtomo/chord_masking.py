"""Helpers for XTOMO chord masking."""

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


def build_array_gradient_spike_mask(
    profile: np.ndarray,
    chord_positions: np.ndarray,
    *,
    max_abs_gradient: float,
) -> np.ndarray:
    """
    Mask chords that are anomalously high relative to adjacent chords.

    Parameters
    ----------
    profile
        Brightness profile for one time slice, shape (nchords,).
    chord_positions
        Signed chord positions used to define adjacent chords physically.
    max_abs_gradient
        Absolute brightness jump [W/m^2] above the larger neighbouring chord
        required to mask a positive spike.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (nchords,), with anomalously high isolated
        chords removed.
    """
    profile_arr = np.asarray(profile, dtype=float).ravel()
    pos_arr = np.asarray(chord_positions, dtype=float).ravel()

    if profile_arr.size != pos_arr.size:
        raise ValueError(
            "profile and chord_positions must have the same length; "
            f"got {profile_arr.size} and {pos_arr.size}."
        )
    if max_abs_gradient < 0.0:
        raise ValueError("max_abs_gradient must be non-negative.")
    if profile_arr.size == 0:
        raise ValueError("profile has zero chords.")
    if profile_arr.size == 1:
        return np.ones(1, dtype=bool)

    order = np.argsort(pos_arr)
    profile_sorted = np.nan_to_num(profile_arr[order], nan=0.0)
    mask_sorted = np.ones(profile_sorted.size, dtype=bool)

    for idx in range(profile_sorted.size):
        if idx == 0:
            neighbour_ref = profile_sorted[1]
        elif idx == profile_sorted.size - 1:
            neighbour_ref = profile_sorted[-2]
        else:
            neighbour_ref = max(profile_sorted[idx - 1], profile_sorted[idx + 1])

        if profile_sorted[idx] - neighbour_ref > max_abs_gradient:
            mask_sorted[idx] = False

    mask = np.ones_like(mask_sorted)
    mask[order] = mask_sorted
    return mask


def build_inversion_chord_mask(
    signals_array1: np.ndarray,
    signals_array3: np.ndarray,
    *,
    apply_zero_mask: bool = True,
    threshold: float = 0.02,
    edge_keep: int = 2,
    profile_array1: np.ndarray | None = None,
    profile_array3: np.ndarray | None = None,
    chord_positions_array1: np.ndarray | None = None,
    chord_positions_array3: np.ndarray | None = None,
    max_gradient_abs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build array-1, array-3, and concatenated inversion masks.
    """
    if apply_zero_mask:
        mask1 = build_array_zero_chord_mask(
            signals_array1, threshold=threshold, edge_keep=edge_keep
        )
        mask3 = build_array_zero_chord_mask(
            signals_array3, threshold=threshold, edge_keep=edge_keep
        )
    else:
        mask1 = np.ones(signals_array1.shape[1], dtype=bool)
        mask3 = np.ones(signals_array3.shape[1], dtype=bool)

    if max_gradient_abs is not None:
        if profile_array1 is None or profile_array3 is None:
            raise ValueError(
                "profile_array1/profile_array3 are required when max_gradient_abs is set."
            )
        if chord_positions_array1 is None or chord_positions_array3 is None:
            raise ValueError(
                "chord_positions_array1/chord_positions_array3 are required when max_gradient_abs is set."
            )

        mask1 &= build_array_gradient_spike_mask(
            profile_array1,
            chord_positions_array1,
            max_abs_gradient=max_gradient_abs,
        )
        mask3 &= build_array_gradient_spike_mask(
            profile_array3,
            chord_positions_array3,
            max_abs_gradient=max_gradient_abs,
        )

    return mask1, mask3, np.concatenate([mask1, mask3])
