"""
Public chord-masking entry points for XTOMO inversions.
Moved to separate script for acess by both brightness and emissivity calculations
"""

from __future__ import annotations

import argparse

import numpy as np

from .chord_masking import build_inversion_chord_mask
from .xtomo_mds import bipolar_radii, read_xray_brightness


def mask_cords(
    shot: int,
    time: float,
    *,
    remove_zero_chords: bool = True,
    zero_chord_threshold: float = 0.02,
    mask_gradient_spikes: bool = False,
    gradient_spike_threshold: float = 1500.0,
    edge_keep: int = 2,
) -> np.ndarray:
    """
    Build the 76-channel inversion mask for arrays 1 and 3.

    The returned mask can be passed directly to
    :func:`xtomo.core_xray_emissivity` via the ``chord_mask`` argument.
    """
    if not (remove_zero_chords or mask_gradient_spikes):
        return np.ones(76, dtype=bool)

    time1_all, b1_all, r1, a1 = read_xray_brightness(shot, array=1, fix_bad_channels=True)
    time3_all, b3_all, r3, a3 = read_xray_brightness(shot, array=3, fix_bad_channels=True)

    t_idx1 = int(np.argmin(np.abs(time1_all - time)))
    t_idx3 = int(np.argmin(np.abs(time3_all - time)))

    _, _, inversion_mask = build_inversion_chord_mask(
        b1_all,
        b3_all,
        apply_zero_mask=remove_zero_chords,
        threshold=zero_chord_threshold,
        edge_keep=edge_keep,
        profile_array1=b1_all[t_idx1, :] if mask_gradient_spikes else None,
        profile_array3=b3_all[t_idx3, :] if mask_gradient_spikes else None,
        chord_positions_array1=bipolar_radii(r1, a1, 1) if mask_gradient_spikes else None,
        chord_positions_array3=bipolar_radii(r3, a3, 3) if mask_gradient_spikes else None,
        max_gradient_abs=gradient_spike_threshold if mask_gradient_spikes else None,
    )
    return inversion_mask


def main() -> None:
    """Console entry point: ``xtomo-mask-cords``."""
    parser = argparse.ArgumentParser(description="Build a 76-channel XTOMO inversion mask")
    parser.add_argument("shot", type=int, help="MDS shot number")
    parser.add_argument("time", type=float, help="Reference time [s] used for profile masking")
    parser.add_argument(
        "--no-zero-filter",
        action="store_true",
        help="Keep near-zero interior chords in the inversion mask",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.02,
        dest="zero_chord_threshold",
        help="Fraction of array max below which interior chords are masked",
    )
    parser.add_argument(
        "--mask-gradient-spikes",
        action="store_true",
        help="Mask isolated high chords using an absolute adjacent-channel threshold",
    )
    parser.add_argument(
        "--gradient-spike-threshold",
        type=float,
        default=1500.0,
        dest="gradient_spike_threshold",
        help="Absolute adjacent-channel spike threshold in W/m^2",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output .npy path to save the boolean mask",
    )
    args = parser.parse_args()

    mask = mask_cords(
        args.shot,
        args.time,
        remove_zero_chords=not args.no_zero_filter,
        zero_chord_threshold=args.zero_chord_threshold,
        mask_gradient_spikes=args.mask_gradient_spikes,
        gradient_spike_threshold=args.gradient_spike_threshold,
    )

    print(f"Using {int(mask.sum())}/{mask.size} chords")
    if args.save:
        np.save(args.save, mask)
        print(f"Mask saved to {args.save}")


if __name__ == "__main__":
    main()
