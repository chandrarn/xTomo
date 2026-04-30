"""
example_xtomo.py
================
End-to-end example for the `xtomo` package.

Steps
-----
1. Run the Fourier-Bessel tomographic inversion for a user-specified shot
   and time window.
2. Plot the 2-D emissivity on the poloidal cross-section (with optional EFIT
   flux contours and vessel geometry).
3. Run the brightness-vs-emissivity comparison: raw chord brightness from
   XTOMO arrays 1 and 3 side-by-side with the inverted midplane profile,
   plus a normalised overlay of all three.

Usage
-----
    # After `pip install -e .` (from the xTomo project root):
    python examples/example_xtomo.py

    # Override shot / time / time-window from the command line:
    python examples/example_xtomo.py --shot 1140221013 --time 1.2 \\
        --tstart 0.8 --tstop 1.6 --dt 0.05 --save-prefix ./out/shot_1140221013
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Package imports — requires `pip install -e .` from the xTomo project root
# ---------------------------------------------------------------------------
try:
    from xtomo import (
        compare_brightness_emissivity,
        core_xray_emissivity,
        plot_core_emissivity,
    )
    from xtomo.chord_masking import build_inversion_chord_mask
    from xtomo.xtomo_mds import bipolar_radii, read_xray_brightness
except ImportError as exc:
    sys.exit(
        f"Cannot import the 'xtomo' package: {exc}\n"
        "Make sure you have run:  pip install -e .\n"
        "from the xTomo project root before running this script."
    )


# ---------------------------------------------------------------------------
# Defaults (edit these directly or override on the command line)
# ---------------------------------------------------------------------------
DEFAULT_SHOT = 1120927023  # 1140221013
DEFAULT_TIME = 1.2  # s  — time slice to visualise
DEFAULT_TSTART = 0.8  # s  — start of inversion window
DEFAULT_TSTOP = 1.6  # s  — end   of inversion window
DEFAULT_DT = 0.05  # s  — time step between reconstructions
DEFAULT_LMAX = 15  #     — max radial Bessel harmonic
DEFAULT_EFIT = "analysis"
DEFAULT_GRADIENT_SPIKE_THRESHOLD = 1500.0  # W/m^2


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def run(
    shot: int = DEFAULT_SHOT,
    time: float = DEFAULT_TIME,
    tstart: float = DEFAULT_TSTART,
    tstop: float = DEFAULT_TSTOP,
    dt: float = DEFAULT_DT,
    lmax: int = DEFAULT_LMAX,
    efit_tree: str = DEFAULT_EFIT,
    logscale: bool = False,
    save_prefix: str = "",
    noflux: bool = False,
    novessel: bool = False,
    remove_zero_chords: bool = True,
    zero_chord_threshold: float = 0.02,
    mask_gradient_spikes: bool = False,
    gradient_spike_threshold: float = DEFAULT_GRADIENT_SPIKE_THRESHOLD,
    mask_inversion_chords: bool = True,
) -> None:
    """
    Run the full example pipeline.

    Parameters
    ----------
    shot         : MDS shot number
    time         : time point [s] for visualisation
    tstart/tstop/dt : inversion time window [s]
    lmax         : max Bessel harmonic index
    efit_tree    : MDSplus tree for EFIT data
    logscale     : use log scale on brightness panels
    save_prefix  : if non-empty, save figures to  <save_prefix>_*.pdf
    noflux       : skip EFIT flux contours on the 2-D plot
    novessel     : skip vessel geometry on the 2-D plot
    remove_zero_chords    : mask interior near-zero chords
    zero_chord_threshold  : chord mask threshold as fraction of array max
    mask_gradient_spikes  : mask isolated high chords using an absolute adjacent
                            brightness-jump threshold on the selected profile
    gradient_spike_threshold : absolute spike threshold [W/m^2]
    mask_inversion_chords : apply chord masking to inversion equations
    """

    print("=" * 60)
    print(f"  XTOMO example  —  shot {shot}")
    print(f"  Inversion window : {tstart:.3f} – {tstop:.3f} s  (dt = {dt:.3f} s)")
    print(f"  Visualisation at : {time:.3f} s")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Tomographic inversion
    # ------------------------------------------------------------------
    print("\n[1/3]  Running tomographic inversion ...")

    inversion_chord_mask = None
    if (remove_zero_chords or mask_gradient_spikes) and mask_inversion_chords:
        print("  Building automatic chord mask for inversion ...")
        time1_all, b1_all, r1, a1 = read_xray_brightness(shot, array=1, fix_bad_channels=True)
        time3_all, b3_all, r3, a3 = read_xray_brightness(shot, array=3, fix_bad_channels=True)
        t_idx1 = int(np.argmin(np.abs(time1_all - time)))
        t_idx3 = int(np.argmin(np.abs(time3_all - time)))
        _mask1, _mask3, inversion_chord_mask = build_inversion_chord_mask(
            b1_all,
            b3_all,
            apply_zero_mask=remove_zero_chords,
            threshold=zero_chord_threshold,
            edge_keep=2,
            profile_array1=b1_all[t_idx1, :] if mask_gradient_spikes else None,
            profile_array3=b3_all[t_idx3, :] if mask_gradient_spikes else None,
            chord_positions_array1=bipolar_radii(r1, a1, 1) if mask_gradient_spikes else None,
            chord_positions_array3=bipolar_radii(r3, a3, 3) if mask_gradient_spikes else None,
            max_gradient_abs=gradient_spike_threshold if mask_gradient_spikes else None,
        )
        print(f"  Inversion will use {int(inversion_chord_mask.sum())}/76 chords")

    emissivity, r, z, t, ok = core_xray_emissivity(
        shot,
        tstart=tstart,
        tstop=tstop,
        dt=dt,
        use_efit_center=True,
        auto_calc_rnorm=True,
        efit_tree=efit_tree,
        lmax=lmax,
        chord_mask=inversion_chord_mask,
    )

    if not ok:
        print("ERROR: Inversion failed — check MDSplus connectivity and shot number.")
        return

    print(f"  Emissivity array shape: {emissivity.shape}  (nr × nz × ntimes)")
    print(f"  R grid : {r[0]:.3f} – {r[-1]:.3f} m   ({len(r)} points)")
    print(f"  Z grid : {z[0]:.3f} – {z[-1]:.3f} m   ({len(z)} points)")
    print(f"  Times  : {t[0]:.3f} – {t[-1]:.3f} s   ({len(t)} steps)")

    # ------------------------------------------------------------------
    # Step 2 — 2-D emissivity plot on poloidal cross-section
    # ------------------------------------------------------------------
    print("\n[2/3]  Plotting 2-D emissivity ...")
    save_2d = f"{save_prefix}_emissivity_2d_{shot}_t{time:.3f}.pdf" if save_prefix else ""
    plot_core_emissivity(
        shot,
        emissivity,
        r,
        z,
        t,
        time=time,
        noflux=noflux,
        novessel=novessel,
        efit_tree=efit_tree,
        halt_on_efit_error=True,
        n_flux_contours=10,
        cmap="hot",
        save=save_2d,
    )
    # ax_2d.get_figure().suptitle(
    #     f"Shot {shot}  —  2-D x-ray emissivity",
    #     fontsize=11,
    #     y=1.01,
    # )

    # ------------------------------------------------------------------
    # Step 3 — Brightness-vs-emissivity comparison
    # ------------------------------------------------------------------
    print("\n[3/3]  Comparing chord brightness with inverted emissivity ...")
    save_cmp = f"{save_prefix}_brightness_comparison_{shot}_t{time:.3f}.pdf" if save_prefix else ""
    fig_cmp, axes_cmp = compare_brightness_emissivity(
        shot,
        time,
        lmax=lmax,
        efit_tree=efit_tree,
        logscale=logscale,
        save=save_cmp,
        remove_zero_chords=remove_zero_chords,
        zero_chord_threshold=zero_chord_threshold,
        mask_gradient_spikes=mask_gradient_spikes,
        gradient_spike_threshold=gradient_spike_threshold,
        mask_inversion_chords=mask_inversion_chords,
    )

    plt.show()

    print("\nDone.  Close the figure windows to exit.")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end XTOMO example: inversion + 2-D plot + comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument(
        "--time", type=float, default=DEFAULT_TIME, help="Time slice to visualise [s]"
    )
    parser.add_argument(
        "--tstart", type=float, default=DEFAULT_TSTART, help="Start of inversion window [s]"
    )
    parser.add_argument(
        "--tstop", type=float, default=DEFAULT_TSTOP, help="End of inversion window [s]"
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Inversion time step [s]")
    parser.add_argument(
        "--lmax", type=int, default=DEFAULT_LMAX, help="Max radial Bessel harmonic index"
    )
    parser.add_argument(
        "--efit-tree",
        dest="efit_tree",
        default=DEFAULT_EFIT,
        help="MDSplus tree containing EFIT equilibrium data",
    )
    parser.add_argument(
        "--logscale", action="store_true", help="Use log scale on brightness panels"
    )
    parser.add_argument("--noflux", action="store_true", help="Skip EFIT flux contours on 2-D plot")
    parser.add_argument("--novessel", action="store_true", help="Skip vessel geometry on 2-D plot")
    parser.add_argument(
        "--save-prefix",
        dest="save_prefix",
        default="",
        metavar="PATH",
        help="Save figures to <PATH>_*.pdf  (default: do not save)",
    )
    parser.add_argument(
        "--no-zero-filter",
        action="store_true",
        help="Keep near-zero interior chords in plots and inversion",
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
        help="Mask isolated high chords using an absolute adjacent-channel gradient threshold",
    )
    parser.add_argument(
        "--gradient-spike-threshold",
        type=float,
        default=DEFAULT_GRADIENT_SPIKE_THRESHOLD,
        dest="gradient_spike_threshold",
        help="Absolute adjacent-channel spike threshold in W/m^2",
    )
    parser.add_argument(
        "--no-inversion-mask",
        action="store_true",
        help="Do not apply chord masking to inversion equations",
    )
    args = parser.parse_args()

    run(
        shot=args.shot,
        time=args.time,
        tstart=args.tstart,
        tstop=args.tstop,
        dt=args.dt,
        lmax=args.lmax,
        efit_tree=args.efit_tree,
        logscale=args.logscale,
        save_prefix=args.save_prefix,
        noflux=args.noflux,
        novessel=args.novessel,
        remove_zero_chords=not args.no_zero_filter,
        zero_chord_threshold=args.zero_chord_threshold,
        mask_gradient_spikes=args.mask_gradient_spikes,
        gradient_spike_threshold=args.gradient_spike_threshold,
        mask_inversion_chords=not args.no_inversion_mask,
    )


if __name__ == "__main__":
    main()
