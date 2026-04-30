"""
compare_brightness_emissivity.py
=================================
Quick-look script to compare:

  1. Raw SXR chord brightness profiles from XTOMO arrays 1 and 3
     (as in plot_xtomo_profile_for_Rice.m, but for both arrays), and

  2. The tomographically inverted midplane emissivity profile E(R, Z≈0)
     from core_xray_emissivity().

Both are shown at the same time point and shot, giving a side-by-side
picture of the line-integrated measurement and the locally inverted profile.

Usage
-----
    python compare_brightness_emissivity.py 1140221013 1.2

    # Optional flags:
    python compare_brightness_emissivity.py 1140221013 1.2 \\
        --lmax 15 --logscale --save comparison.pdf --efit-tree analysis
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from .chord_masking import build_inversion_chord_mask
from .core_xray_emissivity import core_xray_emissivity
from .xtomo_mds import (
    bipolar_radii,
    open_tree,
    read_efit_data,
)

# ---------------------------------------------------------------------------
# Helper: read brightness directly from BRIGHTNESSES subtree (like MATLAB code)
# ---------------------------------------------------------------------------


def read_brightness_profiles(shot: int, array: int, time_target: float):
    """
    Read brightness profiles from \\top.brightnesses.array_N:chord_XX,
    subtract a pre-shot baseline, and return the profile nearest *time_target*.

    Follows the MATLAB code plot_xtomo_profile_for_Rice.m exactly:
      - Reads from the BRIGHTNESSES subtree (already brightness in W/m^2)
      - Baseline = mean of samples at t < 0

    Returns
    -------
    chord_radii  : ndarray (38,)  [m]  unsigned impact parameters
    chord_angles : ndarray (38,)  [rad]
    p_bipolar    : ndarray (38,)  [m]  signed impact parameters
    brightness   : ndarray (38,)  [W/m^2]  profile at nearest time
    timebase     : ndarray        full timebase of array
    all_signals  : ndarray (npts, 38)  all time samples (W/m^2)
    """
    conn = open_tree(shot, "xtomo")

    timebase = np.asarray(
        conn.get(f"dim_of(\\top.brightnesses.array_{array}:chord_01)").data(),
        dtype=float,
    )
    ntimes = len(timebase)
    signals = np.full((ntimes, 38), np.nan)

    chord_radii = np.asarray(
        conn.get(f"\\top.brightnesses.array_{array}:chord_radii").data(), dtype=float
    ).squeeze()
    chord_angles = np.asarray(
        conn.get(f"\\top.brightnesses.array_{array}:chord_angles").data(), dtype=float
    ).squeeze()

    for ichan in range(1, 39):
        node = f"\\top.brightnesses.array_{array}:chord_{ichan:02d}"
        try:
            sig = np.asarray(conn.get(node).data(), dtype=float)
            n = min(len(sig), ntimes)
            signals[:n, ichan - 1] = sig[:n]
        except Exception:
            pass

    conn.closeAllTrees()

    # Subtract pre-shot baseline (t < 0), matching the MATLAB code
    idx_base = np.where(timebase < 0.0)[0]
    if len(idx_base) > 0:
        baseline = np.nanmean(signals[idx_base, :], axis=0)
        signals -= baseline[np.newaxis, :]

    t_idx = int(np.argmin(np.abs(timebase - time_target)))
    brightness_at_t = signals[t_idx, :]

    p_bipolar = bipolar_radii(chord_radii, chord_angles, array).squeeze()  # (38,)

    return chord_radii, chord_angles, p_bipolar, brightness_at_t, timebase, signals


# ---------------------------------------------------------------------------
# Main comparison plot
# ---------------------------------------------------------------------------


def compare_brightness_emissivity(
    shot: int,
    time: float,
    *,
    lmax: int = 15,
    svd_tol: float = 0.1,
    efit_tree: str = "analysis",
    logscale: bool = False,
    save: str = "",
    remove_zero_chords: bool = True,
    zero_chord_threshold: float = 0.02,
    mask_gradient_spikes: bool = False,
    gradient_spike_threshold: float = 1500.0,
    mask_inversion_chords: bool = True,
    use_latex_style: bool = True,
):
    """
    Generate a three-panel comparison figure:

      Panel A (left)  – XTOMO array 1 brightness profiles at *time*
      Panel B (centre) – XTOMO array 3 brightness profiles at *time*
      Panel C (right)  – Tomographically inverted midplane emissivity at *time*

    Each brightness panel distinguishes upper-looking chords (positive p) from
    lower-looking chords (negative p), matching the MATLAB code convention.

    Parameters
    ----------
    shot      : MDS shot number
    time      : time point [s]
    lmax      : max Bessel harmonic (passed to core_xray_emissivity)
    svd_tol   : SVD cutoff  (passed to core_xray_emissivity)
    efit_tree : MDSplus tree for EFIT data
    logscale              : use logarithmic y-axis on brightness panels
    save                  : if non-empty, save figure to this path (e.g. "out.pdf")
    remove_zero_chords    : mask interior chords whose peak amplitude < threshold
    zero_chord_threshold  : fraction of array maximum below which a chord is masked
    mask_gradient_spikes  : mask isolated high chords using an absolute adjacent
                            brightness-jump threshold on the selected profile
    gradient_spike_threshold : absolute spike threshold [W/m^2]
    mask_inversion_chords : apply the same chord mask to the inversion equations
    use_latex_style       : apply Computer Modern serif font style to all figures
    """

    print(f"\n=== compare_brightness_emissivity: shot {shot},  t = {time:.3f} s ===\n")

    # ---- Optional LaTeX-like font style (Computer Modern serif) ---------
    if use_latex_style:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "mathtext.fontset": "cm",
                "font.size": 11,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 9,
            }
        )
        # Tip: add  plt.rcParams['text.usetex'] = True  if pdflatex is available.

    # ------------------------------------------------------------------
    # 1. Read raw brightness for arrays 1 and 3
    # ------------------------------------------------------------------
    print("Reading brightness array 1 ...")
    (chord_r1, chord_a1, p_bip1, bright1, time1, all1) = read_brightness_profiles(shot, 1, time)

    print("Reading brightness array 3 ...")
    (chord_r3, chord_a3, p_bip3, bright3, time3, all3) = read_brightness_profiles(shot, 3, time)

    # Actual time reported in plot title (nearest brightness sample)
    t_idx1 = int(np.argmin(np.abs(time1 - time)))
    t_actual = time1[t_idx1]

    # Upper / lower chord separation per the bipolar convention
    # Array 1: upper = p_bipolar >= 0, lower = p_bipolar < 0
    # Array 3: same convention
    upper1 = p_bip1 >= 0
    lower1 = ~upper1
    upper3 = p_bip3 >= 0
    lower3 = ~upper3

    # ---- Chord masking --------------------------------------------------
    if remove_zero_chords or mask_gradient_spikes:
        mask1, mask3, inversion_mask_auto = build_inversion_chord_mask(
            all1,
            all3,
            apply_zero_mask=remove_zero_chords,
            threshold=zero_chord_threshold,
            edge_keep=2,
            profile_array1=bright1 if mask_gradient_spikes else None,
            profile_array3=bright3 if mask_gradient_spikes else None,
            chord_positions_array1=p_bip1 if mask_gradient_spikes else None,
            chord_positions_array3=p_bip3 if mask_gradient_spikes else None,
            max_gradient_abs=gradient_spike_threshold if mask_gradient_spikes else None,
        )
        n_drop1 = 38 - int(mask1.sum())
        n_drop3 = 38 - int(mask3.sum())
        if n_drop1:
            print(f"  Masking {n_drop1} chord(s) from array 1")
        if n_drop3:
            print(f"  Masking {n_drop3} chord(s) from array 3")
    else:
        mask1 = np.ones(38, dtype=bool)
        mask3 = np.ones(38, dtype=bool)
        inversion_mask_auto = None

    if (remove_zero_chords or mask_gradient_spikes) and mask_inversion_chords:
        if inversion_mask_auto is None:
            raise RuntimeError("Internal error: expected a chord mask but none was built.")
        inversion_chord_mask = inversion_mask_auto
        print(f"  Inversion will use {int(inversion_chord_mask.sum())}/76 chords")
    else:
        inversion_chord_mask = None

    # ------------------------------------------------------------------
    # 2. Compute tomographic emissivity around the requested time
    # ------------------------------------------------------------------
    print("\nRunning tomographic inversion ...")
    dt_inv = 0.005  # 5 ms window around the requested time

    emissivity, r_em, z_em, t_em, ok = core_xray_emissivity(
        shot,
        tstart=time - dt_inv,
        tstop=time + dt_inv,
        dt=dt_inv,
        use_efit_center=True,
        auto_calc_rnorm=True,
        efit_tree=efit_tree,
        lmax=lmax,
        svd_tol=svd_tol,
        chord_mask=inversion_chord_mask,
        verbose=True,
    )

    # Midplane (Z = 0) emissivity profile vs R
    if ok:
        t_em_idx = int(np.argmin(np.abs(t_em - time)))
        z_mid_idx = int(np.argmin(np.abs(z_em)))
        emiss_mid = emissivity[:, z_mid_idx, t_em_idx]  # (nr,)
        t_emiss_plot = t_em[t_em_idx]
        print(f"  Emissivity time slice: {t_emiss_plot:.4f} s  (z ≈ {z_em[z_mid_idx]*100:.1f} cm)")
        # Fetch magnetic axis R for minor-radius axis conversion
        r_magaxis: float | None = None
        try:
            _ef_t, _r_magx, _, _, _, _ = read_efit_data(shot, tree=efit_tree)
            _ef_idx = int(np.argmin(np.abs(_ef_t - t_emiss_plot)))
            r_magaxis = float(_r_magx[_ef_idx])
            print(f"  Magnetic axis: R0 = {r_magaxis:.4f} m")
        except Exception as _exc:
            print(f"  Warning: could not read R0 ({_exc}); using absolute R.")
    else:
        emiss_mid = None
        r_magaxis = None
        print("  WARNING: inversion failed; emissivity panel will be empty.")

    # ------------------------------------------------------------------
    # 3. Build figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        tight_layout=True,
        sharey=False,
    )
    fig.suptitle(f"Shot {shot},  $t \\approx$ {t_actual:.3f} s", fontsize=13, fontweight="bold")

    # ---- Panel A: Array 1 brightness -----------------------------------
    ax1 = axes[0]
    kw_upper = {"marker": ".", "markersize": 4, "linewidth": 1.2}
    kw_lower = {"marker": ".", "markersize": 4, "linewidth": 1.2, "linestyle": "--"}
    u1 = upper1 & mask1
    l1 = lower1 & mask1
    x1_ex = p_bip1[~mask1]

    if logscale:
        b1u = np.clip(bright1[u1] / 1e3, 1e-3, None)
        b1l = np.clip(bright1[l1] / 1e3, 1e-3, None)
        ax1.semilogy(p_bip1[u1], b1u, color="red", label="Outboard", **kw_upper)
        ax1.semilogy(p_bip1[l1], b1l, color="blue", label="Inboard", **kw_lower)
        if x1_ex.size > 0:
            y1_ex = np.clip(bright1[~mask1] / 1e3, 1e-3, None)
            ax1.scatter(
                x1_ex,
                y1_ex,
                s=22,
                facecolors="none",
                edgecolors="0.55",
                linewidths=1.0,
                alpha=0.9,
                label="Excluded datapoints",
                zorder=1,
            )
    else:
        ax1.plot(p_bip1[u1], bright1[u1] / 1e3, color="red", label="Outboard", **kw_upper)
        ax1.plot(p_bip1[l1], bright1[l1] / 1e3, color="blue", label="Inboard", **kw_lower)
        if x1_ex.size > 0:
            ax1.scatter(
                x1_ex,
                bright1[~mask1] / 1e3,
                s=22,
                facecolors="none",
                edgecolors="0.55",
                linewidths=1.0,
                alpha=0.9,
                label="Excluded datapoints",
                zorder=1,
            )

    ax1.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    ax1.set_xlabel(r"Impact parameter  $p$  (m)")
    ax1.set_ylabel(r"Brightness  (kW m$^{-2}$)")
    ax1.set_title("Array 1 (SXR)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- Panel B: Array 3 brightness -----------------------------------
    ax3 = axes[1]
    u3 = upper3 & mask3
    l3 = lower3 & mask3
    x3_ex = p_bip3[~mask3]

    if logscale:
        b3u = np.clip(bright3[u3] / 1e3, 1e-3, None)
        b3l = np.clip(bright3[l3] / 1e3, 1e-3, None)
        ax3.semilogy(p_bip3[u3], b3u, color="red", label="Above midplane", **kw_upper)
        ax3.semilogy(p_bip3[l3], b3l, color="blue", label="Below midplane", **kw_lower)
        if x3_ex.size > 0:
            y3_ex = np.clip(bright3[~mask3] / 1e3, 1e-3, None)
            ax3.scatter(
                x3_ex,
                y3_ex,
                s=22,
                facecolors="none",
                edgecolors="0.55",
                linewidths=1.0,
                alpha=0.9,
                label="Excluded datapoints",
                zorder=1,
            )
    else:
        ax3.plot(p_bip3[u3], bright3[u3] / 1e3, color="red", label="Above midplane", **kw_upper)
        ax3.plot(p_bip3[l3], bright3[l3] / 1e3, color="blue", label="Below midplane", **kw_lower)
        if x3_ex.size > 0:
            ax3.scatter(
                x3_ex,
                bright3[~mask3] / 1e3,
                s=22,
                facecolors="none",
                edgecolors="0.55",
                linewidths=1.0,
                alpha=0.9,
                label="Excluded datapoints",
                zorder=1,
            )

    ax3.axvline(0, color="grey", linewidth=0.5, linestyle=":")
    ax3.set_xlabel(r"Impact parameter  $p$  (m)")
    ax3.set_ylabel(r"Brightness  (kW m$^{-2}$)")
    ax3.set_title("Array 3 (SXR)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ---- Panel C: Midplane emissivity (minor-radius axis) ---------------
    axE = axes[2]
    if ok and emiss_mid is not None:
        emiss_kW = emiss_mid / 1e3
        r_minor_em = (r_em - r_magaxis) if r_magaxis is not None else r_em
        xlabel_em = r"Minor radius  $r = R - R_0$  (m)" if r_magaxis is not None else r"$R$  (m)"
        title_em = (
            f"Midplane emissivity  ($Z \\approx$ {z_em[z_mid_idx]*100:.0f} cm)"
            if r_magaxis is None
            else f"Midplane emissivity  ($R_0 \\approx$ {r_magaxis:.3f} m)"
        )

        if logscale:
            axE.semilogy(
                r_minor_em, np.clip(emiss_kW, 1e-3, None), color="green", linewidth=1.5, marker="."
            )
        else:
            axE.plot(r_minor_em, emiss_kW, color="green", linewidth=1.5, marker=".", markersize=4)
        axE.axvline(0, color="grey", linewidth=0.5, linestyle=":")
        axE.set_xlabel(xlabel_em)
        axE.set_ylabel(r"Emissivity  (kW m$^{-3}$)")
        axE.set_title(title_em)
        axE.grid(True, alpha=0.3)

        # ---- Normalised profile comparison on a shared minor-radius axis ----
        # brightness p_bip3 (signed) and emissivity (R - R0) are both in metres
        # of minor radius, so they now share a common physical x-axis:
        #   negative r → inboard / below-midplane chords
        #   positive r → outboard / above-midplane chords
        fig2, ax_cmp = plt.subplots(figsize=(6.5, 4.5), tight_layout=True)

        b3_max = float(np.nanmax(np.abs(bright3[mask3]))) or 1.0
        em_peak = float(np.nanmax(np.abs(emiss_mid))) or 1.0
        b3_norm = bright3 / b3_max
        em_norm = emiss_mid / em_peak

        # Sort each curve by its x-coordinate for a smooth line
        iu = np.argsort(p_bip3[u3])
        il = np.argsort(p_bip3[l3])
        iem = np.argsort(r_minor_em)

        ax_cmp.plot(
            p_bip3[u3][iu],
            b3_norm[u3][iu],
            color="red",
            linewidth=1.5,
            marker=".",
            markersize=5,
            label="Array 3  (above midplane, norm.)",
        )
        ax_cmp.plot(
            p_bip3[l3][il],
            b3_norm[l3][il],
            color="blue",
            linewidth=1.5,
            marker=".",
            markersize=5,
            linestyle="--",
            label="Array 3  (below midplane, norm.)",
        )
        ax_cmp.plot(
            r_minor_em[iem],
            em_norm[iem],
            color="green",
            linewidth=2.0,
            label="Midplane emissivity  (norm.)",
        )
        ax_cmp.axvline(
            0, color="grey", linewidth=0.6, linestyle=":", label="$r = 0$  (magnetic axis)"
        )
        ax_cmp.set_xlabel(r"Minor radius  $r$  (m)")
        ax_cmp.set_ylabel("Normalised amplitude")
        ax_cmp.set_title(f"Profile comparison:  shot {shot},  $t \\approx$ {t_actual:.3f} s")
        ax_cmp.legend()
        ax_cmp.grid(True, alpha=0.3)

        if save:
            fig2.savefig(save[-4:] + "_normalised.pdf", transparent=True, bbox_inches="tight")
            print("Normalised comparison saved.")
    else:
        axE.text(
            0.5,
            0.5,
            "Inversion failed",
            transform=axE.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="red",
        )

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Main figure saved to {save}")

    return fig, axes


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry point: ``xtomo-compare``."""
    parser = argparse.ArgumentParser(
        description="Compare XTOMO brightness profiles with inverted emissivity"
    )
    parser.add_argument("shot", type=int, help="MDS shot number")
    parser.add_argument("time", type=float, help="Time to plot [s]")
    parser.add_argument("--lmax", type=int, default=15)
    parser.add_argument("--svd-tol", type=float, default=0.1, dest="svd_tol")
    parser.add_argument("--efit-tree", default="analysis", dest="efit_tree")
    parser.add_argument(
        "--logscale", action="store_true", help="Use logarithmic y-axis for brightness panels"
    )
    parser.add_argument("--save", type=str, default="", help="Save figure to this file path")
    parser.add_argument(
        "--no-zero-filter",
        action="store_true",
        help="Keep near-zero interior chords in brightness plots",
    )
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.02,
        dest="zero_chord_threshold",
        help="Fraction of array max below which a chord is masked (default 0.02)",
    )
    parser.add_argument(
        "--mask-gradient-spikes",
        action="store_true",
        help="Mask isolated high chords using an absolute adjacent-channel gradient threshold",
    )
    parser.add_argument(
        "--gradient-spike-threshold",
        type=float,
        default=1500.0,
        dest="gradient_spike_threshold",
        help="Absolute adjacent-channel spike threshold in W/m^2",
    )
    parser.add_argument(
        "--no-latex", action="store_true", help="Do not apply LaTeX/CM serif font style"
    )
    parser.add_argument(
        "--no-inversion-mask",
        action="store_true",
        help="Do not apply chord masking to the inversion system",
    )
    args = parser.parse_args()

    compare_brightness_emissivity(
        args.shot,
        args.time,
        lmax=args.lmax,
        svd_tol=args.svd_tol,
        efit_tree=args.efit_tree,
        logscale=args.logscale,
        save=args.save,
        remove_zero_chords=not args.no_zero_filter,
        zero_chord_threshold=args.zero_chord_threshold,
        mask_gradient_spikes=args.mask_gradient_spikes,
        gradient_spike_threshold=args.gradient_spike_threshold,
        mask_inversion_chords=not args.no_inversion_mask,
        use_latex_style=not args.no_latex,
    )
    plt.show()


if __name__ == "__main__":
    main()
