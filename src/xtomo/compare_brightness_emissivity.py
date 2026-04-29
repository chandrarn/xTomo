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
import numpy as np
import matplotlib.pyplot as plt
import mdsthin as mds

from .xtomo_mds import (
    XTOMO_SERVER,
    open_tree,
    read_xray_brightness,
    bipolar_radii,
)
from .core_xray_emissivity import core_xray_emissivity


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
        conn.get(f'dim_of(\\top.brightnesses.array_{array}:chord_01)').data(),
        dtype=float,
    )
    ntimes   = len(timebase)
    signals  = np.full((ntimes, 38), np.nan)

    chord_radii  = np.asarray(
        conn.get(f'\\top.brightnesses.array_{array}:chord_radii').data(),
        dtype=float)
    chord_angles = np.asarray(
        conn.get(f'\\top.brightnesses.array_{array}:chord_angles').data(),
        dtype=float)

    for ichan in range(1, 39):
        node = f'\\top.brightnesses.array_{array}:chord_{ichan:02d}'
        try:
            sig = np.asarray(conn.get(node).data(), dtype=float)
            n   = min(len(sig), ntimes)
            signals[:n, ichan - 1] = sig[:n]
        except Exception:
            pass

    conn.closeAllTrees()

    # Subtract pre-shot baseline (t < 0), matching the MATLAB code
    idx_base = np.where(timebase < 0.0)[0]
    if len(idx_base) > 0:
        baseline = np.nanmean(signals[idx_base, :], axis=0)
        signals -= baseline[np.newaxis, :]

    t_idx     = int(np.argmin(np.abs(timebase - time_target)))
    brightness_at_t = signals[t_idx, :]

    p_bipolar = bipolar_radii(chord_radii, chord_angles, array)

    return chord_radii, chord_angles, p_bipolar, brightness_at_t, timebase, signals


# ---------------------------------------------------------------------------
# Main comparison plot
# ---------------------------------------------------------------------------

def compare_brightness_emissivity(
    shot:       int,
    time:       float,
    *,
    lmax:       int   = 15,
    svd_tol:    float = 0.1,
    efit_tree:  str   = 'analysis',
    logscale:   bool  = False,
    save:       str   = '',
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
    logscale  : use logarithmic y-axis on brightness panels
    save      : if non-empty, save figure to this path (e.g. "out.pdf")
    """

    print(f'\n=== compare_brightness_emissivity: shot {shot},  t = {time:.3f} s ===\n')

    # ------------------------------------------------------------------
    # 1. Read raw brightness for arrays 1 and 3
    # ------------------------------------------------------------------
    print('Reading brightness array 1 ...')
    (chord_r1, chord_a1, p_bip1, bright1,
     time1, all1) = read_brightness_profiles(shot, 1, time)

    print('Reading brightness array 3 ...')
    (chord_r3, chord_a3, p_bip3, bright3,
     time3, all3) = read_brightness_profiles(shot, 3, time)

    # Actual time reported in plot title (nearest brightness sample)
    t_idx1   = int(np.argmin(np.abs(time1 - time)))
    t_actual = time1[t_idx1]

    phi_deg1 = np.degrees(chord_a1)
    phi_deg3 = np.degrees(chord_a3)

    # Upper / lower chord separation per the bipolar convention
    # Array 1: upper = p_bipolar >= 0, lower = p_bipolar < 0
    # Array 3: same convention
    upper1 = p_bip1 >= 0
    lower1 = ~upper1
    upper3 = p_bip3 >= 0
    lower3 = ~upper3

    # ------------------------------------------------------------------
    # 2. Compute tomographic emissivity around the requested time
    # ------------------------------------------------------------------
    print('\nRunning tomographic inversion ...')
    dt_inv = 0.005   # 5 ms window around the requested time

    emissivity, r_em, z_em, t_em, ok = core_xray_emissivity(
        shot,
        tstart=time - dt_inv,
        tstop=time  + dt_inv,
        dt=dt_inv,
        use_efit_center=True,
        auto_calc_rnorm=True,
        efit_tree=efit_tree,
        lmax=lmax,
        svd_tol=svd_tol,
        verbose=True,
    )

    # Midplane (Z = 0) emissivity profile vs R
    if ok:
        t_em_idx    = int(np.argmin(np.abs(t_em - time)))
        z_mid_idx   = int(np.argmin(np.abs(z_em)))
        emiss_mid   = emissivity[:, z_mid_idx, t_em_idx]   # (nr,)
        t_emiss_plot = t_em[t_em_idx]
        print(f'  Emissivity time slice: {t_emiss_plot:.4f} s  (z ≈ {z_em[z_mid_idx]*100:.1f} cm)')
    else:
        emiss_mid = None
        print('  WARNING: inversion failed; emissivity panel will be empty.')

    # ------------------------------------------------------------------
    # 3. Build figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, 4.5),
        tight_layout=True,
        sharey=False,
    )
    fig.suptitle(
        f'Shot {shot},  t ≈ {t_actual:.3f} s',
        fontsize=13, fontweight='bold')

    plot_fn = axes[0].semilogy if logscale else axes[0].plot

    # ---- Panel A: Array 1 brightness -----------------------------------
    ax1 = axes[0]
    kw_upper = dict(marker='.', markersize=4, linewidth=1.2)
    kw_lower = dict(marker='.', markersize=4, linewidth=1.2, linestyle='--')

    if logscale:
        b1u = np.clip(bright1[upper1] / 1e3, 1e-3, None)
        b1l = np.clip(bright1[lower1] / 1e3, 1e-3, None)
        ax1.semilogy(p_bip1[upper1], b1u, color='red',   label='above midplane', **kw_upper)
        ax1.semilogy(p_bip1[lower1], b1l, color='blue',  label='below midplane', **kw_lower)
    else:
        ax1.plot(p_bip1[upper1], bright1[upper1] / 1e3, color='red',  label='above midplane', **kw_upper)
        ax1.plot(p_bip1[lower1], bright1[lower1] / 1e3, color='blue', label='below midplane', **kw_lower)

    ax1.axvline(0, color='grey', linewidth=0.5, linestyle=':')
    ax1.set_xlabel('Chord radius (m)', fontsize=11)
    ax1.set_ylabel('Brightness (kW/m²)', fontsize=11)
    ax1.set_title('Array 1 (SXR)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- Panel B: Array 3 brightness -----------------------------------
    ax3 = axes[1]
    if logscale:
        b3u = np.clip(bright3[upper3] / 1e3, 1e-3, None)
        b3l = np.clip(bright3[lower3] / 1e3, 1e-3, None)
        ax3.semilogy(p_bip3[upper3], b3u, color='red',  label='above midplane', **kw_upper)
        ax3.semilogy(p_bip3[lower3], b3l, color='blue', label='below midplane', **kw_lower)
    else:
        ax3.plot(p_bip3[upper3], bright3[upper3] / 1e3, color='red',  label='above midplane', **kw_upper)
        ax3.plot(p_bip3[lower3], bright3[lower3] / 1e3, color='blue', label='below midplane', **kw_lower)

    ax3.axvline(0, color='grey', linewidth=0.5, linestyle=':')
    ax3.set_xlabel('Chord radius (m)', fontsize=11)
    ax3.set_ylabel('Brightness (kW/m²)', fontsize=11)
    ax3.set_title('Array 3 (SXR)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ---- Panel C: Midplane emissivity ----------------------------------
    axE = axes[2]
    if ok and emiss_mid is not None:
        emiss_kW = emiss_mid / 1e3
        if logscale:
            axE.semilogy(r_em, np.clip(emiss_kW, 1e-3, None),
                         color='green', linewidth=1.5, marker='.')
        else:
            axE.plot(r_em, emiss_kW, color='green', linewidth=1.5, marker='.',
                     markersize=4)
        axE.set_xlabel('R (m)', fontsize=11)
        axE.set_ylabel('Emissivity (kW/m³)', fontsize=11)
        axE.set_title(f'Midplane emissivity (Z ≈ {z_em[z_mid_idx]*100:.0f} cm)', fontsize=11)
        axE.grid(True, alpha=0.3)

        # ---- Dual axis: normalised brightness scale for visual comparison ---
        # Normalise both curves to their peak to overlay profile shapes
        fig2, ax_cmp = plt.subplots(figsize=(6.0, 4.0), tight_layout=True)

        # Construct a 1-D brightness profile by averaging upper and lower for
        # array 3 (using unsigned radii) as a proxy for the line integral
        b3_norm  = bright3 / np.nanmax(np.abs(bright3))
        em_norm  = emiss_mid / np.nanmax(np.abs(emiss_mid)) if np.nanmax(np.abs(emiss_mid)) > 0 else emiss_mid

        ax_cmp.plot(np.abs(p_bip3[upper3]), b3_norm[upper3],
                    color='red',   linewidth=1.5, marker='.', markersize=5,
                    label='Array 3 brightness (above, normalised)')
        ax_cmp.plot(np.abs(p_bip3[lower3]), b3_norm[lower3],
                    color='blue',  linewidth=1.5, marker='.', markersize=5,
                    linestyle='--', label='Array 3 brightness (below, normalised)')
        ax_cmp.plot(r_em, em_norm,
                    color='green', linewidth=2.0, marker=None,
                    label='Midplane emissivity (normalised)')
        ax_cmp.set_xlabel('R  or  |p|  (m)', fontsize=12)
        ax_cmp.set_ylabel('Normalised amplitude', fontsize=12)
        ax_cmp.set_title(
            f'Profile comparison:  shot {shot},  t ≈ {t_actual:.3f} s',
            fontsize=12)
        ax_cmp.legend(fontsize=9)
        ax_cmp.set_xlim(left=0)
        ax_cmp.grid(True, alpha=0.3)

        if save:
            fig2.savefig(save.replace('.', '_normalised.', 1), dpi=150,
                         bbox_inches='tight')
            print(f'Normalised comparison saved.')
    else:
        axE.text(0.5, 0.5, 'Inversion failed', transform=axE.transAxes,
                 ha='center', va='center', fontsize=12, color='red')

    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f'Main figure saved to {save}')

    return fig, axes


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Console entry point: ``xtomo-compare``."""
    parser = argparse.ArgumentParser(
        description='Compare XTOMO brightness profiles with inverted emissivity')
    parser.add_argument('shot',  type=int,   help='MDS shot number')
    parser.add_argument('time',  type=float, help='Time to plot [s]')
    parser.add_argument('--lmax',    type=int,   default=15)
    parser.add_argument('--svd-tol', type=float, default=0.1,
                        dest='svd_tol')
    parser.add_argument('--efit-tree', default='analysis',
                        dest='efit_tree')
    parser.add_argument('--logscale', action='store_true',
                        help='Use logarithmic y-axis for brightness panels')
    parser.add_argument('--save', type=str, default='',
                        help='Save figure to this file path')
    args = parser.parse_args()

    compare_brightness_emissivity(
        args.shot,
        args.time,
        lmax=args.lmax,
        svd_tol=args.svd_tol,
        efit_tree=args.efit_tree,
        logscale=args.logscale,
        save=args.save,
    )
    plt.show()


if __name__ == '__main__':
    main()
