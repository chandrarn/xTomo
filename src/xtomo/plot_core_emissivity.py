"""
plot_core_emissivity.py
=======================
Python rewrite of plot_core_emissivity.pro (IDL / C-Mod XTOMO).

Plots the 2-D reconstructed x-ray emissivity on the poloidal cross-section,
with optional overlays of EFIT flux contours and the C-Mod vacuum vessel /
tile geometry.

Usage (interactive)
-------------------
    from plot_core_emissivity import plot_core_emissivity
    plot_core_emissivity(shot, emissivity, r, z, t, time=1.0)

Usage (standalone) — computes AND plots for a single time point
-------------------
    python plot_core_emissivity.py 1140221013 1.2
        --tstart 0.5 --tstop 2.0 --dt 0.05
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from .xtomo_mds import read_efit_psi, read_vessel_tiles

# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_core_emissivity(
    shot: int,
    emissivity: np.ndarray,  # (nr, nz, npts)
    r: np.ndarray,  # (nr,)  [m]
    z: np.ndarray,  # (nz,)  [m]
    t: np.ndarray,  # (npts,) [s]
    time: float = 1.0,
    *,
    noflux: bool = False,
    novessel: bool = False,
    nogrid: bool = False,
    efit_tree: str = "analysis",
    halt_on_efit_error: bool = False,
    n_flux_contours: int = 12,
    cmap: str = "hot",
    save: str = "",
    ax: plt.Axes | None = None,
    use_latex_style: bool = True,
) -> plt.Axes:
    """
    Plot the reconstructed 2-D x-ray emissivity for one time slice.

    Parameters
    ----------
    shot       : MDS shot number (used to fetch EFIT / vessel data)
    emissivity : (nr, nz, npts) emissivity array [W/m^3]
    r, z, t    : grid and time arrays from core_xray_emissivity()
    time       : time [s] at which to plot (nearest available slice used)
    noflux     : skip EFIT flux contours
    novessel   : skip vessel / tile outline
    nogrid     : skip axis labels, title, and black background fill
    efit_tree  : MDSplus tree containing EFIT data
    halt_on_efit_error : if True, raise EFIT contour errors for interactive debugging
    n_flux_contours    : number of equally spaced psi_N contours (0 = LCFS only)
    use_latex_style    : apply Computer Modern serif font style
    cmap       : matplotlib colourmap (IDL loadct,39 ≈ 'jet' or 'rainbow')
    save       : if non-empty, save figure to this path
    ax         : if supplied, draw into this Axes object

    Returns
    -------
    ax : matplotlib Axes
    """
    # ---- Optional LaTeX-like font style ---------------------------------
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
            }
        )

    # ---- Select the nearest time slice -----------------------------------
    t_idx = int(np.argmin(np.abs(t - time)))
    t_plot = t[t_idx]

    # ---- Grid bounds used for the axes -----------------------------------
    rmin_ax = 0.40
    rmax_ax = 0.95
    zmin_ax = -0.50
    zmax_ax = 0.50

    # ---- Set up figure if no axes supplied --------------------------------
    if ax is None:
        fig_ratio = (zmax_ax - zmin_ax) / (rmax_ax - rmin_ax)
        fig_w = 5.0
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(fig_w, fig_w * fig_ratio),
            layout="constrained",
        )
        if hasattr(fig.canvas, "manager"):
            fig.canvas.manager.set_window_title(f"X-ray tomography  shot {shot}")
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal")

    # ---- Axis labels and titles ------------------------------------------
    if not nogrid:
        ax.set_xlim(rmin_ax, rmax_ax)
        ax.set_ylim(zmin_ax, zmax_ax)
        ax.set_xlabel("R (m)", fontsize=12)
        ax.set_ylabel("Z (m)", fontsize=12)
        ax.set_title(f"X-ray tomography  shot {shot},  t = {t_plot:.3f} s", fontsize=12)
        ax.tick_params(direction="in")
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        # Black background fill
        ax.set_facecolor("black")

    # ---- pcolormesh plot of emissivity -----------------------------------
    emiss_slice = emissivity[:, :, t_idx] * 1e-6  # (nr, nz) [MW/m^3]

    # Use pcolormesh with the cell-edge coordinates
    dr = r[1] - r[0] if len(r) > 1 else 0.01
    dz = z[1] - z[0] if len(z) > 1 else 0.01
    r_edges = np.append(r - dr / 2, r[-1] + dr / 2)
    z_edges = np.append(z - dz / 2, z[-1] + dz / 2)

    Redge, Zedge = np.meshgrid(r_edges, z_edges, indexing="ij")

    vmin = 0.0
    vmax = emiss_slice.max() if emiss_slice.max() > 0 else 1.0
    pcm = ax.pcolormesh(
        Redge,
        Zedge,
        emiss_slice,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        zorder=1,
    )
    fig.colorbar(pcm, ax=ax, label="Emissivity (MW/m³)", shrink=0.8)

    # ---- EFIT flux contours ----------------------------------------------
    if not noflux:
        # try:
        psi, R_efit, Z_efit, psi_ax, psi_bndry = read_efit_psi(shot, t_plot, tree=efit_tree)

        # Normalise:  psi_N = 0 at axis, 1 at LCFS
        psi_n = (psi - psi_ax) / (psi_bndry - psi_ax)

        Rg, Zg = np.meshgrid(R_efit, Z_efit, indexing="ij")

        levels_inner = (
            np.arange(n_flux_contours + 1) / (n_flux_contours + 1) if n_flux_contours > 0 else []
        )
        levels = np.append(levels_inner, 1.0)  # always include LCFS
        # Note: under some odd conditions, if the EFIT tree fails to converge correctly,
        # There could be no valid psi_n levels, and nothing will be plotted here
        ax.contour(
            Rg,
            Zg,
            psi_n,
            levels=levels,
            colors="white",
            linewidths=0.8,
            alpha=0.7,
            zorder=2,
        )
        # except Exception as exc:
        #     print(f'  Warning: could not plot EFIT contours: {exc}')
        #     if halt_on_efit_error:
        #         raise RuntimeError(
        # 'Halting on EFIT contour error as requested; inspect this exception in debugger.'
        # ) from exc

    # ---- Vessel / tile outlines ------------------------------------------
    if not novessel:
        # try:
        tiles, vessel = read_vessel_tiles(shot, tree=efit_tree)
        for R_seg, Z_seg in tiles:
            ax.plot(R_seg, Z_seg, color="white", linewidth=0.8, alpha=0.9, zorder=3)
        for R_seg, Z_seg in vessel:
            ax.plot(R_seg, Z_seg, color="cyan", linewidth=0.8, alpha=0.7, zorder=3)
        # except Exception as exc:
        #     print(f'  Warning: could not plot vessel geometry: {exc}')

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save}")

    return ax


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console entry point: ``xtomo-plot``."""
    import argparse

    from xtomo.core_xray_emissivity import core_xray_emissivity

    parser = argparse.ArgumentParser(description="Plot 2-D XTOMO emissivity for a C-Mod shot")
    parser.add_argument("shot", type=int, help="MDS shot number")
    parser.add_argument("tplot", type=float, help="Time to plot [s]")
    parser.add_argument("--tstart", type=float, default=None)
    parser.add_argument("--tstop", type=float, default=None)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--lmax", type=int, default=15)
    parser.add_argument("--noflux", action="store_true")
    parser.add_argument("--novessel", action="store_true")
    parser.add_argument("--cmap", type=str, default="hot")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--efit-tree", default="analysis")
    args = parser.parse_args()

    tstart = args.tstart if args.tstart is not None else args.tplot - 0.01
    tstop = args.tstop if args.tstop is not None else args.tplot + 0.01

    print(f"Computing emissivity for shot {args.shot} ...")
    emissivity, r, z, t, ok = core_xray_emissivity(
        args.shot,
        tstart=tstart,
        tstop=tstop,
        dt=args.dt,
        use_efit_center=True,
        auto_calc_rnorm=True,
        efit_tree=args.efit_tree,
        lmax=args.lmax,
    )

    if not ok:
        print("Emissivity calculation failed.")
    else:
        plot_core_emissivity(
            args.shot,
            emissivity,
            r,
            z,
            t,
            time=args.tplot,
            noflux=args.noflux,
            novessel=args.novessel,
            efit_tree=args.efit_tree,
            cmap=args.cmap,
            save=args.save,
        )
        plt.show()


if __name__ == "__main__":
    main()
