"""
xtomo_mds.py
============
MDSplus data reader for the C-Mod XTOMO soft-x-ray diagnostic.

All MDSplus access uses the mdsthin Connection pattern (thin-client),
matching the style used in get_Te_ne.py.

NOTE: All write-to-tree (MDSput) calls from the original IDL code are
      commented out and are NOT executed.  See the block near the bottom
      of core_xray_emissivity.py for the commented-out writes.
"""

import numpy as np
import mdsthin as mds

XTOMO_SERVER = "alcdata"


# ---------------------------------------------------------------------------
# Low-level connection helper
# ---------------------------------------------------------------------------

def open_tree(shot: int, tree: str) -> mds.Connection:
    """Open a named MDSplus tree and return the Connection object."""
    conn = mds.Connection(XTOMO_SERVER)
    conn.openTree(tree, shot)
    return conn


# ---------------------------------------------------------------------------
# Brightness reading  (replaces read_brightness_1.pro / read_brightness_3.pro)
# ---------------------------------------------------------------------------

def read_xray_brightness(shot: int, array: int = 1,
                         fix_bad_channels: bool = True):
    """
    Read x-ray brightness for one XTOMO array (1 or 3).

    Reads raw detector power from \\top.signals.array_N:chord_XX,
    subtracts the pre-shot (t <= 0) baseline, and divides by the stored
    chord conversion factor to obtain brightness in W/m^2.

    Parameters
    ----------
    shot             : MDS shot number
    array            : XTOMO array number (1 or 3)
    fix_bad_channels : replace flagged bad channels with neighbour average

    Returns
    -------
    time         : ndarray (npts,)    – seconds
    brightnesses : ndarray (npts, 38) – W/m^2
    chord_radii  : ndarray (38,)      – impact parameter [m]
    chord_angles : ndarray (38,)      – azimuthal angle  [rad]
    """
    conn = open_tree(shot, "xtomo")

    time = np.asarray(
        conn.get(f'dim_of(\\top.brightnesses.array_{array}:chord_01)').data(),
        dtype=float,
    )
    npts = len(time)
    brightnesses = np.zeros((npts, 38), dtype=float)

    # Bad-channel state flags (True = bad / disabled)
    try:
        bc_raw = conn.get(
            f'getnci("\\top.signals.array_{array}:chord_*","state")'
        ).data()
        bad_channels = np.asarray(bc_raw, dtype=bool)
    except Exception:
        bad_channels = np.zeros(38, dtype=bool)

    # Chord conversion factor:  power [W] -> brightness [W/m^2]
    chord_factor = np.asarray(
        conn.get(f'\\top.brightnesses.array_{array}:chord_factor').data(),
        dtype=float,
    )

    # Read raw detector power from the SIGNALS subtree
    for i in range(1, 39):
        node = f'\\top.signals.array_{array}:chord_{i:02d}'
        try:
            sig = np.asarray(conn.get(node).data(), dtype=float)
            n = min(len(sig), npts)
            brightnesses[:n, i - 1] = sig[:n]
        except Exception as exc:
            print(f"  Warning: chord {i:02d} array {array}: {exc}")

    # Chord geometry stored in the BRIGHTNESSES subtree
    chord_radii = np.asarray(
        conn.get(f'\\top.brightnesses.array_{array}:chord_radii').data(),
        dtype=float,
    )
    chord_angles = np.asarray(
        conn.get(f'\\top.brightnesses.array_{array}:chord_angles').data(),
        dtype=float,
    )

    conn.closeAllTrees()

    # ---- Fix bad channels ------------------------------------------------
    if fix_bad_channels:
        for idet in range(1, 37):  # interior chords (0-indexed: 1..36)
            if bad_channels[idet]:
                print(f'  Chord {idet+1:02d} bad → averaging neighbours')
                brightnesses[:, idet] = (
                    brightnesses[:, idet - 1] + brightnesses[:, idet + 1]
                ) / 2.0
        if bad_channels[0]:
            print('  Chord 01 bad → replacing with chord 02')
            brightnesses[:, 0] = brightnesses[:, 1]
        if bad_channels[37]:
            print('  Chord 38 bad → replacing with chord 37')
            brightnesses[:, 37] = brightnesses[:, 36]

    # ---- Subtract pre-shot baseline (t <= 0) ----------------------------
    base_idx = np.where(time <= 0.0)[0]
    if len(base_idx) > 0:
        baseline = np.mean(brightnesses[base_idx, :], axis=0)
        brightnesses -= baseline[np.newaxis, :]

    # ---- Power → brightness  [divide by geometric chord factor] ---------
    brightnesses /= chord_factor[np.newaxis, :]

    return time, brightnesses, chord_radii, chord_angles


def bipolar_radii(chord_radii: np.ndarray,
                  chord_angles: np.ndarray,
                  array: int) -> np.ndarray:
    """
    Return signed (bipolar) chord impact radii.

    Sign convention matches the original IDL routines:
      - Array 1: negative for angles in (90°, 270°)  [lower-hemisphere]
      - Array 3: negative for angles > 180°           [lower-hemisphere]

    Parameters
    ----------
    chord_radii  : unsigned impact radii [m]
    chord_angles : angles in radians
    array        : 1 or 3

    Returns
    -------
    p_bipolar : signed impact radii [m]
    """
    phi_deg = np.degrees(chord_angles)
    p = chord_radii.copy()
    if array == 1:
        neg = (phi_deg > 90.0) & (phi_deg < 270.0)
    else:  # array 3
        neg = phi_deg > 180.0
    p[neg] = -p[neg]
    return p


# ---------------------------------------------------------------------------
# Detector geometry  (needed by the tomographic inversion)
# ---------------------------------------------------------------------------

def read_xtomo_geometry(shot: int):
    """
    Read XTOMO detector geometry from the tree.

    Returns the positions of the detectors and apertures for all arrays,
    and the per-chord slopes (dZ/dR) along each line of sight.

    Returns
    -------
    num_in_array : ndarray (n_arrays,)
    R_det        : ndarray (max_chords, n_arrays) [m]
    Z_det        : ndarray (max_chords, n_arrays) [m]
    R_ap         : ndarray (n_arrays,)            [m]
    Z_ap         : ndarray (n_arrays,)            [m]
    slopes       : ndarray (max_chords, n_arrays)  – dZ/dR for each chord
    """
    conn = open_tree(shot, "xtomo")
    num_in_array = np.asarray(conn.get('\\top.geometry:num_in_array').data(), dtype=int)
    R_det = np.asarray(conn.get('\\top.geometry:R_detectors').data(), dtype=float)
    Z_det = np.asarray(conn.get('\\top.geometry:Z_detectors').data(), dtype=float)
    R_ap  = np.asarray(conn.get('\\top.geometry:R_aperture').data(),  dtype=float)
    Z_ap  = np.asarray(conn.get('\\top.geometry:Z_aperture').data(),  dtype=float)
    conn.closeAllTrees()

    # Ensure shape is (max_chords, n_arrays); transpose if MDSthin returns
    # (n_arrays, max_chords) due to row-major vs column-major differences.
    if R_det.ndim == 2 and R_det.shape[0] == len(num_in_array):
        R_det = R_det.T
        Z_det = Z_det.T

    n_arrays   = len(num_in_array)
    max_chords = R_det.shape[0]
    slopes = np.zeros((max_chords, n_arrays), dtype=float)
    for ia in range(n_arrays):
        dR = R_det[:, ia] - R_ap[ia]
        dZ = Z_det[:, ia] - Z_ap[ia]
        with np.errstate(divide='ignore', invalid='ignore'):
            slopes[:, ia] = np.where(dR != 0, dZ / dR, 0.0)

    return num_in_array, R_det, Z_det, R_ap, Z_ap, slopes


def chord_radii_for_array(array_idx: int, R_det, Z_det, slopes,
                          rcenter: float, zcenter: float,
                          nchords: int) -> np.ndarray:
    """
    Compute chord impact radii (unsigned, [m]) for a given array.

    The distance from a point (rcenter, zcenter) to the chord defined by
    the line through (R_det, Z_det) with slope dZ/dR = slopes[:, array_idx]
    is computed via the standard point-to-line formula.
    """
    sl = slopes[:nchords, array_idx]
    p = (
        -sl * rcenter + zcenter
        + sl * R_det[:nchords, array_idx]
        - Z_det[:nchords, array_idx]
    ) / np.sqrt(sl ** 2 + 1.0)
    return np.abs(p)


def chord_angles_for_array(array_idx: int, R_det, Z_det, slopes,
                           rcenter: float, zcenter: float,
                           nchords: int) -> np.ndarray:
    """
    Compute chord azimuthal angles (radians, in [0, 2π]) for a given array.

    The sign of the perpendicular distance determines the half-plane, which
    sets whether the chord is in the upper or lower part of the poloidal cross
    section.  This exactly replicates the IDL chord_angles() function.
    """
    sl = slopes[:nchords, array_idx]
    p = (
        -sl * rcenter + zcenter
        + sl * R_det[:nchords, array_idx]
        - Z_det[:nchords, array_idx]
    ) / np.sqrt(sl ** 2 + 1.0)

    phi_pos = np.arctan2(-1.0,  sl)
    phi_neg = np.arctan2( 1.0, -sl)
    phi = np.where(p >= 0, phi_pos, phi_neg)
    phi[phi < 0] += 2.0 * np.pi
    return phi


# ---------------------------------------------------------------------------
# EFIT magnetic equilibrium
# ---------------------------------------------------------------------------

def read_efit_data(shot: int, tree: str = 'analysis'):
    """
    Read EFIT equilibrium data needed for the tomographic inversion.

    Returns
    -------
    efit_times : ndarray (n_efit,)
    r_magx     : ndarray (n_efit,)           [m]   magnetic axis R
    z_magx     : ndarray (n_efit,)           [m]   magnetic axis Z
    rbbbs      : ndarray (n_efit, max_pts)   [m]   LCFS R
    zbbbs      : ndarray (n_efit, max_pts)   [m]   LCFS Z
    nbbbs      : ndarray (n_efit,) int             valid LCFS point count
    """
    conn = open_tree(shot, tree)
    efit_times = np.asarray(conn.get('\\efit_aeqdsk:time').data(),  dtype=float)
    r_magx     = np.asarray(conn.get('\\efit_aeqdsk:rmagx').data(), dtype=float) / 100.0
    z_magx     = np.asarray(conn.get('\\efit_aeqdsk:zmagx').data(), dtype=float) / 100.0
    rbbbs      = np.asarray(conn.get('\\efit_geqdsk:rbbbs').data(), dtype=float)
    zbbbs      = np.asarray(conn.get('\\efit_geqdsk:zbbbs').data(), dtype=float)
    nbbbs      = np.asarray(conn.get('\\efit_geqdsk:nbbbs').data(), dtype=int)
    conn.closeAllTrees()
    return efit_times, r_magx, z_magx, rbbbs, zbbbs, nbbbs


def read_efit_psi(shot: int, time: float, tree: str = 'analysis'):
    """
    Read the EFIT poloidal flux psi(R,Z) at the time slice nearest *time*.

    Returns
    -------
    psi       : ndarray (nR, nZ) – normalised poloidal flux (0 axis = 1 mag axis)
    R_grid    : ndarray (nR,)    [m]
    Z_grid    : ndarray (nZ,)    [m]
    psi_axis  : float – psi value at magnetic axis
    psi_bndry : float – psi value at LCFS
    """
    conn = open_tree(shot, tree)
    efit_times = np.asarray(conn.get('\\efit_aeqdsk:time').data(), dtype=float)
    t_idx      = int(np.argmin(np.abs(efit_times - time)))

    psirz     = np.asarray(conn.get('\\efit_geqdsk:psirz').data(),        dtype=float)
    R_grid    = np.asarray(conn.get('dim_of(\\efit_geqdsk:psirz,0)').data(), dtype=float)
    Z_grid    = np.asarray(conn.get('dim_of(\\efit_geqdsk:psirz,1)').data(), dtype=float)
    psi_ax    = np.asarray(conn.get('\\efit_aeqdsk:ssimag').data(),        dtype=float)
    psi_bndry = np.asarray(conn.get('\\efit_aeqdsk:ssibry').data(),        dtype=float)
    conn.closeAllTrees()

    # psirz can arrive as (n_times, nZ, nR) or (n_times, nR, nZ) from MDSthin.
    # We want shape (nR, nZ); take the time slice and orient accordingly.
    if psirz.ndim == 3:
        psi_slice = psirz[t_idx]
    else:
        psi_slice = psirz

    # Ensure (nR, nZ)
    if psi_slice.shape[0] == len(Z_grid) and psi_slice.shape[1] == len(R_grid):
        psi_slice = psi_slice.T

    return psi_slice, R_grid, Z_grid, float(psi_ax[t_idx]), float(psi_bndry[t_idx])


# ---------------------------------------------------------------------------
# Vessel / tile geometry
# ---------------------------------------------------------------------------

def read_vessel_tiles(shot: int, tree: str = 'analysis'):
    """
    Read vacuum-vessel and tile boundary coordinates from the Analysis tree.

    Returns
    -------
    tiles  : list of (R_array, Z_array) tuples – tile boundary segments
    vessel : list of (R_array, Z_array) tuples – vessel boundary segments
    """
    conn = open_tree(shot, tree)

    def _read_segs(name):
        try:
            nseg  = int(conn.get(f'\\top.limiters.{name}:nseg').data())
            npts  = np.asarray(conn.get(f'\\top.limiters.{name}:pts_per_seg').data(), dtype=int)
            prefix = name[:4]   # 'tile' or 'vess'
            xdata = np.asarray(conn.get(f'\\top.limiters.{name}:x{prefix}').data(), dtype=float)
            ydata = np.asarray(conn.get(f'\\top.limiters.{name}:y{prefix}').data(), dtype=float)
            segs = []
            for i in range(nseg):
                n = npts[i]
                # Handle both (nseg, max_pts) and (max_pts, nseg) shapes
                if xdata.ndim == 2:
                    if xdata.shape[0] == nseg:
                        segs.append((xdata[i, :n], ydata[i, :n]))
                    else:
                        segs.append((xdata[:n, i], ydata[:n, i]))
            return segs
        except Exception as exc:
            print(f"  Warning reading {name} segments: {exc}")
            return []

    tiles  = _read_segs('tiles')
    vessel = _read_segs('vessel')
    conn.closeAllTrees()
    return tiles, vessel


# ---------------------------------------------------------------------------
# Tomography settings from tree
# ---------------------------------------------------------------------------

def read_tomography_settings(shot: int) -> dict:
    """Read inversion parameters from \\top.tomography in the XTOMO tree."""
    conn = open_tree(shot, "xtomo")
    s = {}

    scalar_nodes = {
        'cos_m_vals': '\\top.tomography.input_params:cos_m_vals',
        'sin_m_vals': '\\top.tomography.input_params:sin_m_vals',
        'lmax':       '\\top.tomography.input_params:lmax',
        'rmin':       '\\top.tomography.outputparams:r_min',
        'rmax':       '\\top.tomography.outputparams:r_max',
        'zmin':       '\\top.tomography.outputparams:z_min',
        'zmax':       '\\top.tomography.outputparams:z_max',
        'dr':         '\\top.tomography.outputparams:dr',
        'dz':         '\\top.tomography.outputparams:dz',
        'svd_tol':    '\\top.tomography.input_params:svd_tol',
    }
    for key, node in scalar_nodes.items():
        try:
            s[key] = np.asarray(conn.get(node).data())
        except Exception:
            s[key] = None

    time_dep_nodes = {
        'r_center':    '\\top.tomography.input_params:r_center',
        'z_center':    '\\top.tomography.input_params:z_center',
        'rnorm':       '\\top.tomography.input_params:rnorm',
    }
    for key, node in time_dep_nodes.items():
        try:
            s[key] = np.asarray(conn.get(node).data(), dtype=float)
        except Exception:
            s[key] = None

    for key, node in [('center_times', 'dim_of(\\top.tomography.input_params:r_center)'),
                      ('rnorm_times',  'dim_of(\\top.tomography.input_params:rnorm)')]:
        try:
            s[key] = np.asarray(conn.get(node).data(), dtype=float)
        except Exception:
            s[key] = np.array([0.0])

    conn.closeAllTrees()
    return s
