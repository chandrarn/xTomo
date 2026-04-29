"""
core_xray_emissivity.py
=======================
Python rewrite of core_xray_emissivity.pro (IDL / C-Mod XTOMO).

Computes the 2-D x-ray emissivity profile on a (R, Z) grid via a
Fourier-Bessel SVD tomographic inversion of the soft-x-ray chord
brightness measurements from XTOMO arrays 1 and 3.

Algorithm references
--------------------
  Y. Nagayama,       J. Appl. Phys. 62 (1987) 2702
  L. Wang & R. Granetz, Rev. Sci. Instrum. 62 (1991) 842

MDSplus access uses the mdsthin thin-client pattern (see xtomo_mds.py).

NOTE: Write-to-tree (MDSput) calls present in the original IDL code are
      retained below as comments and are NOT executed.
"""

from __future__ import annotations
import numpy as np
from scipy.special import jv

from .xtomo_mds import (
    XTOMO_SERVER,
    open_tree,
    read_xray_brightness,
    read_xtomo_geometry,
    read_efit_data,
    chord_radii_for_array,
    chord_angles_for_array,
    read_tomography_settings,
)


# ---------------------------------------------------------------------------
# Bessel zeros table
# ---------------------------------------------------------------------------

def bessel_zeros_init():
    """
    Return pre-tabulated zeros of J_m (m = 0, 1, 2) and J'_m at those zeros.

    Data from G. N. Watson, *A Treatise on the Theory of Bessel Functions*,
    2nd ed. (Cambridge, 1958); originally coded by N. Kallivayalil and
    R. Granetz for the IDL version.

    Returns
    -------
    zeros          : ndarray (31, 3)  –  zeros[L, m] = L-th zero of J_m
    dj_at_zeros    : ndarray (31, 3)  –  J'_m(zeros[L, m])
    """
    zeros = np.zeros((31, 3), dtype=float)

    # m = 0
    zeros[:, 0] = [
         2.4048255577,  5.5200781103,  8.6537279129, 11.7915344391,
        14.9309177086, 18.0710639679, 21.2116366299, 24.3524715308,
        27.4934791320, 30.6346064684, 33.7758202136, 36.9170983537,
        40.0584257646, 43.1997917132, 46.3411883717, 49.4826098974,
        52.6240518411, 55.7655107550, 58.9069839261, 62.0484691902,
        65.1899648,    68.3314693,    71.4729816,    74.6145006,
        77.7560256,    80.8975559,    84.0390908,    87.1806298,
        90.3221726,    93.4637188,    96.6052680,
    ]
    # m = 1
    zeros[:, 1] = [
         3.8317060,  7.0155867, 10.1734681, 13.3236919,
        16.4706301, 19.6158585, 22.7600844, 25.9036721,
        29.0468285, 32.1896799, 35.3323076, 38.4747662,
        41.6170942, 44.7593190, 47.9014609, 51.0435352,
        54.1855536, 57.3275254, 60.4694578, 63.6113567,
        66.7532267, 69.8950718, 73.0368952, 76.1786996,
        79.3204872, 82.4622599, 85.6040194, 88.7457671,
        91.8875043, 95.0292318, 98.1709507,
    ]
    # m = 2
    zeros[:, 2] = [
         5.1356223,  8.4172441, 11.6198412, 14.7959518,
        17.9598195, 21.1169971, 24.2701123, 27.4205736,
        30.5692045, 33.7165195, 36.8628565, 40.0084467,
        43.1534538, 46.2979967, 49.4421641, 52.5860235,
        55.7296271, 58.8730158, 62.0162224, 65.1592732,
        68.3021898, 71.4449899, 74.5876882, 77.7302971,
        80.8728269, 84.0152867, 87.1576839, 90.3000252,
        93.4423160, 96.5845614, 99.7267657,
    ]

    # J'_m(x) = [J_{m-1}(x) - J_{m+1}(x)] / 2  (recurrence relation)
    dj_at_zeros = np.zeros_like(zeros)
    for m in range(3):
        for L in range(31):
            x = zeros[L, m]
            dj_at_zeros[L, m] = 0.5 * (jv(m - 1, x) - jv(m + 1, x))

    return zeros, dj_at_zeros


# ---------------------------------------------------------------------------
# Ling Wang kernel and numerical quadrature
# ---------------------------------------------------------------------------

def _bessel_kernel(x: np.ndarray, M: int, bz: float, pnorm: float) -> np.ndarray:
    """
    Integrand of the Ling Wang line-integral identity.

    f(x) = cos(M x) * sin(bz * (cos(x) - pnorm))

    where bz is the relevant zero of J_M and pnorm = p/rnorm is the
    normalised chord impact parameter.
    """
    return np.cos(M * x) * np.sin(bz * (np.cos(x) - pnorm))


def _line_integral_element(M: int, bz: float, pnorm: float,
                           nbins: int = 1001) -> float:
    """
    Numerically integrate the Ling Wang kernel from 0 to arccos(pnorm)
    using Simpson's composite rule with *nbins* points (must be odd).

    Replicates the IDL qsimpson() call in core_xray_emissivity.pro.
    """
    pnorm = float(np.clip(pnorm, 0.0, 1.0 - 1e-12))
    upper = np.arccos(pnorm)
    if upper <= 0.0:
        return 0.0
    if nbins % 2 == 0:
        nbins += 1
    x  = np.linspace(0.0, upper, nbins)
    y  = _bessel_kernel(x, M, bz, pnorm)
    dx = upper / (nbins - 1)
    # Simpson weights: 1, 4, 2, 4, 2, …, 4, 1
    return dx / 3.0 * (
        y[0] + y[-1]
        + 4.0 * y[1:-1:2].sum()   # odd  indices
        + 2.0 * y[2:-2:2].sum()   # even interior indices
    )


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def _build_line_integral_matrix(phi: np.ndarray, pnorm: np.ndarray,
                                cos_m_vals, sin_m_vals, lmax: int,
                                bessel_zeros: np.ndarray,
                                jprimes: np.ndarray) -> np.ndarray:
    """
    Build the Fourier-Bessel line integral matrix A of shape (ndet, num_coeffs).

    A[c, col] = -2 * J'_M(alpha_{LM}) * rnorm * cos/sin(M * phi[c]) * integral

    where the integral is the Ling Wang line integral for chord c.
    (The rnorm factor is already folded into jprimes = rnorm * dJ_at_zeros.)

    Each block of (lmax+1) consecutive columns corresponds to one Fourier
    harmonic M, first all cosine harmonics then all sine harmonics.
    """
    ndet       = len(phi)
    num_coeffs = (len(cos_m_vals) + len(sin_m_vals)) * (lmax + 1)
    A          = np.zeros((ndet, num_coeffs), dtype=float)

    col = 0
    for M in cos_m_vals:
        trig = np.cos(M * phi)                     # (ndet,)
        for L in range(lmax + 1):
            bz = bessel_zeros[L, M]
            jp = jprimes[L, M]
            for c in range(ndet):
                s = _line_integral_element(M, bz, pnorm[c])
                A[c, col + L] = -2.0 * jp * trig[c] * s
        col += lmax + 1

    for M in sin_m_vals:
        trig = np.sin(M * phi)                     # (ndet,)
        for L in range(lmax + 1):
            bz = bessel_zeros[L, M]
            jp = jprimes[L, M]
            for c in range(ndet):
                s = _line_integral_element(M, bz, pnorm[c])
                A[c, col + L] = -2.0 * jp * trig[c] * s
        col += lmax + 1

    return A


def _build_harmonics_matrix(r_polar: np.ndarray, theta: np.ndarray,
                             cos_m_vals, sin_m_vals, lmax: int,
                             bessel_zeros: np.ndarray) -> np.ndarray:
    """
    Build the harmonics evaluation matrix H of shape (ngrid, num_coeffs).

    H[i, col] = cos/sin(M * theta[i]) * J_M(alpha_{LM} * r_polar[i])

    Multiplying H @ coefficients gives the emissivity at each grid point.
    """
    ngrid      = len(r_polar)
    num_coeffs = (len(cos_m_vals) + len(sin_m_vals)) * (lmax + 1)
    H          = np.zeros((ngrid, num_coeffs), dtype=float)

    col = 0
    for M in cos_m_vals:
        trig = np.cos(M * theta)
        for L in range(lmax + 1):
            H[:, col + L] = trig * jv(M, bessel_zeros[L, M] * r_polar)
        col += lmax + 1

    for M in sin_m_vals:
        trig = np.sin(M * theta)
        for L in range(lmax + 1):
            H[:, col + L] = trig * jv(M, bessel_zeros[L, M] * r_polar)
        col += lmax + 1

    return H


# ---------------------------------------------------------------------------
# Main tomographic inversion function
# ---------------------------------------------------------------------------

def core_xray_emissivity(
    shot: int,
    *,
    # Time range (used when use_efit_times=False)
    tstart: float | None = None,
    tstop:  float | None = None,
    dt:     float | None = None,
    # Flags  (mirrors the IDL keyword arguments)
    use_efit_times:        bool = False,
    use_efit_center:       bool = True,
    auto_calc_rnorm:       bool = True,
    read_settings_from_tree: bool = False,
    efit_tree: str = 'analysis',
    # Inversion parameters (used when read_settings_from_tree=False)
    cos_m_vals  = None,    # default [0, 1, 2]
    sin_m_vals  = None,    # default [1]
    lmax:  int   = 15,
    svd_tol: float = 0.1,
    # Fixed centre / normalisation radius (if not using EFIT)
    rcenter: float | None = None,
    zcenter: float | None = None,
    rnorm:   float | None = None,
    # Output grid parameters
    rmin: float = 0.44,
    rmax: float = 0.92,
    zmin: float = -0.42,
    zmax: float =  0.42,
    dr:   float = 0.02,
    dz:   float = 0.02,
    verbose: bool = True,
):
    """
    Compute 2-D soft-x-ray emissivity via Fourier-Bessel tomographic inversion.

    Parameters
    ----------
    shot      : MDS shot number
    tstart / tstop / dt : time range in seconds (required if not use_efit_times)
    use_efit_times  : use EFIT time points as the reconstruction times
    use_efit_center : read plasma magnetic axis from EFIT (recommended)
    auto_calc_rnorm : automatically set rnorm from the LCFS boundary polygon
    read_settings_from_tree : read all inversion parameters from the XTOMO tree
    efit_tree       : MDSplus tree name containing EFIT equilibrium data
    cos_m_vals      : list of cosine Fourier harmonics  (default [0, 1, 2])
    sin_m_vals      : list of sine   Fourier harmonics  (default [1])
    lmax            : max radial Bessel harmonic index
    svd_tol         : absolute singular-value cutoff for the pseudo-inverse
    rcenter / zcenter : plasma centre [m] if not using EFIT (default 0.68, -0.01)
    rnorm           : normalisation radius [m] if not using EFIT (default 0.45)
    rmin/rmax/zmin/zmax : output grid bounds [m]
    dr / dz         : output grid spacing [m]
    verbose         : print progress information

    Returns
    -------
    emissivity : ndarray (nr, nz, npts)  [W/m^3]
    r          : ndarray (nr,)           [m]
    z          : ndarray (nz,)           [m]
    t          : ndarray (npts,)         [s]
    status     : bool  (True = success)
    """
    if cos_m_vals is None:
        cos_m_vals = [0, 1, 2]
    if sin_m_vals is None:
        sin_m_vals = [1]
    cos_m_vals = np.asarray(cos_m_vals, dtype=int)
    sin_m_vals = np.asarray(sin_m_vals, dtype=int)

    # Validate harmonic indices (table supports only m = 0, 1, 2)
    max_m_table = 2
    if cos_m_vals.size > 0 and cos_m_vals.min() < 0:
        cos_m_vals = np.array([], dtype=int)
    if sin_m_vals.size > 0 and sin_m_vals.min() < 0:
        sin_m_vals = np.array([], dtype=int)
    if cos_m_vals.size > 0 and cos_m_vals.max() > max_m_table:
        raise ValueError(f"cos_m_vals contains M > {max_m_table}; extend the Bessel-zeros table first.")
    if sin_m_vals.size > 0 and sin_m_vals.max() > max_m_table:
        raise ValueError(f"sin_m_vals contains M > {max_m_table}; extend the Bessel-zeros table first.")

    bessel_zeros, bessel_zeros_primes = bessel_zeros_init()

    # ------------------------------------------------------------------
    # Read brightness data for arrays 1 and 3
    # ------------------------------------------------------------------
    print(f'Shot = {shot:10d}')
    print('Reading brightness array 1 ...')
    time, brightness1, _, _ = read_xray_brightness(
        shot, array=1, fix_bad_channels=True)
    print('Reading brightness array 3 ...')
    _,    brightness3, _, _ = read_xray_brightness(
        shot, array=3, fix_bad_channels=True)

    # Combined brightness matrix  shape (npts_bright, ndet_total=76)
    brightness = np.concatenate([brightness1, brightness3], axis=1)

    # ------------------------------------------------------------------
    # Optionally override inversion parameters from the XTOMO tree
    # ------------------------------------------------------------------
    if read_settings_from_tree:
        s = read_tomography_settings(shot)
        if s.get('cos_m_vals') is not None: cos_m_vals = np.asarray(s['cos_m_vals'], dtype=int)
        if s.get('sin_m_vals') is not None: sin_m_vals = np.asarray(s['sin_m_vals'], dtype=int)
        if s.get('lmax')       is not None: lmax       = int(s['lmax'])
        if s.get('rmin')       is not None: rmin       = float(s['rmin'])
        if s.get('rmax')       is not None: rmax       = float(s['rmax'])
        if s.get('zmin')       is not None: zmin       = float(s['zmin'])
        if s.get('zmax')       is not None: zmax       = float(s['zmax'])
        if s.get('dr')         is not None: dr         = float(s['dr'])
        if s.get('dz')         is not None: dz         = float(s['dz'])
        if s.get('svd_tol')    is not None: svd_tol    = float(s['svd_tol'])
        if not use_efit_center and s.get('r_center') is not None:
            _rc_arr  = s['r_center']
            _zc_arr  = s['z_center']
            _ct_arr  = s.get('center_times', np.array([0.0]))
        if not auto_calc_rnorm and s.get('rnorm') is not None:
            _rn_arr  = s['rnorm']
            _rnt_arr = s.get('rnorm_times', np.array([0.0]))

    # ------------------------------------------------------------------
    # Read EFIT data if required by any flag
    # ------------------------------------------------------------------
    _efit_times  = None
    _rc_arr = _zc_arr = _ct_arr = None
    _rbbbs  = _zbbbs  = _nbbbs  = _rnt_arr = None
    _rn_arr = None

    if use_efit_times or use_efit_center or auto_calc_rnorm:
        if verbose:
            print(f'Reading EFIT data from tree "{efit_tree}" ...')
        efit_times, r_magx, z_magx, rbbbs, zbbbs, nbbbs = read_efit_data(
            shot, tree=efit_tree)
        _efit_times = efit_times
        if use_efit_times:
            pass  # times will be set below
        if use_efit_center:
            _rc_arr = r_magx
            _zc_arr = z_magx
            _ct_arr = efit_times
        if auto_calc_rnorm:
            _rbbbs  = rbbbs
            _zbbbs  = zbbbs
            _nbbbs  = nbbbs
            _rnt_arr = efit_times

    # Fallback defaults when not using EFIT
    if _rc_arr is None:
        _rc_arr = np.array([rcenter if rcenter is not None else 0.68])
        _zc_arr = np.array([zcenter if zcenter is not None else -0.01])
        _ct_arr = np.array([0.0])
    if _rnt_arr is None and not auto_calc_rnorm:
        _rn_arr  = np.atleast_1d(rnorm if rnorm is not None else 0.45)
        _rnt_arr = np.array([0.0])

    # ------------------------------------------------------------------
    # Build reconstruction time array
    # ------------------------------------------------------------------
    if use_efit_times:
        if _efit_times is None:
            raise RuntimeError("use_efit_times=True but no EFIT data was loaded.")
        times = _efit_times
    else:
        if tstart is None or tstop is None or dt is None:
            raise ValueError(
                "Provide tstart, tstop, dt (seconds) or set use_efit_times=True.")
        npts  = int(np.floor((tstop - tstart) / dt)) + 1
        times = np.linspace(tstart, tstart + (npts - 1) * dt, npts)

    npts = len(times)

    # ------------------------------------------------------------------
    # Build output emissivity grid  (R, Z)
    # ------------------------------------------------------------------
    nr = int(np.floor((rmax - rmin) / dr)) + 1
    r  = np.arange(nr, dtype=float) * dr + rmin

    if abs(zmin + zmax) < 1e-9:          # symmetric: force grid through Z = 0
        nzhalf = int(np.floor(zmax / dz))
        if nzhalf > 0:
            zhalf = np.arange(1, nzhalf + 1, dtype=float) * dz
            z = np.concatenate([-zhalf[::-1], [0.0], zhalf])
        else:
            z = np.array([0.0])
    else:
        nz = int(np.floor((zmax - zmin) / dz)) + 1
        z  = np.arange(nz, dtype=float) * dz + zmin
    nz = len(z)

    r2d, z2d = np.meshgrid(r, z, indexing='ij')   # (nr, nz)
    r1d = r2d.ravel()
    z1d = z2d.ravel()

    # ------------------------------------------------------------------
    # Load detector geometry
    # ------------------------------------------------------------------
    num_in_array, R_det, Z_det, R_ap, Z_ap, slopes = read_xtomo_geometry(shot)

    num_cos_m  = len(cos_m_vals)
    num_sin_m  = len(sin_m_vals)
    num_m      = num_cos_m + num_sin_m
    num_coeffs = num_m * (lmax + 1)
    ndet       = brightness.shape[1]    # 76 (38+38)

    t_arr      = np.zeros(npts, dtype=float)
    emissivity = np.zeros((nr, nz, npts), dtype=float)

    # Cache-invalidation flags: recompute SVD only when centre/rnorm changes
    center_idx_old = -1
    rnorm_idx_old  = -1
    pseudo_inverse = None
    harmonics_mat  = None
    rc = zc = rn = None

    # ------------------------------------------------------------------
    # Loop over reconstruction time slices
    # ------------------------------------------------------------------
    for it in range(npts):

        # Find the brightness sample nearest to times[it]
        time_index  = int(np.argmin(np.abs(times[it] - time)))
        center_idx  = int(np.argmin(np.abs(times[it] - _ct_arr)))
        rnorm_idx   = int(np.argmin(np.abs(times[it] - _rnt_arr))) if _rnt_arr is not None else 0

        t_arr[it] = time[time_index]
        if verbose:
            print(f'Time = {t_arr[it]:9.6f} s  (step {it+1:5d} of {npts:5d})')

        redo_matrix = False

        if center_idx != center_idx_old:
            redo_matrix    = True
            rc             = float(_rc_arr[center_idx])
            zc             = float(_zc_arr[center_idx])
            center_idx_old = center_idx

        if rnorm_idx != rnorm_idx_old:
            redo_matrix   = True
            if auto_calc_rnorm and _rbbbs is not None:
                nb  = int(_nbbbs[rnorm_idx])
                rb  = _rbbbs[rnorm_idx, :nb]
                zb  = _zbbbs[rnorm_idx, :nb]
                rn  = float(np.max(np.sqrt((rb - rc) ** 2 + (zb - zc) ** 2)))
            else:
                rn = float(_rn_arr[rnorm_idx]) if _rn_arr is not None else 0.45
            rnorm_idx_old = rnorm_idx

        # First iteration always requires a full matrix build
        if pseudo_inverse is None:
            redo_matrix = True

        if redo_matrix:
            if verbose:
                print('     Redoing matrix calculations ...')

            # Chord radii and angles (using arrays 1 and 3 → 0-indexed: 0 and 2)
            n1   = int(num_in_array[0])
            n3   = int(num_in_array[2])
            p1   = chord_radii_for_array(0, R_det, Z_det, slopes, rc, zc, n1)
            phi1 = chord_angles_for_array(0, R_det, Z_det, slopes, rc, zc, n1)
            p3   = chord_radii_for_array(2, R_det, Z_det, slopes, rc, zc, n3)
            phi3 = chord_angles_for_array(2, R_det, Z_det, slopes, rc, zc, n3)

            p   = np.concatenate([p1, p3])
            phi = np.concatenate([phi1, phi3])

            # Guard: rnorm must enclose all chords
            if np.any(p > rn):
                rn_new = float(np.max(p))
                print(
                    f'WARNING: requested rnorm ({rn:.3f} m) is smaller than '
                    f'one or more chord radii.  Increasing to {rn_new:.3f} m.')
                rn = rn_new

            pnorm   = p / rn
            # jprimes = rnorm * J'_m(alpha_{LM})  (matches IDL variable name)
            jprimes = rn * bessel_zeros_primes  # shape (31, 3)

            # Build and SVD the line integral matrix
            A = _build_line_integral_matrix(
                phi, pnorm, cos_m_vals, sin_m_vals, lmax,
                bessel_zeros, jprimes)

            U, s_vals, Vh = np.linalg.svd(A, full_matrices=False)
            s_inv = np.where(np.abs(s_vals) >= svd_tol, 1.0 / s_vals, 0.0)
            pseudo_inverse = (Vh.T * s_inv) @ U.T   # shape (num_coeffs, ndet)

            # Polar coordinates of the output grid relative to (rc, zc)
            r_polar = np.sqrt((z1d - zc) ** 2 + (r1d - rc) ** 2) / rn
            theta   = np.arctan2(z1d - zc, r1d - rc)

            # Pre-compute the harmonics evaluation matrix
            harmonics_mat = _build_harmonics_matrix(
                r_polar, theta, cos_m_vals, sin_m_vals, lmax, bessel_zeros)

        # Solve for Fourier-Bessel coefficients
        coefficients = pseudo_inverse @ brightness[time_index, :]   # (num_coeffs,)

        # Evaluate emissivity on the 2-D grid
        emiss1d             = harmonics_mat @ coefficients           # (nr*nz,)
        emiss1d[r_polar >= 1.0] = 0.0      # zero outside the normalisation radius
        emissivity[:, :, it] = emiss1d.reshape(nr, nz)

    # ------------------------------------------------------------------
    # Write results to tree  —  COMMENTED OUT: do not write to tree
    # ------------------------------------------------------------------
    # conn = open_tree(shot, "xtomo")
    # conn.put(
    #     '\\top.tomography:emissivity',
    #     'build_signal('
    #     'build_with_units($1,"watt/meter^3"),*,'
    #     'build_with_units($2,"meter"),'
    #     'build_with_units($3,"meter"),'
    #     'build_with_units($4,"second"))',
    #     emissivity, r, z, t_arr
    # )
    # conn.closeAllTrees()

    return emissivity, r, z, t_arr, True


# ---------------------------------------------------------------------------
# Convenience: read previously stored tomographic result from the tree
# (equivalent to read_core_emissivity.pro)
# ---------------------------------------------------------------------------

def read_core_emissivity(shot: int):
    """
    Read a previously stored tomographic emissivity result from the XTOMO tree.

    Returns
    -------
    emissivity : ndarray (nr, nz, npts)  [W/m^3]
    r          : ndarray (nr,)           [m]
    z          : ndarray (nz,)           [m]
    t          : ndarray (npts,)         [s]
    status     : bool
    """
    try:
        conn = open_tree(shot, "xtomo")
        emiss = np.asarray(conn.get('\\xtomo::top.tomography:emissivity').data(),
                           dtype=float)
        r = np.asarray(conn.get(
            'dim_of(\\xtomo::top.tomography:emissivity,0)').data(), dtype=float)
        z = np.asarray(conn.get(
            'dim_of(\\xtomo::top.tomography:emissivity,1)').data(), dtype=float)
        t = np.asarray(conn.get(
            'dim_of(\\xtomo::top.tomography:emissivity,2)').data(), dtype=float)
        conn.closeAllTrees()
        return emiss, r, z, t, True
    except Exception as exc:
        print(f'ERROR reading emissivity from tree: {exc}')
        return None, None, None, None, False


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Console entry point: ``xtomo-emissivity``."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute C-Mod XTOMO 2-D x-ray emissivity (Fourier-Bessel inversion)')
    parser.add_argument('shot',   type=int,   help='MDS shot number')
    parser.add_argument('tstart', type=float, help='Start time [s]')
    parser.add_argument('tstop',  type=float, help='Stop  time [s]')
    parser.add_argument('dt',     type=float, help='Time step  [s]')
    parser.add_argument('--efit-tree', default='analysis',
                        help='MDSplus tree for EFIT data (default: analysis)')
    parser.add_argument('--no-efit-center', action='store_true',
                        help='Do not use EFIT magnetic axis as coordinate centre')
    parser.add_argument('--no-auto-rnorm', action='store_true',
                        help='Do not auto-compute rnorm from LCFS')
    parser.add_argument('--lmax',    type=int,   default=15)
    parser.add_argument('--svd-tol', type=float, default=0.1)
    parser.add_argument('--save',    type=str,   default='',
                        help='Save results to this .npz file path')
    args = parser.parse_args()

    emissivity, r, z, t, ok = core_xray_emissivity(
        args.shot,
        tstart=args.tstart, tstop=args.tstop, dt=args.dt,
        use_efit_center=not args.no_efit_center,
        auto_calc_rnorm=not args.no_auto_rnorm,
        efit_tree=args.efit_tree,
        lmax=args.lmax,
        svd_tol=args.svd_tol,
    )

    if ok:
        print(f'\nDone.  Emissivity array shape: {emissivity.shape}')
        if args.save:
            np.savez(args.save, emissivity=emissivity, r=r, z=z, t=t, shot=args.shot)
            print(f'Results saved to {args.save}')
    else:
        print('Calculation failed.')


if __name__ == '__main__':
    main()
