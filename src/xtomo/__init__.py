"""
xtomo
=====
Python package for C-Mod XTOMO soft-x-ray tomographic analysis.

Modules
-------
xtomo_mds                  — MDSplus thin-client data readers
core_xray_emissivity       — Fourier-Bessel tomographic inversion
plot_core_emissivity       — 2-D emissivity plot on the poloidal cross-section
compare_brightness_emissivity — Brightness vs inverted profile comparison
"""

from .compare_brightness_emissivity import compare_brightness_emissivity
from .core_xray_emissivity import (
    bessel_zeros_init,
    core_xray_emissivity,
    read_core_emissivity,
)
from .plot_core_emissivity import plot_core_emissivity
from .xtomo_mds import (
    XTOMO_SERVER,
    bipolar_radii,
    open_tree,
    read_efit_data,
    read_efit_psi,
    read_tomography_settings,
    read_vessel_tiles,
    read_xray_brightness,
    read_xtomo_geometry,
)

__all__ = [
    # MDSplus I/O
    "XTOMO_SERVER",
    "open_tree",
    "read_xray_brightness",
    "read_xtomo_geometry",
    "read_efit_data",
    "read_efit_psi",
    "read_vessel_tiles",
    "read_tomography_settings",
    "bipolar_radii",
    # Inversion
    "bessel_zeros_init",
    "core_xray_emissivity",
    "read_core_emissivity",
    # Plotting
    "plot_core_emissivity",
    "compare_brightness_emissivity",
]
