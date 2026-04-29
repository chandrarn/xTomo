# xtomo

Python package for C-Mod XTOMO soft-x-ray tomographic analysis.

Rewrite of the original IDL codebase (`/mnt/home/granetz/xtomo/core_xray_emissivity.pro`)
by R. Granetz (MIT PSFC).  MDSplus access uses the `mdsthin` thin-client
library in place of TDI expressions.

> **Note:** All write-to-tree (`MDSput`) calls present in the original IDL
> code are retained as comments in `core_xray_emissivity.py` and are
> **never executed**.

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd xTomo

# Create and activate a virtual environment [if desired]
python -m venv .venv
source .venv/bin/activate

# Install the package (editable mode recommended during development)
pip install -e ".[dev]"

# Install the pre-commit hooks
pre-commit install
```

---

## Quick start

```python
from xtomo import core_xray_emissivity, plot_core_emissivity

# Run the tomographic inversion
emissivity, r, z, t, ok = core_xray_emissivity(
    1120927023,
    tstart=0.8, tstop=1.4, dt=0.05,
    use_efit_center=True,
    auto_calc_rnorm=True,
)

# Plot a single time slice
import matplotlib.pyplot as plt
plot_core_emissivity(1120927023, emissivity, r, z, t, time=1.2)
plt.show()
```

### Compare brightness profiles with the inverted emissivity

```bash
python -m xtomo.compare_brightness_emissivity 1140221013 1.2
# or
xtomo-compare 1140221013 1.2 --save comparison.pdf
```

---

## Package layout

```
xTomo/
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
└── src/
    └── xtomo/
        ├── __init__.py
        ├── xtomo_mds.py                   # MDSplus I/O layer
        ├── core_xray_emissivity.py        # Fourier-Bessel inversion
        ├── plot_core_emissivity.py        # 2-D emissivity plot
        └── compare_brightness_emissivity.py  # brightness vs emissivity
```

---

## Algorithm

The inversion follows the Fourier-Bessel expansion described in:

- Y. Nagayama,``Tomography of m=1 mode structure in tokamak plasma using least‐square‐fitting method and Fourier–Bessel expansions'' *J. Appl. Phys.* **62** (1987) 2702
- L. Wang & R. Granetz, ``A simplified expression for the Radon transform of Bessel basis functions in tomography'' *Rev. Sci. Instrum.* **62** (1991) 842

The line integrals through each Fourier-Bessel harmonic are evaluated
numerically with Simpson's rule (1001 points), and the resulting system is
solved via SVD with a relative singular-value cutoff (`svd_tol`).
