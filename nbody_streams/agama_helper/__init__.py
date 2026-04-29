"""
nbody_streams.agama_helper
~~~~~~~~~~~~~~~~~~~~~~~~~~
Secondary utilities for fitting, storing, modifying, and loading Agama
expansion-based potentials (Multipole and CylSpline BFEs).

Agama units are set to (Msol, kpc, km/s) at import time when agama is
available.  Time is therefore in kpc / (km/s) ~ 0.978 Gyr.

API overview
------------
``read_*``   — parse coefficient files into dataclasses, or read raw strings.
               Accepts a plain-text file, an HDF5 archive, or a raw string.

``load_*``   — construct Agama potential objects ready for orbit integration.

``write_*``  — pack plain-text coefficient files into compact HDF5 archives.

Quick start
-----------
>>> from nbody_streams import agama_helper as ah
>>>
>>> # --- Fit and save coefficient files ---
>>> snap = ah.create_snapshot_dict(pos_dark, mass_dark, pos_star, mass_star)
>>> ah.fit_potential(snap, nsnap=90, rmax_sel=300.0, save_dir="potential/")
>>>
>>> # --- Pack text files into HDF5 ---
>>> ah.write_snapshot_coefs_to_h5(
...     snapshot_ids=range(90, 101),
...     coef_file_patterns=["potential/{snap:03d}.dark.none_4.coef_mul_DR"],
...     h5_output_paths=["MW_mult.h5"],
... )
>>>
>>> # --- Read coef dataclasses (file, h5, or raw string — all the same call) ---
>>> mc  = ah.read_coefs("potential/090.dark.none_8.coef_mult")   # auto-detect type
>>> mc  = ah.read_coefs("MW_mult.h5", group_name="snap_090")     # from HDF5
>>> cc  = ah.read_coefs("MW_cylsp.h5", group_name="snap_090")    # CylSpline
>>>
>>> # --- Inspect and modify ---
>>> print(mc.lmax, mc.l_values, mc.total_power(2))
>>> mc2 = mc.zeroed(keep_lm=[(0, 0), (2, 0), (2, 2)])
>>>
>>> # --- Load Agama potential (file, h5, string — all the same call) ---
>>> pot = ah.load_agama_potential("potential/090.dark.none_8.coef_mult")
>>> pot = ah.load_agama_potential("MW_mult.h5", group_name="snap_090")
>>> pot = ah.load_agama_potential("MW_mult.h5", group_name="snap_090",
...                               keep_lm_mult=[(0,0),(2,0)])   # selective
>>> pot = ah.load_agama_potential(mc2.to_coef_string())    # from modified dataclass
>>>
>>> # --- Time-evolving potential ---
>>> import numpy as np
>>> times = np.linspace(6.0, 14.0, 101)
>>> pot_ev = ah.load_agama_evolving_potential("MW_mult.h5", times)          # HDF5
>>> pot_ev = ah.load_agama_evolving_potential("potential/MW_mult.ini")       # .ini (times embedded)
"""

# Set Agama units (Msol, kpc, km/s) at import time.
# Time is in kpc/(km/s) ~ 0.978 Gyr.
try:
    import agama as _agama
    _agama.setUnits(mass=1, length=1, velocity=1)
except ImportError:
    pass

from ._fit import create_snapshot_dict, fit_potential
from ._io import (
    write_coef_to_h5,
    write_snapshot_coefs_to_h5,
    read_coef_string,
)
from ._load import (
    load_agama_potential,
    load_agama_evolving_potential,
    create_evolving_ini,
)
from ._coefs import (
    MultipoleCoefs,
    CylSplineCoefs,
    read_coefs,
    generate_lmax_pairs,
)
from ._fire import (
    load_fire_pot,
    read_snapshot_times,
    create_fire_evolving_ini,
)

from ._potential import (
    PotentialGPU,
)
__all__ = [
    # potential class
    "PotentialGPU",
    # snapshot dict + fitting
    "create_snapshot_dict",
    "fit_potential",
    # HDF5 write
    "write_coef_to_h5",
    "write_snapshot_coefs_to_h5",
    # reading coef dataclasses or raw strings
    "read_coefs",           # unified: auto-detects Multipole vs CylSpline
    "read_coef_string",     # raw text (file or h5)
    # coef dataclasses
    "MultipoleCoefs",
    "CylSplineCoefs",
    "generate_lmax_pairs",
    # loading Agama potential objects
    "load_agama_potential",           # single snapshot, any source
    "load_agama_evolving_potential",  # time-evolving, from HDF5
    "create_evolving_ini",
    # FIRE-specific
    "load_fire_pot",
    "read_snapshot_times",
    "create_fire_evolving_ini",
]
