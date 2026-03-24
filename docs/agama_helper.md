# `nbody_streams.agama_helper`

Secondary utilities for fitting, storing, modifying, and loading
[Agama](https://github.com/GalacticDynamics-Oxford/Agama) expansion-based
potentials — specifically Multipole (spherical harmonic BFE) and CylSpline
(azimuthal harmonic + 2-D spline BFE) expansions.

> **Units** — `agama.setUnits(mass=1, length=1, velocity=1)` is called at
> import time (no-op if agama is not installed).  All quantities are in
> **(Msol, kpc, km/s)**; time is in kpc/(km/s) ≈ 0.978 Gyr.

```python
from nbody_streams import agama_helper as ah
```

---

## Contents

- [Overview](#overview)
- [Quick-start workflow](#quick-start-workflow)
- [Source types](#source-types)
- [Coefficient dataclasses](#coefficient-dataclasses)
  - [MultipoleCoefs](#multipolecoefs)
  - [CylSplineCoefs](#cylsplinecoefs)
- [Reading API](#reading-api)
- [HDF5 I/O](#hdf5-io)
- [Loading Agama potentials](#loading-agama-potentials)
- [Center parameter](#center-parameter)
- [FIRE helpers](#fire-helpers)
- [Potential fitting](#potential-fitting)

---

## Overview

The `agama_helper` module sits on top of Agama and provides a Python-native
workflow for the two most common expansion types:

| Expansion | Agama type | File extension | Use case |
|---|---|---|---|
| Multipole | `Multipole` | `.coef_mult` | Dark matter halos, spheroidal components |
| CylSpline | `CylSpline` | `.coef_cylsp` | Stellar discs, baryonic components |

The design separates three concerns:

```
read_*   ->  MultipoleCoefs / CylSplineCoefs   (structured in-memory data)
write_*  ->  HDF5 archives                     (compact storage)
load_*   ->  agama.Potential                   (ready for orbit integration)
```

---

## Quick-start workflow

### 1 — Fit and save coefficient files

```python
snap = ah.create_snapshot_dict(pos_dark, mass_dark, pos_star, mass_star)
ah.fit_potential(snap, nsnap=90, rmax_sel=300.0, save_dir="potential/")
```

### 2 — Pack text files into HDF5

```python
import numpy as np

ah.write_snapshot_coefs_to_h5(
    snapshot_ids=range(90, 101),
    coef_file_patterns=[
        "potential/{snap:03d}.dark.none_8.coef_mult",
        "potential/{snap:03d}.bar.none_8.coef_cylsp",
    ],
    h5_output_paths=["MW_mult.h5", "MW_cylsp.h5"],
    times=np.linspace(6.0, 14.0, 11),   # embed for load_agama_evolving_potential
)
```

### 3 — Inspect and modify coefficients

```python
mc = ah.read_coefs("potential/090.dark.none_8.coef_mult")
print(mc.lmax, mc.l_values)       # 8  [0, 2, 4, 6, 8]
print(mc.total_power(2))          # quadrupole power

# Keep only monopole and quadrupole (all m)
mc_axi = mc.zeroed([0, 2])

# Keep specific (l, m) pairs
mc_sel = mc.zeroed([(0, 0), (2, 0), (2, 2)])
```

### 4 — Load Agama potential

```python
pot     = ah.load_agama_potential("potential/090.dark.none_8.coef_mult")
pot     = ah.load_agama_potential("MW_mult.h5", group_name="snap_090")
pot     = ah.load_agama_potential(mc_axi)                 # from dataclass
pot_ev  = ah.load_agama_evolving_potential("MW_mult.h5")  # time-evolving
pot_ev  = ah.load_agama_evolving_potential("MW_mult.ini") # native .ini
```

---

## Source types

All `read_*` and `load_*` functions transparently accept three (or four) source forms:

| Form | Example | Notes |
|---|---|---|
| Plain-text file | `"potential/090.dark.none_8.coef_mult"` | Agama's native export format |
| HDF5 archive | `"MW_mult.h5"` + `group_name="snap_090"` | Written by `write_coef_to_h5` |
| Raw string | `mc.to_coef_string()` | Result of `.to_coef_string()` or any prior read |
| Coef dataclass | `mc` (a `MultipoleCoefs`) | `load_agama_potential` only |

Detection is automatic — no need to specify the type.

---

## Coefficient dataclasses

### `MultipoleCoefs`

```python
@dataclass
class MultipoleCoefs:
    R_grid   : np.ndarray          # (nR,)      radial grid [kpc]
    lm_labels: list[tuple[int,int]]# [(l,m)...] ordered column labels
    phi      : np.ndarray          # (nR, n_lm) Phi_{l,m}(r)
    dphi_dr  : np.ndarray | None   # (nR, n_lm) dPhi/dr  (None if absent)
    metadata : dict                # header key/value pairs
```

**Properties:**

| Property | Type | Description |
|---|---|---|
| `.lmax` | `int` | Maximum l order present |
| `.l_values` | `list[int]` | Sorted unique l values |
| `.m_values` | `list[int]` | Sorted unique m values (includes negatives) |

**Methods:**

#### `radial_power(l, use_quadrature=True) -> ndarray`

Power spectrum for harmonic order *l* at each radial grid point.

```python
r_power = mc.radial_power(2)          # shape (nR,), co-indexed with mc.R_grid
```

#### `total_power(l, use_quadrature=True) -> float`

Total power for harmonic order *l* summed over all radial bins.

```python
q_power = mc.total_power(2)           # scalar
```

#### `zeroed(keep_lm) -> MultipoleCoefs`

Return a copy with all (l, m) terms **not** in `keep_lm` zeroed.
Negative-m counterparts are included automatically.

```python
# By l order (keep all m for those l)
mc_axi = mc.zeroed([0, 2, 4])

# By specific (l, m) pairs
mc_sel = mc.zeroed([(0, 0), (2, 0)])

# Mixed
mc_mix = mc.zeroed([0, (2, 0), (2, 2)])
```

Passing anything other than `int` or `(int, int)` tuples raises `TypeError`.
Passing an `l` value not present in the expansion emits a `UserWarning`.

#### `to_coef_string() -> str`

Serialise back to Agama's plain-text Multipole format.  Round-trippable:
`read_coefs(mc.to_coef_string())` returns an equivalent dataclass.

---

### `CylSplineCoefs`

```python
@dataclass
class CylSplineCoefs:
    m_values : list[int]           # azimuthal orders present (sorted)
    R_grid   : np.ndarray          # (nR,)    cylindrical R grid [kpc]
    z_grid   : np.ndarray          # (nz,)    vertical grid [kpc]
    phi      : dict[int, ndarray]  # phi[m] shape (nR, nz)
    metadata : dict
```

#### `zeroed(keep_m, include_negative=True) -> CylSplineCoefs`

Return a copy with all m tables **not** in `keep_m` zeroed.

```python
cc_axi  = cc.zeroed([0])           # axisymmetric only
cc_sel  = cc.zeroed([0, 2, 4])     # keep m=0,+/-2,+/-4
cc_nopm = cc.zeroed([0, 2], include_negative=False)  # keep m=0,2 only
```

#### `to_coef_string() -> str`

Serialise back to Agama's CylSpline text format.

---

## Reading API

### `read_coefs(source, group_name="snap_000", dataset_name="coefs")`

Unified entry point.  Auto-detects expansion type from the file header.

```python
mc  = ah.read_coefs("potential/090.dark.none_8.coef_mult")
cc  = ah.read_coefs("potential/090.bar.none_8.coef_cylsp")
mc  = ah.read_coefs("MW_mult.h5",  group_name="snap_090")
cc  = ah.read_coefs("MW_cylsp.h5", group_name="snap_090")
mc  = ah.read_coefs(mc.to_coef_string())   # from raw string
```

Returns `MultipoleCoefs` or `CylSplineCoefs`.

### `read_coef_string(source, group_name="snap_000", dataset_name="coefs") -> str`

Return the raw UTF-8 coefficient text without parsing it.  Useful when you
want to inspect the raw format or pass it to another tool.

---

## HDF5 I/O

### `write_coef_to_h5(h5_path, coef_string, group_name="snap_000", dataset_name="coefs", overwrite=False, metadata=None)`

Store a single coefficient string in an HDF5 group.

```python
ah.write_coef_to_h5(
    "MW_mult.h5",
    Path("potential/090.dark.none_8.coef_mult").read_text(),
    group_name="snap_090",
    metadata={"lmax": 8, "snap": 90},
)
```

### `write_snapshot_coefs_to_h5(snapshot_ids, coef_file_patterns, h5_output_paths, group_fmt="snap_{snap:03d}", dataset_name="coefs", overwrite=True, times=None)`

Batch-write many snapshots.  One HDF5 file per entry in `coef_file_patterns`.

```python
ah.write_snapshot_coefs_to_h5(
    snapshot_ids=range(90, 101),
    coef_file_patterns=[
        "potential/{snap:03d}.dark.none_8.coef_mult",
        "potential/{snap:03d}.bar.none_8.coef_cylsp",
    ],
    h5_output_paths=["MW_mult.h5", "MW_cylsp.h5"],
    times=np.linspace(6.0, 14.0, 11),   # embed for load_agama_evolving_potential
)
```

When `times` is provided, a root-level `"times"` dataset is written to each
HDF5 file.  `load_agama_evolving_potential` reads this automatically when
`times` is not passed explicitly.

**HDF5 structure produced:**

```
MW_mult.h5
+-- times          (float64 array, len = n_snapshots)
+-- snap_090/
|   +-- coefs      (scalar UTF-8 string)
+-- snap_091/
|   +-- coefs
...
```

---

## Loading Agama potentials

### `load_agama_potential(source, group_name="snap_000", dataset_name="coefs", center=None, keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True)`

Load a **single-snapshot** Agama potential.

```python
# From any source type
pot = ah.load_agama_potential("potential/090.dark.none_8.coef_mult")
pot = ah.load_agama_potential("MW_mult.h5", group_name="snap_090")
pot = ah.load_agama_potential(mc.zeroed([0, 2]))   # from modified dataclass

# With in-memory harmonic filtering (Multipole)
pot = ah.load_agama_potential("MW_mult.h5",
                               group_name="snap_090",
                               keep_lm_mult=[0, 2])    # all m for l=0,2
pot = ah.load_agama_potential("MW_mult.h5",
                               group_name="snap_090",
                               keep_lm_mult=[(0,0),(2,0)])  # specific pairs

# With in-memory filtering (CylSpline)
pot = ah.load_agama_potential("MW_cylsp.h5",
                               group_name="snap_090",
                               keep_m_cylspl=[0, 2])
```

**Type-safety:**
- Passing `keep_lm_mult` to a CylSpline source raises `TypeError`.
- Passing `keep_m_cylspl` to a Multipole source raises `TypeError`.
- Passing an Evolving config (`.ini`) raises `TypeError` with a pointer to
  `load_agama_evolving_potential`.

All temporary files are removed in a `finally` block (even on failure).

---

### `load_agama_evolving_potential(source, times=None, *, group_names=None, dataset_name="coefs", center=None, interp_linear=True, keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True)`

Build a **time-evolving** `agama.Potential` from an HDF5 archive or a native
Agama Evolving `.ini` file.

```python
# From HDF5 with embedded times
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5")

# From HDF5, times provided explicitly (overrides embedded times)
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5",
                                           times=np.linspace(6, 14, 11))

# From a native Agama Evolving .ini file
pot_ev = ah.load_agama_evolving_potential("potential/MW_mult.ini")

# With per-snapshot filtering
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5", keep_lm_mult=[0])
pot_ev = ah.load_agama_evolving_potential("MW_cylsp.h5", keep_m_cylspl=[0, 2])
```

The function materialises every snapshot to `/dev/shm`, assembles a new
`Evolving` config, constructs the potential, and cleans up all temporaries.

**Agama `.ini` format** (parsed automatically):

```ini
[Potential]
type = Evolving
interpLinear = True
Timestamps
6.0   /path/to/snap_090.coef_mult
6.8   /path/to/snap_091.coef_mult
...
```

Relative paths in the `.ini` are resolved relative to the `.ini` file's
directory.

---

### `create_evolving_ini(times, coef_paths, output_path, interp_linear=True) -> str`

Write an Agama Evolving `.ini` from explicit file paths.

```python
ini_path = ah.create_evolving_ini(
    times=np.linspace(6, 14, 11),
    coef_paths=[f"potential/{i:03d}.dark.none_8.coef_mult" for i in range(90, 101)],
    output_path="potential/MW_mult_evolving.ini",
)
```

---

## Center parameter

All `load_*` functions accept a `center=` keyword that is forwarded to
`agama.Potential`.  Four forms are supported:

| Form | Type | Columns | Description |
|---|---|---|---|
| Static | length-3 sequence | — | `[x, y, z]` in kpc |
| Time-varying position | (N, 4) ndarray | time, x, y, z | Written to temp file automatically |
| Time-varying pos+vel | (N, 7) ndarray | time, x, y, z, vx, vy, vz | Written to temp file automatically |
| File path | str or Path | — | Passed through to Agama as-is |

```python
# Static centre
pot = ah.load_agama_potential(source, center=[8.1, 0, 0])

# Time-varying centre from an orbit array  shape (N_times, 4)
center_orbit = np.column_stack([times_gyr, x_mw, y_mw, z_mw])
pot = ah.load_agama_potential(source, center=center_orbit)

# From a file
pot = ah.load_agama_potential(source, center="orbit_mw.txt")
```

The temporary file created for 2-D arrays is cleaned up in the same `finally`
block as the coefficient temporary.

---

## FIRE helpers

These functions encode the FIRE simulation path conventions
(`potential/10kpc/`, `snapshot_times.txt`) used in Arora et al. 2022.

### `read_snapshot_times(sim_dir, sep=r'\s+') -> pandas.DataFrame`

Read `snapshot_times.txt` from a FIRE simulation directory.  Returns a
DataFrame with canonical columns `snap`, `scale-factor`, `redshift`,
`time[Gyr]`, `time_width[Myr]`.

```python
df = ah.read_snapshot_times("/data/m12i_res7100/")
times_gyr = df["time[Gyr]"].values
```

> **Note:** `pandas` is a lazy dependency of this function only.  If pandas
> is not installed, a clear `ImportError` is raised with an install hint.

### `create_fire_evolving_ini(sim_dir, model_pattern, output_filename, snap_range=None, verbose=True) -> str`

Write an Agama Evolving `.ini` from a FIRE simulation directory.

```python
ini = ah.create_fire_evolving_ini(
    sim_dir="/data/m12i_res7100/",
    model_pattern="*.dark.none_4.coef_mul_DR",
    output_filename="MW_dark_evolving.ini",
    snap_range=(500, 600),
)
```

### `load_fire_pot(sim_dir, nsnap, sym="n", lmax=4, kind="whole", keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, file_ext="DR", halo=None, verbose=True, return_coefs=False, save_modified=False, save_dir=None)`

Load a FIRE potential snapshot as an `agama.Potential`.

```python
# Full potential (dark + baryonic)
pot = ah.load_fire_pot("/data/m12i_res7100/", nsnap=600)

# Dark matter only, axisymmetric
pot = ah.load_fire_pot("/data/m12i_res7100/", nsnap=600,
                        kind="dark", lmax=8, keep_lm_mult=[0])

# Return the CylSpline dataclass instead of a potential
cc = ah.load_fire_pot("/data/m12i_res7100/", nsnap=600,
                       kind="bar", return_coefs=True)
```

**`kind` values:**

| `kind` | Returns |
|---|---|
| `"whole"` | `agama.Potential(dark, bar)` combined |
| `"dark"` | Multipole potential only |
| `"bar"` | CylSpline potential only |

When `return_coefs=True`:
- `kind="dark"` -> `MultipoleCoefs`
- `kind="bar"` -> `CylSplineCoefs`
- `kind="whole"` -> `(MultipoleCoefs, CylSplineCoefs)`

---

## Potential fitting

### `create_snapshot_dict(pos_dark, mass_dark, pos_star=None, mass_star=None, pos_gas=None, mass_gas=None, temperature_gas=None) -> dict`

Pack particle arrays into the dictionary format expected by `fit_potential`.

### `fit_potential(snap, nsnap, *, sym="n", lmax=4, rmax_sel=300.0, rmax_exp=None, file_ext="", save_dir="potential/", halo=None, kind="both", center=None, rotation=None, verbose=True, subsample_factor=1, cold_temp_log10_thresh=4.5) -> dict[str, list[str]]`

Fit Agama BFE potentials to a particle snapshot and save coefficient files.

```python
snap = ah.create_snapshot_dict(pos_dark, mass_dark, pos_star, mass_star)
paths = ah.fit_potential(
    snap,
    nsnap=90,
    sym="n",         # 'n'=none, 'a'=axisymmetric, 's'=spherical
    lmax=8,
    rmax_sel=300.0,  # selection radius [kpc]
    save_dir="potential/",
    verbose=True,
)
# paths["dark"] -> list of written Multipole file paths
# paths["bar"]  -> list of written CylSpline file paths
```

Dark matter and hot gas are fitted with a Multipole expansion; stars and cold
gas are fitted with a CylSpline expansion.

---

## Full example: MW potential pipeline

```python
import numpy as np
from nbody_streams import agama_helper as ah

SIM = "/data/m12i_res7100/"

# 1. Pack 10 snapshots into compact HDF5 archives
snap_ids = range(590, 601)
snap_times = np.linspace(13.0, 14.0, 11)   # Gyr

ah.write_snapshot_coefs_to_h5(
    snapshot_ids=snap_ids,
    coef_file_patterns=[
        SIM + "potential/10kpc/{snap}.dark.none_8.coef_mul_DR",
        SIM + "potential/10kpc/{snap}.bar.none_8.coef_cylsp_DR",
    ],
    h5_output_paths=["MW_dark.h5", "MW_bar.h5"],
    times=snap_times,
)

# 2. Inspect and sanity-check
mc = ah.read_coefs("MW_dark.h5", group_name="snap_590")
print("lmax:", mc.lmax)
print("l=2 power:", mc.total_power(2))

# 3. Load static potential (axisymmetric approximation)
pot_axi = ah.load_agama_potential("MW_dark.h5", group_name="snap_595",
                                   keep_lm_mult=[0, 2, 4])

# 4. Load full time-evolving potential
pot_ev = ah.load_agama_evolving_potential("MW_dark.h5")

# 5. Evaluate (same Agama interface as always)
import agama
xv = np.array([[8.1, 0, 0, 0, 220, 0]])
acc, phi = pot_ev(xv[:, :3], t=13.5, der=1)
```
