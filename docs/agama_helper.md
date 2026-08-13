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

> **Example notebook** — `examples/coef_time_axis.ipynb` walks through the
> coefficient time axis end to end (reading, manipulating, writing, FIRE
> resampling) against real FIRE fixtures.

---

## Contents

- [Overview](#overview)
- [Quick-start workflow](#quick-start-workflow)
- [Source types](#source-types)
- [Coefficient dataclasses](#coefficient-dataclasses)
  - [MultipoleCoefs](#multipolecoefs)
  - [CylSplineCoefs](#cylsplinecoefs)
- [Coefficient time axis](#coefficient-time-axis)
- [Writers and materialization](#writers-and-materialization)
- [Reading API](#reading-api)
- [HDF5 I/O](#hdf5-io)
- [Loading Agama potentials](#loading-agama-potentials)
- [GPU potential evaluation (PotentialGPU)](#gpu-potential-evaluation-potentialgpu)
- [Center parameter](#center-parameter)
- [FIRE helpers](#fire-helpers)
- [Potential fitting](#potential-fitting)
- [Gotchas](#gotchas)

---

## Overview

The `agama_helper` module sits on top of Agama and provides a Python-native
workflow for the two most common expansion types, plus a GPU acceleration layer:

| Expansion | Agama type | File extension | Use case |
|---|---|---|---|
| Multipole | `Multipole` | `.coef_mult` | Dark matter halos, spheroidal components |
| CylSpline | `CylSpline` | `.coef_cylsp` | Stellar discs, baryonic components |

The design separates four concerns:

```
read_*        ->  MultipoleCoefs / CylSplineCoefs   (structured in-memory data)
write_*       ->  HDF5 archives                     (compact storage)
load_*        ->  agama.Potential  (default)         (ready for CPU orbit integration)
              ->  PotentialGPU    (gpu=True)          (GPU-accelerated, drop-in replacement)
PotentialGPU  ->  GPU potential from any source      (type=, file=, dataclass, agama.Potential)
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
    phi      : np.ndarray          # (nR, n_lm) or (nR, n_lm, nt)  Phi_{l,m}(r)
    dphi_dr  : np.ndarray | None   # same shape as phi (None if absent)
    metadata : dict = field(default_factory=dict)   # header key/value pairs
    times    : np.ndarray | None = None             # (nt,); None = no time axis
```

`times` is the last field, defaulting to `None`, so every existing positional
construction (`MultipoleCoefs(R_grid, lm_labels, phi, dphi_dr, metadata)`)
keeps working unchanged. See [Coefficient time axis](#coefficient-time-axis).

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
    m_values : list[int]                  # azimuthal orders present (sorted)
    R_grid   : np.ndarray                 # (nR,) cylindrical R grid [kpc]
    z_grid   : np.ndarray                 # (nz,) vertical grid [kpc]
    phi      : dict[int, ndarray]         # phi[m] shape (nR, nz) or (nR, nz, nt)
    metadata : dict = field(default_factory=dict)
    dphi_dR  : dict[int, ndarray] | None = None   # same shapes as phi; None if absent
    dphi_dz  : dict[int, ndarray] | None = None   # same shapes as phi; None if absent
    times    : np.ndarray | None = None           # (nt,); None = no time axis
```

A real `Potential.export()` carries `#Phi`, `#dPhi/dR` and `#dPhi/dz`
sections; `dphi_dR`/`dphi_dz` are populated from those and are `None` when the
source only had `#Phi` (see [Gotchas](#gotchas)). `times` is the last field,
defaulting to `None` — see [Coefficient time axis](#coefficient-time-axis).

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

## Coefficient time axis

`MultipoleCoefs` and `CylSplineCoefs` hold **either a single snapshot or a
whole time series**, in the same class. Time, when present, is always the
**last** array axis, and is always **optional**. There is no separate series
class and no separate series reader — a single-snapshot read behaves exactly
as it always has (`times is None`), and every multi-snapshot form (see
[Reading API](#reading-api)) returns the same dataclass with a trailing time
axis attached.

| field | no time axis | with time axis |
|---|---|---|
| `MultipoleCoefs.phi` | `(nR, n_lm)` | `(nR, n_lm, nt)` |
| `MultipoleCoefs.dphi_dr` | `(nR, n_lm)` or `None` | `(nR, n_lm, nt)` or `None` |
| `CylSplineCoefs.phi[m]` | `(nR, nz)` | `(nR, nz, nt)` |
| `CylSplineCoefs.dphi_dR[m]` | `(nR, nz)` or `None` | `(nR, nz, nt)` or `None` |
| `CylSplineCoefs.dphi_dz[m]` | `(nR, nz)` or `None` | `(nR, nz, nt)` or `None` |
| `times` | `None` | `(nt,)` |

`times` is the **last** dataclass field, defaulting to `None`, so existing
positional construction of either class is unaffected.

### Introspection properties

| Property | Type | Description |
|---|---|---|
| `.has_time_axis` | `bool` | `True` iff `times is not None` |
| `.n_times` | `int \| None` | `len(times)`, or `None` without a time axis |

### Methods (both classes unless noted)

#### `copy() -> MultipoleCoefs \| CylSplineCoefs`

Deep copy — no array, list or dict is shared with the original (including
`times`).

#### `snapshot(i) -> MultipoleCoefs \| CylSplineCoefs`  (alias: `obj[i]`)

Time-less view at time index *i* (negative indices allowed). The result
**shares memory** with the parent — every returned array is a NumPy view — so
writing through it edits the parent. Call `.copy()` on the result for an
independent object. Raises if the object has no time axis.

```python
s3 = ser.snapshot(3)     # == ser[3]
s3.phi.base is ser.phi   # True -- a view, not a copy
```

#### `column(l, m) -> int`  (Multipole only)

Index of the `(l, m)` column in `phi`, `dphi_dr` and `lm_labels`. Raises
`KeyError` if the pair is absent.

```python
idx = ser.column(2, 2)
ser.phi[:, idx, :] *= 1.5     # arbitrary manual surgery on one harmonic
```

#### `validate() -> None`

Assert internal shape consistency between the grids/labels, `phi`,
`dphi_dr`/`dphi_dR`/`dphi_dz` and `times`, naming the offending field on
failure. **Direct field assignment is legal** — `validate()` is what catches
the damage afterwards, and it runs automatically at the top of every write and
materialize entry point (`to_coef_string`, `to_coef_strings`, `to_coef_files`,
`to_h5`, `to_evolving_ini`, `materialize_potential`).

```python
ser.phi = ser.phi[:, :-1]   # drop a column by hand -- now inconsistent
ser.validate()              # ValueError naming MultipoleCoefs.phi and its shape
```

#### `with_times(times, *, phi=None, dphi_dr=None) -> MultipoleCoefs`
#### `with_times(times, *, phi=None, dphi_dR=None, dphi_dz=None) -> CylSplineCoefs`

Attach or relabel the time axis. **Nothing is interpolated, smoothed or
resampled here** — compute your own time-resampled arrays (or use
[`spline_resample_coefs`](#fire-helpers)) and attach the result.

- Omitting `phi` only **relabels** an existing time axis: `len(times)` must
  equal the current `n_times`.
- Supplying `phi` **attaches or replaces** the time axis: `len(times)` may
  differ freely from the current `n_times`.
- `MultipoleCoefs.phi` must be `(nR, n_lm, nt)` — the leading two axes must
  match the existing `R_grid`/`lm_labels` exactly; grids are never
  interpolated.
- `CylSplineCoefs.phi`/`dphi_dR`/`dphi_dz` are **dicts keyed by exactly the
  existing `m_values`** (not a stacked array), each entry `(nR, nz, nt)`.
- `dphi_dr` (or `dphi_dR`/`dphi_dz`) is required alongside `phi` whenever the
  object already carries one — there is no way to silently drop a derivative
  block via `with_times`.

```python
new_phi = my_spline(mc.phi, old_times, new_times)      # your resampling code
new_dphi = my_spline(mc.dphi_dr, old_times, new_times)
ser2 = mc.with_times(new_times, phi=new_phi, dphi_dr=new_dphi)
```

### Module-level: `stack_coefs(items, times) -> MultipoleCoefs | CylSplineCoefs`

Stack time-less snapshots into one object carrying a time axis.

```python
c0 = mc.copy()
c1 = mc.copy()
c1.phi = c1.phi * 2.0
c1.dphi_dr = c1.dphi_dr * 2.0
ser = ah.stack_coefs([c0, c1], times=[0.0, 1.0])
ser.phi.shape   # (nR, n_lm, 2)
```

**Grids and labels are never interpolated or reconciled.** Stacking snapshots
whose `R_grid`, `z_grid`, `lm_labels`, `m_values` or header `metadata` differ
is a hard `ValueError`, and mixing `MultipoleCoefs` with `CylSplineCoefs` is a
hard `TypeError`. Each item must itself be time-less; stacking an
already-series object raises.

### `radial_power` / `total_power` with a time axis

The **meaning** of the quantity is unchanged — only a trailing time axis is
appended when present:

```python
mc.radial_power(2).shape    # (nR,)       -- no time axis
mc.total_power(2)           # float       -- no time axis

ser.radial_power(2).shape   # (nR, nt)    -- with a time axis
ser.total_power(2).shape    # (nt,)       -- with a time axis
```

---

## Writers and materialization

Both coefficient classes can turn themselves back into live potentials or
on-disk files, iterating over every time sample when a time axis is present.

### `materialize_potential(*, center=None, interp_linear=True, gpu=False)`

Build a live potential from the *current* (possibly hand-edited) arrays.
Without a time axis this delegates to `load_agama_potential` (see
[Loading Agama potentials](#loading-agama-potentials)); with one it delegates
to `load_agama_evolving_potential` using `times=self.times`. Either way
`validate()` runs first.

```python
pot = ser.materialize_potential()                 # agama.Potential (Evolving)
pot = ser.materialize_potential(gpu=True)          # EvolvingPotentialGPU
pot = mc.materialize_potential()                   # no time axis -> single snapshot
```

### `to_coef_string(t=None) -> str`

Serialise one snapshot. `t` is **required** when the object carries a time
axis and **rejected** (`ValueError`) when it does not.

```python
s3 = ser.to_coef_string(t=3)     # one time index
mc.to_coef_string()              # time-less object, no t=
```

### `to_coef_strings() -> list[str]`

Serialise every time sample — a one-element list when there is no time axis.

### `to_coef_files(out_dir, name_fmt="snap_{i:04d}{ext}") -> list[str]`

Write one plain-text coefficient file per time sample. `name_fmt` receives
`i` (time index) and `ext` (`.coef_mult` or `.coef_cylsp`); it must produce a
distinct name per sample or a `ValueError` is raised (a fixed name is fine for
a time-less object, since there is only one sample).

### `to_h5(path, group_fmt="snap_{i:04d}", dataset_name="coefs", overwrite=True, write_times=True) -> str`

Write every time sample into one HDF5 archive, in the layout produced by
`write_snapshot_coefs_to_h5` — round-trips through
`read_coefs(path, group_name="all")`. **Appends** to an existing archive
(groups from an earlier, longer write survive); delete the file first for a
clean archive. `group_fmt` must contain `{i}` or raises, same as `name_fmt`
above. `write_times=False` skips the root `"times"` dataset (ignored without
a time axis).

### `to_evolving_ini(ini_path, out_dir=None, interp_linear=True) -> str`

Write the per-snapshot coefficient files plus an Agama `Evolving` `.ini`.
Requires a time axis (`ValueError` otherwise — an Evolving config is a list of
`(time, file)` pairs). `out_dir` defaults to `ini_path`'s parent; file names
are prefixed with the `.ini` stem.

```python
ser.to_h5("series.h5")                 # round-trips via group_name="all"
ser.to_evolving_ini("series.ini")      # native Agama Evolving config
```

---

## Reading API

### `read_coefs(source, group_name="snap_000", dataset_name="coefs", times=None)`

Unified entry point.  Auto-detects expansion type from the file header.
`read_mult_coefs` and `read_cylspl_coefs` share this exact signature — only
the expansion type is fixed instead of auto-detected. A single-group read
returns exactly what it always has (`times is None`); any multi-snapshot form
returns the same dataclass with a trailing time axis attached.

```python
mc  = ah.read_coefs("potential/090.dark.none_8.coef_mult")
cc  = ah.read_coefs("potential/090.bar.none_8.coef_cylsp")
mc  = ah.read_coefs("MW_mult.h5",  group_name="snap_090")
cc  = ah.read_coefs("MW_cylsp.h5", group_name="snap_090")
mc  = ah.read_coefs(mc.to_coef_string())   # from raw string
```

Returns `MultipoleCoefs` or `CylSplineCoefs`.

**`group_name`** selects HDF5 groups:

| Form | Result |
|---|---|
| plain `str` (default `"snap_000"`) | **one** group; no time axis (unchanged default behaviour) |
| `"all"` | every group in the archive, numerically sorted (`"snap_0042"` sorts as 42); time axis present |
| sequence of `str` | exactly those groups, in the given order; time axis present |

**`source`** additionally accepts two more forms, for which `group_name` is
**refused** (a `ValueError`, not silently ignored) since it has no meaning:

- An Agama Evolving `.ini` path — every listed snapshot is read.
- A sequence of file paths / raw coef strings / coef objects, in the given
  order — `np.sort(glob(...))` or a generator both work.

```python
ser = ah.read_coefs("MW_mult.h5", group_name="all")                # whole archive
ser = ah.read_coefs("MW_mult.h5", group_name=["snap_090", "snap_095"],
                     times=[6.0, 6.5])
ser = ah.read_coefs("potential/MW_mult.ini")                        # times from the .ini
ser = ah.read_coefs(sorted(glob("potential/*.coef_mult")), times=t_gyr)
```

**`times`** resolves in this order: the explicit argument, then the `.h5`
root `"times"` dataset, then the `.ini` timestamps. If a time axis is
requested and none of those yield times, this **raises** — an index-based
axis is never invented. The root `"times"` dataset is co-indexed with the
archive's numerically sorted group order, so an explicit `group_name` that
subsets or reorders the archive still gets each group's own time, not the
time at its position in the request.

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

### `write_snapshot_coefs_to_h5(snapshot_ids, coef_file_patterns, h5_output_paths, group_fmt="snap_{snap:03d}", dataset_name="coefs", overwrite=True, encoding="utf-8", times=None)`

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

### `load_agama_potential(source, group_name="snap_000", dataset_name="coefs", center=None, keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, gpu=False)`

Load a **single-snapshot** potential.  By default returns `agama.Potential`;
pass `gpu=True` for a GPU-accelerated drop-in replacement.

```python
# --- CPU (agama.Potential) ---
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

# --- GPU (PotentialGPU) ---
pot_gpu = ah.load_agama_potential("potential/090.dark.none_8.coef_mult", gpu=True)
pot_gpu = ah.load_agama_potential("potential/090.bar.none_8.coef_cylsp", gpu=True)
pot_gpu = ah.load_agama_potential("MW_mult.h5", group_name="snap_090", gpu=True)
pot_gpu = ah.load_agama_potential(mc.zeroed([(0,0),(2,0)]), gpu=True)

# Filtering still applies before GPU construction
pot_gpu = ah.load_agama_potential("MW_mult.h5", group_name="snap_090",
                                   keep_lm_mult=[0, 2], gpu=True)

# center= wraps in ShiftedPotentialGPU (array, not file path)
pot_gpu = ah.load_agama_potential("MW_mult.h5", gpu=True,
                                   center=np.array([1.0, 0.0, 0.0]))
pot_gpu = ah.load_agama_potential("MW_mult.h5", gpu=True,
                                   center=lmc_trajectory[:, :4])  # (T,4) [t,x,y,z]
```

**Type-safety:**
- Passing `keep_lm_mult` to a CylSpline source raises `TypeError`.
- Passing `keep_m_cylspl` to a Multipole source raises `TypeError`.
- Passing an Evolving config raises `TypeError` — use `load_agama_evolving_potential`.

All temporary files are removed in a `finally` block (even on failure).

---

### `load_agama_evolving_potential(source, times=None, *, group_names=None, dataset_name="coefs", center=None, interp_linear=True, keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, gpu=False)`

Build a **time-evolving** potential from an HDF5 archive, a native Agama
Evolving `.ini` file, a coefficient object that already carries a time axis,
or a sequence of coef objects / paths / raw coef strings.

```python
# --- CPU (agama.Potential) ---
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5")
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5",
                                           times=np.linspace(6, 14, 11))
pot_ev = ah.load_agama_evolving_potential("potential/MW_mult.ini")
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5", keep_lm_mult=[0])
pot_ev = ah.load_agama_evolving_potential("MW_cylsp.h5", keep_m_cylspl=[0, 2])

# --- From a coefficient object carrying a time axis ---
ser = ah.read_coefs("MW_mult.h5", group_name="all")
pot_ev = ah.load_agama_evolving_potential(ser)          # times taken from ser.times
pot_ev = ah.load_agama_evolving_potential(ser, times=my_times)  # overrides ser.times

# --- GPU (EvolvingPotentialGPU) ---
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5", gpu=True)
pot_ev = ah.load_agama_evolving_potential("MW_mult.h5",
                                           times=np.linspace(6, 14, 11),
                                           keep_lm_mult=[0, 2], gpu=True)
pot_ev = ah.load_agama_evolving_potential(ser, gpu=True)   # skips the text round-trip
```

With `gpu=True` the function builds one `PotentialGPU` per snapshot and
returns an `EvolvingPotentialGPU` with linear time-interpolation (controlled
by `interp_linear`).  All filtering is applied before GPU construction.  When
`source` is a coefficient object, the GPU path feeds its arrays straight to
the GPU builders (`_build_multipole_data`/`_build_cylspline_data`), skipping
the text round-trip entirely — more precise than the string path (which
truncates through `%.13g`/`%.14g`), not merely faster.

A coefficient object with **no** time axis (`times is None`) raises
`TypeError` — use `load_agama_potential` for that, or attach one first with
`with_times()` / `stack_coefs()` / `read_coefs(..., group_name="all")`.

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

## GPU potential evaluation (PotentialGPU)

`PotentialGPU` is a GPU-accelerated drop-in for `agama.Potential` that targets
N ≥ 50 k particles where GPU throughput gives a 5–10× speedup.  It is
importable directly or reached via `load_agama_potential(gpu=True)`.

```python
from nbody_streams.agama_helper import PotentialGPU
```

### Supported types

| Source | GPU class |
|---|---|
| Multipole `.coef_mul` / `.coef_mul_DR` | `MultipolePotentialGPU` |
| CylSpline `.coef_cylsp_DR` | `CylSplinePotentialGPU` |
| Analytic: `NFW`, `Plummer`, `Hernquist`, `Isochrone` | Direct CuPy kernel |
| Analytic: `MiyamotoNagai`, `LogHalo`/`Logarithmic` | Direct CuPy kernel |
| Analytic: `DehnenSpherical` (γ ∈ [0,2)) | Direct CuPy kernel |
| Analytic: `UniformAcceleration` (constant or time-dependent) | Direct CuPy kernel |
| `Disk` | `CompositePotentialGPU(DiskAnsatzGPU + MultipolePotentialGPU)` |
| `Spheroid`, `King` | Agama CPU export → `MultipolePotentialGPU` |
| Multi-component `.ini` / any text file with `[Potential]` | `CompositePotentialGPU` |
| `EvolvingPotentialGPU` | Linear lerp of snapshot GPU potentials |

### Factory: `PotentialGPU(...)`

Mirrors `agama.Potential` — accepts the same argument styles:

```python
# Analytic type (case-insensitive type= and kwargs)
pot = PotentialGPU(type='NFW', mass=1e12, scaleRadius=20)
pot = PotentialGPU(type='nfw', Mass=1e12, ScaleRadius=20)   # case-insensitive

# BFE from file (auto-detects Multipole vs CylSpline by content)
pot = PotentialGPU(file='snap.coef_mul_DR')
pot = PotentialGPU(file='bar.coef_cylsp_DR')

# Multi-section INI / any text file with [Potential ...] headers
pot = PotentialGPU(file='potential.ini')        # .ini extension
pot = PotentialGPU(file='potential.dat')        # any extension: content-detected
pot = PotentialGPU(file='potential')            # no extension: also works

# Arbitrary section header names (case-insensitive, Agama-compatible)
# potential.ini:
#   [Potential halo]
#   type = NFW
#   mass = 1e12
#   scaleRadius = 20
#   [Potential disk]
#   type = MiyamotoNagai
#   mass = 5e10
#   scaleRadius = 3
#   scaleHeight = 0.3
pot = PotentialGPU(file='potential.ini')   # → CompositePotentialGPU([NFW, MN])

# From MultipoleCoefs / CylSplineCoefs dataclass
pot = PotentialGPU(mc)
pot = PotentialGPU(mc.zeroed([(0,0),(2,0)]))

# From agama.Potential (exports to BFE)
pot = PotentialGPU(agama.Potential(type='Spheroid', ...))
# Only types Agama actually exports coefficients for (Multipole, CylSpline,
# Spheroid, King, ...).  Agama analytic types store no parameters on export,
# so those raise with a message telling you to use type= instead.

# Non-inertial frame: spatially uniform, time-dependent acceleration
acc = np.loadtxt('accMW')                       # (T,4): t, ax, ay, az
pot = PotentialGPU(type='UniformAcceleration', file=acc)
pot = PotentialGPU(type='UniformAcceleration', file='accMW')      # or a path
pot = PotentialGPU(type='UniformAcceleration', ax=0.01, ay=0., az=0.)  # constant

# Composite: variadic positional args
pot = PotentialGPU(pot_halo, pot_disk, pot_lmc)

# Modifiers
pot = PotentialGPU(mc, center=[x0, y0, z0])           # static shift
pot = PotentialGPU(mc, center=lmc_traj[:, :4])        # time-varying [t,x,y,z]
pot = PotentialGPU(mc, center=lmc_traj[:, :7])        # Hermite spline [t,x,y,z,vx,vy,vz]
pot = PotentialGPU(mc, scale=2.0, ampl=0.5)           # scaled

# + operator composes into CompositePotentialGPU
pot = pot_halo + pot_disk + pot_lmc
```

### API (matches `agama.Potential`)

All methods accept CuPy or NumPy arrays of shape `(N, 3)` or `(3,)`:

```python
phi  = pot.potential(xyz, t=0.)       # (N,)   [km/s]^2
F    = pot.force(xyz, t=0.)           # (N,3)  [km/s]^2/kpc  (= -grad Phi)
rho  = pot.density(xyz, t=0.)         # (N,)   [Msol/kpc^3]
F, dF = pot.forceDeriv(xyz, t=0.)    # (N,3), (N,6)
phi, F, dF = pot.evalDeriv(xyz, t=0.)
phi  = pot.eval(xyz, pot=True)        # Agama-compatible eval
F, dF = pot.eval(xyz, acc=True, der=True)
```

`forceDeriv` returns `dF = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]`
matching `agama.Potential.forceDeriv` exactly.

### `UniformAcceleration` — non-inertial reference frame

Φ(**x**, t) = −**a**(t)·**x**, so the force is **a**(t) everywhere and both the
Hessian and the density are identically zero (Agama returns zero for these
too — its density is the Laplacian of a potential that is linear in **x**).

This is the term you add when integrating in the MW disc frame while the halo
is being accelerated by an infalling satellite:

```python
import numpy as np
from nbody_streams.agama_helper import PotentialGPU

accMW = np.loadtxt('accMW_McM17streams')   # (T, 4): t, ax, ay, az

pot = (PotentialGPU(file='MW_mult.coef_mul_DR')
       + PotentialGPU(coefs_lmc, center=lmc_traj[:, :4])
       + PotentialGPU(type='UniformAcceleration', file=accMW))

F = pot.force(xyz, t=-3.5)     # t is required — the acceleration varies with it
```

**Input forms** (all identical to `agama.Potential(type='UniformAcceleration', file=...)`):

| `file=` | Interpretation |
|---|---|
| `(T, 4)` array or path | `[t, ax, ay, az]` → regularized natural cubic spline |
| `(T, 7)` array or path | `[t, ax, ay, az, dax/dt, day/dt, daz/dt]` → cubic Hermite spline |
| omitted, with `ax=/ay=/az=` | constant acceleration, no time dependence |

An INI section works too:

```ini
[Potential accel]
type = UniformAcceleration
file = accMW_McM17streams      ; resolved relative to the INI file
```

**Time interpolation.**  The 4-column form uses a **natural** cubic spline with
Agama's Hyman (1983) regularization filter — matching
`agama.Spline(t, a, reg=True)` to machine precision.  This is *not*
`scipy.interpolate.CubicSpline`'s default `not-a-knot` spline; the two differ
noticeably near the endpoints and around sharp jumps.  Outside the tabulated
range the acceleration is extrapolated linearly from the endpoint value and
slope, again as Agama does.

**Cost.**  Interpolation happens once per call on the CPU (O(log T), ~1.5 µs
for a 900-row table); the CuPy kernels receive three plain floats.  A
time-dependent instance therefore costs a flat ~2–7 µs more per `force()` call
than a constant one, independent of N — under 1 % at N ≥ 10⁶.

**What is not supported.**  `UniformAccelerationGPU.from_agama()` and
`PotentialGPU(<agama UniformAcceleration object>)` both raise: Agama does not
expose the acceleration table through the Python object, so there is nothing to
read back. Rebuild from the same array or file you gave Agama.

### Accuracy (vs Agama CPU)

| Component | phi rel err | force rel err |
|---|---|---|
| Multipole l=0 (monopole) | ~1e-12 | ~1e-12 |
| Multipole l>0 harmonics | ~1e-7 | ~1e-5 |
| Disk composite (DiskAnsatz + Multipole lmax=32) | ~1e-6 | ~2e-6 |
| Analytic (NFW, Hernquist, etc.) | ~1e-15 | ~1e-15 |
| `UniformAcceleration` (constant and time-dependent) | ~1e-14 | ~1e-15 |

l>0 errors are a numerical floor from log-scaling derivative cancellation —
both GPU and Agama CPU hit the same floor.  BFE fitting error for N-body data
is typically > 1 %, so this precision floor has no practical impact.

### Requirements

Always required (CPU-only stack):

- `numpy`, `h5py`
- `scipy` (quintic spline construction; graceful fallback if absent)
- `agama` (for `Disk` / `Spheroid` / `King` types that use CPU export)

Required **only** for the `*PotentialGPU` classes:

- CUDA GPU (tested on NVIDIA L40)
- `cupy >= 10.0` (matching CUDA version) — `pip install 'nbody_streams[cuda]'`
- `nvcc` accessible on PATH

### CuPy is optional

`import nbody_streams.agama_helper` works without CuPy, and so does every part
of the subpackage that never touches the GPU: `read_coefs`, the coefficient
dataclasses, `write_*_h5`, `load_agama_potential` /
`load_agama_evolving_potential` (CPU path), and the FIRE helpers.

```python
from nbody_streams import agama_helper as ah

ah.CUPY_AVAILABLE                                       # False on CPU-only installs
mc  = ah.read_coefs("MW_mult.h5", group_name="snap_090")  # works
pot = ah.load_agama_potential("MW_mult.h5", group_name="snap_090")  # works (CPU)
```

Touching a GPU path without CuPy raises an `ImportError` naming the missing
package rather than an `AttributeError` from an undefined symbol:

```python
pot = ah.PotentialGPU(type='NFW', mass=1e12, scaleRadius=20)
# ImportError: NFWPotentialGPU requires CuPy.
# CuPy is required for the GPU potential classes of nbody_streams.agama_helper, ...
# Install with:  pip install cupy-cuda12x        (adjust to your CUDA version)
#            or:  pip install 'nbody_streams[cuda]'
```

The shim lives in `agama_helper/_cupy.py`; import it as
`from ._cupy import CUPY_AVAILABLE, cp, require_cupy` in any new GPU module
instead of importing `cupy` directly.

---

## Center parameter

All `load_*` functions accept a `center=` keyword that is forwarded to
`agama.Potential` (CPU path) or `ShiftedPotentialGPU` (GPU path).
Four forms are supported:

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

### Time interpolation of `center=` and `scale=`

Agama reads `center=`, `scale=` and the `UniformAcceleration` table through the
same `readTimeDependentArray`, so the GPU path uses one shared interpolator for
all three:

| Input | Interpolation |
|---|---|
| values only — `center` `(T,4)`, `scale` `(T,2)`/`(T,3)` | **natural** cubic spline + Hyman (1983) regularization filter |
| values + derivatives — `center` `(T,7)` | cubic Hermite spline |
| outside the tabulated range | linear, from the endpoint value and slope |

This matches `agama.Spline(t, y, reg=True)` to machine precision. It is *not*
`scipy.interpolate.CubicSpline`'s default `not-a-knot` spline — on a 40-sample
trajectory the two differ by tens of pc, shrinking to sub-pc by ~300 samples, so
the gap matters most when a trajectory is subsampled (e.g. `traj[0::100, :4]`).

One API difference worth noting: Agama's `scale=` always carries *two* values
`A(t), S(t)` — a static value is the string `"A S"` and a time table is `(T,3)`
`[t, ampl, scale]`. The `(T,2)` `[t, scale]` form accepted by
`ScaledPotentialGPU` is a GPU-side convenience that takes `ampl` from the
separate keyword.

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

### `load_fire_pot(sim_dir, nsnap, sym="n", lmax=4, kind="whole", keep_lm_mult=None, keep_m_cylspl=None, include_negative_m=True, file_ext="DR", out_acc=False, halo=None, verbose=True, return_coefs=False, save_modified=False, save_dir=None)`

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

### `refine_times(times, factor=10) -> ndarray`

Subdivide every interval of *times* by *factor*, keeping the original nodes.

FIRE snapshot cadence is **uneven** — on `m12i` the spacing between snapshots
varies by ~12x across the run — so a plain `np.linspace` over the full time
range would not land back on the original sample times. `refine_times`
subdivides interval-by-interval instead, which keeps every original node at
indices `0, factor, 2*factor, ...`.

```python
ah.refine_times([0.0, 1.0, 3.0], factor=2)
# array([0. , 0.5, 1. , 2. , 3. ])
```

### `spline_resample_coefs(coefs, times_new) -> MultipoleCoefs | CylSplineCoefs`

Resample a coefficient time series onto a new time grid with cubic splines,
using one `agama.Spline` per coefficient series along the time axis. Grids
and labels are untouched — only the trailing time axis changes — and
derivative blocks (`dphi_dr`, or `dphi_dR`/`dphi_dz`) are resampled alongside
`phi` whenever present.

**The `agama_helper` package itself never interpolates coefficients** —
`with_times()`, `stack_coefs()` and every reader leave that entirely to the
caller. `spline_resample_coefs` is an **opt-in helper** for exactly that job,
built on top of the same `with_times()` hand-off.

```python
ser = ah.read_coefs("mult_halo.h5", group_name="all")
fine = ah.spline_resample_coefs(ser, ah.refine_times(ser.times, factor=10))
fine.n_times                                    # 10 * (ser.n_times - 1) + 1
np.allclose(fine.phi[..., ::10], ser.phi)        # True -- original nodes preserved exactly
pot = fine.materialize_potential()
```

> **Why agama and not scipy:** `agama.Spline` is a **natural** cubic spline.
> `scipy.interpolate.CubicSpline` matches it to ~4e-16 relative **only** with
> `bc_type="natural"` — its **default** `"not-a-knot"` differs by up to ~6e-3
> relative near the endpoints on real coefficient series. `spline_resample_coefs`
> therefore requires `agama` and has no scipy fallback: a silent
> `not-a-knot` fallback would be a materially different spline, not merely a
> less-precise one.

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

## Gotchas

- **A real CylSpline export carries three sections.** `Potential.export()`
  writes `#Phi`, `#dPhi/dR` and `#dPhi/dz`, each repeating the full set of
  `\t#m` blocks. `read_cylspl_coefs` splits on the section markers *before*
  scanning `#m` blocks, so `phi` always comes from `#Phi` regardless of how
  many sections follow it. Older `#Phi`-only files (e.g. the 2021-era FIRE
  archives) still parse exactly as before, with `dphi_dR is None` and
  `dphi_dz is None`.

- **The coef text format truncates float64.** `to_coef_string` writes
  `%.13g` (Multipole) / `%.14g` (CylSpline), so a value -> text -> value
  round-trip loses roughly the last digit (~1e-13 relative).
  `materialize_potential(gpu=True)` (and `load_agama_evolving_potential(gpu=True)`)
  on a coef object feed the arrays straight into the GPU builders and skip this
  round-trip entirely, which is why that path is *more* precise, not merely
  faster.

- **A Multipole file needs `#dPhi/dr`; a CylSpline file does not.** Agama
  fails with `RuntimeError: Error loading Multipole potential` when asked to
  load a `#Phi`-only Multipole file, so `MultipoleCoefs.to_coef_string()`
  raises `ValueError` when `dphi_dr is None` rather than writing a file Agama
  can't load. A `#Phi`-only **CylSpline** file *does* load — Agama
  reconstructs the derivatives from the spline, at ~7e-5 relative error on
  `Phi` — so that stays a legal (if slightly less accurate) fallback.

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
                                   keep_lm_mult=[(0,0), (2,0), (4,0)])

# 4. Load full time-evolving potential
pot_ev = ah.load_agama_evolving_potential("MW_dark.h5")

# 5. Evaluate (same Agama interface as always)
import agama
xv = np.array([[8.1, 0, 0, 0, 220, 0]])
acc, phi = pot_ev(xv[:, :3], t=13.5, der=1)
```
