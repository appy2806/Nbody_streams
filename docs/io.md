# `nbody_streams.nbody_io` — HDF5 Snapshot I/O

I/O utilities for N-body snapshots and restart data.

Required dependency: **h5py**.

---

## HDF5 snapshot format

All snapshots are written to HDF5 files (`.h5`) under an output directory.
The internal layout is:

```
/snapshots/
    snap.000  : ndarray (N_total, 6) float64  [x, y, z, vx, vy, vz]
    snap.001  : ...
    snap.NNN  : ...
    (attribute) snap_time.NNN : float  [physical time of this snapshot]

/properties/
    time_step : float
    n_species : int             [multi-species schema only]
    species_names : bytes[]     [multi-species schema only]
    <species_name>/
        N     : int
        m     : float           [scalar when mass is uniform]
        m_array : ndarray (N,)  [per-particle; written when mass varies]
        eps   : float           [scalar when softening is uniform]
        eps_array : ndarray (N,)[per-particle; written when softening varies]
```

Two schemas are supported:

- **Multi-species** (current): any number of species, written when
  `species=` is passed to `_save_snapshot`.  Identified by the `n_species`
  attribute on `/properties`.
- **Legacy two-species**: `/properties/dark` and `/properties/star`
  sub-groups.  `ParticleReader` reads both schemas transparently.

Units: positions in kpc, velocities in km/s, masses in M_sun.

---

## `ParticleReader`

```python
ParticleReader(sim_pattern, times_file_path=None, verbose=False)
```

Read HDF5-based snapshot sets produced by `run_nbody_gpu`, `run_nbody_cpu`,
or `run_nbody_gpu_tree`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `sim_pattern` | str | Path or glob pattern for HDF5 files (e.g., `'output/snapshot.h5'` or `'output/snapshot.*.h5'`). |
| `times_file_path` | str or Path, optional | Path to the snapshot times file. If not provided, the reader looks for `snapshot.times` in the same directory as the first matched file. |
| `verbose` | bool | Print status messages. Default False. |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `species_list` | list of `Species` | Ordered list of species read from file properties. |
| `num_dark` | int | Backward-compatible: number of dark particles. |
| `num_star` | int | Backward-compatible: number of star particles. |
| `mass_dark`, `mass_star` | float | Representative mass for dark/star species. |
| `Snapshots` | ndarray (int) | Sorted array of available snapshot indices. |
| `Times` | SimpleNamespace or None | `.snap` (int array) and `.time` (float array) when a times file is found. |

### `read_snapshot`

```python
reader.read_snapshot(identifier)
```

Read a single snapshot by index or physical time.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `identifier` | int or float | int: snapshot index. float: physical time in Gyr (requires `snapshot.times`). |

**Returns**

`types.SimpleNamespace` with attributes:

- `part.dark` / `part.star` — dicts with keys `'posvel'` (ndarray, shape `(N, 6)`) and `'mass'` (ndarray, shape `(N,)`).
- `part.species` — dict mapping species name to the same dict format.
- `part.snap` — int, snapshot index.
- `part.time` — float or None, physical time in Gyr.

**Example**

```python
from nbody_streams.nbody_io import ParticleReader

reader = ParticleReader("output/snapshot.h5", verbose=True)
snap = reader.read_snapshot(50)       # by index
snap = reader.read_snapshot(7.5)      # by time (Gyr)

pos_dark = snap.dark["posvel"][:, :3]
vel_dark = snap.dark["posvel"][:, 3:]
```

### `extract_orbits`

```python
reader.extract_orbits(
    particle_type="star",
    min_parallel_workers=4,
    snap_indices=None,
)
```

Extract orbits for selected particle types across all (or selected)
snapshots into RAM, using parallel worker processes that write directly into
shared memory.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `particle_type` | str or bool | `'all'` or `True` — all species. `False` — return None. Species name (e.g. `'dark'`, `'star'`) — single species. |
| `min_parallel_workers` | int | Maximum number of parallel worker processes. Actual workers = min(this, cpu_count, num_snapshots). |
| `snap_indices` | array-like or None | Snapshot indices to extract. None extracts all. |

**Returns**

`types.SimpleNamespace`:

- `orbits.species` — dict mapping species name to ndarray, shape `(num_snapshots, N_k, 6)`.
- `orbits.dark` / `orbits.star` — backward-compatible aliases.
- `orbits.Times` — 1-D ndarray of physical times, or None.
- `orbits.Snaps` — 1-D int ndarray of snapshot indices.

**Memory warning**: a `ResourceWarning` is emitted if the estimated
allocation exceeds 4 GB.  Use `snap_indices` to load a subset, or iterate
over `read_snapshot` instead.

**Example**

```python
orbits = reader.extract_orbits("dark", snap_indices=range(0, 100, 5))
# orbits.dark shape: (20, N_dark, 6)
print(orbits.Times)     # physical times array
```

---

## Internal I/O functions

These functions are used internally by the integrators and are not part of
the stable public API.  They are documented here for completeness.

### `_save_snapshot`

```python
_save_snapshot(
    phase_space,
    snap_index,
    time,
    output_dir,
    *,
    species=None,
    num_dark=None,
    num_star=None,
    mass_dark=None,
    mass_star=None,
    time_step=None,
    eps_dark=None,
    eps_star=None,
    single_file=None,
    num_files_to_write=None,
    total_expected_snapshots=None,
)
```

Write a snapshot compatible with `ParticleReader`.

- **Multi-species path** (`species` kwarg provided): writes the current
  schema with `n_species` / `species_names` attributes and per-species
  sub-groups.  Mass and softening are stored as scalars when uniform, or as
  compressed `m_array` / `eps_array` datasets when per-particle.
- **Legacy path** (`species=None`): writes `/properties/dark` and
  `/properties/star` sub-groups for backward compatibility.

File distribution: `num_files_to_write=N` distributes snapshots across
`snapshot.000.h5 … snapshot.(N-1).h5`.  Existing datasets are never
overwritten.

### `_save_restart`

```python
_save_restart(
    phase_space,
    time,
    step,
    output_dir,
    snapshot_counter,
    *,
    mass_arr=None,
    softening_arr=None,
    species_names=None,
    species_N=None,
)
```

Save a restart checkpoint to `<output_dir>/restart.npz` for crash recovery.
Old restart files lacking multi-species keys are loaded gracefully by
`_load_restart`.

### `_load_restart`

```python
_load_restart(output_dir) -> tuple | None
```

Load a restart checkpoint.

**Returns**

`(phase_space, time, step, snapshot_counter, mass_arr, softening_arr, species_names, species_N)`
or `None` if no restart file exists.  The last four elements are `None` for
files written by older package versions.

---

## `snapshot.times` file

A plain-text two-column file written alongside HDF5 output:

```
# snap_index time
0 0.0000000000e+00
1 1.0000000000e-02
...
```

`ParticleReader` reads this file automatically and uses it for time-based
snapshot lookup (`read_snapshot(7.5)`).
