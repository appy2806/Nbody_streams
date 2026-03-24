# `nbody_streams.run`

Low-level KDK leapfrog integrators and initial-condition generator.  For most use cases, prefer `run_simulation` from the main package.  Use these functions directly when you need fine control over softening kernels, floating-point precision, or per-particle softening.

```python
from nbody_streams import run_nbody_gpu, run_nbody_cpu, make_plummer_sphere
from nbody_streams.run import G_DEFAULT, NBODY_UNITS
```

> **Optional dependencies:** `run_nbody_gpu` requires CuPy.  `run_nbody_cpu` with `method='tree'` requires pyfalcon.  External potentials require Agama.

---

## Contents

- [run_nbody_gpu](#run_nbody_gpu)
- [run_nbody_cpu](#run_nbody_cpu)
- [make_plummer_sphere](#make_plummer_sphere)
- [Constants](#constants)

---

## run_nbody_gpu

```python
final_xv = run_nbody_gpu(
    phase_space,                 # ndarray (N, 6)
    masses,                      # ndarray (N,)
    time_start,                  # float
    time_end,                    # float
    dt,                          # float
    softening,                   # float or ndarray (N,)
    G=4.300917e-6,               # float
    precision='float32_kahan',   # 'float32' | 'float64' | 'float32_kahan'
    kernel='spline',             # str
    external_potential=None,     # agama.Potential or None
    external_update_interval=1,  # int
    output_dir='./output',       # str
    save_snapshots=True,         # bool
    snapshots=10,                # int
    num_files_to_write=1,        # int
    restart_interval=1000,       # int
    continue_run=False,          # bool
    overwrite=False,             # bool
    verbose=True,                # bool
    debug_energy=False,          # bool
    species=None,                # list[Species] or None
)
```

GPU-accelerated N-body simulation using KDK (kick-drift-kick) leapfrog integration.  Self-gravity is computed via CUDA direct-sum kernels; positions and velocities are maintained in float64.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `phase_space` | `ndarray (N, 6)` | required | Initial `[x, y, z, vx, vy, vz]` in float64 |
| `masses` | `ndarray (N,)` | required | Particle masses [M_sun] |
| `time_start` | `float` | required | Start time |
| `time_end` | `float` | required | End time |
| `dt` | `float` | required | Fixed timestep |
| `softening` | `float` or `ndarray (N,)` | required | Gravitational softening length(s) [kpc] |
| `G` | `float` | `4.300917e-6` | Gravitational constant |
| `precision` | `str` | `'float32_kahan'` | Force computation precision (`'float32'`, `'float64'`, `'float32_kahan'`) |
| `kernel` | `str` | `'spline'` | Softening kernel: `'newtonian'`, `'plummer'`, `'dehnen_k1'`, `'dehnen_k2'`, `'spline'` |
| `external_potential` | `agama.Potential` or `None` | `None` | Time-varying external potential. **Note:** host-satellite dynamical friction is not included — see `force_extra` |
| `external_update_interval` | `int` | `1` | Recompute external forces every N steps. Use `5–10` for slowly-varying potentials |
| `force_extra` | `callable` or `None` | `None` | Extra acceleration added after gravity + external at every step. Signature: `force_extra(pos, vel, masses, t) -> (N,3)`. On GPU paths `pos`/`vel` are **CuPy** arrays; user is responsible for hardware path |
| `output_dir` | `str` | `'./output'` | Output directory for snapshots and restart files |
| `save_snapshots` | `bool` | `True` | Write HDF5 snapshots to disk |
| `snapshots` | `int` | `10` | Number of evenly-spaced output snapshots |
| `num_files_to_write` | `int` | `1` | Split snapshots across this many HDF5 files |
| `restart_interval` | `int` | `1000` | Save restart checkpoint every N steps |
| `continue_run` | `bool` | `False` | Resume from existing restart file |
| `overwrite` | `bool` | `False` | Delete existing snapshots before starting |
| `verbose` | `bool` | `True` | Print progress information |
| `debug_energy` | `bool` | `False` | Print virial ratio Q=KE/|PE| and energy drift dE/E at each output interval. Requires one extra O(N^2) potential pass per output interval |
| `species` | `list[Species]` or `None` | `None` | Multi-species descriptors for per-species HDF5 output |

### Returns

`ndarray (N, 6)` — final phase-space coordinates in float64.

### Raises

| Exception | Condition |
|---|---|
| `ImportError` | CuPy not installed |
| `ValueError` | `phase_space` not shape `(N, 6)` |

### Integration scheme

KDK leapfrog:

1. Half-kick: `vel += 0.5 * dt * acc`
2. Drift: `pos += dt * vel`
3. Recompute `acc`
4. Half-kick: `vel += 0.5 * dt * acc`

Positions and velocities are kept in float64 on the GPU throughout.  Force calls use `precision`-selected dtype.

### GPU memory management

- Positions/velocities stay on the GPU for the full integration.
- Data is transferred to CPU only for snapshot and restart writes.
- External forces are computed on CPU (Agama) and transferred to GPU.

### Snapshot and restart behaviour

Snapshots are written as HDF5 files in `output_dir`.  Restart files (`restart.npz`) are written every `restart_interval` steps.  Setting `continue_run=True` loads the most recent restart file and resumes from that step.

### Example: basic self-gravity

```python
import numpy as np
from nbody_streams import run_nbody_gpu, make_plummer_sphere

xv, masses = make_plummer_sphere(10_000, M_total=1e9, a=1.0)

final = run_nbody_gpu(
    xv, masses,
    time_start=0.0, time_end=0.5, dt=1e-4,
    softening=0.1,
    precision='float32_kahan',
    kernel='spline',
    output_dir='./output/gpu_direct',
    snapshots=50,
)
```

### Example: with external potential

```python
import agama
agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(type='NFW', mass=1e12, scaleRadius=20.0)

final = run_nbody_gpu(
    xv, masses,
    time_start=0.0, time_end=1.0, dt=1e-3,
    softening=0.1,
    external_potential=pot,
    external_update_interval=5,
    output_dir='./output/ext_pot',
    snapshots=100,
)
```

### Example: resume from crash

```python
final = run_nbody_gpu(
    xv, masses,
    time_start=0.0, time_end=1.0, dt=1e-3,
    softening=0.1,
    output_dir='./output/resumable',
    continue_run=True,
)
```

---

## run_nbody_cpu

```python
final_xv = run_nbody_cpu(
    phase_space,              # ndarray (N, 6)
    masses,                   # ndarray (N,)
    time_start,               # float
    time_end,                 # float
    dt,                       # float
    softening,                # float or ndarray (N,)
    G=4.302e-6,               # float
    method='direct',          # 'direct' | 'tree'
    theta=0.6,                # float
    kernel=1,                 # int or str
    nthreads=None,            # int or None
    external_potential=None,  # agama.Potential or None
    output_dir='./',          # str
    save_snapshots=True,      # bool
    snapshots=1,              # int
    num_files_to_write=1,     # int
    restart_interval=1000,    # int
    continue_run=False,       # bool
    overwrite=False,          # bool
    verbose=True,             # bool
    debug_energy=False,       # bool
    species=None,             # list[Species] or None
)
```

CPU-based N-body simulation using KDK leapfrog.  Gravity solver is either Numba direct O(N^2) or pyfalcon falcON tree O(N).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `phase_space` | `ndarray (N, 6)` | required | Initial `[x, y, z, vx, vy, vz]` |
| `masses` | `ndarray (N,)` | required | Particle masses [M_sun] |
| `time_start` | `float` | required | Start time |
| `time_end` | `float` | required | End time |
| `dt` | `float` | required | Fixed timestep |
| `softening` | `float` or `ndarray (N,)` | required | Softening length(s) [kpc] |
| `G` | `float` | `4.302e-6` | Gravitational constant |
| `method` | `'direct'` or `'tree'` | `'direct'` | Gravity solver: `'direct'` = Numba O(N^2); `'tree'` = pyfalcon falcON O(N) |
| `theta` | `float` | `0.6` | Opening angle for tree method |
| `kernel` | `int` or `str` | `1` | Softening kernel. For `method='tree'`: integer (`0`=Plummer, `1`=dehnen_k1, `2`=dehnen_k2). For `method='direct'`: string (`'newtonian'`, `'plummer'`, `'spline'`, `'dehnen_k1'`, `'dehnen_k2'`) |
| `nthreads` | `int` or `None` | `None` | Numba thread count for direct method. `None` = auto |
| `external_potential` | `agama.Potential` or `None` | `None` | Optional external time-varying potential. Dynamical friction is not included |
| `force_extra` | `callable` or `None` | `None` | Extra acceleration after gravity + external. Signature: `force_extra(pos, vel, masses, t) -> (N,3)`. `pos`/`vel` are NumPy arrays on CPU path |
| `output_dir` | `str` | `'./'` | Output directory |
| `save_snapshots` | `bool` | `True` | Write HDF5 snapshots |
| `snapshots` | `int` | `1` | Number of evenly-spaced output snapshots |
| `num_files_to_write` | `int` | `1` | Split snapshots across N HDF5 files |
| `restart_interval` | `int` | `1000` | Checkpoint interval in steps |
| `continue_run` | `bool` | `False` | Resume from restart file |
| `overwrite` | `bool` | `False` | Delete existing snapshots |
| `verbose` | `bool` | `True` | Print progress |
| `debug_energy` | `bool` | `False` | Print virial ratio and energy drift. For `method='tree'` the potential is returned for free by falcON — zero overhead |
| `species` | `list[Species]` or `None` | `None` | Multi-species descriptors |

### Returns

`ndarray (N, 6)` — final phase-space coordinates.

### Raises

| Exception | Condition |
|---|---|
| `ImportError` | pyfalcon not installed and `method='tree'` |
| `ValueError` | Invalid method or kernel combination |

### Method selection guidance

| System size | Recommended method |
|---|---|
| N < 20 000 | `method='direct'` — Numba JIT at full parallelism |
| N > 20 000 | `method='tree'` — falcON O(N) FMM; requires pyfalcon |

### Example: tree method

```python
from nbody_streams import run_nbody_cpu, make_plummer_sphere

xv, masses = make_plummer_sphere(50_000, M_total=1e10, a=5.0)

final = run_nbody_cpu(
    xv, masses,
    time_start=0.0, time_end=2.0, dt=2e-3,
    softening=0.2,
    method='tree',
    theta=0.5,
    kernel=2,           # dehnen_k2
    output_dir='./output/cpu_tree',
    snapshots=100,
    debug_energy=True,  # free for tree method
)
```

### Example: direct method with specific kernel

```python
final = run_nbody_cpu(
    xv, masses,
    time_start=0.0, time_end=0.1, dt=1e-4,
    softening=0.05,
    method='direct',
    kernel='dehnen_k2',
    nthreads=32,
    output_dir='./output/cpu_direct',
    snapshots=20,
)
```

---

## make_plummer_sphere

See [main.md — make_plummer_sphere](main.md#make_plummer_sphere) for the full reference.  This function is imported at the top level and also accessible from `nbody_streams.run`.

```python
from nbody_streams.run import make_plummer_sphere

xv, masses = make_plummer_sphere(N=10_000, M_total=1e9, a=1.0, seed=42)
```

---

## Constants

```python
from nbody_streams.run import G_DEFAULT, NBODY_UNITS

print(G_DEFAULT)      # 4.300917270069976e-06
print(NBODY_UNITS)
# {
#   'kpc': 1.0,
#   'Msun': 1.0,
#   'kpc / (km/s)': 1.0,
#   'km/s': 1,
#   'G': 4.300917270069976e-06,
# }
```

`G_DEFAULT` is the gravitational constant in kpc / (km/s)^2 / M_sun units.  `NBODY_UNITS` documents the full unit system used throughout the package.
