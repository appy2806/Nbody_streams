# `nbody_streams` — main package

The top-level package exports the primary simulation entry point, species descriptor, IC generator, and low-level force/integration functions.

```python
import nbody_streams
from nbody_streams import run_simulation, Species, make_plummer_sphere
```

---

## Contents

- [Species dataclass](#species-dataclass)
- [run_simulation](#run_simulation)
- [make_plummer_sphere](#make_plummer_sphere)
- [Backend dispatch table](#backend-dispatch-table)
- [PerformanceWarning](#performancewarning)

---

## Species dataclass

```python
from nbody_streams import Species
```

### Definition

```python
@dataclass
class Species:
    name      : str
    N         : int
    mass      : float | ndarray (N,)
    softening : float | ndarray (N,)  # default 0.0
```

`Species` describes one particle component.  Multiple species are concatenated into a single phase-space array and passed to `run_simulation` together.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Unique species identifier (e.g. `'dark'`, `'star'`, `'bh'`) |
| `N` | `int` | Number of particles in this species |
| `mass` | `float` or `ndarray (N,)` | Particle mass(es) in M_sun |
| `softening` | `float` or `ndarray (N,)` | Gravitational softening length(s) in kpc. Default `0.0` |

### Named constructors

```python
dm    = Species.dark(N=10_000, mass=1e6, softening=0.1)
stars = Species.star(N=5_000,  mass=1e5, softening=0.05)
```

`Species.dark(...)` creates `Species(name='dark', ...)`.
`Species.star(...)` creates `Species(name='star', ...)`.

### Per-particle mass and softening

```python
import numpy as np
m_arr = np.linspace(1e5, 2e5, 1000)
gas = Species(name='gas', N=1000, mass=m_arr, softening=0.05)
```

### Validation

- `name` must be a non-empty string.
- `N` must be > 0.
- If `mass` is an array, it must have shape `(N,)`.
- If `softening` is an array, it must have shape `(N,)`.
- Duplicate species names in the same list raise `ValueError`.

---

## run_simulation

```python
from nbody_streams import run_simulation

result = run_simulation(
    phase_space,           # ndarray (N_total, 6)
    species,               # list[Species]
    time_start,            # float
    time_end,              # float
    dt,                    # float
    G=4.300917e-6,         # float
    architecture='gpu',    # 'cpu' | 'gpu'
    method='direct',       # 'direct' | 'tree'
    external_potential=None,
    output_dir='./output',
    save_snapshots=True,
    snapshots=100,
    num_files_to_write=1,
    restart_interval=1000,
    continue_run=False,
    overwrite=False,
    verbose=True,
    debug_energy=False,
    **kwargs,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `phase_space` | `ndarray (N_total, 6)` | required | Initial phase-space `[x, y, z, vx, vy, vz]` for all particles concatenated in the same order as `species` |
| `species` | `list[Species]` | required | Ordered list of species descriptors; `sum(s.N)` must equal `N_total` |
| `time_start` | `float` | required | Start time [kpc/(km/s)] |
| `time_end` | `float` | required | End time [kpc/(km/s)] |
| `dt` | `float` | required | Fixed timestep [kpc/(km/s)] |
| `G` | `float` | `4.300917e-6` | Gravitational constant |
| `architecture` | `'cpu'` or `'gpu'` | `'gpu'` | Compute backend |
| `method` | `'direct'` or `'tree'` | `'direct'` | Gravity solver |
| `external_potential` | `agama.Potential` or `None` | `None` | Optional external time-varying potential (Agama). The host is treated as a smooth field; granularity-driven scattering is absent unless `dynamical_friction=True` |
| `dynamical_friction` | `bool` | `False` | Apply Chandrasekhar DF to the satellite centre of mass. Requires `external_potential` (raises `ValueError` otherwise). When enabled, builds a DF closure via `_chandrasekhar.make_df_force_extra` and passes it as `force_extra`. Tunable via the `df_*` kwargs below |
| `output_dir` | `str` | `'./output'` | Directory for snapshot and restart files |
| `save_snapshots` | `bool` | `True` | Write HDF5 snapshots to disk |
| `snapshots` | `int` | `100` | Number of evenly-spaced output snapshots |
| `num_files_to_write` | `int` | `1` | Split snapshots across this many HDF5 files |
| `restart_interval` | `int` | `1000` | Save a restart checkpoint every N steps |
| `continue_run` | `bool` | `False` | Resume from existing restart file |
| `overwrite` | `bool` | `False` | Delete existing snapshot files before starting |
| `verbose` | `bool` | `True` | Print progress information |
| `debug_energy` | `bool` | `False` | Print virial ratio Q=KE/|PE| and energy drift dE/E at each output interval |
| `**kwargs` | | | Backend-specific advanced options (see below) |

### Backend-specific kwargs

| kwarg | Backend | Default | Description |
|---|---|---|---|
| `theta` | tree (gpu+cpu) | `0.6` | Barnes-Hut / falcON opening angle |
| `nthreads` | cpu direct | `None` | CPU thread count for Numba; `None` = auto |
| `external_update_interval` | gpu (direct+tree) | `1` | Recompute external forces every N steps |
| `precision` | gpu direct | `'float32_kahan'` | Force computation precision: `'float32'`, `'float64'`, or `'float32_kahan'` |
| `nleaf` | gpu tree | `64` | Min leaf node size. Must be in {16, 24, 32, 48, 64} |
| `ncrit` | gpu tree | `64` | Group criticality threshold |
| `level_split` | gpu tree | `5` | Tree level for spatial grouping |
| `step_timeout_s` | gpu tree | `60.0` | Per-step watchdog timeout (seconds) before raising `RuntimeError` |

### Dynamical friction kwargs

These are only active when `dynamical_friction=True`.

| kwarg | Type | Default | Description |
|---|---|---|---|
| `df_M_sat` | `float` | `None` | Total satellite mass [M_sun]. If `None`, summed from `species` masses |
| `df_coulomb_mode` | `str` | `'variable'` | Coulomb log mode: `'variable'` uses local density to compute ln(Λ); `'fixed'` uses `df_fixed_ln_lambda` |
| `df_fixed_ln_lambda` | `float` | `3.0` | Fixed Coulomb logarithm. Only used when `df_coulomb_mode='fixed'` |
| `df_core_gamma` | `float` | `1.0` | Core-stalling suppression exponent γ (BT2008 eq. 8.13 variant). Set to `0.0` to disable stalling |
| `df_r_core` | `float` | `None` | Core radius [kpc] for stalling suppression. If `None`, estimated from the potential |
| `df_update_interval` | `int` | `1` | Recompute the DF force every N steps. Useful for slowly-evolving orbits |
| `df_shrink_n_iter` | `int` | `10` | Number of shrinking-sphere CoM iterations per step |
| `df_shrink_frac` | `float` | `0.7` | Shrinking factor for CoM estimator sphere radius per iteration |
| `df_sigma_grid_r` | `ndarray` or `None` | `None` | Custom radial grid for σ(r) evaluation. If `None`, a default log-spaced grid is used |

### Returns

```python
dict[str, ndarray]
```

Final phase-space coordinates, keyed by species name.  Each value has shape `(N_k, 6)`.

```python
result['dark']   # (N_dark, 6)
result['star']   # (N_star, 6)
```

### Raises

| Exception | Condition |
|---|---|
| `ValueError` | `species` inconsistent with `phase_space` shape |
| `FileExistsError` | Snapshot files exist in `output_dir`, `overwrite=False`, `continue_run=False` |
| `ImportError` | CuPy unavailable when `architecture='gpu'`; or `libtreeGPU.so` not built when `method='tree'` |
| `TypeError` | Unrecognised kwargs for the selected backend |

### Softening kernels (hardcoded per backend)

| Backend | Kernel |
|---|---|
| GPU direct | `'spline'` (Monaghan 1992 cubic spline) |
| CPU direct | `'spline'` |
| CPU tree (pyfalcon/falcON) | `'dehnen_k1'` (integer 1) |
| GPU tree (Barnes-Hut) | Plummer softening (hardcoded in C++) |

### Example: single species

```python
from nbody_streams import run_simulation, Species, make_plummer_sphere

xv, _ = make_plummer_sphere(1000, M_total=1e9, a=1.0)
dm = Species.dark(N=1000, mass=1e6, softening=0.1)

result = run_simulation(
    xv, [dm],
    time_start=0.0, time_end=1.0, dt=1e-3,
    architecture='cpu', method='direct',
    save_snapshots=False, verbose=False,
)
print(result['dark'].shape)   # (1000, 6)
```

### Example: multi-species dark matter + stars

```python
import numpy as np
from nbody_streams import run_simulation, Species, make_plummer_sphere

xv_dm,   _ = make_plummer_sphere(800, M_total=1e9, a=2.0)
xv_star, _ = make_plummer_sphere(200, M_total=1e7, a=0.5)
xv_all = np.vstack([xv_dm, xv_star])

dm   = Species.dark(N=800, mass=1.25e6, softening=0.2)
star = Species.star(N=200, mass=5e4,    softening=0.05)

result = run_simulation(
    xv_all, [dm, star],
    time_start=0.0, time_end=0.5, dt=1e-3,
    architecture='gpu', method='direct',
    output_dir='./output/mw', snapshots=100,
)
print(result.keys())   # dict_keys(['dark', 'star'])
```

### Example: GPU tree backend

```python
result = run_simulation(
    xv_all, [dm, star],
    time_start=0.0, time_end=5.0, dt=1e-4,
    architecture='gpu', method='tree',
    theta=0.6, step_timeout_s=60.0,
    output_dir='./output/tree', snapshots=500,
)
```

### Example: with external Agama potential

```python
import agama
agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(type='NFW', mass=1e12, scaleRadius=20.0)

result = run_simulation(
    xv, [dm],
    time_start=0.0, time_end=1.0, dt=1e-3,
    architecture='gpu', method='direct',
    external_potential=pot,
    external_update_interval=5,
)
```

### Example: with Chandrasekhar dynamical friction

```python
import agama
agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(type='NFW', mass=1e12, scaleRadius=15.0,
                      outerCutoffRadius=300.0)

result = run_simulation(
    xv, [dm],
    time_start=0.0, time_end=5.0, dt=1e-3,
    architecture='gpu', method='direct',
    external_potential=pot,
    dynamical_friction=True,
    df_M_sat=5e9,
    df_coulomb_mode='variable',
)
```

See [docs/dynamical_friction.md](dynamical_friction.md) for a full reference including the low-level `force_extra` interface.

---

## make_plummer_sphere

```python
from nbody_streams import make_plummer_sphere

xv, masses = make_plummer_sphere(
    N,                     # int
    M_total=10_000,        # float [M_sun]
    a=0.01,                # float [kpc]
    seed=42069,            # int
    G=4.300917e-6,         # float
)
```

Generates a Plummer-sphere initial condition in virial equilibrium.

Positions are sampled from the Plummer density profile:

```
rho(r) = (3 M / 4 pi a^3) * (1 + r^2/a^2)^(-5/2)
```

Velocities are sampled via rejection sampling (Aarseth, Henon & Wielen 1974).  The centre of mass and bulk velocity are corrected to zero before returning.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `N` | `int` | required | Number of particles |
| `M_total` | `float` | `10_000` | Total mass [M_sun] |
| `a` | `float` | `0.01` | Plummer scale radius [kpc] |
| `seed` | `int` | `42069` | Random seed |
| `G` | `float` | `4.300917e-6` | Gravitational constant |

### Returns

| Name | Shape | Description |
|---|---|---|
| `phase_space` | `(N, 6)` float64 | Positions and velocities `[x, y, z, vx, vy, vz]` |
| `masses` | `(N,)` float64 | Equal-mass particles summing to `M_total` |

### Example

```python
xv, masses = make_plummer_sphere(10_000, M_total=5e5, a=0.005, seed=42)
# xv.shape   -> (10000, 6)
# masses.sum() -> 5e5
```

---

## Backend dispatch table

`run_simulation` dispatches to the following backend functions:

| `architecture` | `method` | Backend function | Notes |
|---|---|---|---|
| `'gpu'` | `'direct'` | `run_nbody_gpu` | CuPy + CUDA kernels; O(N^2) |
| `'gpu'` | `'tree'` | `run_nbody_gpu_tree` | C++/CUDA Barnes-Hut; O(N log N); requires `libtreeGPU.so` |
| `'cpu'` | `'direct'` | `run_nbody_cpu` (direct) | Numba `@njit(parallel=True)`; O(N^2) |
| `'cpu'` | `'tree'` | `run_nbody_cpu` (tree) | pyfalcon falcON FMM; O(N); requires pyfalcon |

---

## PerformanceWarning

`PerformanceWarning` is a `UserWarning` subclass emitted automatically when particle counts exceed recommended thresholds:

| Condition | Warning |
|---|---|
| CPU direct, N > 20 000 | Slow O(N^2); suggest `method='tree'` or `architecture='gpu'` |
| GPU direct, N > 500 000 | May be slow; suggest `method='tree'` |
| Any method, N > 2 000 000 and `method != 'tree'` | Direct summation at this scale is very slow |
| Total satellite mass > 1e10 M_sun with `external_potential` set and `dynamical_friction=False` | Dynamical friction may be significant at this mass scale — consider setting `dynamical_friction=True` |

To suppress:

```python
import warnings
from nbody_streams import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)
```

---

## Top-level exports

The following names are available directly under `nbody_streams`:

```python
from nbody_streams import (
    # Simulation entry point
    run_simulation,
    # Species
    Species, PerformanceWarning,
    # IC generator
    make_plummer_sphere,
    # Low-level integrators
    run_nbody_gpu, run_nbody_cpu,
    # Force / potential kernels
    compute_nbody_forces_gpu, compute_nbody_forces_cpu,
    compute_nbody_potential_gpu, compute_nbody_potential_cpu,
    get_gpu_info,
    # I/O
    ParticleReader,
    # Constants
    G_DEFAULT, NBODY_UNITS,
    # GPU tree code (when libtreeGPU.so is built)
    tree_gravity_gpu, TreeGPU, cuda_alive, run_nbody_gpu_tree,
    # Subpackages
    utils, coords, fast_sims, viz, agama_helper,
)
```
