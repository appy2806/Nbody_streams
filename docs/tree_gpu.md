# `nbody_streams.tree_gpu` — GPU Barnes-Hut Tree Code

GPU-accelerated O(N log N) gravity via the Barnes-Hut tree algorithm
(monopole + quadrupole moments).  The tree code is implemented in CUDA C++
and exposed to Python through a ctypes interface backed by `libtreeGPU.so`.

Optional dependency: **CuPy** (required at import time).

---

## Building the shared library

The shared library must be compiled before the subpackage can be imported:

```bash
cd nbody_streams/tree_gpu
make -j$(nproc)          # builds libtreeGPU.so alongside the source files
```

The Makefile auto-detects the GPU architecture via `nvidia-smi` / CuPy.  If
`libtreeGPU.so` is absent, importing `nbody_streams.tree_gpu` raises an
`ImportError` with the build command printed in the message.

---

## Availability flag

The top-level `nbody_streams` module exposes:

```python
nbody_streams._TREE_GPU_AVAILABLE  # True once libtreeGPU.so is loaded
```

---

## Public API

### `tree_gravity_gpu`

```python
tree_gravity_gpu(
    pos,
    mass,
    eps,
    G=4.300917e-6,
    *,
    theta=0.6,
    nleaf=64,
    ncrit=64,
    level_split=5,
    verbose=False,
    tree=None,
)
```

Compute gravitational accelerations and potential using the GPU Barnes-Hut
tree (one-shot: allocates and frees the tree per call).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | `cp.ndarray`, shape `(N, 3)` | Particle positions, float32 on GPU. |
| `mass` | float or `cp.ndarray (N,)` | Particle masses. |
| `eps` | float or `cp.ndarray (N,)` | Plummer softening length(s). Scalar gives uniform softening; array enables per-particle softening for multi-species runs. |
| `G` | float | Gravitational constant. Default: kpc / (km/s)^2 / Msun. |
| `theta` | float | Barnes-Hut opening angle. 0.75 = fast; 0.5 = accurate; 0.3 = very accurate. Default 0.6. |
| `nleaf` | int | Max particles per leaf cell. Must be in {16, 24, 32, 48, 64}. Default 64. |
| `ncrit` | int | Group size for the tree walk. Default 64. |
| `level_split` | int | Tree level for spatial grouping. Default 5. |
| `verbose` | bool | Print timing / interaction counts from the C++ layer. |
| `tree` | `TreeGPU` or None | Pre-allocated tree handle. Skips ~27 ms malloc/free per call. |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `acc` | `cp.ndarray`, shape `(N, 3)` | Gravitational accelerations in original particle order. |
| `phi` | `cp.ndarray`, shape `(N,)` | Gravitational potential per particle. |

**Softening convention** (matches nbody_streams direct-sum):

- Direct particle-particle: `eps2_ij = max(eps_i^2, eps_j^2)`
- Cell approximation: `eps2_ij = max(eps_i^2, eps_cell_max^2)`

Both use the Plummer kernel: `r^2 -> r^2 + eps2`.

**Example**

```python
import cupy as cp
from nbody_streams.tree_gpu import tree_gravity_gpu

pos  = cp.random.randn(10_000, 3, dtype=cp.float32)
mass = cp.ones(10_000, dtype=cp.float32) * 1e5
eps  = 0.05   # uniform softening, kpc

acc, phi = tree_gravity_gpu(pos, mass, eps)
```

---

### `TreeGPU`

```python
TreeGPU(n, eps=0.0, theta=0.6, verbose=False)
```

Pre-allocated GPU tree handle for time-stepping loops.  Avoids ~27 ms of
GPU malloc/free overhead per step.  Must be closed explicitly or used as a
context manager.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `n` | int | Number of particles. Must not exceed available VRAM (80% free, capped at 28-bit Particle4 index). |
| `eps` | float | Default fallback softening length. |
| `theta` | float | Barnes-Hut opening angle. |
| `verbose` | bool | Print timing from C++ layer. |

**Methods**

- `close()` — free GPU memory.
- Context manager: `__enter__` / `__exit__` call `close()`.

**Example**

```python
from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU

with TreeGPU(N, eps=0.05, theta=0.6) as tree:
    for step in range(n_steps):
        acc, phi = tree_gravity_gpu(pos, mass, eps=0.05, tree=tree)
        # ... leapfrog update ...
```

---

### `cuda_alive`

```python
cuda_alive() -> bool
```

Lightweight CUDA context health check.  Calls `cudaGetLastError()` via
ctypes — zero GPU overhead, no synchronisation (reads a thread-local error
flag only).

Returns `True` if the CUDA context has no pending errors, `False` if there
is a pending error.  Returns `True` (assume OK) if `libcudart` cannot be
loaded.

Recommended call frequency: every few hundred steps alongside energy checks,
not every single step.

```python
from nbody_streams.tree_gpu import cuda_alive
if not cuda_alive():
    raise RuntimeError("CUDA context has a pending error.")
```

---

### `run_nbody_gpu_tree`

```python
run_nbody_gpu_tree(
    phase_space,
    masses,
    time_start,
    time_end,
    dt,
    softening,
    G=4.300917e-6,
    theta=0.6,
    nleaf=64,
    ncrit=64,
    level_split=5,
    step_timeout_s=60.0,
    external_potential=None,
    external_update_interval=1,
    output_dir="./output",
    save_snapshots=True,
    snapshots=100,
    num_files_to_write=1,
    restart_interval=1000,
    continue_run=False,
    overwrite=False,
    verbose=True,
    debug_energy=False,
    species=None,
)
```

KDK leapfrog integrator using the GPU Barnes-Hut tree code.  Matches the
call signature of `run_nbody_gpu` (direct-sum) but routes all force
evaluations through `tree_gravity_gpu`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `phase_space` | `ndarray (N, 6)` | Initial conditions `[x, y, z, vx, vy, vz]`. |
| `masses` | `ndarray (N,)` | Per-particle masses. |
| `time_start`, `time_end` | float | Integration interval. |
| `dt` | float | Fixed timestep. |
| `softening` | float or `ndarray (N,)` | Plummer softening length(s). |
| `G` | float | Gravitational constant. |
| `theta` | float | Opening angle (0.75 fast — 0.5 accurate). Default 0.6. |
| `nleaf` | int | Max particles per leaf cell (must be in {16,24,32,48,64}). Default 64. |
| `ncrit` | int | Walk group size. Default 64. |
| `level_split` | int | Tree level for spatial grouping. Default 5. |
| `step_timeout_s` | float | Seconds before the watchdog fires `KeyboardInterrupt` for a hanging CUDA kernel. Default 60. |
| `external_potential` | `agama.Potential` or None | External time-varying potential (requires Agama). |
| `external_update_interval` | int | Evaluate external potential every N steps. Default 1. |
| `output_dir` | str | Snapshot output directory. Default `'./output'`. |
| `save_snapshots` | bool | Write HDF5 snapshots. Default True. |
| `snapshots` | int | Number of snapshots to write. Default 100. |
| `num_files_to_write` | int | Distribute snapshots across this many HDF5 files. Default 1. |
| `restart_interval` | int | Save restart checkpoint every N steps. Default 1000. |
| `continue_run` | bool | Resume from an existing restart file. Default False. |
| `overwrite` | bool | Delete existing snapshot files before starting. Default False. |
| `verbose` | bool | Print header, progress (rate/ETA), and snapshot messages. Default True. |
| `debug_energy` | bool | Print virial ratio Q=K/\|W\| and relative energy drift dE/E. Overhead ~0.5 ms per print (float64 reduction). Default False. |
| `species` | list of `Species` or None | Multi-species descriptor list. When provided, `masses` and `softening` are built from it internally. |

**Returns**

`ndarray, shape (N, 6)` — final phase-space coordinates in float64.

**Raises**

- `ImportError` if CuPy is not installed.
- `FileExistsError` if snapshot files already exist and `overwrite=False`.
- `NotImplementedError` if an external potential is requested but Agama is not installed.

**Example**

```python
import numpy as np
from nbody_streams.tree_gpu import run_nbody_gpu_tree

rng = np.random.default_rng(42)
N = 50_000
pos = rng.normal(0, 5, (N, 3))
vel = rng.normal(0, 50, (N, 3))
phase_space = np.hstack([pos, vel])
masses = np.ones(N) * 1e4

final = run_nbody_gpu_tree(
    phase_space, masses,
    time_start=0.0, time_end=1.0, dt=0.001,
    softening=0.1,
    theta=0.6,
    output_dir="./output_tree",
    snapshots=10,
)
```

---

## `_StepWatchdog`

A background daemon thread that fires `KeyboardInterrupt` in the main thread
if a single integration step exceeds a configurable timeout.  Used internally
by `run_nbody_gpu_tree`; also available for users who run the tree loop
manually.

**Per-step overhead: ~1 microsecond** (two lock acquires + monotonic timestamp).
The interrupt fires even when the main thread is blocked inside a C extension
(`cudaDeviceSynchronize`).

```python
from nbody_streams.tree_gpu.run_gpu_tree import _StepWatchdog

watchdog = _StepWatchdog(timeout_s=60.0)
for step in range(n_steps):
    with watchdog:
        acc, phi = tree_gravity_gpu(pos, mass, eps=0.05, tree=tree)
        vel += 0.5 * dt * acc
        pos += dt * vel
watchdog.close()
```

---

## Float32 note

All force evaluations inside the tree code operate in **float32**.  Positions
and velocities in `run_nbody_gpu_tree` are also kept in float32 on the GPU
(consistent with the tree-code internals).  Energy diagnostics are computed
in float64.

---

## `G_DEFAULT`

```python
from nbody_streams.tree_gpu import G_DEFAULT  # 4.300917e-6 kpc (km/s)^2 / Msun
```
