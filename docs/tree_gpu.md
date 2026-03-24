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

---

## Implementation internals

This section documents the CUDA C++ internals for contributors and users who
want to understand the accuracy/performance trade-offs or port the code.

### Data structures

**`Particle4<float>`** (`Particle4.h`)

The fundamental particle type is an AoS (Array-of-Structures) `float4`:

```
packed_data = { x, y, z, mass/idx }
```

The `.w` field is dual-purpose: during the tree walk it holds the particle
mass.  During tree construction the lower 28 bits encode the sorted particle
index and the lowest 4 bits encode the octant, using `__float_as_int` /
`__int_as_float` bit-casting.  This limits particle count to 2^28 ≈ 268M.

**`CellData`** (`Treecode.h`)

Each tree cell is packed into a `uint4` (128 bits):

```
packed_data.x  bits[31:27] = tree level (5 bits)
               bits[26:0]  = parent cell index
packed_data.y  bits[31:29] = n_children - 1 (3 bits, 0=leaf)
               bits[28:0]  = first child index (or 0xFFFFFFFF for leaf)
packed_data.z  = particle begin index (pbeg)
packed_data.w  = particle end index (pend)
```

`isLeaf()` returns true when `.y == 0xFFFFFFFF`.

**`Quadrupole<T>`** (`Treecode.h`)

The traceless quadrupole tensor is stored in 6 independent components packed
as `float4 + float2`:

```
q0 = { Qxx, Qyy, Qzz, Qxy }
q1 = { Qxz, Qyz }
```

`Qzz` is stored explicitly (the tensor is not assumed traceless in storage,
but the physics kernel enforces trace-free accumulation in `computeMultipoles`).

**`GroupData`**

Groups are `int2{pbeg, np}` — a contiguous block of particles sorted for
coalesced global-memory access during the tree walk.

---

### The four-phase pipeline

Each call to `tree_gravity_gpu` (or each step of `run_nbody_gpu_tree`)
executes four CUDA kernels in sequence:

#### Phase 1 — `buildTree`

1. Bounding-box reduction over all particle positions (warp-level `fminf`/`fmaxf`, then inter-warp shared-memory reduction).
2. Morton key assignment — each particle's (x,y,z) is mapped to a 30-bit Morton Z-order key relative to the bounding box.
3. Radix sort of particles by Morton key (Thrust `device_sort`).
4. Octree construction: **host-driven iterative build** (replaced CDP1 recursive kernel launches for CUDA 12 compatibility).  A `PendingWork` queue on the host drives per-level CUDA kernel launches; each kernel processes one octant level in parallel.  The work queue has a compile-time depth of `BUILD_MAX_WORK = 262144` (safe for N ≤ 4M balanced trees; increase and recompile for highly clustered configurations).
5. A permutation array `d_origIdx[tree_idx] = original_user_idx` is built so output forces can be returned in user order.

#### Phase 2 — `computeMultipoles`

Bottom-up cell moment accumulation using warp-level shuffle reductions
(`shfl_xor`):

- **Monopole**: total mass and centre-of-mass position accumulated in
  double precision per warp, then reduced to float and stored as `(cx, cy, cz, mass)` in `d_cellMonopole`.
- **Quadrupole**: 6 independent components of `Q_ij = Σ_k m_k (r_ki r_kj - δ_ij r_k²/3)` accumulated in double per warp, stored as `float4 + float2` in `d_cellQuad0/1`.
- **Per-cell max softening**: `d_cellEpsMax[cell] = max(eps_k)` over all particles in the cell, propagated bottom-up alongside the moments.  Used in the force kernel for the cell-approximation softening.

#### Phase 3 — `makeGroups`

Particles are re-sorted into *interaction groups* by splitting the tree at
`level_split` (default 5).  Each group is a contiguous block of ≤ `ncrit`
particles from the same sub-tree, which gives coalesced global-memory reads
in the force kernel.  The sorted-group index array `d_ptclEpsGrp` is
populated here from `d_ptclEpsTree` via the group permutation.

#### Phase 4 — `computeForces`

Each interaction group walks the tree in parallel.  The walk uses a
per-warp ring-buffer (size `CELL_LIST_MEM_PER_WARP = 4096 × 32`) to hold
the pending cell list, avoiding global atomics.

**Opening criterion** — improved Barnes-Hut (`split_node_grav_impbh`):

```
dr = max(0, |r_group_center - r_cell_center| - r_group_half_size)
split if  |dr|² < |cellSize.w|
```

This is the *min-distance* criterion: a cell is only approximated if the
closest point of the interaction group's bounding box is far enough from the
cell centre.  It is more conservative (and more accurate) than the
classic single-particle criterion `d < θ × l`.

**Interaction types** (once a cell passes the opening criterion):

| Condition | Kernel | Cost |
|-----------|--------|------|
| Leaf cell (`isLeaf`) | Direct pair-wise with Plummer softening | O(nleaf) FMAs per group particle |
| Internal cell, `#QUADRUPOLE` defined | Monopole + quadrupole approximation | 1 interaction per cell |
| Internal cell, monopole only | Monopole approximation | 1 interaction per cell |

`#QUADRUPOLE` is enabled by default (see `Treecode.h` line 15).

**Flush-to-zero**: `computeForces.o` is compiled with `-ftz=true` to flush
denormal floats to zero in force accumulation.  This improves throughput on
subnormals at the cost of ~1 ULP accuracy for very small separations (covered
by softening in practice).

---

### Softening convention

Both direct pair-particle and cell-approximation interactions use the same
**max convention**:

```
eps²_ij = max(eps²_i,  eps²_j)          # direct
eps²_ij = max(eps²_i,  eps²_cell_max)   # cell approx
```

where `eps_cell_max` is the maximum softening of any particle in the cell
(stored in `d_cellEpsMax`).  This matches the convention used by the
nbody_streams direct-sum kernels (`fields.py`) and is standard practice for
multi-species N-body (prevents over-softening of heavy dark-matter particles
onto light stellar particles).

---

### Per-particle softening pipeline

Per-particle softening is threaded through the entire tree sort without extra
arrays:

```
user eps[i]
  → k_load_pos_mass_eps: ptclVel[i].x = eps[i]
  → buildTree (Morton sort): d_ptclEpsTree[tree_order_i] extracted
  → makeGroups (group sort): d_ptclEpsGrp[group_order_i] extracted
  → computeForces: reads d_ptclEpsGrp (query eps_i), d_ptclEpsTree (source eps_j)
```

The uniform-softening path (`tree_gravity_gpu(eps=scalar)`) uses the same
pipeline; `k_load_pos_mass` broadcasts the scalar into every `ptclVel.x`.

---

### GPU memory layout (per-step allocations)

`TreeGPU.alloc(N)` makes a single bulk allocation; there are no per-step
`cudaMalloc` / `cudaFree` calls.  The main allocations:

| Buffer | Size | Purpose |
|--------|------|---------|
| `d_ptclPos`, `d_ptclAcc` | N × 16 bytes (float4) | Particle positions, accelerations |
| `d_ptclEpsTree`, `d_ptclEpsGrp` | N × 4 bytes each | Softening in two sort orders |
| `d_stack_memory_pool` | `max(N/5, 65536) × 96 × 4` bytes | Build-tree stack (≈ 24 MB for N=1M) |
| `d_cellDataList` | N × 16 bytes | Cell metadata |
| `d_cellMonopole`, `d_cellQuad0/1` | N × 16 + 8 bytes | Monopole + quadrupole moments |
| `d_cellEpsMax` | N × 4 bytes | Per-cell max softening |
| Build scratch (`d_build_workQueue`) | `262144 × 64` bytes ≈ 16 MB | Host-driven build queue |

Total VRAM for N = 1M particles: approximately **700 MB** before quadrupole
storage.  `TreeGPU` checks that 80% of free VRAM is available before allocating.

---

### Build system details

The Makefile (`nbody_streams/tree_gpu/Makefile`) handles three non-trivial
concerns:

**GPU architecture detection** — in priority order:
1. `nvidia-smi --query-gpu=compute_cap` (bare-metal, clusters)
2. `python3 -c "import cupy; ..."` (WSL2, no `nvidia-smi` in PATH)
3. Hard-coded `sm_80` (Ampere) as last resort

All detected architectures get individual `-gencode=arch=compute_XX,code=sm_XX`
flags plus a PTX blob for forward compatibility with future hardware.

**GCC 11 workaround** — NVCC's EDG frontend misparses a `std::sample`
pattern in GCC 11's `stl_algo.h` as a lambda introducer, producing
`"expected a ']'"` errors in `buildTree.cu` and `makeGroups.cu`.  The
Makefile detects GCC 11 as the system `g++` and automatically redirects
`-ccbin` to `g++-12` or `g++-13` if available:

```
Host compiler : g++-13 (GCC 11 workaround — see Makefile comment)
```

If no alternative is found, a warning is printed and the build proceeds
(and will fail).  Fix: `sudo apt install g++-13`.

**Optional flags**:

```bash
make FAST_MATH=1      # adds -use_fast_math (rsqrtf, etc.)
make VERBOSE=1        # echo full compile commands
make clean            # remove .o files
make clean_all        # remove .o files + libtreeGPU.so
```

`FAST_MATH` is **on by default** (see Makefile line 10).  Disable for
higher floating-point accuracy at ~5-10% throughput cost.

---

### ctypes interface (`_force.py`)

The Python layer calls `libtreeGPU.so` via `ctypes`.  Exported C symbols:

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `tree_new` | `(eps, theta) → Tree*` | Allocate Treecode struct |
| `tree_delete` | `(Tree*)` | Free struct + GPU buffers |
| `tree_alloc` | `(Tree*, n)` | `cudaMalloc` all buffers |
| `tree_set_pos_mass_eps_device` | `(Tree*, n, x, y, z, mass, eps)` | Load per-particle eps |
| `tree_set_pos_mass_device` | `(Tree*, n, x, y, z, mass)` | Uniform-eps path |
| `tree_build` | `(Tree*, nleaf)` | Run buildTree |
| `tree_multipoles` | `(Tree*)` | Run computeMultipoles |
| `tree_groups` | `(Tree*, level_split, ncrit)` | Run makeGroups |
| `tree_forces` | `(Tree*)` | Run computeForces |
| `tree_get_acc_pot_device` | `(Tree*, ax, ay, az, pot)` | Copy results to output arrays |

All pointer arguments are raw CUDA device pointers obtained from CuPy
via `arr.data.ptr`.  The Python wrapper (`tree_gravity_gpu`) handles
argument marshalling and the two-sort permutation to restore user order.

---

### Known limitations

- **Single GPU only** — the ctypes interface always operates on device 0.  Multi-GPU support would require per-device `Tree*` instances and explicit `cudaSetDevice` calls in the C interface.
- **float32 positions** — precision loss accumulates for simulations spanning dynamic ranges > ~10^6 (e.g. a cluster embedded in a cosmological box).  Consider recentring particles at each snapshot output for long-duration runs.
- **Fixed timestep only** — `run_nbody_gpu_tree` uses a global fixed dt (KDK leapfrog).  Individual/block timesteps are not implemented.
- **No periodic boundary conditions** — the bounding box is recomputed from actual particle positions each step.
