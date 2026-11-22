# `nbody_streams.tree_gpu` — GPU Barnes-Hut Tree Code

GPU-accelerated O(N log N) gravity via the Barnes-Hut tree algorithm
(monopole + quadrupole moments).  The tree code is implemented in CUDA C++
and exposed to Python through a ctypes interface backed by `libtreeGPU.so`.

Optional dependency: **CuPy** (required at import time).

---

## Building the shared library

`pip install` builds the shared library automatically when `nvcc` and `make`
are in `PATH` (see `setup.py`; `NBODY_STREAMS_BUILD_TREE=1` makes the build
mandatory, `0` skips it).  To rebuild in place, in either an editable or a
regular install:

```bash
nbody-build-tree                 # make -j$(nproc) in the installed tree_gpu/
nbody-build-tree --clean-all     # remove objects and the library first
```

or run `make -j$(nproc)` in `nbody_streams/tree_gpu/` of a source checkout.

The Makefile detects the GPU architecture with `nvidia-smi` (or CuPy) and
falls back to `sm_80` plus PTX when no GPU is visible, so on a cluster build
on a GPU node.  If `libtreeGPU.so` is absent, importing
`nbody_streams.tree_gpu` raises an `ImportError` with the build command in
the message.

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
| `pos` | `cp.ndarray` or `np.ndarray`, shape `(N, 3)` | Particle positions.  float64 input is recentred before the float32 cast (see [Precision](#precision)); float32 input is used as given. |
| `mass` | float or `cp.ndarray (N,)` | Particle masses. |
| `eps` | float or `cp.ndarray (N,)` | Plummer softening length(s). Scalar gives uniform softening; array enables per-particle softening for multi-species runs. |
| `G` | float | Gravitational constant. Default: kpc / (km/s)^2 / Msun. |
| `theta` | float or None | Barnes-Hut opening angle. 0.75 = fast; 0.5 = accurate; 0.3 = very accurate. Default: `tree.theta` when a handle is passed (a value passed together with `tree` updates the handle), else 0.6. |
| `nleaf` | int | Max particles per leaf cell. Must be in {16, 24, 32, 48, 64}. Default 64. |
| `ncrit` | int | Group size for the tree walk. Default 64. |
| `level_split` | int | Unused; accepted for backward compatibility (groups are cut from tree cells, see Phase 3). Default 5. |
| `verbose` | bool | Print timing / interaction counts from the C++ layer. |
| `tree` | `TreeGPU` or None | Pre-allocated tree handle. Skips the per-call allocation (~40 ms at N = 1M on an RTX 3080 Laptop, more than the force pass itself). |
| `center` | array_like `(3,)` or None | Centre subtracted from float64 `pos` before the float32 cast.  Default: subsample centre of mass.  Ignored for float32 `pos`. |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `acc` | `cp.ndarray`, shape `(N, 3)` | Gravitational accelerations in original particle order. |
| `phi` | `cp.ndarray`, shape `(N,)` | Gravitational potential per particle, excluding the particle's interaction with itself (same convention as the direct-sum backend). |

**Softening convention** (matches nbody_streams direct-sum):

- Direct particle-particle: `eps2_ij = max(eps_i^2, eps_j^2)`
- Cell approximation: `eps2_ij = max(eps_i^2, eps_cell_max^2)`

Both use the Plummer kernel: `r^2 -> r^2 + eps2`.  Softening must be
strictly positive (`ValueError` otherwise).  Zero-mass particles (tracers)
are allowed: they feel forces but exert none.

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

Pre-allocated GPU tree handle for time-stepping loops.  Avoids ~40 ms (N = 1M) of
GPU malloc/free overhead per step.  Must be closed explicitly or used as a
context manager.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `n` | int | Number of particles. `ValueError` if the tree would not fit in 80% of the free GPU memory (`set_memory_fraction` changes the fraction), or above 2^29 (the particle-index limit); see [GPU memory layout](#gpu-memory-layout). |
| `eps` | float | Default fallback softening length. |
| `theta` | float | Barnes-Hut opening angle, in (0, 1.15). |
| `verbose` | bool | Print timing from C++ layer. |

**Attributes and methods**

- `theta` — settable property; the new value applies from the next force call.
- `build_stats()` — dict with the last build's `cells`, `nodes`, `leaves`,
  `levels`, `work_items` and the build capacities `cell_capacity`,
  `node_capacity`, `work_capacity` (these grow when a build overflows them).
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
| `level_split` | int | Unused; accepted for backward compatibility (groups are cut from tree cells, see Phase 3). Default 5. |
| `step_timeout_s` | float | Seconds before the watchdog fires `KeyboardInterrupt` for a hanging CUDA kernel. Default 60. |
| `external_potential` | `agama.Potential` or None | External time-varying potential (requires Agama). |
| `external_update_interval` | int | Evaluate external potential every N steps. Default 1. |
| `output_dir` | str | Snapshot output directory. Default `'./output'`. |
| `save_snapshots` | bool | Write HDF5 snapshots. Default True. |
| `snapshots` | int | Number of snapshots to write. Exactly this many datasets are produced, ids `000`..`snapshots-1`, the last one at `time_end`. Default 100. |
| `num_files_to_write` | int | Distribute snapshots across this many HDF5 files. Default 1. |
| `restart_interval` | int or None | Save restart checkpoint every N steps and at the end. Default 1000; `None` or 0 writes no restart files. The closing summary reports integration and output time separately. |
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

**Snapshot schedule**

Identical to `run_nbody_gpu` / `run_nbody_cpu`. With `n_steps = round((time_end - time_start) / dt)`, the output steps are

```python
snapshot_steps = np.round(np.linspace(0, n_steps, snapshots)).astype(int)   # snapshots > 1
snapshot_steps = np.array([n_steps])                                       # snapshots == 1
```

so `snapshots=N` always yields exactly `N` datasets with 0-based ids `snap.000` … `snap.{N-1:03d}`, the first at `time_start` (for `N > 1`) and the last at `time_end` — including when `snapshots` does not divide `n_steps`. Snapshot ids and times therefore match the direct-sum backends step for step for the same `(time_start, time_end, dt, snapshots)`, and `ParticleReader` can be pointed at either backend's output interchangeably.

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

## Precision

All force evaluations inside the tree code operate in **float32**.  float32
spacing grows with distance from the origin (about 4e-3 pc at 40 kpc), so the
tree should see positions measured from the system it resolves:

- `run_nbody_gpu_tree` integrates positions and velocities in **float64**.
  Each step it passes `pos - c` to the tree, where `c` is the float64
  position of the particle with the lowest self-gravity potential at the
  previous step (the density peak of the bound progenitor; the mass-weighted
  centre before the first step).  Self-gravity is translation invariant, so
  `c` does not change the physics.  The external potential is evaluated at the
  absolute float64 positions.  Measured cost: about 1 ms/step at N = 1M.
- `tree_gravity_gpu` does the same for **float64** `pos` (NumPy or CuPy):
  it subtracts `center` in float64 before the float32 cast.  The default
  centre is the mass-weighted mean of a ~4096-particle strided subsample
  (~0.2 ms at N = 1M, within timing noise of the 39 ms call); pass
  `center=` (e.g. the progenitor position) when most of the mass has been
  stripped and the mean lies far from the bound core.  For a 1-pc Hernquist
  cluster at 40 kpc the 99th-percentile force error is 4.4e-4 with float64
  input, the same as at the origin, against 1.6e-2 for absolute float32 input.
- float32 `pos` is used as given: its precision was fixed when it was
  rounded, so recentring it would not help.  Rescaling units does not help
  either (float rounding is relative); only translating does.

Energy diagnostics are computed in float64.

### Passes around the tree

At N = 1M the work outside the tree walk is 0.8 ms per step (it was 1.9 ms):

- `tree_gravity_gpu`: one kernel subtracts the centre, casts to float32 and
  writes the `(3, N)` layout the tree reads (`to_float32_relative_soa`); one
  kernel after the walk transposes the accelerations, removes the self term
  from `phi` and multiplies by `G`, with the same float32 operations as
  before (outputs are bitwise unchanged). With a `TreeGPU` handle, an `eps`
  array is checked for positive values only the first time that array object
  is passed (the check costs a host sync).
- `run_nbody_gpu_tree`: positions, velocities and the total float64
  acceleration are updated in place by two kernels per step, kick + drift
  and (self-gravity + external) + kick. The kick increment `h a` is formed in
  float64 in both cases.

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
mass.  During tree construction its bits hold `(idx << 3) | octant`: the
original particle index in bits [31:3] and the octant (0..7) in bits [2:0],
via `__float_as_uint` / `__uint_as_float` (the `double` specialisation uses
the low 32 bits of the double's bit pattern).  This limits the particle count
to 2^29 = 536,870,912 (`PARTICLE4_MAX_N`; was 2^28 with a 4-bit octant field).

**`CellData`** (`Treecode.h`)

Each tree cell is packed into a `uint4` (128 bits):

```
packed_data.x  = parent cell index (32 bits)
packed_data.y  bits[31:29] = n_children - 1 (3 bits)
               bits[28:0]  = first child index (or 0xFFFFFFFF for leaf)
packed_data.z  = particle begin index (pbeg)
packed_data.w  = particle end index (pend)
```

The cell's tree level is not stored: the build writes it straight into the
level-sort key array (`d_key`), which `buildTree` sorts to order the cells by
level.  (It used to occupy bits [31:27] of `.x`, which limited the parent
index, and so the cell count, to 2^27.)  The 29-bit first-child index limits
the cells to 2^29 - 1 (`CELLDATA_MAX_CELLS`); below that a node's `.y` can
never equal the leaf marker.  `Particle4` and `CellData` round trips at these
limits are checked by `tree_selftest_packing()` (`tests/test_limits.py`).

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
2. Octree construction: **host-driven iterative build** (replaced CDP1 recursive kernel launches for CUDA 12 compatibility).  Each split of a cell scatters its particles into octants (no particle sort by space-filling-curve key happens here) and appends work items for children that need further splitting to a `PendingWork` queue.  Per level, the host reads the queue back, buckets the items by grid size (a power of two, at most 10 buckets), uploads them to a second queue and launches **one batched kernel per bucket** (`buildOctantBatch`: `blockIdx.z` = work item, `blockIdx.y` = octant).  This replaced one launch per work item (~7000 launches per build at N = 1M; build time 30 ms -> 6 ms on an RTX 3080 Laptop).  The queues hold `max(16384, N/32 + 1)` work items initially (grown on overflow, see [GPU memory layout](#gpu-memory-layout)); the most measured is 0.017 N, for a very thin straight filament (<= 0.0074 N in 3-D).
   Depth is capped at level 30 (`TREE_MAX_LEVEL`): children created there become leaves even if they hold more than `nleaf` particles, so particles that cannot be separated (exact duplicates, or closer than the float resolution of the box) end up in one oversized leaf whose forces the walk sums directly.  Balanced trees are ~15-20 levels deep, so the cap never triggers in normal use.
3. Cells are then stable-sorted by tree level (Thrust `stable_sort_by_key`) so each level is contiguous.
4. A permutation array `d_origIdx[tree_idx] = original_user_idx` is built so output forces can be returned in user order.

#### Phase 2 — `computeMultipoles`

Bottom-up, O(N + nCells):

1. `leafMultipoles` — one warp per leaf reduces the leaf's particles with
   warp shuffles (`shfl_xor`).  Moments are taken about the leaf's first
   particle (so `<xx> - <x><x>` does not cancel for small cells far from the
   origin) with per-warp float partial sums accumulated in double.
2. `nodeMultipoles` — one thread per node, one launch per level from the
   deepest up, combines the (at most 8) children with the parallel-axis
   theorem: `M = Σ m_c`, `com = Σ m_c com_c / M`,
   `Q = Σ m_c (Q_c + d_c d_c^T) / M` with `d_c = com_c - com`.  Centres of
   mass are carried between levels in double (`d_cellComD`).

Per cell this stores:

- **Monopole** `(com_x, com_y, com_z, mass)` in `d_cellMonopole`.
- **Quadrupole**: the 6 components of the mass-normalised central second
  moment `Q_ij = <x_i x_j> - <x_i><x_j>` (not trace-free; the force kernel
  uses the trace explicitly), stored as `float4 + float2` in `d_cellQuad0/1`.
- **Opening radius** `(l/theta + s)^2` in `d_cellSize.w` (see below), from the
  tight particle bounding box (union of the children's boxes for nodes).
- **Max softening** `d_cellEpsMax[cell] = max(eps_k)` over the cell's particles.

A cell whose particles are all massless gets zero mass and quadrupole and a
finite COM inside the cell, so it contributes nothing to the forces.

#### Phase 3 — `makeGroups`

Builds the *interaction groups*: sets of <= `ncrit` particles that walk the
tree together and share one opening test against the group's bounding box,
so they must be spatially compact.  Groups are cut from *container* cells:
cells with <= 32 * `ncrit` particles whose parent has more (or top-level
cells).  Each container's contiguous particle range is split into
ceil(n / `ncrit`) equal chunks, and groups are sorted by particle offset so
neighbouring warps walk nearby cells.  A group can only straddle regions inside
one container, which bounds the worst case: no group can open the whole tree.

Groups index the tree-ordered particle arrays directly; buildTree sorts the
particles inside each leaf by original index, so group membership, and the
forces, are bit-for-bit reproducible between calls.

This replaced Bonsai's Peano-Hilbert grouping (fixed `ncrit` chunks along a
PH ordering, split at `level_split` key boundaries).  With far outliers, one
level-5 cell held the whole system and a chunk could straddle the dense core
and the sparse halo: that group opened every cell, one warp summed over all N
particles, and the force pass waited on it (2M-particle Plummer: max 2.0M
direct interactions per particle against ~4000 on average, 17x slower than
1M).  `level_split` is still accepted but unused.

#### Phase 4 — `computeForces`

Each interaction group walks the tree in parallel.  The kernel is
persistent: `nblock = (number of SMs) × (resident blocks per SM from
cudaOccupancyMaxActiveBlocksPerMultiprocessor)` blocks of 4 warps pull groups
from an atomic counter (capped so the ring buffers below use at most 1/4 of
free memory).  This replaced a fixed `8*13 = 104` blocks, which left large
GPUs mostly idle.  `__launch_bounds__(128, 6)` allows ~85 registers so the
quadrupole walk spills little.  Each warp has its own ring buffer (size
`CELL_LIST_MEM_PER_WARP = 4096 × 32` ints) holding the pending cell list,
avoiding global atomics.

**Opening criterion** — Bonsai group MAC with the Salmon-Warren offset (`split_node_grav_impbh`):

```
dr = max(0, |r_group_center - r_cell_com| - r_group_half_size)   (per axis)
split if  |dr|^2 < cellSize.w
cellSize.w = (l/theta + s)^2
```

`dr` is the per-axis minimum distance from the interaction group's bounding
box to the cell's centre of mass (COM).  In `computeMultipoles`, `l` is the
longest side of the cell's tight particle bounding box (clamped to >= 1e-6)
and `s` is the distance between that bounding-box centre and the COM
(Salmon-Warren offset).  Because the test uses the group box rather than a
single particle, it is more conservative than the classic criterion
`d < l/theta`.  Cells containing fewer than 3 particles are always opened.

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
  → buildTree (octant scatter): d_ptclEpsTree[tree_order_i] extracted
  → computeForces: reads d_ptclEpsTree for both query (eps_i) and source (eps_j)
```

The uniform-softening path (`tree_gravity_gpu(eps=scalar)`) uses the same
pipeline; `k_load_pos_mass` broadcasts the scalar into every `ptclVel.x`.

---

### GPU memory layout

`TreeGPU.alloc(N)` makes a bulk allocation of the persistent tree buffers
listed below.  Nothing is allocated per step in steady state: the tree-walk
ring-buffer pool (`d_gmem_pool`) is allocated on the first force call and
reused, and Thrust sort
scratch goes through a caching allocator owned by the tree.  The cell arrays
grow (never shrink) with the number of cells.  The main allocations:

| Buffer | Size | Bytes/particle |
|--------|------|----------------|
| `d_ptclPos`, `d_ptclVel`, `d_ptclPos_tmp`, `d_ptclAcc` | 4 × N × 16 (float4) | 64 |
| `d_ptclEpsTree`, `d_origIdx` | 2 × N × 4 | 8 |
| `d_stack_memory_pool` (node pool) | `max(N/8, 16384)` × 96 × 4 initially (grown on overflow) | 48 |
| `d_cellDataList` (+ `_tmp`, `d_key`, `d_value`) | `max(0.6 N + 64, 65536)` × (16+16+4+4) initially (grown on overflow) | 24 |
| `d_leafList` | `min(N, cell capacity)` × 4 | 2.4 |
| Build work queues (`d_build_workQueue`, `...In`) | 2 × `max(N/32 + 1, 16384)` × 64 initially (grown on overflow) | 4 |
| **at allocation** | | **151** (was 200) |
| Cell multipoles, sizes, eps, COM, boxes (8 arrays) | nCells × 124, allocated by the first call | 7 (nleaf 64) - 45 (nleaf 16) |
| `d_groupList` | N × 8 | 8 |
| Thrust scratch (level sort, group sort) | ~16 × nCells, cached | 1 - 6 |
| Walk pool (`d_gmem_pool`) | 2 MiB per persistent block, 6 blocks per SM, capped at 1/4 of free memory | fixed: 576 MiB (48 SMs), 1.7 GiB (L40, 142 SMs), 1.6 GiB (H200) |

Measured with `cudaMemGetInfo` around `TreeGPU(N)` plus one
`tree_gravity_gpu` call (slope between N = 1e7 and 5e7, Plummer, L40):
**188 B/particle at nleaf = 16, 166 at nleaf = 64** (before: 238 / 216),
plus the fixed walk pool; the most-cells tree measured (a cubic lattice at
5e7, nleaf = 16, 0.37 N cells) takes 208 (before: 258).  Each `tree_gravity_gpu` call also allocates, from the CuPy
pool, 41 B/particle: float32 SoA positions (12), SoA acceleration (12) and
potential (4), and the returned (N, 3) acceleration (12); the returned acc
and phi (16) stay with the caller.

Usage measured with `TreeGPU.build_stats()` (maximum over N = 1e5 - 5e7,
nleaf = 16; with nleaf = 64 cells are <= 0.083 N):

| Input | cells / N | nodes / N | work items / N |
|-------|-----------|-----------|----------------|
| Plummer | 0.243 | 0.032 | 0.0020 |
| Hernquist | 0.237 | 0.031 | 0.0012 |
| Dehnen gamma = 1.5 cusp | 0.237 | 0.031 | 0.0012 |
| Cold stream (arc, width 0.05 of 20) + core | 0.268 | 0.038 | 0.0057 |
| Thin exponential disc (h_z / R_d = 0.02) | 0.251 | 0.044 | 0.0048 |
| Uniform cube | 0.349 | 0.045 | 0.0073 |
| Cubic lattice | 0.369 | 0.046 | 0.0074 |
| Straight filament (width 1e-4 of length) | 0.274 | 0.067 | 0.0174 |
| **initial capacity** | **0.6** | **0.125** | **0.031** |

The initial capacities are >= 1.5x every measured case (1.63x for cells,
1.86x for nodes, 1.8x for work items), so realistic inputs never grow them;
0.6 N cells also covers the ~0.54 N of a uniform distribution with ~17
particles in every cell of one level.  The cell multipole arrays are sized
to the actual cell count, not the capacity.

**Memory guard.**  `TreeGPU(N)` raises `ValueError` unless

    N × (216 + 40) + 64 MiB + walk pool  <=  f × free GPU memory

and N <= 2^29: 216 B/particle covers the tree after a call for trees of up
to 0.4 N cells (the most measured is 0.37 N), 40 the per-call arrays above;
the remaining 1 - f (default f = 0.8) absorbs allocator rounding and CuPy
pool growth.  (It used
52 B/particle, admitting ~5x more particles than fit.)  `run_nbody_gpu_tree`
checks before it creates its handle (its float64 pos/vel, float32/float64
masses and float32 softening, 64 B/particle, already exist and count as used)
with a further 128 B/particle: the float64 total acceleration (24), the
previous step's float32 acc and phi, alive during the next tree call (16),
the (N, 6) float64 pos/vel concatenation for snapshots and restarts (48) and
CuPy pool fragmentation between them (~40; measured, the pool peaks at 232
B/particle with snapshots, 196 without), plus 24 for the float64 external
acceleration when `external_potential` or `force_extra` is given (the
potential's own scratch is not counted).

Largest N the guard admits, from total memory M with ~0.5 GB taken by the
CUDA context and the inputs already on the GPU (a one-off call with float64
positions: 32 B/particle; `run_nbody_gpu_tree`: 64):

| GPU memory | `tree_gravity_gpu` | `run_nbody_gpu_tree` (no external) | (with external) |
|------------|--------------------|------------------------------------|-----------------|
| 16 GB (RTX 3080 Laptop, 48 SMs) | ~45M | ~29M | ~28M |
| 48 GB (L40, 45 GiB usable, 142 SMs) | ~129M | ~84M | ~79M |
| 141 GB (H200, 132 SMs) | ~422M | ~273M | ~259M |
| 192 GB (B200, 148 SMs) | 537M (2^29 index limit; memory ~577M) | ~374M | ~354M |

On a dedicated GPU, raise f to plan for more of the memory:

```python
import nbody_streams.tree_gpu as tg
tg.set_memory_fraction(0.95)   # returns the previous value; get_memory_fraction() reads it
```

The 216 B/particle term is the worst distribution measured (a cubic lattice
at `nleaf = 16`); smooth distributions need ~10-20% less, so 0.95 is safe for
them.  Above the default an unusual distribution can run out of memory inside
the build, which ends the process.

Checked on an L40 with 25 GiB free (other users held the rest): the largest
N the guard admits for a one-off call (72M, cubic lattice, nleaf = 16) peaked
at 78% of the free memory.

With the old 52 B/particle guard the laptop admitted ~220M particles (capped
at 2^28) for a tree that fit only up to ~45M even with all of the memory.

The node pool, cell list and work queues are sized for typical trees, not
for the worst case: a rigorous bound would need ~2.7 N cells and ~1.7 N node
slots (~65 GB of node pool at N = 1e8), because a clump of more than `nleaf`
(nearly) coincident particles becomes a chain of ~25 single-child nodes down
to the depth cap.  Inputs with thousands of such clumps (e.g. 4000 clusters of
17 identical positions, N = 68k) exceed them.  The build then recovers: a
block whose allocation would exceed a capacity writes no tree data (no cell,
node-pool slot, work item or leaf), finalises its particles as leaves would,
and sets an overflow flag; the counters keep counting.  `buildTree` reads the
flag with the counters it always reads back (no extra synchronisation),
restores the input particle order, grows each overflowed buffer to
`max(2 × capacity, 1.5 × count reached)` and builds again.  The buffers stay
grown on the handle, so a `TreeGPU` reused across steps pays this once; the
rebuilt tree is bitwise the one a big-enough handle builds.  With
`verbose=True` each round prints a `treeGPU: tree build overflowed` line to
stderr.  If a buffer cannot grow (index-packing cap: 2^29 - 1 cells, which
also bounds nodes and work items; or out of GPU memory) `tree_gravity_gpu` raises
`RuntimeError`.  For testing, `TREEGPU_DEBUG_BUILD_CAPACITY=k` (read when a
handle is allocated) starts all three buffers at k entries.

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

`FAST_MATH` is **on by default** (`FAST_MATH ?= 1`, Makefile line 15; the Makefile header comment saying "default: off" is stale).  Disable for
higher floating-point accuracy at ~5-10% throughput cost.

---

### ctypes interface (`_force.py`)

The Python layer calls `libtreeGPU.so` via `ctypes`.  Exported C symbols:

| Symbol | Signature | Notes |
|--------|-----------|-------|
| `tree_new` | `(eps, theta) → Tree*` | Allocate Treecode struct |
| `tree_delete` | `(Tree*)` | Free struct + GPU buffers |
| `tree_alloc` | `(Tree*, n)` | `cudaMalloc` all buffers |
| `tree_set_verbose` | `(Tree*, verbose)` | Timing / interaction-count output |
| `tree_set_theta` | `(Tree*, theta)` | Change the opening angle of an existing handle |
| `tree_set_pos_mass_eps_device` | `(Tree*, n, x, y, z, mass, eps)` | Load per-particle eps |
| `tree_set_pos_mass_device` | `(Tree*, n, x, y, z, mass)` | Uniform-eps path |
| `tree_build` | `(Tree*, nleaf)` | Run buildTree |
| `tree_compute_multipoles` | `(Tree*)` | Run computeMultipoles |
| `tree_make_groups` | `(Tree*, level_split, ncrit)` | Run makeGroups |
| `tree_compute_forces` | `(Tree*)` | Run computeForces (interaction statistics only when verbose) |
| `tree_get_acc_device` | `(Tree*, n, ax, ay, az, pot)` | Copy results to output arrays in user order |

All pointer arguments are raw CUDA device pointers obtained from CuPy
via `arr.data.ptr`.  The Python wrapper (`tree_gravity_gpu`) handles
argument marshalling and the two-sort permutation to restore user order.

---

### Known limitations

- **Single GPU only** — the ctypes interface always operates on device 0.  Multi-GPU support would require per-device `Tree*` instances and explicit `cudaSetDevice` calls in the C interface.
- **float32 tree** — the tree itself works in float32; see [Precision](#precision) for how `run_nbody_gpu_tree` keeps this from limiting compact structure far from the origin, and what direct callers of `tree_gravity_gpu` should do.
- **Device-side failures terminate the process** — device-side checks call `assert` / `exit`, which kills the Python process rather than raising an exception.  (Build-buffer overflow is not one of them: it is detected and the buffers grow, see [GPU memory layout](#gpu-memory-layout).)  Pipeline phases are not synchronised individually; set the environment variable `TREE_SYNC_CHECK=1` to synchronise after each phase so an execution error is reported at the kernel that caused it.
- **Fixed timestep only** — `run_nbody_gpu_tree` uses a global fixed dt (KDK leapfrog).  Individual/block timesteps are not implemented.
- **No periodic boundary conditions** — the bounding box is recomputed from actual particle positions each step.
