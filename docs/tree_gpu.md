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
| `pos` | `cp.ndarray` or `np.ndarray`, shape `(N, 3)` | Particle positions.  float64 input is recentred before the float32 cast (see [Precision](#precision)); float32 input is used as given. |
| `mass` | float or `cp.ndarray (N,)` | Particle masses. |
| `eps` | float or `cp.ndarray (N,)` | Plummer softening length(s). Scalar gives uniform softening; array enables per-particle softening for multi-species runs. |
| `G` | float | Gravitational constant. Default: kpc / (km/s)^2 / Msun. |
| `theta` | float or None | Barnes-Hut opening angle. 0.75 = fast; 0.5 = accurate; 0.3 = very accurate. Default: `tree.theta` when a handle is passed (a value passed together with `tree` updates the handle), else 0.6. |
| `nleaf` | int | Max particles per leaf cell. Must be in {16, 24, 32, 48, 64}. Default 64. |
| `ncrit` | int | Group size for the tree walk. Default 64. |
| `level_split` | int | Tree level for spatial grouping. Default 5. |
| `verbose` | bool | Print timing / interaction counts from the C++ layer. |
| `tree` | `TreeGPU` or None | Pre-allocated tree handle. Skips ~27 ms malloc/free per call. |
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

Pre-allocated GPU tree handle for time-stepping loops.  Avoids ~27 ms of
GPU malloc/free overhead per step.  Must be closed explicitly or used as a
context manager.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `n` | int | Number of particles. Must not exceed available VRAM (80% free, capped at 28-bit Particle4 index). |
| `eps` | float | Default fallback softening length. |
| `theta` | float | Barnes-Hut opening angle, in (0, 1.15). |
| `verbose` | bool | Print timing from C++ layer. |

**Attributes and methods**

- `theta` — settable property; the new value applies from the next force call.
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
| `snapshots` | int | Number of snapshots to write. Exactly this many datasets are produced, ids `000`..`snapshots-1`, the last one at `time_end`. Default 100. |
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
2. Octree construction: **host-driven iterative build** (replaced CDP1 recursive kernel launches for CUDA 12 compatibility).  Each split of a cell scatters its particles into octants (no particle sort by space-filling-curve key happens here) and appends work items for children that need further splitting to a `PendingWork` queue.  Per level, the host reads the queue back, buckets the items by grid size (a power of two, at most 10 buckets), uploads them to a second queue and launches **one batched kernel per bucket** (`buildOctantBatch`: `blockIdx.z` = work item, `blockIdx.y` = octant).  This replaced one launch per work item (~7000 launches per build at N = 1M; build time 30 ms -> 6 ms on an RTX 3080 Laptop).  The queues have a compile-time depth of `BUILD_MAX_WORK = 262144` (safe for N ≤ 4M balanced trees; increase and recompile for highly clustered configurations).
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

| Buffer | Size | Purpose |
|--------|------|---------|
| `d_ptclPos`, `d_ptclAcc` | N × 16 bytes (float4) | Particle positions, accelerations |
| `d_ptclEpsTree` | N × 4 bytes | Per-particle softening (tree order) |
| `d_stack_memory_pool` | `max(N/5, 65536) × 96 × 4` bytes | Build-tree stack (≈ 24 MB for N=1M) |
| `d_cellDataList` | N × 16 bytes | Cell metadata |
| `d_cellMonopole`, `d_cellQuad0/1` | N × 16 + 8 bytes | Monopole + quadrupole moments |
| `d_cellEpsMax` | N × 4 bytes | Per-cell max softening |
| `d_cellComD`, `d_cellBoxMin/Max` | nCells × 64 bytes | Bottom-up multipole scratch |
| Build scratch (`d_build_workQueue`, `d_build_workQueueIn`) | 2 × `262144 × 64` bytes ≈ 32 MB | Host-driven build queues |
| Walk pool (`d_gmem_pool`) | 2 MB per persistent block | e.g. 384 blocks ≈ 768 MB on a 48-SM GPU; capped at 1/4 of free memory |

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
- **Device-side failures terminate the process** — device-side checks call `assert` / `exit`, which kills the Python process rather than raising an exception.  Pipeline phases are not synchronised individually; set the environment variable `TREE_SYNC_CHECK=1` to synchronise after each phase so an execution error is reported at the kernel that caused it.
- **Fixed timestep only** — `run_nbody_gpu_tree` uses a global fixed dt (KDK leapfrog).  Individual/block timesteps are not implemented.
- **No periodic boundary conditions** — the bounding box is recomputed from actual particle positions each step.
