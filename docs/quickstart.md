# Quick start

> **Units throughout this package:** length = kpc, velocity = km/s, mass = M_sun.
> The gravitational constant in these units is `G = 4.300917e-6 kpc (km/s)^2 / M_sun`.

---

## Installation

### Core package

```bash
git clone https://github.com/appy2806/Nbody_streams.git
cd Nbody_streams
pip install -r requirements.txt
pip install -e .
```

### Optional extras

| Extra | What it enables | Install command |
|---|---|---|
| `agama` | External potentials, `fast_sims`, `agama_helper`, sky coordinate conversion | `pip install -e ".[agama]" --no-build-isolation` |
| `falcon` | CPU tree / FMM backend via pyfalcon | `pip install -e ".[falcon]" --no-build-isolation` |
| `healpy` | Mollweide sky projections in `viz.plot_mollweide` | `pip install -e ".[healpy]"` |
| `all` | All of the above | `pip install -e ".[all]" --no-build-isolation` |

### GPU support (CuPy)

Install CuPy for the CUDA version matching your system:

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda13x   # CUDA 13.x
```

### GPU tree code (one-time build)

The GPU Barnes-Hut tree code is a C++/CUDA shared library and must be compiled once before use.  Requires `nvcc` in `PATH`:

```bash
cd nbody_streams/tree_gpu
make -j$(nproc)
cd ../..
```

This produces `nbody_streams/tree_gpu/libtreeGPU.so`.  Re-run `make` after pulling updates that touch the `.cu` source files.

---

## Units convention

| Quantity | Unit |
|---|---|
| Position | kpc |
| Velocity | km/s |
| Mass | M_sun (solar masses) |
| Time | kpc / (km/s) ≈ 0.978 Gyr |
| Gravitational constant G | 4.300917e-6 kpc (km/s)^2 / M_sun |

All input arrays, output arrays, softening lengths, timesteps, and times use these units unless explicitly stated otherwise.  When using an external Agama potential, call `agama.setUnits(mass=1, length=1, velocity=1)` once before integration.

---

## Hello world: Plummer sphere self-gravity

This example creates a Plummer-sphere initial condition, runs a short GPU simulation, and reads back the final state.

```python
import numpy as np
from nbody_streams import run_simulation, Species, make_plummer_sphere

# --- Initial conditions ---
# 5 000 equal-mass particles, total mass 1e9 Msun, scale radius 1 kpc
N = 5_000
xv, masses = make_plummer_sphere(N, M_total=1e9, a=1.0, seed=42)

# --- Species descriptor ---
dm = Species.dark(N=N, mass=1e9 / N, softening=0.1)

# --- Run simulation ---
result = run_simulation(
    xv, [dm],
    time_start=0.0,
    time_end=1.0,          # 1 kpc/(km/s) ~ 1 Gyr
    dt=1e-3,
    architecture='gpu',    # 'cpu' if no GPU
    method='direct',
    output_dir='./output/plummer',
    snapshots=50,
    verbose=True,
)

# result['dark'] is the final (N, 6) phase-space array
print(result['dark'].shape)   # (5000, 6)
```

### CPU fallback (no GPU required)

```python
result = run_simulation(
    xv, [dm],
    time_start=0.0,
    time_end=1.0,
    dt=1e-3,
    architecture='cpu',
    method='direct',        # or 'tree' for falcON O(N) (requires pyfalcon)
    output_dir='./output/plummer_cpu',
    snapshots=10,
)
```

### Reading snapshots back

```python
from nbody_streams import ParticleReader

reader = ParticleReader('./output/plummer/snapshot.h5', verbose=True)
snap = reader.read_snapshot(25)        # snapshot index 25

pos  = snap.species['dark']['posvel'][:, :3]
vel  = snap.species['dark']['posvel'][:, 3:]
mass = snap.species['dark']['mass']

print(f"time = {snap.time}")           # physical time at this snapshot
```

---

## Two-species example: dark matter + stars

```python
import numpy as np
from nbody_streams import run_simulation, Species, make_plummer_sphere

N_dm   = 4_000
N_star = 1_000

xv_dm,   _ = make_plummer_sphere(N_dm,   M_total=1e9, a=2.0)
xv_star, _ = make_plummer_sphere(N_star, M_total=1e7, a=0.3)

xv_all = np.vstack([xv_dm, xv_star])

dm   = Species.dark(N=N_dm,   mass=1e9 / N_dm,   softening=0.2)
star = Species.star(N=N_star, mass=1e7 / N_star, softening=0.05)

result = run_simulation(
    xv_all, [dm, star],
    time_start=0.0, time_end=2.0, dt=2e-3,
    architecture='gpu', method='direct',
    output_dir='./output/two_species', snapshots=100,
)

print(result['dark'].shape)   # (4000, 6)
print(result['star'].shape)   # (1000, 6)
```

---

## Choosing a backend

| Particle count | Recommended | Notes |
|---|---|---|
| N < 500 | `cpu`, `direct` | Lowest latency; no GPU kernel-launch overhead |
| 1K – 100K | `gpu`, `direct` | ~10-100x faster than CPU direct |
| 100K – 500K | `gpu`, `direct` or `gpu`, `tree` | Tree saves GPU memory |
| 500K – 2M | `gpu`, `tree` | O(N log N); requires `make` in `tree_gpu/` |
| N > 2M | `cpu`, `tree` (falcON) | True O(N) FMM |

---

## Next steps

- Full API for `run_simulation` and `Species`: [main.md](main.md)
- Force/potential kernels: [fields.md](fields.md)
- Low-level integrators: [run.md](run.md)
- GPU tree code: [tree_gpu.md](tree_gpu.md)
- HDF5 I/O: [io.md](io.md)
- Stream generation: [fast_sims.md](fast_sims.md)
- Visualization: [viz.md](viz.md)
- Analysis utilities: [utils.md](utils.md)
- Coordinate transforms: [coords.md](coords.md)
- Agama potential helpers: [agama_helper.md](agama_helper.md)
