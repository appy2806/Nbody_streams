# nbody_streams

Direct N-body simulator and utilities for collisionless systems (single particle type).
Designed for research and prototyping with a minimal API and optional GPU acceleration.

- Direct N-body code with both CPU and optional GPU implementations.
- GPU backend uses custom CUDA kernels (CuPy) and Kahan-style corrections for improved float32 accuracy.
- CPU fallback uses NumPy/Numba where available.
- Optional tree/FMM backend (pyfalcon) and optional external potentials via Agama.
- Intended for collisionless systems with up to ~100k particles (benchmarks depend on hardware).
- Fast stream-generation methods (particle spray, restricted N-body) via AGAMA.

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e .          # editable (development) install

# Optional extras
pip install nbody_streams[gpu]     # CuPy for CUDA kernels
pip install nbody_streams[agama]   # AGAMA for fast_sims & external potentials
pip install nbody_streams[falcon]  # pyfalcon tree/FMM backend
pip install nbody_streams[healpy]  # HEALPix for mollweide projections
```

---

## Package overview

| Subpackage | Description |
|---|---|
| `nbody_streams.fields` | Direct-force kernels (GPU and CPU) |
| `nbody_streams.run` | Leapfrog (KDK) integrator with optional external potentials |
| `nbody_streams.io` | `ParticleReader` for HDF5 snapshots, save/load helpers |
| `nbody_streams.utils` | Profile fitting (Dehnen, Plummer, double-power-law), iterative shape measurement, energy-based unbinding |
| `nbody_streams.fast_sims` | Fast stream generation: particle spray and restricted N-body (requires AGAMA) |
| `nbody_streams.coords` | Coordinate transforms and stream-to-observable conversions |
| `nbody_streams.viz` | Mollweide projections, surface density, stream evolution plots |
| `nbody_streams.cuda_kernels` | CUDA kernel templates used by CuPy |

---

## API

### Direct N-body (`fields`, `run`)

```python
from nbody_streams.fields import compute_nbody_forces_gpu, compute_nbody_potential_gpu
from nbody_streams.run import run_leapfrog

# Compute accelerations (GPU, float32 with Kahan correction)
acc = compute_nbody_forces_gpu(
    pos, masses, softening=0.01,
    precision='float32_kahan', kernel='spline',
)

# Full integration with external potential
run_leapfrog(
    pos, vel, masses, dt=0.001, n_steps=1000,
    softening=0.01, precision='float32',
    ext_potential=agama_potential,    # optional
    snapshot_interval=100,
    output_file='sim.h5',
)
```

### Snapshots (`io`)

```python
from nbody_streams.io import ParticleReader

r = ParticleReader("path/to/sim.*.h5", verbose=True)
snap = r.read_snapshot(10)  # returns dict with pos, vel, mass, time
```

### Fast stream generation (`fast_sims`)

All methods require [AGAMA](https://github.com/GalacticDynamics-Oxford/Agama) (`pip install agama`).

#### Particle spray

```python
import agama
agama.setUnits(mass=1, length=1, velocity=1)

from nbody_streams.fast_sims import create_particle_spray_stream

pot = agama.Potential("MWPotential.ini")

result = create_particle_spray_stream(
    pot_host=pot,
    initmass=20_000,                    # Msun
    sat_cen_present=[-10.9, -3.4, -22.2, 70.4, 188.6, 95.8],  # [kpc, km/s]
    scaleradius=0.01,                   # kpc
    num_particles=10_000,
    time_total=3.0,                     # Gyr lookback
    time_end=13.78,                     # Gyr present epoch
    save_rate=5,                        # number of output snapshots
    prog_pot_kind='Plummer',            # or 'King', 'Plummer_withRcut'
)
# result['times']   — (5,) save times
# result['prog_xv'] — (5, 6) progenitor trajectory
# result['part_xv'] — (N, 5, 6) stream particles at each snapshot
```

**Custom stripping times** (e.g. pericenter-weighted / episodic release):

```python
N = num_particles // 2 + 1
t_strip = my_pericenter_times(N)  # length N, within [time_end - time_total, time_end]

result = create_particle_spray_stream(
    ...,
    time_stripping=t_strip,  # duplicates are handled automatically
)
```

**Fardal+2015 IC method:**

```python
from nbody_streams.fast_sims import (
    create_particle_spray_stream,
    create_ic_particle_spray_fardal2015,
)

result = create_particle_spray_stream(
    ...,
    create_ic_method=create_ic_particle_spray_fardal2015,
)
```

**Adding a subhalo perturber:**

```python
result = create_particle_spray_stream(
    ...,
    add_perturber={
        'mass': 1e8,              # Msun
        'scaleRadius': 0.5,       # kpc (NFW)
        'w_subhalo_impact': [x, y, z, vx, vy, vz],  # impact phase-space
        'time_impact': 1.5,       # Gyr ago
    },
)
```

#### Restricted N-body

```python
from nbody_streams.fast_sims import run_restricted_nbody

result = run_restricted_nbody(
    pot_host=pot,
    initmass=20_000,
    sat_cen_present=[-10.9, -3.4, -22.2, 70.4, 188.6, 95.8],
    scaleradius=0.01,
    num_particles=10_000,
    time_total=3.0,
    time_end=13.78,
    step_size=50,           # ODE steps per potential update
    save_rate=5,
    trajsize_each_step=5,
    prog_pot_kind='Plummer',
    dynFric=True,           # optional dynamical friction
)
# result['times']      — save times
# result['prog_xv']    — progenitor trajectory
# result['part_xv']    — particle phase-space
# result['bound_mass'] — bound mass at each save time
```

**Pre-existing particles** (skip rewinding/sampling):

```python
result = run_restricted_nbody(
    pot_host=pot,
    initmass=20_000,
    sat_cen_present=com_xv,
    xv_init=my_particles,   # shape (N, 6)
    time_total=0.5,
    time_end=13.78,
    ...
)
```

### Analysis utilities (`utils`)

```python
from nbody_streams.utils import (
    empirical_density_profile,
    empirical_circular_velocity_profile,
    empirical_velocity_dispersion_profile,
    fit_dehnen_profile,
    fit_plummer_profile,
    fit_iterative_ellipsoid,
    compute_iterative_boundness,
)

# Radial profiles
r_bins, rho = empirical_density_profile(positions, masses, bins=50)

# Profile fitting
gamma, a, M, r_fit, rho_fit = fit_dehnen_profile(positions, masses)
a, M, r_fit, rho_fit = fit_plummer_profile(positions, masses)

# Iterative shape measurement
axes, axis_ratios, eigvecs = fit_iterative_ellipsoid(positions, masses)

# Energy-based unbinding
bound_mask = compute_iterative_boundness(positions, velocities, masses, potential)
```

### Coordinates (`coords`)

```python
from nbody_streams.coords import (
    galactocentric_to_galactic,
    compute_stream_coords,
    compute_proper_motions,
)

l, b, d = galactocentric_to_galactic(x, y, z)
phi1, phi2 = compute_stream_coords(l, b, pole_l, pole_b)
```

### Visualization (`viz`)

```python
from nbody_streams.viz import (
    plot_mollweide,
    plot_density_map,
    plot_stream_evolution,
)

fig, ax = plot_mollweide(l, b, nside=64)
fig, ax = plot_density_map(x, y, masses, bins=200)
```

---

## Performance

GPU benchmarks (N = 10,240 particles, RTX 3080):

| Kernel | Time/Step | Throughput | Energy Conservation |
|--------|-----------|------------|---------------------|
| Float32 + float4 | **2.3 ms** | 45.7 Gint/s | < 0.001% |
| Float32_kahan + float4 | 2.4 ms | 44.2 Gint/s | < 0.001% |
| Float64 | 23.4 ms | 4.5 Gint/s | < 0.001% |

All precision modes achieve memory bandwidth utilization of **99% of theoretical peak** (760 GB/s).

---

<details>
<summary><strong>Float32 precision analysis (Newton's 3rd Law at small scales)</strong></summary>

### Summary

Float32 kernels are mathematically correct but suffer from precision loss when particle positions are at small scales (< 0.1 length units). This manifests as violation of Newton's 3rd Law at the ~1% level for scales around 0.01.

**Important:** Even without scaling, the "asymmetric" float32 forces **do not cause energy drift** over long integrations because:
1. Integration is performed in float64 (positions, velocities)
2. Force errors are small and largely random (not systematic)
3. Symplectic integrators are robust to small force errors

### Scale-dependent error

| Scale Factor | Typical |r| | Float32 Net Force | Status |
|---|---|---|---|
| 0.1 | 10 | 6.4e-9 | Excellent |
| 1 | 1.0 | 1.4e-6 | Good |
| 10 | 0.1 | 1.2e-4 | Marginal |
| 100 | 0.01 | 1.0e-2 | Broken |

### Solutions

**Option 1 — Unit scaling (recommended):** Rescale positions to O(1) before computing forces. Gives perfect Newton's 3rd Law + full float32 speed.

```python
scale = prog_scaleradius  # e.g. 0.01 kpc
pos_scaled = pos / scale
acc_scaled = compute_nbody_forces_gpu(pos_scaled.astype(np.float32), ...)
acc_physical = acc_scaled / (scale**2)
```

**Option 2 — Float32 as-is:** Simplest code, energy still conserved to < 0.001%. Newton's 3rd Law violated by ~1% (cosmetic only).

**Option 3 — Float64:** Accurate at all scales, ~10x slower. Best for small N (< 5000).

### Comparison: direct N-body vs tree methods

| Method | Force Accuracy | Energy Conservation | Speed |
|---|---|---|---|
| Direct N^2 (float32) | ~1% asymmetry | < 0.001% drift | 2.3 ms/step |
| Tree (float32, theta=0.5) | 1-5% force errors | ~0.01-0.1% drift | 5 ms/step |
| Tree (float64, theta=0.5) | 1-5% force errors | ~0.001% drift | 10 ms/step |

Direct N-body with float32 gives **better energy conservation** than tree methods because tree approximation errors are systematic (monopole bias), while float32 rounding errors are random and cancel statistically.

</details>

---

## License

MIT
