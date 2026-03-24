# `nbody_streams.fast_sims` — Fast Stream Generation

Lightweight alternatives to full N-body integration for generating stellar
streams: particle spray and restricted N-body.  Both methods use
[Agama](https://github.com/GalacticDynamics-Oxford/Agama) for the host
potential and orbit integration.

Required dependency: **agama**.  Both functions raise `ImportError` with
install instructions if Agama is not available.

```python
from nbody_streams import fast_sims
```

---

## Public API

### `create_particle_spray_stream`

```python
create_particle_spray_stream(
    pot_host,
    initmass,
    sat_cen_present,
    scaleradius,
    num_particles=10000,
    prog_pot_kind="King",
    time_total=3.0,
    time_end=13.78,
    time_stripping=None,
    save_rate=1,
    dynFric=False,
    pot_for_dynFric_sigma=None,
    gala_modified=True,
    add_perturber=None,
    create_ic_method=create_ic_particle_spray_chen2025,
    verbose=False,
    accuracy_integ=1e-7,
    eigenvalue_method=True,
    **kwargs,
)
```

Create a stellar stream using the particle-spray method.

The progenitor orbit is **rewound** from its present-day phase-space
(`sat_cen_present`) by `time_total` Gyr, then particles are progressively
released at the tidal radius as the satellite evolves forward to `time_end`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pot_host` | `agama.Potential` | Host gravitational potential. |
| `initmass` | float | Initial satellite mass (M_sun, must be > 0). |
| `sat_cen_present` | array_like, shape `(6,)` | Present-day satellite `[x, y, z, vx, vy, vz]` (kpc, km/s). |
| `scaleradius` | float | Satellite scale radius (kpc, must be > 0). |
| `num_particles` | int | Number of stream particles. Default 10000. |
| `prog_pot_kind` | str | Progenitor potential profile: `'King'`, `'Plummer'`, or `'Plummer_withRcut'`. |
| `time_total` | float | Look-back time to rewind the orbit (Gyr, >= 0). Default 3.0. |
| `time_end` | float | Present-day epoch (Gyr). Default 13.78. |
| `time_stripping` | ndarray `(N,)` or None | Custom particle-release times. `N = num_particles // 2 + 1`. Values must lie in `[time_end - time_total, time_end]`. If None, uniform stripping along the orbit. |
| `save_rate` | int | Number of output snapshots (1 = final only). |
| `dynFric` | bool | Enable Chandrasekhar dynamical friction on the progenitor orbit. Default False. |
| `pot_for_dynFric_sigma` | `agama.Potential` or None | Potential for velocity-dispersion computation (DF friction). |
| `gala_modified` | bool | Use Gala-modified dispersion parameters (Fardal method only). Default True. |
| `add_perturber` | dict or None | Perturber properties. Required keys: `'mass'` (M_sun), `'scaleRadius'` (kpc), `'w_subhalo_impact'` (shape `(6,)`), `'time_impact'` (Gyr). The perturber is a truncated NFW on a self-consistent orbit. Set to None to disable. |
| `create_ic_method` | Callable | IC generator function. Defaults to `create_ic_particle_spray_chen2025`. Can be replaced with `create_ic_particle_spray_fardal2015`. |
| `verbose` | bool | Print progress messages. Default False. |
| `accuracy_integ` | float | Orbit integrator accuracy. Default 1e-7. |
| `eigenvalue_method` | bool | Use tidal-tensor eigenvalues for Jacobi radius (more accurate). Default True. |
| `**kwargs` | | Extra arguments for the progenitor potential (e.g. `W0`, `trunc` for King profiles). |

**Returns**

`dict` with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `'times'` | `(Nsaves,)` | Save times (Gyr). |
| `'prog_xv'` | `(Nsaves, 6)` | Progenitor phase-space at each save time. |
| `'part_xv'` | `(Nparticles, Nsaves, 6)` or `(Nparticles, 6)` if `save_rate=1` | Stream particle states. Contains NaN where particles were not yet released. |

**Example**

```python
import agama
import numpy as np
from nbody_streams import fast_sims

agama.setUnits(mass=1, length=1, velocity=1)
pot_host = agama.Potential(type="NFW", mass=1e12, scaleRadius=20)

result = fast_sims.create_particle_spray_stream(
    pot_host,
    initmass=5e8,
    sat_cen_present=[20, 0, 5, -30, 200, 10],
    scaleradius=0.3,
    num_particles=2000,
    time_total=5.0,
    time_end=13.78,
)

stream_pos = result["part_xv"][:, :3]   # save_rate=1, so shape (N, 6)
```

---

### `create_ic_particle_spray_chen2025`

```python
create_ic_particle_spray_chen2025(
    orbit_sat,
    mass_sat,
    rj,
    R,
    G=None,
)
```

Create spray initial conditions using the Chen+2025 correlated phase-space
model.  Covariance matrix and mean offsets are calibrated from N-body
simulations.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `orbit_sat` | ndarray `(N, 6)` | Satellite orbit (kpc, km/s). |
| `mass_sat` | float | Satellite mass (M_sun). |
| `rj` | ndarray `(N,)` | Jacobi radii (kpc). |
| `R` | ndarray `(N, 3, 3)` | Rotation matrices to the satellite frame (rows: radial, azimuthal, angular-momentum). |
| `G` | float or None | Gravitational constant. Defaults to `agama.G`. |

**Returns**

`ndarray, shape (2N, 6)` — initial conditions for leading and trailing stream particles.

---

### `create_ic_particle_spray_fardal2015`

```python
create_ic_particle_spray_fardal2015(
    orbit_sat,
    rj,
    vj,
    R,
    gala_modified=True,
)
```

Create spray initial conditions using the Fardal+2015 method.
Generates asymmetric Gaussian offsets in position and velocity from
the satellite Lagrange points.

Reference: Fardal, M. A., et al. 2015, MNRAS, 452, 301.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `orbit_sat` | ndarray `(N, 6)` | Satellite orbit (kpc, km/s). |
| `rj` | ndarray `(N,)` | Jacobi radii (kpc). |
| `vj` | ndarray `(N,)` | Velocity scales (km/s). |
| `R` | ndarray `(N, 3, 3)` | Rotation matrices to the satellite frame. |
| `gala_modified` | bool | Use Gala's modified dispersion parameters. Default True. |

**Returns**

`ndarray, shape (2N, 6)` — initial conditions for leading/trailing stream particles.

---

### `run_restricted_nbody`

```python
run_restricted_nbody(
    pot_host,
    initmass,
    sat_cen_present,
    scaleradius=None,
    num_particles=10000,
    prog_pot_kind="King",
    xv_init=None,
    dynFric=False,
    pot_for_dynFric_sigma=None,
    time_total=3.0,
    time_end=0.0,
    step_size=10,
    save_rate=300,
    trajsize_each_step=10,
    add_perturber=None,
    verbose=False,
    accuracy_integ=1e-7,
    **kwargs,
)
```

Run a restricted (collisionless) N-body simulation.

Test particles representing the satellite's stars move in the combined
potential of the host galaxy and the evolving satellite.  The satellite
potential is rebuilt from bound particles at every `step_size` integration
steps using a monopole expansion.

By default the progenitor orbit is **rewound** from `sat_cen_present` by
`time_total`; particles are sampled from the chosen profile and integrated
forward.  When `xv_init` is provided the given particles are integrated
forward directly (no rewinding or sampling).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pot_host` | `agama.Potential` | Host gravitational potential. |
| `initmass` | float | Initial satellite mass (M_sun, must be > 0). |
| `sat_cen_present` | array_like, shape `(6,)` | Present-day satellite `[x, y, z, vx, vy, vz]` (kpc, km/s). When `xv_init` is provided this is taken as the progenitor COM. |
| `scaleradius` | float or None | Satellite scale radius (kpc). Ignored when `xv_init` is given. |
| `num_particles` | int | Number of test particles. Default 10000. |
| `prog_pot_kind` | str | Progenitor potential profile: `'King'`, `'Plummer'`, or `'Plummer_withRcut'`. |
| `xv_init` | ndarray `(N, 6)` or None | Pre-existing particle phase-space. When provided, integrates forward directly without rewinding or sampling. |
| `dynFric` | bool | Enable dynamical friction on the progenitor orbit. Default False. |
| `pot_for_dynFric_sigma` | `agama.Potential` or None | Potential for velocity-dispersion computation (DF friction). |
| `time_total` | float | Look-back time / integration duration (Gyr, >= 0). Default 3.0. |
| `time_end` | float | Present-day epoch (Gyr). Default 0.0. |
| `step_size` | int | Number of ODE steps grouped per potential-update iteration. Default 10. |
| `save_rate` | int | Number of interpolated output snapshots (1 = final only). Default 300. |
| `trajsize_each_step` | int | Trajectory points saved per integration step. Default 10. |
| `add_perturber` | dict or None | Perturber properties (same format as `create_particle_spray_stream`). |
| `verbose` | bool | Print progress messages. Default False. |
| `accuracy_integ` | float | Orbit integrator accuracy. Default 1e-7. |
| `**kwargs` | | Extra arguments for the progenitor potential (e.g. `W0`, `trunc` for King). |

**Returns**

`dict` with keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `'times'` | `(Nsaves,)` | Saved snapshot times (Gyr). |
| `'prog_xv'` | `(Nsaves, 6)` | Progenitor phase-space at saved times. |
| `'part_xv'` | `(N, Nsaves, 6)` | Particle phase-space at saved times. |
| `'bound_mass'` | `(Nsaves,)` | Bound mass at each saved time (M_sun). |

**Example**

```python
from nbody_streams import fast_sims

result = fast_sims.run_restricted_nbody(
    pot_host,
    initmass=5e8,
    sat_cen_present=[20, 0, 5, -30, 200, 10],
    scaleradius=0.3,
    num_particles=5000,
    time_total=5.0,
    save_rate=100,
)

# Access final snapshot particle positions
final_pos = result["part_xv"][:, -1, :3]
```

---

## Perturber potential

Both `create_particle_spray_stream` and `run_restricted_nbody` accept an
`add_perturber` dict.  The perturber is placed on a self-consistent orbit
in `pot_host`: it is rewound from its impact phase-space to the simulation
start and integrated forward.  By default a truncated NFW profile is used
(spheroid with outer cutoff at 10 scale radii).

Required dict keys:

```python
add_perturber = {
    "mass": 1e9,              # M_sun
    "scaleRadius": 1.0,       # kpc
    "w_subhalo_impact": [...], # shape (6,) phase-space at impact
    "time_impact": 11.5,       # Gyr (epoch of closest approach)
}
```

---

## Dynamical friction

Set `dynFric=True` to apply Chandrasekhar dynamical friction to the
progenitor orbit.  The Coulomb logarithm is computed in `'variable'` mode
by default (`ln_Lambda = log(b_max / b_min)` where `b_max = r`,
`b_min = G M / v^2`).  Provide `pot_for_dynFric_sigma` if a specific
potential should be used for the velocity-dispersion profile; otherwise
a Milky Way fallback profile is used.
