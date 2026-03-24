# Dynamical friction

Chandrasekhar dynamical friction (DF) models the deceleration a massive
satellite experiences as it moves through a background distribution of lighter
field particles.  In nbody_streams it is implemented as a smooth-field
approximation: the host is represented by an Agama potential (not by live N-body
particles), and the DF force is applied to the satellite centre of mass (CoM) at
every timestep via the `force_extra` hook in the low-level integrators.

---

## Contents

- [When to use DF](#when-to-use-df)
- [Physics and formulae](#physics-and-formulae)
- [sigma(r) computation](#sigmar-computation)
- [Centre-of-mass detection](#centre-of-mass-detection)
- [Usage via run_simulation](#usage-via-run_simulation)
- [Usage via the low-level force_extra hook](#usage-via-the-low-level-force_extra-hook)
- [df_* kwargs reference](#df_-kwargs-reference)
- [Dynamical friction timescales](#dynamical-friction-timescales)
- [Caveats](#caveats)
- [When NOT to use DF](#when-not-to-use-df)

---

## When to use DF

The Chandrasekhar inspiral timescale at orbital radius r scales as

```
t_df  ~  1.17 * (M_host / M_sat) * (r / V_c) / ln(Lambda)
```

| M_sat (M_sun) | Example | t_df at 50 kpc in MW halo | Safe to ignore? |
|---|---|---|---|
| < 1e8 | Globular cluster | > 2000 Gyr | Yes, always |
| 1e8 – 1e9 | Ultra-faint dwarf | 200 – 2000 Gyr | Yes |
| 1e9 – 1e10 | SMC-class | 20 – 200 Gyr | Marginal over few-Gyr runs |
| > 1e10 | LMC-class | < 20 Gyr | No — DF is important |

Use `dynamical_friction=True` whenever `M_sat > ~1e10 M_sun` or when the
simulation spans a significant fraction of `t_df`.

`run_simulation` emits a `PerformanceWarning` when the total satellite mass
exceeds `1e10 M_sun` and `external_potential` is set but `dynamical_friction`
is `False`.

---

## Physics and formulae

### Chandrasekhar formula (BT2008 eq. 8.13)

The deceleration on a satellite of mass `M_sat` moving at velocity `v` through
a background with local density `rho` and velocity dispersion `sigma(r)` is

```
a_DF = -4 pi G^2 M_sat rho ln(Lambda) / v^3
         * [ erf(X) - 2X/sqrt(pi) * exp(-X^2) ] * v_hat
```

where

```
X = v / (sqrt(2) * sigma(r))
```

`erf(X)` and the exponential factor together give the fraction of field
particles slower than the satellite.

### Coulomb logarithm

Two modes are supported, controlled by `df_coulomb_mode`:

- `'variable'` (default): `ln(Lambda) = ln(M_host / M_sat)` evaluated locally
  using the enclosed mass from the potential at the current CoM radius.
- `'fixed'`: `ln(Lambda) = df_fixed_ln_lambda` (a constant; default `3.0`).

The variable mode is generally more physical but requires the potential to
support enclosed-mass queries.

### Core-stalling suppression

In cored potentials, DF stalls near the core because the density of slower
particles drops to zero.  The formula is modified by a suppression factor

```
S = 1 - exp(-(r / r_core)^gamma)
```

controlled by `df_core_gamma` (exponent; default `1.0`) and `df_r_core` (core
radius in kpc; estimated from the potential if `None`).  Set
`df_core_gamma=0.0` to disable the suppression entirely.

---

## sigma(r) computation

The velocity dispersion profile `sigma(r)` is needed to evaluate `X` at each
step.  `_chandrasekhar.compute_sigma_r` builds it in two stages:

1. **Quasispherical DF**: if the potential supports it, the isotropic
   distribution function `f(E)` is computed and integrated to give `sigma^2(r)`
   via the standard Jeans integral.

2. **Jeans-equation fallback**: if the DF route fails (non-spherical potential,
   numerical issues), `_jeans_sigma_r` integrates the spherically-averaged
   Jeans equation outward from a boundary condition at large r:

   ```
   d(rho * sigma^2) / dr + rho * d Phi / dr = 0
   ```

   where `rho(r)` is estimated from the potential via Poisson's equation.

The result is cached for the entire run.  A custom grid can be supplied via
`df_sigma_grid_r`.

---

## Centre-of-mass detection

The DF force depends on the instantaneous satellite CoM position and velocity.
For a live N-body satellite the CoM drifts as particles are stripped, so a
naive mean is biased by unbound tidal tails.

`_chandrasekhar._shrinking_sphere_com` implements an iterative shrinking-sphere
estimator:

1. Start with all particles within a sphere of radius `r_0` (set to the
   half-mass radius).
2. Compute the mean position of particles inside the sphere.
3. Shrink the sphere by a factor `df_shrink_frac` (default `0.7`) around the
   new centre.
4. Repeat for `df_shrink_n_iter` (default `10`) iterations.

`make_df_force_extra` wraps this in a predictor-corrector scheme: the CoM
velocity used in the Chandrasekhar formula is the half-step (drift-midpoint)
estimate, which reduces the phase error from O(dt) to O(dt^2).

---

## Usage via run_simulation

```python
import agama
import numpy as np
from nbody_streams import run_simulation, Species, make_plummer_sphere

agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(type='NFW', mass=1e12, scaleRadius=15.0,
                      outerCutoffRadius=300.0)

# Build satellite ICs (LMC-class, 5e9 Msun)
xv, _ = make_plummer_sphere(10_000, M_total=5e9, a=3.0)
# Offset to 50 kpc with circular-ish velocity
xv[:, 0] += 50.0
xv[:, 4] += 80.0
dm = Species.dark(N=10_000, mass=5e5, softening=0.3)

result = run_simulation(
    xv, [dm],
    time_start=0.0, time_end=5.0, dt=1e-3,
    architecture='gpu', method='direct',
    external_potential=pot,
    dynamical_friction=True,
    df_M_sat=5e9,
    df_coulomb_mode='variable',
    df_core_gamma=1.0,
    df_update_interval=1,
)
# result['dark'] -> (10000, 6) final phase-space
```

`dynamical_friction=True` without `external_potential` raises `ValueError`.

---

## Usage via the low-level force_extra hook

For full control — custom sigma grids, non-standard CoM estimators, or
multi-satellite setups — call `make_df_force_extra` directly and pass the
resulting closure as `force_extra`:

```python
from nbody_streams._chandrasekhar import make_df_force_extra
from nbody_streams import run_nbody_gpu

import agama
agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(type='NFW', mass=1e12, scaleRadius=15.0,
                      outerCutoffRadius=300.0)

df_fn = make_df_force_extra(
    pot,
    M_sat=5e9,
    t_start=0.0,
    t_end=5.0,
    coulomb_mode='variable',
    core_gamma=1.0,
)

final = run_nbody_gpu(
    xv, masses,
    time_start=0.0, time_end=5.0, dt=1e-3,
    softening=0.3,
    external_potential=pot,
    force_extra=df_fn,
    output_dir='./output/df_run',
    snapshots=500,
)
```

On GPU paths `pos` and `vel` passed to `force_extra` are CuPy arrays.
`make_df_force_extra` handles the transfer internally; the returned acceleration
array is a NumPy array transferred back to GPU by the integrator.

---

## df_* kwargs reference

| kwarg | Type | Default | Description |
|---|---|---|---|
| `df_M_sat` | `float` | `None` | Total satellite mass [M_sun]. If `None`, summed from `species` masses |
| `df_coulomb_mode` | `str` | `'variable'` | `'variable'` — ln(Λ) from enclosed mass; `'fixed'` — use `df_fixed_ln_lambda` |
| `df_fixed_ln_lambda` | `float` | `3.0` | Fixed Coulomb logarithm (only when `df_coulomb_mode='fixed'`) |
| `df_core_gamma` | `float` | `1.0` | Core-stalling suppression exponent γ. Set to `0.0` to disable |
| `df_r_core` | `float` | `None` | Core radius [kpc] for stalling suppression. If `None`, estimated from the potential |
| `df_update_interval` | `int` | `1` | Recompute DF force every N steps |
| `df_shrink_n_iter` | `int` | `10` | Number of shrinking-sphere iterations for CoM estimator |
| `df_shrink_frac` | `float` | `0.7` | Sphere-radius shrink factor per iteration |
| `df_sigma_grid_r` | `ndarray` or `None` | `None` | Custom radial grid [kpc] for sigma(r) evaluation. `None` = default log-spaced grid |

---

## Dynamical friction timescales

Reference timescales for a circular orbit at 50 kpc in a Milky-Way-like halo
(M_host = 1e12 M_sun, V_c = 220 km/s, ln(Λ) ~ 3):

| M_sat (M_sun) | Example object | t_df (Gyr) |
|---|---|---|
| 1e7 | Large globular cluster | > 10 000 |
| 1e8 | Small dwarf galaxy | ~ 1 000 |
| 1e9 | SMC-class | ~ 100 |
| 5e9 | Intermediate | ~ 20 |
| 1e10 | LMC-class | ~ 10 |
| 5e10 | Massive satellite | ~ 2 |

---

## Caveats

- **Smooth-field approximation**: DF is derived assuming a smooth, infinite,
  homogeneous background.  Granularity effects (resonant heating, stochastic
  scattering) are not modelled.

- **No tidal stripping feedback**: `df_M_sat` is a fixed input.  As the
  satellite loses mass through tidal stripping, the true DF force weakens; this
  is not automatically accounted for.  For runs where mass loss is significant,
  update `df_M_sat` manually or use a time-varying wrapper around `force_extra`.

- **sigma evaluated at midpoint**: the dispersion `sigma(r)` is sampled at the
  CoM radius at the half-step (predictor CoM).  This is second-order accurate in
  `dt` but may introduce small errors if `sigma(r)` varies sharply.

- **Cost**: the DF force is O(1) per step (one CoM estimation + one Agama density
  call).  Overhead is typically ~1% of the total integration time for GPU direct
  runs with N >= 10 000.

- **CPU path**: on CPU paths `pos`/`vel` are NumPy arrays; `make_df_force_extra`
  works identically.  There is no GPU-specific code inside the closure.

---

## When NOT to use DF

- `M_sat < 1e9 M_sun` and the simulation is shorter than a few Gyr: inspiral
  timescale greatly exceeds the run time; DF is negligible.
- The satellite is not embedded in a smooth potential (`external_potential=None`):
  DF requires a background potential to compute `rho(r)` and `sigma(r)`.
- You are running a pure self-gravity test (Plummer stability, energy
  conservation benchmarks): DF will spuriously decelerate the CoM.
- The satellite orbit is unbound (hyperbolic passage): DF is small and the
  shrinking-sphere CoM estimator may be unreliable.
