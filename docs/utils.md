# `nbody_streams.utils` — Analysis Utilities

Utility functions for particle-based analysis of N-body simulations.

```python
from nbody_streams import utils
```

Optional dependencies:

- **scipy** — required for profile fitting, boundness, and velocity-dispersion functions.
- **numba** — accelerates structure-tensor and distance computations; pure NumPy fallback is used when absent.
- **agama** — used by `fit_double_spheroid_profile` (Agama Potential fit path) and `find_center` / `iterative_unbinding` (density-peak solver).

---

## Grid / binning helpers

### `make_uneven_grid`

```python
make_uneven_grid(xmin, xmax=None, nbins=10)
```

Create a 1-D grid with unequally spaced nodes.  The grid starts at 0, the
second node sits at `xmin`, and the last node sits at `xmax`.  Spacing grows
geometrically.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `xmin` | float | Location of the first non-zero node (must be > 0). |
| `xmax` | float or None | Location of the last node (must be > `xmin`). If None, returns a uniform grid with spacing `xmin`. |
| `nbins` | int | Total number of bins (>= 3). |

**Returns** `ndarray, shape (nbins,)`

---

## Empirical radial profiles

All profile functions bin particles using `make_uneven_grid` (log-spaced
shells) and return `(radius, quantity)` arrays at bin centres.

### `empirical_density_profile`

```python
empirical_density_profile(pos, mass, nbins=50, rmin=0.1, rmax=600)
```

Compute the mass-density radial profile rho(r).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | array_like `(N, 3)` or `(N,)` | Particle positions (Cartesian) or pre-computed radii. |
| `mass` | scalar or array_like `(N,)` | Particle masses. |
| `nbins` | int | Number of radial bins. Default 50. |
| `rmin`, `rmax` | float | Inner / outer grid nodes. |

**Returns** `(radius, density)` — bin centres and mass density in each shell.

---

### `empirical_circular_velocity_profile`

```python
empirical_circular_velocity_profile(pos, mass, nbins=50, rmin=0.1, rmax=600, G=G_DEFAULT)
```

Compute v_circ(r) = sqrt(G * M(<r) / r).

**Returns** `(radius, v_circ)` — bin centres and circular velocity in km/s.

---

### `empirical_velocity_dispersion_profile`

```python
empirical_velocity_dispersion_profile(pos, vel, nbins=50, rmin=0.1, rmax=600)
```

Compute the velocity dispersion sigma_v(r) (standard deviation of speed in
each radial bin).

**Parameters**: `pos` (positions or radii), `vel` (velocities or speed magnitudes).

**Returns** `(radius, vel_disp)`

---

### `empirical_velocity_rms_profile`

```python
empirical_velocity_rms_profile(pos, vel, nbins=50, rmin=0.1, rmax=600)
```

Compute the root-mean-square velocity v_rms(r).

**Returns** `(radius, vel_rms)`

---

### `empirical_velocity_anisotropy_profile`

```python
empirical_velocity_anisotropy_profile(pos, vel, mass=None, nbins=50, rmin=0.1, rmax=None)
```

Compute the velocity anisotropy parameter beta(r) = 1 - sigma_t^2 / (2 * sigma_r^2).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | array_like `(N, 3)` | Particle positions (Cartesian, required for radial/tangential decomposition). |
| `vel` | array_like `(N, 3)` | Particle velocities. |
| `mass` | scalar, array_like, or None | Particle masses. None gives equal mass. |
| `rmax` | float or None | Maximum radius. None -> 90th percentile of `|pos|`. |

**Returns** `(r_centres, beta)`

---

## Density-profile fitting

### `fit_double_spheroid_profile`

```python
fit_double_spheroid_profile(
    r_centers=np.array([]),
    rho_vals=np.array([]),
    pos=np.array([]),
    mass=np.array([]),
    bins=20,
    axis_y=1.0,
    axis_z=1.0,
    weighting="uniform",
    plot_results=False,
    return_profiles=False,
    rcut=None,
    cutoff_strength=2.0,
)
```

Fit a Zhao (1996) double-power-law (spheroid) density profile.

If `r_centers` and `rho_vals` are not supplied, the radial density profile
is estimated from `pos` and `mass` using ellipsoidal radii
`r_tilde = sqrt(x^2 + (y/axis_y)^2 + (z/axis_z)^2)`.

When Agama is available, the fit uses `agama.Potential(type='Spheroid')`.
Otherwise a pure-Python double-power-law model is used.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `r_centers`, `rho_vals` | array_like | Pre-computed radial profile. If empty, derived from `pos` and `mass`. |
| `pos` | array_like `(N, 3)` | Particle positions. |
| `mass` | scalar or array_like | Particle masses. |
| `bins` | int | Number of radial bins when estimating density from particles. |
| `axis_y`, `axis_z` | float | Axis ratios b/a and c/a for ellipsoidal radius. Default 1.0 (spherical). |
| `weighting` | str or array_like | Weight scheme: `'uniform'`, `'inner'`, `'outer'`, `'sqrt'`, `'inverse_sqrt'`, or custom array. |
| `plot_results` | bool | Show diagnostic plots (requires matplotlib). |
| `return_profiles` | bool | If True, also return raw profile arrays. |
| `rcut` | float or None | Optional outer truncation radius. |
| `cutoff_strength` | float | Steepness of the exponential cutoff. Default 2.0. |

**Returns**

`(M_fit, a_fit, alpha_fit, beta_fit, gamma_fit)` — or additionally
`(r_centers, rho_vals, rho_residuals, r2_rho_vals)` when `return_profiles=True`.

---

### `fit_dehnen_profile`

```python
fit_dehnen_profile(pos, mass, axis_y=1.0, axis_z=1.0, bins=50)
```

Fit a triaxial Dehnen (1993) density profile using log-spaced radial shells
and `scipy.optimize.curve_fit`.

**Returns** `(M_fit, a_fit, gamma_fit, r_centers, rho_vals)` where `gamma_fit`
is the inner slope (0 <= gamma < 3).

---

### `fit_plummer_profile`

```python
fit_plummer_profile(pos, mass, bins=30)
```

Fit a spherical Plummer profile rho(r) = 3M/(4*pi*b^3) * (1 + r^2/b^2)^(-5/2)
in log-space via `scipy.optimize.curve_fit`.

**Returns** `(M_fit, b_fit, r_centers, rho_vals)` where `b_fit` is the Plummer
scale radius.

---

## Morphological diagnostics

### `fit_iterative_ellipsoid`

```python
fit_iterative_ellipsoid(
    pos,
    mass=None,
    vel=None,
    Rmin=0.0,
    Rmax=1.0,
    reduced_structure=True,
    orient_with_momentum=True,
    tol=1e-4,
    max_iter=50,
    verbose=False,
    return_ellip_triax=False,
)
```

Fit an adaptive ellipsoid to a particle distribution and compute shape
diagnostics.  Iteratively selects particles inside an adaptive ellipsoid and
diagonalises the (optionally reduced/weighted) structure tensor.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | array_like `(N, 3)` | Particle coordinates. |
| `mass` | array_like `(N,)` or None | Particle masses. None -> unit mass. |
| `vel` | array_like `(N, 3)` or None | Particle velocities. Required when `orient_with_momentum=True`. |
| `Rmin`, `Rmax` | float | Inner / outer radii for the initial spherical selection. |
| `reduced_structure` | bool | Use iterative reduced inertia tensor (weights 1/R^2_sph). Default True. |
| `orient_with_momentum` | bool | Align the minor axis with the angular-momentum vector each iteration. Default True. |
| `tol` | float | Convergence tolerance on axis-ratio changes. Default 1e-4. |
| `max_iter` | int | Maximum iterations. Default 50. |
| `return_ellip_triax` | bool | If True, also return ellipticity and triaxiality. |

**Returns**

- `abc` — ndarray `(3,)`, normalised semi-axis lengths `[1, b/a, c/a]`.
- `transform` — ndarray `(3, 3)`, rows are unit eigenvectors `[e_a, e_b, e_c]`.
- `ellip` — float `1 - c/a` (only when `return_ellip_triax=True`).
- `triax` — float `(a^2-b^2)/(a^2-c^2)` (only when `return_ellip_triax=True`).

---

## Spherical grid generators

### `uniform_spherical_grid`

```python
uniform_spherical_grid(radius=1.0, num_pts=500)
```

Generate uniformly random points on the surface of a sphere.

**Returns** `ndarray, shape (num_pts, 3)` — Cartesian coordinates.

---

### `spherical_spiral_grid`

```python
spherical_spiral_grid(radius=1.0, proj="Cart")
```

Load a pre-defined spherical spiral grid from
`nbody_streams/data/spherical_grid_unit.xyz`, scaled to `radius`.

**Parameters**

| `proj` | Output coordinate system: `'Cart'` (Cartesian), `'Sph'` (spherical `(r, theta, phi)`), or `'Cyl'` (cylindrical `(R, phi, z)`). |

**Returns** `ndarray, shape (M, 3)`

---

## Centre finding

### `find_center`

```python
find_center(
    pos,
    mass=None,
    vel=None,
    method="density_peak",
    return_velocity=False,
    vel_aperture=5.0,
    **kwargs,
)
```

Find the centre (and optionally the velocity centre) of a particle distribution.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | array_like `(N, 3)` or `(N, 6)` | Particle positions. If shape `(N, 6)`, velocities are split internally. |
| `mass` | scalar, array_like, or None | Particle masses. |
| `vel` | array_like `(N, 3)` or None | Required when `return_velocity=True`. |
| `method` | str | Algorithm: `'density_peak'`, `'shrinking_sphere'`, or `'kde'`. |
| `return_velocity` | bool | If True, also return the velocity centre. |
| `vel_aperture` | float | Aperture radius (kpc) for velocity centering. Default 5.0. |
| `**kwargs` | | Method-specific options (see below). |

**Method options**

- `'density_peak'`: gravitational potential minimum. Tries GPU tree (`tree_gpu`), then pyfalcon, then agama — raises `ImportError` if none are available. Options: `softening` (0.1 kpc), `top_fraction` (0.01), `theta` (0.4), `lmax` (8, agama fallback only).
- `'shrinking_sphere'`: iterative shrinking sphere. Options: `r_init` (30), `shrink_factor` (0.9), `min_particles` (100).
- `'kde'`: Gaussian KDE density peak. Slow for large N.

**Returns**

`centre_pos` — ndarray `(3,)`, or `(centre_pos, centre_vel)` when `return_velocity=True`.

**Example**

```python
from nbody_streams import utils
import numpy as np

pos = np.random.randn(10_000, 3) * 5
mass = np.ones(10_000)

centre = utils.find_center(pos, mass, method="shrinking_sphere")
centre_pos, centre_vel = utils.find_center(
    pos, mass, vel=np.random.randn(10_000, 3),
    method="shrinking_sphere", return_velocity=True,
)
```

### `find_center_position` (deprecated)

Deprecated alias for `find_center`. Use `find_center` instead.

---

## Iterative boundness

### `iterative_unbinding`

```python
iterative_unbinding(
    pos_dark,
    vel_dark,
    mass_dark,
    pos_star=None,
    vel_star=None,
    mass_star=None,
    center_position=[],
    center_velocity=[],
    recursive_iter_converg=50,
    potential_compute_method="tree",
    softening=0.03,
    G=G_DEFAULT,
    center_on="dark",
    vel_aperture=5.0,
    tol_frac_change=0.0001,
    verbose=True,
    return_history=False,
    **kwargs,
)
```

Iteratively determine bound particles by computing the gravitational
potential, evaluating total energy E = phi + 0.5 * v^2 per particle, and
removing unbound (E > 0) particles until convergence.

Supports multi-component systems (dark matter + stars).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos_dark`, `vel_dark` | array_like `(N_d, 3)` | Dark-matter positions (kpc) and velocities (km/s). |
| `mass_dark` | scalar or array_like | Dark-matter masses (M_sun). |
| `pos_star`, `vel_star`, `mass_star` | optional | Stellar component (same conventions). |
| `center_position`, `center_velocity` | array_like `(3,)` | Pre-computed centre. If empty, determined automatically. |
| `recursive_iter_converg` | int | Maximum iterations. Default 50. |
| `potential_compute_method` | str | Potential solver: `'tree'` (pyfalcon), `'tree_gpu'` (GPU Barnes-Hut), `'bfe'` (agama multipole), `'direct'` (O(N^2) CPU), `'direct_gpu'` (GPU direct). |
| `softening` | float | Gravitational softening (kpc). Default 0.03. |
| `G` | float | Gravitational constant. |
| `center_on` | `'dark'`, `'star'`, or `'both'` | Component used for position/velocity centering. |
| `vel_aperture` | float | Aperture radius (kpc) for velocity centering. Default 5.0. |
| `tol_frac_change` | float | Convergence tolerance on bound-fraction change. Default 0.0001. |
| `verbose` | bool | Print diagnostics. Default True. |
| `return_history` | bool | Return per-iteration bound masks. |
| `**kwargs` | | Extra keyword arguments forwarded to the potential solver (e.g. `theta=0.4`, `lmax=8`). |

**Returns**

`(results, center_position, center_velocity)` where:

- `results` is `(bound_dark,)` or `(bound_dark, bound_star)` — integer masks (1 = bound).

**Example**

```python
from nbody_streams import utils
import numpy as np

pos = np.random.randn(10_000, 3) * 5
vel = np.random.randn(10_000, 3) * 50
mass = np.ones(10_000) * 1e4

(bound,), cpos, cvel = utils.iterative_unbinding(
    pos, vel, mass,
    potential_compute_method="tree_gpu",
    softening=0.1,
)
print(f"Bound fraction: {bound.mean():.3f}")
```

### `compute_iterative_boundness` (deprecated)

Deprecated alias for `iterative_unbinding`. Use `iterative_unbinding` instead.

---

## `G_DEFAULT`

```python
from nbody_streams.utils.main import G_DEFAULT  # 4.300917e-6 kpc (km/s)^2 / Msun
```
