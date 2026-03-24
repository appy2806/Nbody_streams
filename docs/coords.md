# `nbody_streams.coords` — Coordinate Transforms

Cartesian/spherical/cylindrical position and vector field conversions,
stream-aligned coordinate generation, and conversion to observable sky
coordinates.

```python
from nbody_streams import coords
```

Optional dependencies:

- **agama** — required for `get_observed_stream_coords` (galactocentric-to-ICRS transformation).
- **scipy** — required for `generate_stream_coords` with `optimizer_fit=True`.

---

## Coordinate conventions

| System | Components | Notes |
|--------|-----------|-------|
| Cartesian (`cart`) | (x, y, z) | |
| Spherical (`sph`) | (rho, theta, phi) | rho = radial distance; theta = polar angle (colatitude) from +z, in [0, pi]; phi = azimuthal angle, [0, 2*pi) or (-pi, pi] with `mollweide=True` |
| Cylindrical (`cyl`) | (R, phi, z) | R = sqrt(x^2 + y^2); phi = azimuthal angle [0, 2*pi); z = height |

---

## `convert_coords`

```python
convert_coords(data, from_sys, to_sys, *, mollweide=False)
```

Convert positions between coordinate systems.  Supports arbitrary leading
batch dimensions: input shape `(..., 3)`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | array_like `(..., 3)` | Input coordinates. |
| `from_sys` | `'cart'`, `'sph'`, or `'cyl'` | Source coordinate system. |
| `to_sys` | `'cart'`, `'sph'`, or `'cyl'` | Target coordinate system. |
| `mollweide` | bool | Only relevant for cart<->sph conversions. If True, phi wraps to (-pi, pi] (Healpy convention). Default False. |

**Returns** `ndarray, shape (..., 3)` — converted coordinates.

**Supported pairs**: cart<->sph, cart<->cyl, sph<->cyl.  sph<->cyl is routed
through Cartesian internally.

**Example**

```python
from nbody_streams import coords
import numpy as np

pos_cart = np.random.randn(1000, 3) * 10

pos_sph = coords.convert_coords(pos_cart, "cart", "sph")
pos_cyl = coords.convert_coords(pos_cart, "cart", "cyl")
pos_back = coords.convert_coords(pos_sph, "sph", "cart")

# Mollweide (healpy) phi convention: (-pi, pi]
pos_sph_moll = coords.convert_coords(pos_cart, "cart", "sph", mollweide=True)
```

---

## `convert_vectors`

```python
convert_vectors(pos, vec, from_sys, to_sys)
```

Rotate vector fields between coordinate systems.  Converts both the
positions and the associated vectors (velocities, forces, etc.) simultaneously.

The rotation is performed by building the point-wise Jacobian matrix for each
particle and applying it to the vector field.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pos` | array_like `(..., 3)` | Positions in the source coordinate system. |
| `vec` | array_like `(..., 3)` | Vector field values in the source coordinate system. |
| `from_sys` | `'cart'`, `'sph'`, or `'cyl'` | Source coordinate system. |
| `to_sys` | `'cart'`, `'sph'`, or `'cyl'` | Target coordinate system. |

**Returns** `(pos_new, vec_new)` — both as ndarray `(..., 3)`.

**Example**

```python
pos_sph, vel_sph = coords.convert_vectors(pos_cart, vel_cart, "cart", "sph")
# vel_sph components: (v_rho, v_theta, v_phi)

pos_cyl, F_cyl = coords.convert_vectors(pos_cart, F_cart, "cart", "cyl")
# F_cyl components: (F_R, F_phi, F_z)
```

---

## `convert_to_vel_los`

```python
convert_to_vel_los(xv, reference_xv=None)
```

Compute the line-of-sight (radial) velocity by projecting the velocity
vector onto the unit radial direction: `v_los = v . r_hat`.

Optionally subtracts a reference phase-space vector first (e.g. the
progenitor centre-of-mass).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `xv` | array_like `(..., 6)` | Phase-space coordinates `[x, y, z, vx, vy, vz]`. Positions in kpc, velocities in km/s. Accepts `(6,)`, `(N, 6)`, `(M, N, 6)`, or any higher batch dimensions. |
| `reference_xv` | array_like or None | Reference phase-space point(s) to subtract before computing v_los. Must be broadcastable to the shape of `xv`. Default None (no subtraction). |

**Returns** `float` or `ndarray` — line-of-sight velocity in km/s.  Scalar when input is `(6,)`, otherwise shape `(...)`.

**Raises**

`ValueError` if position vectors have zero magnitude.

**Example**

```python
from nbody_streams import coords
import numpy as np

xv = np.array([8.0, 0.0, 0.0, 0.0, 220.0, 0.0])
v_los = coords.convert_to_vel_los(xv)   # -> 0.0 (circular orbit, tangential velocity)

xv_batch = np.random.randn(100, 6)
v_los_batch = coords.convert_to_vel_los(xv_batch)  # shape (100,)

# Subtract progenitor frame before computing v_los
xv_stream = np.random.randn(5, 1000, 6)
xv_prog   = np.random.randn(5, 1, 6)
v_los_rel = coords.convert_to_vel_los(xv_stream, reference_xv=xv_prog)  # shape (5, 1000)
```

---

## `generate_stream_coords`

```python
generate_stream_coords(
    xv,
    xv_prog=None,
    degrees=True,
    optimizer_fit=False,
    fit_kwargs=None,
)
```

Convert galactocentric phase-space into stream-aligned coordinates (phi1,
phi2) for single or multiple streams.

The stream frame is defined by the angular momentum vector of the progenitor
orbit.  Particles are projected into a rotated frame where phi1 runs along
the stream and phi2 measures the perpendicular offset.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `xv` | ndarray `(N, 6)` or `(S, N, 6)` | Galactocentric phase-space. S = number of streams / time steps, N = particles. |
| `xv_prog` | ndarray `(6,)`, `(S, 6)`, or None | Progenitor phase-space vector(s). If None, the particle closest to the median position of each stream is used. |
| `degrees` | bool | Return angles in degrees; otherwise radians. Default True. |
| `optimizer_fit` | bool | If True, apply a scipy.optimize rotation in the phi1-phi2 plane to minimise the spread in phi2 (aligns the stream along phi1). |
| `fit_kwargs` | dict or None | Extra kwargs forwarded to `scipy.optimize.minimize` when `optimizer_fit=True`. |

**Returns**

- `phi1` — ndarray, shape `(N,)` or `(S, N)`. Stream longitude.
- `phi2` — ndarray, shape `(N,)` or `(S, N)`. Stream latitude.

**Example**

```python
from nbody_streams import coords
import numpy as np

xv = np.random.randn(1000, 6)
xv[:, :3] *= 20   # kpc
xv[:, 3:] *= 100  # km/s

xv_prog = xv[0]
phi1, phi2 = coords.generate_stream_coords(xv, xv_prog, degrees=True)
```

---

## `get_observed_stream_coords`

```python
get_observed_stream_coords(
    xv,
    xv_prog=None,
    degrees=True,
    optimizer_fit=False,
    fit_kwargs=None,
    galcen_distance=8.122,
    galcen_v_sun=(12.9, 245.6, 7.78),
    z_sun=0.0208,
)
```

Convert galactocentric phase-space coordinates to observed sky coordinates
(RA, Dec, v_los) and stream coordinates (phi1, phi2).

Uses Agama for the galactocentric -> galactic -> ICRS transformation.
Requires **agama**.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `xv` | ndarray `(N, 6)` or `(S, N, 6)` | Galactocentric Cartesian phase-space [x, y, z, vx, vy, vz]. Positions in kpc, velocities in km/s. |
| `xv_prog` | ndarray or None | Progenitor phase-space. See `generate_stream_coords`. |
| `degrees` | bool | Return angles in degrees. Default True. |
| `optimizer_fit` | bool | Optimise the stream-frame rotation. Default False. |
| `galcen_distance` | float | Sun-Galactic-centre distance in kpc. Default 8.122. |
| `galcen_v_sun` | tuple | Solar motion (U, V, W) in km/s. Default (12.9, 245.6, 7.78). |
| `z_sun` | float | Height of the Sun above the midplane in kpc. Default 0.0208. |

**Returns**

`(ra, dec, v_los, phi1, phi2)` — all as ndarray.  Shape `(N,)` for a single
stream, `(S, N)` for a batch.

**Raises**

`ImportError` if Agama is not installed.

**Example**

```python
from nbody_streams import coords

ra, dec, v_los, phi1, phi2 = coords.get_observed_stream_coords(
    xv_stream,
    xv_prog=xv_prog,
    degrees=True,
    galcen_distance=8.122,
)

import matplotlib.pyplot as plt
plt.plot(ra, dec, "k.", ms=0.5)
plt.xlabel("RA [deg]")
plt.ylabel("Dec [deg]")
```
