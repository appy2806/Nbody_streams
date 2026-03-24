# `nbody_streams.fields`

Force and potential kernels for direct N-body gravitational calculations.  Provides GPU (CuPy + CUDA RawKernel) and CPU (Numba `@njit(parallel=True)`) backends with identical Python interfaces.

```python
from nbody_streams import (
    compute_nbody_forces_gpu,
    compute_nbody_forces_cpu,
    compute_nbody_potential_gpu,
    compute_nbody_potential_cpu,
    get_gpu_info,
)
```

> **Optional dependencies:** GPU functions require CuPy (`pip install cupy-cudaXXX`). CPU functions require Numba (`pip install numba`).

---

## Contents

- [Softening kernels](#softening-kernels)
- [compute_nbody_forces_gpu](#compute_nbody_forces_gpu)
- [compute_nbody_potential_gpu](#compute_nbody_potential_gpu)
- [compute_nbody_forces_cpu](#compute_nbody_forces_cpu)
- [compute_nbody_potential_cpu](#compute_nbody_potential_cpu)
- [get_gpu_info](#get_gpu_info)
- [Precision modes](#precision-modes)
- [Softening convention](#softening-convention)

---

## Softening kernels

All force and potential functions accept a `kernel` parameter that selects the gravitational softening scheme:

| `kernel` | Type | Description |
|---|---|---|
| `'newtonian'` | Infinite | Pure 1/r^2 with small epsilon regularization |
| `'plummer'` | Infinite | Plummer: 1/(r^2 + h^2)^(3/2) |
| `'dehnen_k1'` | Infinite | Dehnen P1 kernel, C2 continuity (falcON default) |
| `'dehnen_k2'` | Infinite | Dehnen P2 kernel, C4 continuity |
| `'spline'` | Compact | Monaghan 1992 cubic spline; Newtonian for r >= h |

`'spline'` is the default in `run_simulation`.  For `method='tree'` via pyfalcon the kernel is specified as an integer (`0`=Plummer, `1`=dehnen_k1, `2`=dehnen_k2).

---

## compute_nbody_forces_gpu

```python
acc = compute_nbody_forces_gpu(
    pos,                            # array_like (N, 3)
    mass,                           # array_like (N,) or float
    softening=0.0,                  # float or array_like (N,)
    G=4.300917e-6,                  # float
    precision='float32_kahan',      # 'float32' | 'float64' | 'float32_kahan'
    kernel='spline',                # str, see Softening kernels table
    return_cupy=False,              # bool
    skip_validation=False,          # bool
)
```

Compute gravitational accelerations for all N particles using O(N^2) direct pairwise summation on the GPU.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pos` | `array_like (N, 3)` | required | Particle positions in kpc |
| `mass` | `array_like (N,)` or `float` | required | Particle masses in M_sun. Scalar = equal mass |
| `softening` | `float` or `array_like (N,)` | `0.0` | Softening length(s) in kpc. Per-particle effective length uses `max(h_i, h_j)` |
| `G` | `float` | `4.300917e-6` | Gravitational constant |
| `precision` | `str` | `'float32_kahan'` | Floating-point mode (see Precision modes) |
| `kernel` | `str` | `'spline'` | Softening kernel |
| `return_cupy` | `bool` | `False` | If `True`, return CuPy array (stays on GPU). If `False`, return NumPy array |
| `skip_validation` | `bool` | `False` | Skip input validation (hot-loop optimisation) |

### Returns

`ndarray (N, 3)` — gravitational accelerations [km/s per kpc/(km/s)].  NumPy by default; CuPy when `return_cupy=True`.

### Raises

| Exception | Condition |
|---|---|
| `ImportError` | CuPy not installed |
| `ValueError` | Input shape mismatch |
| `RuntimeError` | CUDA error during computation |

### Example

```python
import numpy as np
from nbody_streams import compute_nbody_forces_gpu

N = 10_000
pos  = np.random.randn(N, 3).astype(np.float32)
mass = np.ones(N)

acc = compute_nbody_forces_gpu(pos, mass, softening=0.01, precision='float32_kahan')
# acc.shape -> (10000, 3)
```

Per-particle softening:

```python
h = np.linspace(0.005, 0.02, N)
acc = compute_nbody_forces_gpu(pos, mass, softening=h, kernel='plummer')
```

Keep result on GPU:

```python
acc_gpu = compute_nbody_forces_gpu(pos, mass, softening=0.01, return_cupy=True)
# acc_gpu is a cupy.ndarray
```

---

## compute_nbody_potential_gpu

```python
phi = compute_nbody_potential_gpu(
    pos,                            # array_like (N, 3)
    mass,                           # array_like (N,) or float
    softening=0.0,                  # float or array_like (N,)
    G=4.300917e-6,                  # float
    precision='float32_kahan',      # str
    kernel='spline',                # str
    return_cupy=False,              # bool
    skip_validation=False,          # bool
)
```

Compute gravitational potential Phi(r_i) = G * sum_j m_j * kernel(|r_i - r_j|) at each particle position.  Self-potential is excluded.

### Parameters

Same as `compute_nbody_forces_gpu`.

### Returns

`ndarray (N,)` — gravitational potential per particle.  Self-potential excluded.

### Notes

- Total potential energy: `PE = 0.5 * sum(mass * phi)` (factor 0.5 avoids double counting).
- ~20–30% faster than force computation for the same N.

### Example

```python
phi = compute_nbody_potential_gpu(pos, mass, softening=0.01)

KE = 0.5 * np.sum(mass * np.sum(vel**2, axis=1))
PE = 0.5 * np.sum(mass * phi)
E_total = KE + PE
```

---

## compute_nbody_forces_cpu

```python
acc = compute_nbody_forces_cpu(
    pos,                  # array_like (N, 3)
    mass,                 # array_like (N,) or float
    softening=0.0,        # float or array_like (N,)
    G=4.300917e-6,        # float
    kernel='spline',      # str
    nthreads=None,        # int or None
    precision='float64',  # str
)
```

Compute N-body accelerations using Numba `@njit(parallel=True)` CPU parallelization.  O(N^2) pairwise summation.  Compiled and cached on first call.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pos` | `array_like (N, 3)` | required | Particle positions |
| `mass` | `array_like (N,)` or `float` | required | Particle masses |
| `softening` | `float` or `array_like (N,)` | `0.0` | Softening length(s) |
| `G` | `float` | `4.300917e-6` | Gravitational constant |
| `kernel` | `str` | `'spline'` | Softening kernel |
| `nthreads` | `int` or `None` | `None` | Number of Numba threads. `None` = auto (all cores) |
| `precision` | `str` | `'float64'` | Computation dtype. `'float64'` or `'float32'` |

### Returns

`ndarray (N, 3)` — gravitational accelerations.

### Notes

- Compiled on first call; subsequent calls use the cached JIT.
- Softening uses max convention: `h_ij = max(h_i, h_j)`.
- For N > 100 000, prefer the GPU or tree backends.

### Example

```python
from nbody_streams import compute_nbody_forces_cpu

N = 1000
pos  = np.random.randn(N, 3)
mass = np.ones(N) * 1e6

acc = compute_nbody_forces_cpu(pos, mass, softening=0.05, nthreads=8)
```

---

## compute_nbody_potential_cpu

```python
phi = compute_nbody_potential_cpu(
    pos,                  # array_like (N, 3)
    mass,                 # array_like (N,) or float
    softening=0.0,        # float or array_like (N,)
    G=4.300917e-6,        # float
    kernel='spline',      # str
    nthreads=None,        # int or None
    precision='float64',  # str
)
```

Compute gravitational potential per particle using Numba CPU parallelization.  Same API as `compute_nbody_potential_gpu`.

### Returns

`ndarray (N,)` — potential per particle.

---

## get_gpu_info

```python
from nbody_streams import get_gpu_info

info = get_gpu_info()
```

Return a dictionary with information about the available GPU.

### Returns

```python
{
    'available'          : bool,          # whether CuPy and a GPU are present
    'device_name'        : str,           # GPU model name
    'compute_capability' : (int, int),    # (major, minor)
    'memory_total'       : int,           # total VRAM in bytes
    'memory_free'        : int,           # free VRAM in bytes
}
```

### Example

```python
info = get_gpu_info()
if info['available']:
    print(f"GPU: {info['device_name']}")
    print(f"Free VRAM: {info['memory_free'] / 1e9:.1f} GB")
```

---

## Precision modes

The `precision` parameter of the GPU functions controls both storage dtype and the summation algorithm:

| `precision` | Storage | Summation | Use case |
|---|---|---|---|
| `'float32'` | float32 | Standard | Fastest; small position-scale error at r << 0.1 |
| `'float32_kahan'` | float32 | Kahan compensated | Best float32 precision; ~15% slower than plain float32 |
| `'float64'` | float64 | Standard | Accurate at all scales; ~10x slower than float32 |

`run_simulation` defaults to `'float32_kahan'` for GPU direct.  Float64 is recommended for small-N (< 5000) or sub-kpc scale systems.

---

## Softening convention

For all backends, the effective pairwise softening length uses the **max convention**:

```
h_ij = max(h_i, h_j)
```

This ensures momentum is conserved: the force on particle i from particle j uses the same softening as the force on j from i.
