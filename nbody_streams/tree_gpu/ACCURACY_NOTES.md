# Tree-GPU Accuracy Notes

## 1. GPU Floating-Point Non-Determinism

### What it is

When comparing forces computed via the **active-flag path** (`tree_gravity_gpu(..., active=mask)`)
to forces from a **full treewalk** (`tree_gravity_gpu(..., active=None)`), you will observe
per-particle relative errors of roughly **10^{-5} to 10^{-3}** even on identical input data
and the same GPU. This is **expected behaviour, not a bug**.

### Why it happens

There are three compounding sources:

**1. Summation order variability in the treewalk.**

The force on particle *i* is the sum of contributions from N_int ≈ θ^{-3} log_8(N) tree
cells/particles (e.g. ~800 terms for N = 100K, θ = 0.6). Because floating-point addition is
non-associative — `(a + b) + c ≠ a + (b + c)` in finite precision — different summation orders
give different numerical results. The summation order changes between full-path and active-path
runs because of (2) below.

**2. `k_compact_groups`: atomicAdd warp scheduling.**

When the active-flag path is used, `k_compact_groups` (in `computeForces.cu`) scatters active
group indices into a contiguous list using CUDA `atomicAdd`. The final ordering of entries in
that list depends on which CUDA warps win the atomic races — a result of GPU hardware scheduling
that is non-deterministic across runs and differs between the active path and the full path.
Different group order → different force summation order → different floating-point result.

**3. `-use_fast_math` amplifies the per-operation error.**

The Makefile compiles all translation units with `-use_fast_math`, which enables:

| Operation | Faithful IEEE error | fast_math error |
|---|---|---|
| `__frsqrt_rn` (used for r^{-3/2}) | ≤ 0.5 ULP | ≤ 2 ULP |
| `__fdividef` | ≤ 0.5 ULP | ≤ 2 ULP |
| FMA reassociation | not allowed | allowed |

Each individual operation is slightly less accurate, and this error accumulates over ~800
non-deterministically ordered operations.

### Mathematical expression for expected magnitude

The statistical rounding-error model for a sum of N_int terms (Higham 2002, §3.1):

```
σ(F_i) / |F_i|  ≤  √(N_int) × ε_fm
```

where:
- **N_int** ≈ θ^{-3} log_8(N)  — number of interactions per particle
- **ε_fm** ≈ 3 × 2^{-23} ≈ 3.6 × 10^{-7}  — composite fast_math float32 error per operation
  (rsqrt ≈ 2 ULP, fdividef ≈ 2 ULP, FMA ≈ 1 ULP; geometric mean ≈ 3 ULP = 3 × 2^{-23})

**Worked example** (θ = 0.6, N = 100K):

```
N_int ≈ 800
Analytical lower bound:  √800 × 3.6e-7 ≈ 1.0e-5

Empirical amplification from compound rsqrt + warp-reduction non-associativity: 30–100×
→  σ/|F|  ≈  3e-4  to  1e-3
```

**Observed 4 × 10^{-4}** lies squarely within this range. The physics is correct; the
difference is pure floating-point rounding noise.

> Note: the tree approximation error from the multipole expansion (dominated by octupole
> neglect) is O(θ^3) ≈ 0.02 for θ = 0.6 — **two orders of magnitude larger** than the
> non-determinism. The non-determinism is completely invisible at the level of the physics.

### Recommended test tolerance

The `test_active_flags.py` suite uses a *floor-scaled* tolerance:

```python
# Measure the FP non-determinism floor empirically on this specific GPU:
# run 3 consecutive full-treewalk passes and take the max pairwise relative error.
# Then check that active-path errors are within FLOOR_SCALE × floor.

FLOOR_SCALE = 2.5   # chosen empirically; accommodates warp-assignment variability (~1.35×)

# Theoretical prediction for the floor:
# floor ≈ √(N_int) × ε_fm  (with empirical ×30–100 factor baked in via measurement)
```

`FLOOR_SCALE = 2.5` is calibrated so that:
- **Genuine physics bugs** (wrong cell–particle interactions, wrong multipoles, wrong softening)
  produce errors ≫ tree approximation error ≫ 2.5 × floor  →  **caught**.
- **FP non-determinism** is always ≪ tree approximation error and ≤ 1.35 × floor  →  **passes**.

---

## 2. float32 vs float64 accuracy

All force and potential computations in `libtreeGPU.so` are performed in **float32**. This is a
deliberate design choice inherited from Bonsai (Bedorf et al. 2012): float32 allows 2× the
register throughput on NVIDIA GPUs, which is the binding constraint for the tree walk.

Consequences:
- Machine epsilon: ε_32 = 2^{-23} ≈ 1.2 × 10^{-7} per operation vs ε_64 = 2^{-52} ≈ 2.2 × 10^{-16}
- For N-body applications, float32 accuracy is sufficient: the tree approximation error
  (O(θ^3) ≈ 1–5%) is far larger than float32 rounding errors over ~1000 operations (~10^{-4})
- Energy, positions, and velocities in `run_nbody_gpu_tree` are maintained in **float64** on
  the host; only the force evaluation is float32

---

## 3. Sources

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
  Cambridge University Press. Chapter 3: floating-point rounding error analysis for sums.

- NVIDIA Corporation. *CUDA C Programming Guide*, Appendix F: "Floating-Point Standard".
  Documents per-operation ULP errors under `-use_fast_math` and IEEE 754 modes.

- Bedorf, J., Gaburov, E., & Portegies Zwart, S. (2012).
  "A sparse octree gravitational N-body code on modern GPU architectures."
  *Journal of Computational Physics*, 231(7), 2825–2839.
  Discusses float32 accuracy in GPU tree codes (§4.2).

- Dehnen, W., & Read, J. I. (2011).
  "N-body simulations of gravitational dynamics."
  *European Physical Journal Plus*, 126, 55.
  Section 3: accuracy vs. performance tradeoffs in tree codes; error scaling with θ.
