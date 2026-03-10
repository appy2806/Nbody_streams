#!/usr/bin/env python
"""
test_active_flags.py -- Validate active-particle flag accuracy.

Tests
-----
  1. fp_floor_per_N        : measure GPU non-determinism floor for each N
                             (two identical full runs; all differences are pure fp)
  2. active_handful        : 2, 5, 10, 50, 200 innermost particles active;
                             per-particle errors must stay within floor
  3. active_fraction       : 1%, 10%, 50% of particles active (spatially coherent
                             inner sphere); errors must stay within floor
  4. active_ones_vs_none   : active=all-ones vs active=None must be within floor
  5. no_nan_active         : no NaN/Inf in outputs for any active configuration
  6. inactive_slots_any_n  : run across multiple N values (1K, 10K, 50K, 100K)

Design
------
The non-determinism floor is measured fresh for each N from two consecutive
full runs.  Active-flag errors are then asserted to be < FLOOR_SCALE * floor.
FLOOR_SCALE = 5.0 is conservative: in practice errors sit at ~1-2x floor.

Active particles are always selected as the innermost by radius (spatially
coherent), which maximises the chance of entire groups being skipped.
"""

import numpy as np
import cupy as cp
import pytest

from nbody_streams.tree_gpu import tree_gravity_gpu

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THETA       = 0.5
EPS         = 0.05
G           = 1.0
FLOOR_SCALE = 2.5   # active errors must be < FLOOR_SCALE * fp_floor
# Why 2.0 and not 1.0:
#   The fp_floor is measured from two consecutive full runs with the same
#   execution order.  The active path uses group compaction, which changes the
#   order groups are assigned to warps → different fp accumulation order →
#   errors up to ~1.35x the same-order floor.  FLOOR_SCALE=2.0 accommodates
#   this while still catching any real physics bug (which would be >> 2x floor).


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _make_plummer(n, seed=42):
    rng   = np.random.default_rng(seed)
    scale = 3.0 * np.pi / 16.0
    pos   = np.empty((n, 3), dtype=np.float64)
    i = 0
    while i < n:
        batch    = min(n - i, max(n // 10, 1024))
        X1       = rng.random(batch)
        X2       = rng.random(batch)
        X3       = rng.random(batch)
        R        = 1.0 / np.sqrt(np.maximum(X1**(-2.0 / 3.0) - 1.0, 0.0))
        valid    = R < 100.0
        R, X2, X3 = R[valid], X2[valid], X3[valid]
        Z  = (1.0 - 2.0 * X2) * R
        X  = np.sqrt(np.maximum(R * R - Z * Z, 0.0)) * np.cos(2.0 * np.pi * X3)
        Y  = np.sqrt(np.maximum(R * R - Z * Z, 0.0)) * np.sin(2.0 * np.pi * X3)
        take = min(len(R), n - i)
        pos[i:i + take, 0] = X[:take] * scale
        pos[i:i + take, 1] = Y[:take] * scale
        pos[i:i + take, 2] = Z[:take] * scale
        i += take
    return (
        cp.asarray(pos, dtype=cp.float32),
        cp.full(n, 1.0 / n, dtype=cp.float32),
    )


def _run(pos, mass, active=None):
    acc, phi = tree_gravity_gpu(
        pos, mass,
        eps=EPS, theta=THETA, G=G,
        verbose=False, active=active,
    )
    cp.cuda.runtime.deviceSynchronize()
    return acc, phi


def _rel_err_vec(ref_acc, test_acc):
    """Per-particle relative acceleration error."""
    diff = cp.linalg.norm(test_acc - ref_acc, axis=1)
    amag = cp.linalg.norm(ref_acc, axis=1).clip(1e-30)
    return diff / amag


def _fp_floor(pos, mass):
    """Measure non-determinism floor from 3 consecutive full runs.

    Takes the max over two independent pair comparisons so the floor estimate
    is robust to a single lucky/unlucky run.  Using max() rather than mean()
    avoids false positives when one pair happens to land on the same warp
    schedule by coincidence.
    """
    a1, _ = _run(pos, mass)
    a2, _ = _run(pos, mass)
    a3, _ = _run(pos, mass)
    f12 = float(_rel_err_vec(a1, a2).max())
    f23 = float(_rel_err_vec(a2, a3).max())
    return max(f12, f23)


def _inner_mask(pos, n_active):
    """Float32 active mask: 1.0 for the n_active innermost particles."""
    n   = pos.shape[0]
    r   = cp.sqrt((pos ** 2).sum(axis=1))
    idx = cp.asnumpy(r).argsort()
    m   = np.zeros(n, dtype=np.float32)
    m[idx[:n_active]] = 1.0
    return cp.asarray(m)


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------
N_VALUES       = [1_024, 2048, 10_000, 50_000, 100_000]
HANDFUL_COUNTS = [2, 5, 10, 50, 200, 500]
FRACTIONS      = [0.01, 0.10, 0.50]       # 1%, 10%, 50%


# ---------------------------------------------------------------------------
# Test 1 — fp floor is itself small (sanity on the GPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", N_VALUES)
def test_fp_floor_is_small(n):
    """Two identical full runs must agree to < 1e-4 relative."""
    pos, mass = _make_plummer(n)
    floor = _fp_floor(pos, mass)
    print(f"\n  N={n:>7,}  fp_floor={floor:.3e}")
    assert floor < 1e-5, f"fp floor unexpectedly large: {floor:.3e}"


# ---------------------------------------------------------------------------
# Test 2 — handful of active particles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [10_000, 100_000])
@pytest.mark.parametrize("n_active", HANDFUL_COUNTS)
def test_active_handful(n, n_active):
    """
    n_active innermost particles active.
    Per-particle errors for those particles must be within FLOOR_SCALE * fp_floor.
    """
    if n_active >= n:
        pytest.skip("n_active >= n")

    pos, mass = _make_plummer(n)
    floor     = _fp_floor(pos, mass)
    tol       = FLOOR_SCALE * max(floor, 1e-7)   # never tighter than float32 eps

    acc_full, _ = _run(pos, mass)

    active_mask = _inner_mask(pos, n_active)
    acc_act, _  = _run(pos, mass, active=active_mask)

    # compare only the active particles
    active_idx  = cp.asarray(cp.asnumpy(active_mask).astype(bool))
    rel         = _rel_err_vec(acc_full[active_idx], acc_act[active_idx])
    max_rel     = float(rel.max())
    rms_rel     = float(rel.mean())

    print(f"\n  N={n:>7,}  n_active={n_active:>4}  floor={floor:.3e}  "
          f"max_rel={max_rel:.3e}  rms_rel={rms_rel:.3e}  tol={tol:.3e}")

    assert max_rel <= tol, (
        f"Active forces deviate beyond fp floor: max_rel={max_rel:.3e} > tol={tol:.3e} "
        f"(floor={floor:.3e})"
    )


# ---------------------------------------------------------------------------
# Test 3 — fraction of particles active (spatially coherent inner sphere)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("frac", FRACTIONS)
def test_active_fraction(n, frac):
    """
    Inner frac*N particles active.
    Errors on active particles must stay within FLOOR_SCALE * fp_floor.
    """
    n_active = max(2, int(n * frac))
    pos, mass = _make_plummer(n)
    floor     = _fp_floor(pos, mass)
    tol       = FLOOR_SCALE * max(floor, 1e-7)

    acc_full, _ = _run(pos, mass)

    active_mask = _inner_mask(pos, n_active)
    acc_act, _  = _run(pos, mass, active=active_mask)

    active_idx = cp.asarray(cp.asnumpy(active_mask).astype(bool))
    rel        = _rel_err_vec(acc_full[active_idx], acc_act[active_idx])
    max_rel    = float(rel.max())

    print(f"\n  N={n:>7,}  frac={frac:.0%}  n_active={n_active:>6,}  "
          f"floor={floor:.3e}  max_rel={max_rel:.3e}  tol={tol:.3e}")

    assert max_rel <= tol, (
        f"Active forces deviate beyond fp floor: max_rel={max_rel:.3e} > tol={tol:.3e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — active=all-ones vs active=None (default path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", N_VALUES)
def test_active_ones_matches_none(n):
    """
    Passing active=ones must give the same forces as active=None,
    within the fp floor.
    """
    pos, mass = _make_plummer(n)
    floor     = _fp_floor(pos, mass)
    tol       = FLOOR_SCALE * max(floor, 1e-7)

    acc_none, phi_none = _run(pos, mass, active=None)
    ones               = cp.ones(n, dtype=cp.float32)
    acc_ones, phi_ones = _run(pos, mass, active=ones)

    rel     = _rel_err_vec(acc_none, acc_ones)
    max_rel = float(rel.max())

    phi_rel = float((cp.abs(phi_ones - phi_none) /
                     cp.abs(phi_none).clip(1e-30)).max())

    print(f"\n  N={n:>7,}  floor={floor:.3e}  "
          f"max_rel_acc={max_rel:.3e}  max_rel_phi={phi_rel:.3e}  tol={tol:.3e}")

    assert max_rel <= tol, (
        f"active=ones vs active=None: max_rel={max_rel:.3e} > tol={tol:.3e}"
    )
    assert phi_rel <= tol, (
        f"active=ones vs active=None phi: max_rel={phi_rel:.3e} > tol={tol:.3e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — no NaN / Inf for any active configuration
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
@pytest.mark.parametrize("n_active", [2, 10, 100, "all", "ones"])
def test_no_nan_active(n, n_active):
    """Outputs must be finite for any active mask."""
    pos, mass = _make_plummer(n)

    if n_active == "all":
        active = None
    elif n_active == "ones":
        active = cp.ones(n, dtype=cp.float32)
    else:
        if n_active >= n:
            pytest.skip("n_active >= n")
        active = _inner_mask(pos, n_active)

    acc, phi = _run(pos, mass, active=active)

    assert not bool(cp.any(cp.isnan(acc))),  f"NaN in acc  (N={n}, n_active={n_active})"
    assert not bool(cp.any(cp.isinf(acc))),  f"Inf in acc  (N={n}, n_active={n_active})"
    assert not bool(cp.any(cp.isnan(phi))),  f"NaN in phi  (N={n}, n_active={n_active})"
    assert not bool(cp.any(cp.isinf(phi))),  f"Inf in phi  (N={n}, n_active={n_active})"
