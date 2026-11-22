"""
test_small_n.py -- Regression test: the GPU tree with very few particles.

The cell buffer used to be sized to exactly N cells, but the root split alone
needs up to min(8, N) cells (plus a chain of nodes for coincident particles),
so N = 1-2 always and N = 3-4 sometimes hit a device-side assert in
buildOctant.  A device assert kills the CUDA context, so the cases run in a
subprocess and the parent only reads their results.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import cupy as cp
import pytest

from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_CHILD = textwrap.dedent(r"""
    import json
    import numpy as np
    import cupy as cp
    from nbody_streams.tree_gpu import tree_gravity_gpu
    from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu

    def positions(kind, n, rng):
        if kind == "plummer":
            r = 1.0 / np.sqrt(rng.uniform(0.01, 1.0, n) ** (-2.0 / 3.0) - 1.0)
            v = rng.normal(size=(n, 3))
            return v / np.linalg.norm(v, axis=1)[:, None] * r[:, None]
        if kind == "coincident":          # all at one point but one
            x = np.full((n, 3), 0.3)
            x[:1] = (1.0, -2.0, 0.5)
            return x
        if kind == "pair":                # close pair far from the rest
            x = rng.normal(size=(n, 3))
            k = min(n, 2)
            x[:k] = 50.0 + 0.02 * rng.normal(size=(k, 3))
            return x
        raise ValueError(kind)

    for kind in ("plummer", "coincident", "pair"):
        for n in range(0, 9):
            for nleaf in (16, 32, 64):
                rng = np.random.default_rng(1000 * n + nleaf)
                x = positions(kind, n, rng)
                m = rng.uniform(0.5, 1.5, n)
                eps = rng.uniform(0.01, 0.05, n)
                acc, phi = tree_gravity_gpu(cp.asarray(x), cp.asarray(m), cp.asarray(eps),
                                            G=1.0, theta=0.3, nleaf=nleaf)
                cp.cuda.Device().synchronize()
                res = dict(kind=kind, n=n, nleaf=nleaf, acc_shape=list(acc.shape),
                           phi_shape=list(phi.shape), dtypes=[str(acc.dtype), str(phi.dtype)],
                           finite=bool(cp.isfinite(acc).all() and cp.isfinite(phi).all()))
                acc = cp.asnumpy(acc).astype(np.float64)
                phi = cp.asnumpy(phi).astype(np.float64)
                if n >= 1:
                    a_ref = cp.asnumpy(compute_nbody_forces_gpu(
                        x, m, softening=eps, G=1.0, precision="float64",
                        kernel="plummer", return_cupy=True))
                    p_ref = cp.asnumpy(compute_nbody_potential_gpu(
                        x, m, softening=eps, G=1.0, precision="float64",
                        kernel="plummer", return_cupy=True))
                    a_scale = max(np.linalg.norm(a_ref, axis=1).max(), 1e-30)
                    p_scale = max(np.abs(p_ref).max(), 1e-30)
                    res["acc_err"] = float(np.linalg.norm(acc - a_ref, axis=1).max() / a_scale) if n > 1 else 0.0
                    res["phi_err"] = float(np.abs(phi - p_ref).max() / p_scale) if n > 1 else 0.0
                    res["acc_max"] = float(np.abs(acc).max())
                    res["phi_max"] = float(np.abs(phi).max())
                    res["self_scale"] = float((m / eps).max())
                print("RESULT " + json.dumps(res), flush=True)
    print("DONE", flush=True)
""")


@pytest.fixture(scope="module")
def small_n_results():
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True,
                          env=env, cwd=_REPO, timeout=600)
    lines = proc.stdout.splitlines()
    assert proc.returncode == 0 and "DONE" in lines, (
        f"tiny-N subprocess failed (rc={proc.returncode}); last results:\n"
        + "\n".join(lines[-3:]) + "\nstderr:\n" + proc.stderr[-3000:])
    return [json.loads(s[len("RESULT "):]) for s in lines if s.startswith("RESULT ")]


def test_small_n_no_crash_and_shapes(small_n_results):
    assert len(small_n_results) == 3 * 9 * 3
    for r in small_n_results:
        assert r["finite"], r
        assert r["acc_shape"] == [r["n"], 3] and r["phi_shape"] == [r["n"]], r
        assert r["dtypes"] == ["float32", "float32"], r


def test_small_n_single_particle_is_zero(small_n_results):
    # phi: only the float32 rounding left after removing the m/eps self term
    for r in small_n_results:
        if r["n"] == 1:
            assert r["acc_max"] == 0.0, r
            assert r["phi_max"] <= 1e-6 * r["self_scale"], r


def test_small_n_matches_direct_sum(small_n_results):
    for r in small_n_results:
        # acc tolerance: cell approximations soften with max(eps_i, eps_cell_max),
        # not the pairwise max, an O(eps^2/r^2) ~ 1e-4 difference here
        if r["n"] >= 2:
            assert r["acc_err"] < 2e-3 and r["phi_err"] < 2e-4, r


def test_zero_particles_returns_empty():
    acc, phi = tree_gravity_gpu(cp.empty((0, 3)), cp.empty(0), 0.01, G=1.0)
    assert acc.shape == (0, 3) and phi.shape == (0,)
    assert acc.dtype == cp.float32 and phi.dtype == cp.float32
    with TreeGPU(0) as tree:
        acc, phi = tree_gravity_gpu(np.empty((0, 3), dtype=np.float32), np.empty(0), 0.01,
                                    tree=tree)
        assert acc.shape == (0, 3) and phi.shape == (0,)


@pytest.mark.parametrize("n", [-1, 2.5])
def test_treegpu_bad_n_raises(n):
    with pytest.raises(ValueError):
        TreeGPU(n)
