"""
test_overflow.py -- The GPU tree build recovers from buffer overflow.

The node pool, cell list and work queue of the tree build are sized for
typical trees, not for the worst case (a clump of > nleaf coincident
particles becomes a ~25-deep chain of single-child nodes).  On overflow the
build writes nothing out of bounds, raises a flag, and buildTree grows the
buffers and rebuilds; they stay grown on the handle.  This used to be a
device-side assert, which kills the CUDA context, so the cases run in a
subprocess and the parent only reads their results.

TREEGPU_DEBUG_BUILD_CAPACITY=k (read when a handle is allocated) starts all
three buffers at k entries, forcing many grow-and-rebuild rounds.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_CHILD = textwrap.dedent(r"""
    import json, os, sys
    import numpy as np
    import cupy as cp
    from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU
    from nbody_streams import compute_nbody_forces_gpu, compute_nbody_potential_gpu

    VERBOSE = len(sys.argv) > 1
    CAP = "TREEGPU_DEBUG_BUILD_CAPACITY"

    def clumps(k, npc, rng):           # k clusters of npc exactly coincident particles
        return np.repeat(rng.uniform(-1, 1, (k, 3)), npc, axis=0)

    def plummer(n, rng):
        r = 1.0 / np.sqrt(rng.uniform(0.01, 1.0, n) ** (-2.0 / 3.0) - 1.0)
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1)[:, None] * r[:, None]

    def handle(n, cap=None):
        if cap is None:
            os.environ.pop(CAP, None)
        else:
            os.environ[CAP] = str(cap)
        try:
            return TreeGPU(n, verbose=VERBOSE)
        finally:
            os.environ.pop(CAP, None)

    def tree(x, m, eps, h):
        acc, phi = tree_gravity_gpu(cp.asarray(x), cp.asarray(m), cp.asarray(eps),
                                    G=1.0, theta=0.3, nleaf=16, tree=h)
        return cp.asnumpy(acc), cp.asnumpy(phi)

    def errors(x, m, eps, acc, phi):
        a_ref = cp.asnumpy(compute_nbody_forces_gpu(
            x, m, softening=eps, G=1.0, precision="float64", kernel="plummer", return_cupy=True))
        p_ref = cp.asnumpy(compute_nbody_potential_gpu(
            x, m, softening=eps, G=1.0, precision="float64", kernel="plummer", return_cupy=True))
        acc = acc.astype(np.float64); phi = phi.astype(np.float64)
        return (float(np.linalg.norm(acc - a_ref, axis=1).max() / np.linalg.norm(a_ref, axis=1).max()),
                float(np.abs(phi - p_ref).max() / np.abs(p_ref).max()))

    def same(r1, r2):
        return bool(np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1]))

    def report(case, x, m, eps, res, ref, reuse):
        acc, phi = res
        out = dict(case=case, n=len(x), finite=bool(np.isfinite(acc).all() and np.isfinite(phi).all()),
                   equal_ref=same(res, ref), reuse_equal=all(same(res, r) for r in reuse))
        out["acc_err"], out["phi_err"] = errors(x, m, eps, acc, phi)
        print("RESULT " + json.dumps(out), flush=True)

    rng = np.random.default_rng(7)
    BIG = 4_000_000                                   # never overflows here

    # 1. 4000 clusters of 17 coincident particles: the default sizes overflow
    #    (node pool, then cells); grown handle reused for 3 calls.
    x = clumps(4000, 17, rng); n = len(x)
    m = rng.uniform(0.5, 1.5, n) / n; eps = np.full(n, 0.01)
    with handle(n) as h:
        runs = [tree(x, m, eps, h) for _ in range(3)]
    with handle(n, BIG) as h:
        ref = tree(x, m, eps, h)
    report("clumps68k", x, m, eps, runs[0], ref, runs[1:])

    # 2. forced tiny capacity (many rounds, all three buffers), clumps plus a
    #    Plummer background with per-particle eps (equal eps in the clumps:
    #    cells soften with their max eps, not pairwise, see test_small_n)
    x = np.concatenate([clumps(300, 20, rng), plummer(4000, rng)]); n = len(x)
    m = rng.uniform(0.5, 1.5, n) / n
    eps = np.concatenate([np.full(6000, 0.01), rng.uniform(0.01, 0.02, 4000)])
    with handle(n, 1) as h:
        runs = [tree(x, m, eps, h) for _ in range(2)]
    with handle(n, BIG) as h:
        ref = tree(x, m, eps, h)
    report("tiny_cap_mixed", x, m, eps, runs[0], ref, runs[1:])

    # 3. forced small capacity, plain Plummer, then the grown handle reused on
    #    different positions (each must match a never-overflowed handle)
    xa = plummer(30000, rng); xb = clumps(1500, 20, rng); n = len(xa)
    m = rng.uniform(0.5, 1.5, n) / n; eps = np.full(n, 0.01)
    with handle(n, 64) as h:
        ra1 = tree(xa, m, eps, h); rb = tree(xb, m, eps, h); ra2 = tree(xa, m, eps, h)
    with handle(n, BIG) as h:
        refa = tree(xa, m, eps, h); refb = tree(xb, m, eps, h)
    report("small_cap_plummer", xa, m, eps, ra1, refa, [ra2])
    report("small_cap_reuse_clumps", xb, m, eps, rb, refb, [])
    print("DONE", flush=True)
""")


def _run_child(*args):
    env = dict(os.environ)
    env.pop("TREEGPU_DEBUG_BUILD_CAPACITY", None)
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", _CHILD, *args], capture_output=True, text=True,
                          env=env, cwd=_REPO, timeout=900)
    lines = proc.stdout.splitlines()
    assert proc.returncode == 0 and "DONE" in lines, (
        f"overflow subprocess failed (rc={proc.returncode}); last results:\n"
        + "\n".join(lines[-3:]) + "\nstderr:\n" + proc.stderr[-3000:])
    return [json.loads(s[len("RESULT "):]) for s in lines if s.startswith("RESULT ")], proc.stderr


@pytest.fixture(scope="module")
def overflow_results():
    return _run_child()


def test_overflow_cases_complete(overflow_results):
    results, stderr = overflow_results
    assert [r["case"] for r in results] == [
        "clumps68k", "tiny_cap_mixed", "small_cap_plummer", "small_cap_reuse_clumps"]
    for r in results:
        assert r["finite"], r
    # growth notes only when verbose; no device assert
    assert "overflowed" not in stderr and "Assertion" not in stderr, stderr[-2000:]


def test_overflow_matches_direct_sum(overflow_results):
    for r in overflow_results[0]:
        assert r["acc_err"] < 2e-3 and r["phi_err"] < 2e-4, r


def test_overflow_matches_unoverflowed_build(overflow_results):
    # a rebuilt tree is the tree a big-enough handle builds: bitwise equal,
    # also on later calls with the grown handle
    for r in overflow_results[0]:
        assert r["equal_ref"] and r["reuse_equal"], r


def test_overflow_note_when_verbose():
    results, stderr = _run_child("verbose")
    assert len(results) == 4
    assert "treeGPU: tree build overflowed" in stderr
    assert "Assertion" not in stderr and "ERROR" not in stderr
