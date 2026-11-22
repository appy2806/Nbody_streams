"""
Source-split direct sum (small-N GPU fill).

With S > 1 the source loop is split across ``blockIdx.y`` slices whose partial
sums are reduced in a fixed order. Checks, for every softening kernel and
precision, forces and potentials:

1. any forced S agrees with S = 1 to summation-order rounding;
2. repeated calls with S > 1 are bitwise identical (no atomics);
3. the automatic rule keeps S = 1 once the grid fills the GPU, and the hot
   path accepts the ``_split`` override.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import nbody_streams.fields as fields
from nbody_streams.fields import (
    compute_nbody_forces_gpu, compute_nbody_potential_gpu, _direct_forces_gpu_soa,
    _choose_split, _split_launch_info, KERNEL_MAP, _PRECISION_MAP,
)

PRECISIONS = ["float32", "float32_kahan", "float64"]
TOL = {"float32": 1e-5, "float32_kahan": 1e-6, "float64": 1e-13}
SPLITS = (2, 3, 8)


def _system(N, seed=7):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(N, 3))
    m = rng.uniform(0.5, 1.5, N) / N
    h = rng.uniform(0.01, 0.1, N)
    return pos, m, h


def _eval(what, pos, m, h, precision, kernel, split, monkeypatch):
    monkeypatch.setattr(fields, "_DIRECT_SPLIT", split)
    fn = compute_nbody_forces_gpu if what == "force" else compute_nbody_potential_gpu
    return fn(pos, m, h, precision=precision, kernel=kernel)


@pytest.mark.parametrize("soft", ["scalar", "array"])
@pytest.mark.parametrize("N", [1, 5, 127, 129, 1000, 3000])
@pytest.mark.parametrize("what", ["force", "potential"])
@pytest.mark.parametrize("precision", PRECISIONS)
@pytest.mark.parametrize("kernel", list(KERNEL_MAP))
def test_split_matches_unsplit(kernel, precision, what, N, soft, monkeypatch):
    pos, m, h = _system(N)
    if soft == "scalar":
        h = 0.05
    ref = _eval(what, pos, m, h, precision, kernel, 1, monkeypatch)
    scale = np.abs(ref).max()
    for S in SPLITS:
        got = _eval(what, pos, m, h, precision, kernel, S, monkeypatch)
        assert got.dtype == ref.dtype and got.shape == ref.shape
        err = np.abs(got.astype(np.float64) - ref).max()
        assert err <= TOL[precision] * scale, (S, err, scale)


@pytest.mark.parametrize("precision", PRECISIONS)
def test_hot_path_split_matches_unsplit(precision):
    pos, m, h = _system(3000)
    pos = pos + [30.0, -4.0, 2.0]                 # float32: relative-to-centre packing
    dt = _PRECISION_MAP[precision][0]
    args = (cp.asarray(pos), cp.asarray(m, dtype=dt), cp.asarray(h, dtype=dt), 1.0,
            precision, KERNEL_MAP["spline"])
    ref = cp.asnumpy(cp.stack(_direct_forces_gpu_soa(*args, _split=1)))
    for S in SPLITS:
        got = cp.asnumpy(cp.stack(_direct_forces_gpu_soa(*args, _split=S)))
        assert np.abs(got - ref).max() <= TOL[precision] * np.abs(ref).max()


@pytest.mark.parametrize("what", ["force", "potential"])
@pytest.mark.parametrize("precision", PRECISIONS)
def test_split_is_deterministic(precision, what, monkeypatch):
    pos, m, h = _system(3000, seed=11)
    for S in (3, 8):
        a = _eval(what, pos, m, h, precision, "spline", S, monkeypatch)
        for _ in range(3):
            b = _eval(what, pos, m, h, precision, "spline", S, monkeypatch)
            np.testing.assert_array_equal(a, b)


def test_auto_split_rule(monkeypatch):
    monkeypatch.setattr(fields, "_DIRECT_SPLIT", None)
    for kernel_id in KERNEL_MAP.values():
        for precision in PRECISIONS:
            resident = _split_launch_info("force", precision,
                                          precision == "float32_kahan", kernel_id)[3]
            assert resident >= cp.cuda.Device().attributes["MultiProcessorCount"]
            # A grid that already fills the device is never split.
            full = int(128 * fields._SPLIT_FILL * resident)
            for N in (full, full + 1, 4 * full, 10**8):
                assert _choose_split(N, resident) == 1
            assert _choose_split(full - 128, resident) > 1
            # A small grid is split, within the tile count and the cap.
            for N in (1, 129, 1000, 16384):
                tiles = (N + 127) // 128
                S = _choose_split(N, resident)
                assert 1 <= S <= min(tiles, fields._SPLIT_MAX)
                if tiles > 1 and tiles < fields._SPLIT_FILL * resident:
                    assert S > 1
    # The override wins over the automatic rule.
    assert _choose_split(10**7, 100, split=4) == 4
    monkeypatch.setattr(fields, "_DIRECT_SPLIT", 3)
    assert _choose_split(10**7, 100) == 3


def test_auto_split_large_n_is_unsplit(monkeypatch):
    """At large N the automatic choice is the plain kernel (bitwise S = 1)."""
    resident = _split_launch_info("force", "float32", False, KERNEL_MAP["plummer"])[3]
    N = int(128 * fields._SPLIT_FILL * resident) + 5
    pos, m, h = _system(N, seed=5)
    monkeypatch.setattr(fields, "_DIRECT_SPLIT", None)
    auto = compute_nbody_forces_gpu(pos, m, h, precision="float32", kernel="plummer")
    monkeypatch.setattr(fields, "_DIRECT_SPLIT", 1)
    one = compute_nbody_forces_gpu(pos, m, h, precision="float32", kernel="plummer")
    np.testing.assert_array_equal(auto, one)
