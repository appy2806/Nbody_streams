"""Restart files: ``restart_interval=None`` turns them off, the final restart is
written atomically and loads, and the summary separates integration from output."""
import numpy as np
import pytest

from nbody_streams.nbody_io import _load_restart
from nbody_streams.run import run_nbody_cpu, make_plummer_sphere

try:
    import cupy as cp
    cp.cuda.Device(0).compute_capability
    from nbody_streams.run import run_nbody_gpu
    from nbody_streams.tree_gpu import run_nbody_gpu_tree
    GPU = True
except Exception:  # no CuPy, no device, or no tree library
    GPU = False

G = 4.300917270038e-06


def _ic(n=256):
    ps, m = make_plummer_sphere(n, M_total=1e5, a=0.01, seed=3)
    return ps, np.asarray(m, dtype=np.float64)


def _runners():
    out = [pytest.param("cpu", id="cpu")]
    for name in ("gpu", "tree"):
        out.append(pytest.param(name, id=name,
                                marks=pytest.mark.skipif(not GPU, reason="needs a GPU")))
    return out


def _run(kind, tmp_path, restart_interval, verbose=False):
    ps, m = _ic()
    kw = dict(time_start=0.0, time_end=12 * 1e-5, dt=1e-5, softening=0.001, G=G,
              save_snapshots=False, restart_interval=restart_interval,
              output_dir=str(tmp_path), verbose=verbose)
    if kind == "cpu":
        return run_nbody_cpu(ps, m, kernel="plummer", **kw)
    if kind == "gpu":
        return run_nbody_gpu(ps, m, **kw)
    return run_nbody_gpu_tree(ps, m, **kw)


@pytest.mark.parametrize("kind", _runners())
@pytest.mark.parametrize("interval", [None, 0])
def test_restart_off(kind, interval, tmp_path):
    out = _run(kind, tmp_path, interval)
    assert np.isfinite(out).all()
    assert not (tmp_path / "restart.npz").exists()
    assert not list(tmp_path.glob("restart*"))


@pytest.mark.parametrize("kind", _runners())
def test_restart_on(kind, tmp_path):
    out = _run(kind, tmp_path, 5)
    assert (tmp_path / "restart.npz").exists()
    assert not list(tmp_path.glob("*.tmp"))
    r = _load_restart(tmp_path)
    assert r is not None
    xv, t = r[0], r[1]
    assert t == pytest.approx(12 * 1e-5)
    np.testing.assert_allclose(xv, out, rtol=0, atol=0)


@pytest.mark.parametrize("kind", _runners())
def test_summary_separates_output(kind, tmp_path, capsys):
    _run(kind, tmp_path, 5, verbose=True)
    text = capsys.readouterr().out
    assert "integration" in text and "output" in text


@pytest.mark.parametrize("mass, eps, species_N, stored", [
    (np.full(10, 2.0), np.full(10, 0.1), None, "scalar"),
    (np.r_[np.full(4, 1.0), np.full(6, 3.0)], np.r_[np.full(4, 0.2), np.full(6, 0.5)],
     [4, 6], "per_species"),
    (np.linspace(1, 2, 10), np.linspace(0.1, 0.2, 10), [4, 6], "array"),
])
def test_restart_mass_softening_compaction(tmp_path, mass, eps, species_N, stored):
    from nbody_streams.nbody_io import _save_restart
    xv = np.random.default_rng(0).normal(size=(10, 6))
    names = ["a", "b"] if species_N else None
    _save_restart(xv, 1.0, 7, tmp_path, 2, mass_arr=mass, softening_arr=eps,
                  species_names=names, species_N=species_N)
    with np.load(tmp_path / "restart.npz") as d:
        keys = set(d.files)
        if stored == "scalar":
            assert d["mass_arr"].ndim == 0 and d["softening_arr"].ndim == 0
        elif stored == "per_species":
            assert {"mass_arr_per_species", "softening_arr_per_species"} <= keys
            assert "mass_arr" not in keys
        else:
            assert d["mass_arr"].shape == (10,)
    r = _load_restart(tmp_path)
    np.testing.assert_array_equal(r[0], xv)
    np.testing.assert_array_equal(r[4], mass)
    np.testing.assert_array_equal(r[5], eps)
    assert r[7] == species_N


def test_run_nbody_cpu_defaults(tmp_path):
    """run_nbody_cpu with its default method/kernel (direct, spline) runs."""
    ps, m = _ic(128)
    out = run_nbody_cpu(ps, m, 0.0, 5e-5, 1e-5, 0.001, G=G, output_dir=str(tmp_path),
                        save_snapshots=False, restart_interval=None, verbose=False)
    assert np.isfinite(out).all()


def test_run_nbody_cpu_tree_default_kernel(tmp_path):
    pytest.importorskip("pyfalcon")
    ps, m = _ic(128)
    out = run_nbody_cpu(ps, m, 0.0, 5e-5, 1e-5, 0.001, G=G, method="tree",
                        output_dir=str(tmp_path), save_snapshots=False,
                        restart_interval=None, verbose=False)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("kind", _runners())
def test_continue_run_matches_straight_run(kind, tmp_path):
    """6 steps + continue_run to 12 equals 12 straight steps (compacted restart)."""
    ps, m = _ic()
    kw = dict(time_start=0.0, dt=1e-5, softening=0.001, G=G, save_snapshots=False,
              restart_interval=3, verbose=False)
    fn = {"cpu": lambda *a, **k: run_nbody_cpu(*a, kernel="plummer", **k)}.get(kind)
    if fn is None:
        fn = run_nbody_gpu if kind == "gpu" else run_nbody_gpu_tree
    straight = fn(ps.copy(), m, time_end=12e-5, output_dir=str(tmp_path / "a"), **kw)
    fn(ps.copy(), m, time_end=6e-5, output_dir=str(tmp_path / "b"), **kw)
    cont = fn(ps.copy(), m, time_end=12e-5, output_dir=str(tmp_path / "b"),
              continue_run=True, **kw)
    # The tree recentres positions before its float32 force evaluation; a
    # continued run re-derives that centre, so it agrees to float32 rounding.
    rtol = 1e-6 if kind == "tree" else 1e-9
    np.testing.assert_allclose(cont, straight, rtol=rtol, atol=1e-12)
