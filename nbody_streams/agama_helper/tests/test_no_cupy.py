"""
test_no_cupy.py
~~~~~~~~~~~~~~~
Regression tests for the optional-CuPy contract of ``agama_helper``.

CuPy is an optional extra (``pip install nbody_streams[cuda]``).  A CPU-only
install must still be able to::

    import nbody_streams
    from nbody_streams import agama_helper as ah
    coefs = ah.read_coefs("...")          # pure NumPy / h5py

and must get a *clear* ImportError - not an AttributeError, NameError or
segfault - the moment a GPU code path is touched.

The check runs in a subprocess with an import hook that makes ``import cupy``
fail, because this test module usually runs alongside tests that have already
imported CuPy for real.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


_BLOCK_CUPY = """
import sys

class _BlockCuPy:
    def find_spec(self, name, path=None, target=None):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("No module named 'cupy' (blocked by test)")
        return None

sys.meta_path.insert(0, _BlockCuPy())
for _m in [m for m in sys.modules if m == "cupy" or m.startswith("cupy.")]:
    del sys.modules[_m]

try:
    import cupy  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError("cupy import was not blocked")
"""


def _run_without_cupy(body: str) -> subprocess.CompletedProcess:
    """Execute *body* in a fresh interpreter where ``import cupy`` fails."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_CUPY + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _check(body: str) -> str:
    proc = _run_without_cupy(body)
    assert proc.returncode == 0, (
        f"subprocess failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout


def test_package_imports_without_cupy():
    """``import nbody_streams.agama_helper`` must not require CuPy."""
    out = _check(
        """
        import nbody_streams
        from nbody_streams import agama_helper as ah

        assert ah.CUPY_AVAILABLE is False
        assert nbody_streams._AGAMA_HELPER_AVAILABLE is True, (
            "agama_helper was swallowed by the ImportError guard in "
            "nbody_streams/__init__.py"
        )
        for name in ah.__all__:
            assert hasattr(ah, name), f"missing export: {name}"
        print("OK")
        """
    )
    assert "OK" in out


def test_analytic_potentials_module_imports_without_cupy():
    """Module-level ``cp.ElementwiseKernel`` calls must not fire at import."""
    out = _check(
        """
        from nbody_streams.agama_helper import _analytic_potentials as ap
        from nbody_streams.agama_helper._potential import (
            _AgamaTimeSpline, _GPUPotBase,
        )
        print("OK")
        """
    )
    assert "OK" in out


def test_cpu_only_helpers_still_work_without_cupy():
    """Coefficient dataclasses, text and HDF5 round-trips are pure NumPy."""
    out = _check(
        """
        import os, tempfile
        import numpy as np
        from nbody_streams import agama_helper as ah

        mc = ah.MultipoleCoefs(
            R_grid=np.array([1.0, 2.0, 4.0]),
            lm_labels=[(0, 0), (2, 0)],
            phi=np.array([[-1.0, 0.1], [-0.5, 0.05], [-0.25, 0.02]]),
            dphi_dr=np.array([[1.0, -0.1], [0.25, -0.02], [0.06, -0.005]]),
            metadata={"lmax": 2, "gridSizeR": 3, "symmetry": "t",
                      "type": "Multipole"},
        )

        # text round-trip
        back = ah.read_coefs(mc.to_coef_string())
        assert np.allclose(back.phi, mc.phi)

        # HDF5 round-trip
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "coefs.h5")
            mc.to_h5(path, group_fmt="snap_{i:04d}")
            back = ah.read_coefs(path, group_name="snap_0000")
            assert np.allclose(back.phi, mc.phi)

        # CPU-side spline helper lives in _potential.py but needs no GPU
        from nbody_streams.agama_helper._potential import _AgamaTimeSpline
        spl = _AgamaTimeSpline(np.linspace(0.0, 1.0, 5),
                               np.linspace(2.0, 3.0, 5)[:, None])
        assert abs(float(np.asarray(spl(0.5))[0]) - 2.5) < 1e-9
        print("OK")
        """
    )
    assert "OK" in out


def test_gpu_paths_raise_clear_import_error_without_cupy():
    """Every GPU entry point fails loudly with an actionable message."""
    out = _check(
        """
        import numpy as np
        from nbody_streams.agama_helper import _analytic_potentials as ap
        from nbody_streams.agama_helper._cupy import cp, require_cupy

        def expect(label, fn):
            try:
                fn()
            except ImportError as exc:
                msg = str(exc)
                assert "cupy" in msg.lower(), msg
                assert "nbody_streams[cuda]" in msg, msg
            else:
                raise AssertionError(f"{label} did not raise ImportError")

        # constructing any GPU potential
        expect("NFWPotentialGPU", lambda: ap.NFWPotentialGPU(1e12, 20.0))
        expect("PlummerPotentialGPU", lambda: ap.PlummerPotentialGPU(1e10, 1.0))
        expect("UniformAccelerationGPU",
               lambda: ap.UniformAccelerationGPU(ax=1.0))
        # array plumbing and kernel launches
        expect("_prep_xyz", lambda: ap._prep_xyz(np.zeros(3)))
        expect("kernel call",
               lambda: ap._nfw_phi_kernel(1.0, 1.0, 1.0, 1.0, 1.0))
        expect("cp attribute", lambda: cp.asarray([1.0]))
        expect("require_cupy", lambda: require_cupy("thing"))
        print("OK")
        """
    )
    assert "OK" in out


def test_cupy_flag_matches_reality():
    """With the real environment, ``CUPY_AVAILABLE`` tracks the import."""
    from nbody_streams import agama_helper as ah

    try:
        import cupy  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    assert ah.CUPY_AVAILABLE is installed
