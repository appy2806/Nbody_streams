"""Build hook: compile the GPU tree library during ``pip install``.

All package metadata lives in setup.cfg.  This file only adds one step to
``build_py``: run the Makefile in ``nbody_streams/tree_gpu`` so that
``libtreeGPU.so`` is built and installed with the package.

Controlled by the environment variable ``NBODY_STREAMS_BUILD_TREE``:

    auto (default)  build when nvcc and make are found; otherwise skip with a
                    warning (the CPU and CuPy backends do not need the library)
    1               build, and fail the install if the build fails
    0               skip the build

Rebuild later (new GPU, new CUDA, or on a cluster's GPU node) with
``nbody-build-tree``.
"""
import os
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py

TREE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "nbody_streams", "tree_gpu")


def _build_tree_library():
    mode = os.environ.get("NBODY_STREAMS_BUILD_TREE", "auto").strip().lower()
    if mode in ("0", "no", "false", "off"):
        print("nbody_streams: NBODY_STREAMS_BUILD_TREE=0, skipping the GPU tree build.")
        return
    required = mode in ("1", "yes", "true", "on")

    def give_up(reason):
        msg = ("nbody_streams: GPU tree library not built (" + reason + "). "
               "The CPU and CuPy direct-sum backends work without it; build it "
               "later with `nbody-build-tree`.")
        if required:
            raise RuntimeError(msg)
        print("WARNING: " + msg)

    missing = [tool for tool in ("nvcc", "make") if shutil.which(tool) is None]
    if missing:
        give_up(" and ".join(missing) + " not found in PATH")
        return
    print("nbody_streams: building libtreeGPU.so in " + TREE_DIR)
    rc = subprocess.call(["make", "-j%d" % (os.cpu_count() or 1)], cwd=TREE_DIR)
    if rc != 0:
        give_up("make exited with status %d" % rc)


class build_py_with_tree(build_py):
    def run(self):
        _build_tree_library()
        super().run()


setup(cmdclass={"build_py": build_py_with_tree})
