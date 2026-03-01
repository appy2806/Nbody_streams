"""
nbody_streams.tree_gpu._build
Console-script entry point: nbody-build-tree

Compiles libtreeGPU.so in-place inside the installed package directory
using the Makefile that ships alongside the CUDA sources.

Usage (after pip install):
    nbody-build-tree              # auto nproc
    nbody-build-tree --jobs 8     # explicit parallelism
    nbody-build-tree --clean      # clean objects first, then build
    nbody-build-tree --clean-all  # remove .so too, then build
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys


def _tree_gpu_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nbody-build-tree",
        description="Build libtreeGPU.so (GPU Barnes-Hut tree code) in-place.",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=multiprocessing.cpu_count(),
        metavar="N",
        help="Parallel make jobs (default: nproc)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run 'make clean' before building (removes .o files)",
    )
    parser.add_argument(
        "--clean-all",
        action="store_true",
        help="Run 'make clean_all' before building (removes .o and .so)",
    )
    parser.add_argument(
        "--fast-math",
        type=int,
        choices=[0, 1],
        default=1,
        metavar="{0,1}",
        help="Pass FAST_MATH=0 or FAST_MATH=1 to make (default: 1)",
    )
    args = parser.parse_args(argv)

    tree_dir = _tree_gpu_dir()
    print(f"Building libtreeGPU.so in: {tree_dir}")

    def run(target: str | None = None) -> int:
        cmd = ["make", f"-j{args.jobs}", f"FAST_MATH={args.fast_math}"]
        if target:
            cmd.append(target)
        return subprocess.call(cmd, cwd=tree_dir)

    if args.clean_all:
        rc = run("clean_all")
        if rc != 0:
            return rc
    elif args.clean:
        rc = run("clean")
        if rc != 0:
            return rc

    rc = run()
    if rc == 0:
        so_path = os.path.join(tree_dir, "libtreeGPU.so")
        if os.path.exists(so_path):
            print(f"\nSuccess: {so_path}")
        else:
            print("\nWARNING: make exited 0 but libtreeGPU.so not found.")
    else:
        print("\nBuild failed. Check CUDA installation and nvcc availability.", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
