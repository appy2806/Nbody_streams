"""Time per force evaluation against N (data for ``plot_timings.py``).

    python benchmarks/timing_sweep.py --method gpu_tree --label h200 [--out DIR]

Methods: ``cpu_direct`` (Numba, float64), ``fmm`` (pyfalcon, one core),
``gpu_direct`` (``compute_nbody_forces_gpu``), ``gpu_tree`` (``tree_gravity_gpu``
with a reused ``TreeGPU`` handle).  Plummer sphere (G = M = a = 1), Plummer
softening 0.01 a, theta = 0.6; positions are passed as float64 (the recentred
path the integrators use).  Each point records the minimum and median of up to
``--reps`` calls after one warm-up call (fewer once the calls add up to 2 s);
the sweep stops after the first N whose call exceeds ``--max-call`` seconds,
or at the first failure (e.g. out of memory).  Output:
``<out>/sweep_<label>_<method>.json``, rewritten after every point.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

NS = [2 ** k for k in range(7, 27)] + [100_000_000, 200_000_000, 300_000_000,
                                         400_000_000, 500_000_000]


def plummer(n, seed=1, xp=np):
    """Plummer positions (a = 1, radii capped at 100 a), float64, chunked."""
    rng = xp.random.default_rng(seed) if xp is np else xp.random.RandomState(seed)
    out = xp.empty((n, 3), dtype=xp.float64)
    for i in range(0, n, 20_000_000):
        k = min(20_000_000, n - i)
        u = rng.uniform(1e-10, 1.0, k) if xp is np else rng.uniform(1e-10, 1.0, k)
        r = xp.minimum(1.0 / xp.sqrt(u ** (-2.0 / 3.0) - 1.0), 100.0)
        v = rng.normal(size=(k, 3)) if xp is np else rng.normal(size=(k, 3))
        out[i:i + k] = r[:, None] * v / xp.linalg.norm(v, axis=1)[:, None]
    return out


def device_info(method):
    info = {"host": platform.node(), "python": platform.python_version()}
    try:
        info["commit"] = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    if method.startswith("gpu"):
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(0)
        info.update(gpu=p["name"].decode(), sms=p["multiProcessorCount"],
                    gpu_mem_gb=round(p["totalGlobalMem"] / 2 ** 30, 1))
    else:
        info["cpu"] = platform.processor() or platform.machine()
    return info


def make_call(method, n, args):
    eps = 0.01
    if method == "cpu_direct":
        from nbody_streams import compute_nbody_forces_cpu
        x = plummer(n); m = np.full(n, 1.0 / n)
        return lambda: compute_nbody_forces_cpu(x, m, softening=eps, G=1.0, kernel="plummer",
                                                nthreads=args.threads, precision="float64"), None
    if method == "fmm":
        import pyfalcon
        x = plummer(n); m = np.full(n, 1.0 / n)
        return lambda: pyfalcon.gravity(x, m, eps, theta=0.6, kernel=0), None
    import cupy as cp
    sync = cp.cuda.Device().synchronize
    P = plummer(n, xp=cp)
    if method == "gpu_direct":
        from nbody_streams import compute_nbody_forces_gpu
        M = cp.full(n, 1.0 / n)
        return (lambda: (compute_nbody_forces_gpu(P, M, softening=eps, G=1.0, kernel="plummer",
                                                  precision=args.precision, return_cupy=True),
                         sync())), None
    from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU
    M = cp.full(n, 1.0 / n, dtype=cp.float32)
    tree = TreeGPU(n, theta=0.6)
    return (lambda: (tree_gravity_gpu(P, M, eps, G=1.0, tree=tree), sync())), tree


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", required=True, choices=["cpu_direct", "fmm", "gpu_direct", "gpu_tree"])
    ap.add_argument("--label", required=True, help="machine label, e.g. rtx3080, l40, h200")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--nmin", type=float, default=128)
    ap.add_argument("--nmax", type=float, default=5e8)
    ap.add_argument("--max-call", type=float, default=30.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--precision", default="float32_kahan", help="gpu_direct precision")
    ap.add_argument("--threads", type=int, default=None, help="cpu_direct threads")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"sweep_{args.label}_{args.method}.json")
    meta = {"label": args.label, "method": args.method, "precision": args.precision,
            "info": device_info(args.method), "points": []}
    print(json.dumps(meta["info"]), flush=True)
    for n in NS:
        if n < args.nmin:
            continue
        if n > args.nmax:
            break
        try:
            call, handle = make_call(args.method, n, args)
            call()                                   # warm-up / JIT / compile
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter(); call(); ts.append(time.perf_counter() - t0)
                if sum(ts) > 2.0:
                    break
        except Exception as e:                       # out of memory, size limits
            print(f"{args.method} N={n:>12,}: FAILED ({type(e).__name__}: {e})", flush=True)
            meta["failed_at"] = {"n": n, "error": f"{type(e).__name__}: {e}"}
            json.dump(meta, open(out, "w"), indent=1)
            break
        t_min, t_med = float(np.min(ts)), float(np.median(ts))
        meta["points"].append({"n": n, "t_min": t_min, "t_median": t_med, "reps": len(ts)})
        print(f"{args.method} N={n:>12,}: {1e3 * t_min:11.3f} ms/call (min of {len(ts)})  "
              f"{t_min / n:.3e} s/particle", flush=True)
        json.dump(meta, open(out, "w"), indent=1)
        if handle is not None:
            handle.close()
        del call, handle
        if args.method.startswith("gpu"):
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        if t_min > args.max_call:
            break


if __name__ == "__main__":
    sys.exit(main())
