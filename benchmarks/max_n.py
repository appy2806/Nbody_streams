"""How many particles fit on one GPU, and how fast: GPU tree at large N.

    python benchmarks/max_n.py --mode call --label h200 --n 1e8 2e8 3e8 ...
    python benchmarks/max_n.py --mode run  --label h200 --n 1e8 2e8 ...

``call``: one ``TreeGPU`` handle + ``tree_gravity_gpu`` on float64 positions
already on the GPU (an analysis call).  Records the time per call (min of 3
after a warm-up) and the GPU memory in use at the peak (``cudaMemGetInfo``
total - free, so CuPy's pool and the tree's buffers both count).

``run``: ``run_nbody_gpu_tree`` (no snapshots, no external field), float64
pos/vel kept on the GPU.  Time per step from runs of 2 and 2 + ``--steps``
steps after a 1-step warm-up (the difference removes setup and transfer).
Host memory: the phase-space array is (N, 6) float64 on the host.

GPU memory is polled every 20 ms in a background thread; bytes per particle
count what the tree (and integrator) add on top of the inputs (call mode:
float64 positions + float32 masses already on the GPU, reported separately).  Each N runs in a
fresh subprocess so an out-of-memory failure (or a crash)
ends only that point; results are appended to
``<out>/maxn_<label>_<mode>.json``.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def child(mode, n, steps, mem_fraction=None):
    import numpy as np
    import cupy as cp
    if mem_fraction is not None:
        import nbody_streams.tree_gpu as tg
        tg.set_memory_fraction(mem_fraction)
    sys.path.insert(0, HERE)
    from timing_sweep import plummer

    import threading
    dev = cp.cuda.Device()
    pool = cp.get_default_memory_pool()
    free0, total = cp.cuda.runtime.memGetInfo()
    peak = [0]
    stop = threading.Event()

    def used():
        f, _ = cp.cuda.runtime.memGetInfo()
        return total - f

    def sample():
        peak[0] = max(peak[0], used())

    def poll():                                   # catches peaks inside library calls
        dev.use()
        while not stop.is_set():
            sample(); time.sleep(0.02)

    res = {"n": n, "mode": mode, "mem_fraction": mem_fraction, "gpu_mem_total_gb": total / 2 ** 30,
           "gpu_mem_used_before_gb": (total - free0) / 2 ** 30}
    if mode == "call":
        from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU
        P = plummer(n, xp=cp)                     # inputs already on the GPU, as in an analysis
        M = cp.full(n, 1.0 / n, dtype=cp.float32)
        dev.synchronize(); pool.free_all_blocks() # drop generation temporaries
        base = used(); peak[0] = base
        res["input_bytes_per_particle"] = (base - (total - free0)) / n
        threading.Thread(target=poll, daemon=True).start()
        t0 = time.perf_counter()
        tree = TreeGPU(n, theta=0.6); sample()
        acc, phi = tree_gravity_gpu(P, M, 0.01, G=1.0, tree=tree); sample()
        res["t_first_call_incl_alloc"] = time.perf_counter() - t0
        del acc, phi
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            acc, phi = tree_gravity_gpu(P, M, 0.01, G=1.0, tree=tree)
            dev.synchronize(); ts.append(time.perf_counter() - t0)
            sample(); del acc, phi
        res["t_call_min"] = min(ts)
        tree.close()
    else:
        from nbody_streams.tree_gpu import run_nbody_gpu_tree
        x = cp.asnumpy(plummer(n, xp=cp))
        dev.synchronize(); pool.free_all_blocks()
        base = used(); peak[0] = base
        threading.Thread(target=poll, daemon=True).start()
        ps = np.zeros((n, 6)); ps[:, :3] = x; del x
        ps[:, 3:] = np.random.default_rng(2).normal(scale=0.3, size=(n, 3))
        m = np.full(n, 1.0 / n)
        dt = 1e-4
        walls = {}
        for k in (1, 2, 2 + steps):               # k = 1: warm-up (compile, allocate)
            t0 = time.perf_counter()
            run_nbody_gpu_tree(ps, m, 0.0, k * dt, dt, 0.01, G=1.0, theta=0.6,
                               save_snapshots=False, restart_interval=None, verbose=False,
                               output_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "maxn_out"),
                               overwrite=True)
            walls[k] = time.perf_counter() - t0
            sample()
        res["t_step"] = (walls[2 + steps] - walls[2]) / steps
        res["wall"] = walls
    stop.set(); time.sleep(0.05); sample()
    res["gpu_mem_peak_gb"] = peak[0] / 2 ** 30
    res["gpu_bytes_per_particle"] = (peak[0] - base) / n   # tree (+ integrator), excl. inputs
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["call", "run"], required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n", type=float, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="nbody_streams.tree_gpu.set_memory_fraction (default: library default)")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._child:
        return child(args.mode, int(args.n[0]), args.steps, args.mem_fraction)

    os.makedirs(args.out, exist_ok=True)
    tag = "" if args.mem_fraction is None else f"_f{args.mem_fraction:g}"
    out = os.path.join(args.out, f"maxn_{args.label}_{args.mode}{tag}.json")
    data = json.load(open(out)) if os.path.exists(out) else {"label": args.label, "mode": args.mode,
                                                            "points": []}
    for n in args.n:
        n = int(n)
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--_child", "--mode", args.mode,
               "--label", args.label, "--n", str(n), "--steps", str(args.steps)]
        if args.mem_fraction is not None:
            cmd += ["--mem-fraction", str(args.mem_fraction)]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                r = json.loads(line[-1][7:])
            else:
                tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
                r = {"n": n, "error": " | ".join(tail)[-400:], "returncode": p.returncode}
        except subprocess.TimeoutExpired:
            r = {"n": n, "error": f"timeout after {args.timeout} s"}
        r["wall_total"] = time.perf_counter() - t0
        print(json.dumps(r), flush=True)
        data["points"] = [q for q in data["points"] if q["n"] != n] + [r]
        json.dump(data, open(out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
