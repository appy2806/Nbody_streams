"""H200 check of the GPU tree above the old 2^28-particle limit.

  PYTHONPATH=<repo> python benchmarks/large_n_check.py 4e8 3.1e8 <scratch dir> [mem_fraction]

1. tree_gravity_gpu, N1 = 4e8 Plummer (float32 inputs), nleaf 16 and 64:
   no buffer growth, finite output, a repeat call bitwise equal, and forces
   of 512 particles with input index >= 2^28 (and 512 below) against a
   float64 direct sum over all N1 sources (a truncated index would hand
   those particles another particle's force: errors of order 1).
2. run_nbody_gpu_tree, N2 (e.g. 3.1e8 with mem_fraction 0.95; 2.7e8 fits the default 0.8),
   3 steps, no snapshots: a sanity run that must finish with finite output.
"""
import sys, time, tempfile, numpy as np, cupy as cp
from nbody_streams.tree_gpu import tree_gravity_gpu, TreeGPU, run_nbody_gpu_tree, _force, set_memory_fraction

N1, N2, OUTDIR = int(float(sys.argv[1])), int(float(sys.argv[2])), sys.argv[3]
FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else None
assert N1 > 2**28 and N2 > 2**28
if FRAC is not None:
    set_memory_fraction(FRAC)
    print(f"memory fraction {FRAC}")
print("guard max_n", _force._max_n(), "selftest", _force._lib.tree_selftest_packing())

def plummer(n, seed):            # generated on the GPU in chunks (host RAM)
    rs = cp.random.RandomState(seed)
    x = cp.empty((n, 3), cp.float32)
    for a in range(0, n, 50_000_000):
        b = min(n, a + 50_000_000)
        r = 1.0 / cp.sqrt(rs.uniform(1e-4, 1.0, b - a) ** (-2.0 / 3.0) - 1.0)
        v = rs.normal(size=(b - a, 3)); v /= cp.linalg.norm(v, axis=1)[:, None]
        x[a:b] = (v * r[:, None]).astype(cp.float32)
    return x

direct = cp.RawKernel(r'''
extern "C" __global__ void direct(const float* x, const float* m, const float* e, long long n,
                                  const long long* tgt, int nt, double* acc) {
  // one block per target; float64 sum over all sources, max-eps Plummer softening
  __shared__ double sh[3][256];
  const long long i = tgt[blockIdx.x];
  const double xi = x[3*i], yi = x[3*i+1], zi = x[3*i+2], ei = e[i];
  double ax = 0, ay = 0, az = 0;
  for (long long j = threadIdx.x; j < n; j += blockDim.x) {
    const double dx = x[3*j] - xi, dy = x[3*j+1] - yi, dz = x[3*j+2] - zi;
    const double ej = e[j] > ei ? e[j] : ei;
    const double r2 = dx*dx + dy*dy + dz*dz + ej*ej, ri = rsqrt(r2), f = m[j]*ri*ri*ri;
    ax += f*dx; ay += f*dy; az += f*dz;
  }
  sh[0][threadIdx.x] = ax; sh[1][threadIdx.x] = ay; sh[2][threadIdx.x] = az; __syncthreads();
  for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) for (int k = 0; k < 3; k++) sh[k][threadIdx.x] += sh[k][threadIdx.x+s]; __syncthreads(); }
  if (threadIdx.x == 0) for (int k = 0; k < 3; k++) acc[3*blockIdx.x+k] = sh[k][0];
}''', "direct")

x = plummer(N1, 1)
m = cp.full(N1, 1.0 / N1, cp.float32); e = cp.full(N1, 1e-3, cp.float32)
rng = np.random.default_rng(0)
tgt = cp.asarray(np.concatenate([rng.integers(2**28, N1, 512), rng.integers(0, 2**28, 512)]).astype(np.int64))
ref = cp.empty((len(tgt), 3), cp.float64)
direct((len(tgt),), (256,), (x, m, e, np.int64(N1), tgt, np.int32(len(tgt)), ref))
for nleaf in (16, 64):
    with TreeGPU(N1, theta=0.5) as t:
        cap0 = t.build_stats()
        t0 = time.perf_counter()
        acc, phi = tree_gravity_gpu(x, m, e, G=1.0, nleaf=nleaf, tree=t); cp.cuda.Device().synchronize()
        t1 = time.perf_counter()
        acc2, _ = tree_gravity_gpu(x, m, e, G=1.0, nleaf=nleaf, tree=t); cp.cuda.Device().synchronize()
        t2 = time.perf_counter()
        st = t.build_stats()
    err = cp.linalg.norm(acc[tgt].astype(cp.float64) - ref, axis=1) / cp.linalg.norm(ref, axis=1)
    hi, lo = cp.asnumpy(err[:512]), cp.asnumpy(err[512:])
    print(f"nleaf={nleaf} calls {t1-t0:.1f}/{t2-t1:.1f}s stats={st} finite={bool(cp.isfinite(acc).all())} "
          f"repeat_equal={bool(cp.array_equal(acc, acc2))} err(idx>=2^28) p50/p99/max="
          f"{np.median(hi):.2e}/{np.percentile(hi, 99):.2e}/{hi.max():.2e} err(idx<2^28) p99={np.percentile(lo, 99):.2e}")
    assert all(st[k] == cap0[k] for k in ("cell_capacity", "node_capacity", "work_capacity"))
    assert np.percentile(hi, 99) < 1e-2 and hi.max() < 5e-2 and np.percentile(lo, 99) < 1e-2
    del acc, acc2, phi, err
    cp.get_default_memory_pool().free_all_blocks()   # the next handle's guard sees this memory
del x, m, e; cp.get_default_memory_pool().free_all_blocks()

x = cp.asnumpy(plummer(N2, 2)).astype(np.float64)
ps = np.concatenate([x, np.zeros_like(x)], 1); del x
with tempfile.TemporaryDirectory(dir=OUTDIR) as d:    # the final restart file is ~13 GB
    t0 = time.perf_counter()
    out = run_nbody_gpu_tree(ps, np.full(N2, 1.0 / N2), 0.0, 3e-4, 1e-4, 1e-3, G=1.0, theta=0.6, nleaf=64,
                             output_dir=d, save_snapshots=False, snapshots=2, verbose=True, restart_interval=None)
print(f"run N2={N2:,}: {time.perf_counter()-t0:.1f}s, finite={bool(np.isfinite(out).all())}")
print("H200_DONE")
