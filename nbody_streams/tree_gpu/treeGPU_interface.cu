/* GPU interface implementation for nbody_streams tree_gpu library.
 *
 * Per-particle softening:
 *   eps is carried through the pipeline via ptclVel.x.
 *   buildTree extracts it into d_ptclEpsTree (tree order).
 *   makeGroups builds walk groups from tree cells, so groups index the
 *   tree-ordered arrays directly (no group-order copies).
 *   computeForces reads both:
 *     - g_ptclEpsGrp for query particles (eps_i)
 *     - g_ptclEpsTree for source particles (eps_j)
 *   Direct interaction uses: eps2_ij = max(eps_i^2, eps_j^2)
 *   Cell approximation uses: eps2_ij = max(eps_i^2, eps_cell_max^2)
 */
#include "Treecode.h"
#include <cstdio>

typedef float real_t;
typedef Treecode<real_t> Tree;

/* ---------- GPU kernels (must be outside extern "C") ---------- */

/* Load pos + mass + per-particle eps (multi-species path).
 * eps_arr[i] → ptclVel[i].x   (carried through tree sort)
 * mass[i]    → ptclVel[i].w   (required by buildOctant leaf finalisation)
 */
__global__ void k_load_pos_mass_eps(
    int n,
    float4* ptclPos,
    float4* ptclVel,
    const real_t* x, const real_t* y, const real_t* z,
    const real_t* mass, const real_t* eps_arr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptclPos[i] = make_float4(x[i], y[i], z[i], mass[i]);
    ptclVel[i] = make_float4(eps_arr[i], 0.0f, 0.0f, mass[i]);
}

/* Legacy load: positions + mass only.  Uses the global eps from the Tree
 * struct (sqrt(eps2)) so the single-softening path still works via the
 * same per-particle pipeline without any code-path divergence. */
__global__ void k_load_pos_mass(
    int n, float global_eps,
    float4* ptclPos,
    float4* ptclVel,
    const real_t* x, const real_t* y, const real_t* z,
    const real_t* mass)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptclPos[i] = make_float4(x[i], y[i], z[i], mass[i]);
    ptclVel[i] = make_float4(global_eps, 0.0f, 0.0f, mass[i]);
}

/* AoS float4 (tree order) -> SoA in user order. */
__global__ void k_extract_unsorted(
    int n,
    const int* __restrict__ d_origIdx,
    const float4* __restrict__ sorted_acc,
    real_t* ax, real_t* ay, real_t* az, real_t* pot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int orig_idx = d_origIdx[i];
    float4 a = sorted_acc[i];

    ax[orig_idx]  = a.x;
    ay[orig_idx]  = a.y;
    az[orig_idx]  = a.z;
    pot[orig_idx] = a.w;
}


/* Packing self-test (tree_selftest_packing): one thread per case, counts
 * failed round trips instead of asserting (a device assert would kill the
 * context).
 *   Particle4<float/double>: every octant 0..7 with indices near 0, 2^28 and
 *     2^29 - 1, set in both orders over arbitrary prior bits of .w.
 *   CellData: nodes with first near CELLDATA_MAX_CELLS and every child count
 *     1..8, leaves, parents near 2^32, update_first/update_parent. */
template<typename T>
__device__ int check_particle4(const int idx, const int oct, const unsigned long long junk)
{
    typedef typename vec<4,T>::type T4;
    int bad = 0;
    T4 d;
    d.x = T(1); d.y = T(2); d.z = T(3);
    if (sizeof(T) == 4) { float w = __uint_as_float((unsigned int)junk); d.w = w; }
    else                { double w = __longlong_as_double((long long)junk); d.w = w; }
    Particle4<T> p(d), q(d);
    p.set_idx(idx); p.set_oct(oct);           /* build order: index, then octant */
    q.set_oct(oct); q.set_idx(idx);
    bad += p.get_idx() != idx || p.get_oct() != oct;
    bad += q.get_idx() != idx || q.get_oct() != oct;
    p.set_oct(7 - oct);                       /* re-set keeps the index */
    bad += p.get_idx() != idx || p.get_oct() != 7 - oct;
    p.set_idx(idx ^ 1);
    bad += p.get_idx() != (idx ^ 1) || p.get_oct() != 7 - oct;
    bad += p.x() != T(1) || p.y() != T(2) || p.z() != T(3);
    return bad;
}

__global__ void k_selftest_packing(const int ncase, unsigned int *nbad)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncase) return;
    const int k = t / 64, oct = t % 8, sub = (t / 8) % 8;
    /* indices: 0..63, 2^28-32..2^28+31, 2^29-64..2^29-1, ~2^29/3 */
    const int bases[4] = {0, (1 << 28) - 32, PARTICLE4_MAX_N - 64, PARTICLE4_MAX_N / 3};
    const int idx = bases[k % 4] + ((k / 4) * 8 + sub) % 64;      /* base .. base+63 */
    unsigned long long junk = 0x9E3779B97F4A7C15ull * (unsigned long long)(t + 1);
    int bad = 0;
    if (idx >= 0 && idx < PARTICLE4_MAX_N)
    {
        bad += check_particle4<float >(idx, oct, junk);
        bad += check_particle4<double>(idx, oct, junk);
    }

    /* CellData */
    const int n = oct + 1;                                   /* 1..8 children */
    const unsigned int firstBases[3] = {0u, (1u << 28), (unsigned int)CELLDATA_MAX_CELLS - 8u - 63u};
    const unsigned int first = firstBases[k % 3] + (unsigned int)((t / 8) % 64);
    if (first + n <= (unsigned int)CELLDATA_MAX_CELLS)
    {
        const unsigned int parent = 0xFFFFFFFFu - (unsigned int)t;        /* full 32 bits */
        const unsigned int pbeg = (unsigned int)PARTICLE4_MAX_N - 1u - t, pend = pbeg + 1u;
        CellData c(parent & 0x7FFFFFFFu, pbeg, pend, first, n - 1);
        bad += !c.isNode() || c.isLeaf();
        bad += c.first() != (int)first || c.n() != n;
        bad += c.parent() != (int)(parent & 0x7FFFFFFFu);
        bad += c.pbeg() != (int)pbeg || c.pend() != (int)pend;
        c.update_parent(CELLDATA_MAX_CELLS - 1 - t);
        bad += c.parent() != CELLDATA_MAX_CELLS - 1 - t || c.first() != (int)first || c.n() != n;
        const int first2 = CELLDATA_MAX_CELLS - n - (t % 17);
        c.update_first(first2);
        bad += c.first() != first2 || c.n() != n || !c.isNode();
        bad += c.parent() != CELLDATA_MAX_CELLS - 1 - t || c.pbeg() != (int)pbeg || c.pend() != (int)pend;

        const CellData leaf(first, pbeg, pend);
        bad += !leaf.isLeaf() || leaf.isNode();
        bad += leaf.parent() != (int)first || leaf.pbeg() != (int)pbeg || leaf.pend() != (int)pend;
    }
    if (bad) atomicAdd(nbad, (unsigned int)bad);
}

/* ---------- Exported C interface ---------- */
extern "C" {

    Tree* tree_new(real_t eps, real_t theta) { return new Tree(eps, theta); }
    void tree_delete(Tree* tree) { if (tree) delete tree; }
    void tree_alloc(Tree* tree, int n) { tree->alloc(n); }
    int  tree_get_nptcl(Tree* tree) { return tree->get_nPtcl(); }

    /* ------------------------------------------------------------------ */
    /* Primary: load positions + mass + per-particle softening             */
    /* eps_arr: GPU pointer to float32 array of length n                   */
    /*   scalar eps → Python broadcasts: cp.full(n, eps, dtype=float32)   */
    /* ------------------------------------------------------------------ */
    void tree_set_pos_mass_eps_device(Tree* tree, int n,
                                      real_t* x, real_t* y, real_t* z,
                                      real_t* mass, real_t* eps_arr)
    {
        if (!tree->d_ptclPos || !tree->d_ptclVel) {
            fprintf(stderr, "Error: Tree device buffers not allocated.\n");
            return;
        }
        int block = 256;
        int grid  = (n + block - 1) / block;
        k_load_pos_mass_eps<<<grid, block>>>(
            n, (float4*)tree->d_ptclPos.ptr, (float4*)tree->d_ptclVel.ptr,
            x, y, z, mass, eps_arr);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "CUDA Error (set_pos_mass_eps): %s\n", cudaGetErrorString(err));
    }

    /* ------------------------------------------------------------------ */
    /* Legacy: load positions + mass only (uses global eps from constructor)*/
    /* ------------------------------------------------------------------ */
    void tree_set_pos_mass_device(Tree* tree, int n,
                                  real_t* x, real_t* y, real_t* z,
                                  real_t* mass)
    {
        if (!tree->d_ptclPos || !tree->d_ptclVel) {
            fprintf(stderr, "Error: Tree device buffers not allocated.\n");
            return;
        }
        const float global_eps = sqrtf(tree->eps2);
        int block = 256;
        int grid  = (n + block - 1) / block;
        k_load_pos_mass<<<grid, block>>>(
            n, global_eps, (float4*)tree->d_ptclPos.ptr, (float4*)tree->d_ptclVel.ptr,
            x, y, z, mass);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "CUDA Error (set_pos_mass): %s\n", cudaGetErrorString(err));
    }

    /* Pipeline stages */
    /* 0 on success; -1 if the build overflowed and the buffers could not
     * grow (message on stderr) */
    int  tree_build(Tree* tree, int nL) { return tree->buildTree(nL); }
    void tree_compute_multipoles(Tree* tree) { tree->computeMultipoles(); }
    void tree_make_groups(Tree* tree, int levelSplit, int nGroup) {
        tree->makeGroups(levelSplit, nGroup);
    }

    struct Res { double x, y, z, w; };
    Res tree_compute_forces(Tree* tree) {
        /* Interaction statistics need extra atomics in the walk; only collect
         * them when verbose (the Python wrapper discards the return value). */
        double4 r = tree->computeForces(tree->verbose != 0);
        return {r.x, r.y, r.z, r.w};
    }

    /* Extract results in original particle order. */
    void tree_get_acc_device(Tree* tree, int n,
                             real_t* ax, real_t* ay, real_t* az, real_t* pot)
    {
        if (!tree->d_origIdx.ptr || !tree->d_ptclAcc.ptr) {
            fprintf(stderr, "Error: Tree data not ready for extraction.\n");
            return;
        }
        int block = 256;
        int grid  = (n + block - 1) / block;
        k_extract_unsorted<<<grid, block>>>(
            n,
            tree->d_origIdx.ptr,
            (const float4*)tree->d_ptclAcc.ptr,
            ax, ay, az, pot);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "CUDA Error (get_acc): %s\n", cudaGetErrorString(err));
    }

    /* index-packing limits: particles (Particle4 index) and cells (CellData first) */
    int tree_max_particles(void) { return PARTICLE4_MAX_N; }
    int tree_max_cells(void)     { return CELLDATA_MAX_CELLS; }

    /* Last build's usage and the capacities (8 values, see Treecode::get_build_stats). */
    void tree_get_build_stats(Tree* tree, long long* out) { tree->get_build_stats(out); }

    /* Run the Particle4/CellData packing round trips on the device; returns
     * the number of failures (0 = pass), or -1 on a CUDA error. */
    int tree_selftest_packing(void)
    {
        const int ncase = 64 * 4 * 64;     /* 64 index groups x 4 bases x (8 sub x 8 octants) */
        unsigned int *d_bad = NULL, h_bad = 0;
        if (cudaMalloc(&d_bad, sizeof(unsigned int)) != cudaSuccess) return -1;
        cudaMemset(d_bad, 0, sizeof(unsigned int));
        k_selftest_packing<<<(ncase + 255) / 256, 256>>>(ncase, d_bad);
        cudaError_t err = cudaMemcpy(&h_bad, d_bad, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(d_bad);
        return err == cudaSuccess ? (int)h_bad : -1;
    }

    void tree_set_verbose(Tree* tree, int verbose) {
        if (tree) tree->verbose = verbose;
    }

    /* theta only enters computeMultipoles (cellOp = l/theta + s), so changing
     * it between force calls on a reused handle takes effect immediately. */
    void tree_set_theta(Tree* tree, real_t theta) {
        if (tree) tree->theta = theta;
    }
}
