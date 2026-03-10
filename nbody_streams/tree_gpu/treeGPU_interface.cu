/* GPU interface implementation for nbody_streams tree_gpu library.
 *
 * Per-particle softening:
 *   eps is carried through the pipeline via ptclVel.x.
 *   buildTree extracts it into d_ptclEpsTree (tree order).
 *   makeGroups shuffles it into d_ptclEpsGrp  (group order).
 *   computeForces reads both:
 *     - g_ptclEpsGrp for query particles (eps_i)
 *     - g_ptclEpsTree for source particles (eps_j)
 *   Direct interaction uses: eps2_ij = 0.5*(eps_i^2 + eps_j^2)
 *   Cell approximation uses: eps_i^2  (query particle's own softening)
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

/* AoS float4 → SoA extraction, composing two sort mappings. */
__global__ void k_extract_unsorted(
    int n,
    const int* __restrict__ d_value,
    const int* __restrict__ d_origIdx,
    const float4* __restrict__ sorted_acc,
    real_t* ax, real_t* ay, real_t* az, real_t* pot)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int orig_idx = d_origIdx[d_value[i]];
    float4 a = sorted_acc[i];

    ax[orig_idx]  = a.x;
    ay[orig_idx]  = a.y;
    az[orig_idx]  = a.z;
    pot[orig_idx] = a.w;
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
    void tree_build(Tree* tree, int nL) { tree->buildTree(nL); }
    void tree_compute_multipoles(Tree* tree) { tree->computeMultipoles(); }
    void tree_make_groups(Tree* tree, int levelSplit, int nGroup) {
        tree->makeGroups(levelSplit, nGroup);
    }

    struct Res { double x, y, z, w; };
    Res tree_compute_forces(Tree* tree) {
        double4 r = tree->computeForces();
        return {r.x, r.y, r.z, r.w};
    }

    /* Extract results in original particle order. */
    void tree_get_acc_device(Tree* tree, int n,
                             real_t* ax, real_t* ay, real_t* az, real_t* pot)
    {
        if (!tree->d_value.ptr || !tree->d_origIdx.ptr || !tree->d_ptclAcc.ptr) {
            fprintf(stderr, "Error: Tree data not ready for extraction.\n");
            return;
        }
        int block = 256;
        int grid  = (n + block - 1) / block;
        k_extract_unsorted<<<grid, block>>>(
            n,
            tree->d_value.ptr,
            tree->d_origIdx.ptr,
            (const float4*)tree->d_ptclAcc.ptr,
            ax, ay, az, pot);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "CUDA Error (get_acc): %s\n", cudaGetErrorString(err));
    }

    void tree_set_verbose(Tree* tree, int verbose) {
        if (tree) tree->verbose = verbose;
    }
}
