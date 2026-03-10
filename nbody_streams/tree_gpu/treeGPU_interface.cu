/* GPU interface implementation for nbody_streams tree_gpu library.
 *
 * Auxiliary buffer layout (d_ptclAux per particle):
 *   .x = eps_i     (per-particle softening)
 *   .y = active    (1.0 = active, 0.0 = inactive; default 1.0)
 *   .z = reserved  (0.0)
 *   .w = mass      (required by buildOctant leaf finalisation)
 *
 * Per-particle softening pipeline:
 *   eps is carried through the tree sort via d_ptclAux.x.
 *   buildTree extracts it into d_ptclEpsTree (tree order).
 *   makeGroups shuffles it into d_ptclEpsGrp  (group order).
 *   computeForces reads both:
 *     - g_ptclEpsGrp for query particles (eps_i)
 *     - g_ptclEpsTree for source particles (eps_j)
 *   Direct interaction uses: eps2_ij = max(eps_i^2, eps_j^2)
 *   Cell approximation uses: max(eps_i^2, eps_cell_max^2)
 *
 * Active-flag pipeline (block-timestep support):
 *   active_flag is carried through the tree sort via d_ptclAux.y.
 *   buildTree extracts it into d_ptclActiveTree (tree order).
 *   makeGroups shuffles it into d_ptclActiveGrp (group order).
 *   computeForces compacts active groups → d_activeGroupListData,
 *   then runs treewalk only over active groups.
 *   Pass active=NULL (or omit) to process all particles (default).
 */
#include "Treecode.h"
#include <cstdio>

typedef float real_t;
typedef Treecode<real_t> Tree;

/* ---------- GPU kernels (must be outside extern "C") ---------- */

/* Load pos + mass + eps + active flag.
 * ptclAux layout: .x=eps  .y=active(0/1)  .z=0  .w=mass
 * When active_arr is NULL, all particles are set active (1.0f). */
__global__ void k_load_pos_mass_eps_active(
    int n,
    float4* ptclPos,
    float4* ptclAux,
    const real_t* x, const real_t* y, const real_t* z,
    const real_t* mass, const real_t* eps_arr, const float* active_arr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptclPos[i] = make_float4(x[i], y[i], z[i], mass[i]);
    float af = (active_arr != nullptr) ? active_arr[i] : 1.0f;
    ptclAux[i] = make_float4(eps_arr[i], af, 0.0f, mass[i]);
}

/* Legacy load: positions + mass only (uses global eps, all-active). */
__global__ void k_load_pos_mass(
    int n, float global_eps,
    float4* ptclPos,
    float4* ptclAux,
    const real_t* x, const real_t* y, const real_t* z,
    const real_t* mass)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptclPos[i] = make_float4(x[i], y[i], z[i], mass[i]);
    ptclAux[i] = make_float4(global_eps, 1.0f, 0.0f, mass[i]);
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
    /* Primary: load positions + mass + eps (all particles active)         */
    /* eps_arr: GPU pointer to float32 array of length n                   */
    /*   scalar eps → Python broadcasts: cp.full(n, eps, dtype=float32)   */
    /* ------------------------------------------------------------------ */
    void tree_set_pos_mass_eps_device(Tree* tree, int n,
                                      real_t* x, real_t* y, real_t* z,
                                      real_t* mass, real_t* eps_arr)
    {
        if (!tree->d_ptclPos || !tree->d_ptclAux) {
            fprintf(stderr, "Error: Tree device buffers not allocated.\n");
            return;
        }
        tree->use_active = false;
        int block = 256;
        int grid  = (n + block - 1) / block;
        k_load_pos_mass_eps_active<<<grid, block>>>(
            n, (float4*)tree->d_ptclPos.ptr, (float4*)tree->d_ptclAux.ptr,
            x, y, z, mass, eps_arr, /*active_arr=*/nullptr);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "CUDA Error (set_pos_mass_eps): %s\n", cudaGetErrorString(err));
    }

    /* ------------------------------------------------------------------ */
    /* Active-flag path: load positions + mass + eps + per-particle active */
    /* active_arr: GPU pointer to float32 array (1.0=active, 0.0=inactive)*/
    /* ------------------------------------------------------------------ */
    void tree_set_pos_mass_eps_active_device(Tree* tree, int n,
                                             real_t* x, real_t* y, real_t* z,
                                             real_t* mass, real_t* eps_arr,
                                             float* active_arr)
    {
        if (!tree->d_ptclPos || !tree->d_ptclAux) {
            fprintf(stderr, "Error: Tree device buffers not allocated.\n");
            return;
        }
        tree->use_active = true;
        int block = 256;
        int grid  = (n + block - 1) / block;
        k_load_pos_mass_eps_active<<<grid, block>>>(
            n, (float4*)tree->d_ptclPos.ptr, (float4*)tree->d_ptclAux.ptr,
            x, y, z, mass, eps_arr, active_arr);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "CUDA Error (set_pos_mass_eps_active): %s\n", cudaGetErrorString(err));
    }

    /* ------------------------------------------------------------------ */
    /* Legacy: load positions + mass only (uses global eps, all active)    */
    /* ------------------------------------------------------------------ */
    void tree_set_pos_mass_device(Tree* tree, int n,
                                  real_t* x, real_t* y, real_t* z,
                                  real_t* mass)
    {
        if (!tree->d_ptclPos || !tree->d_ptclAux) {
            fprintf(stderr, "Error: Tree device buffers not allocated.\n");
            return;
        }
        tree->use_active = false;
        const float global_eps = sqrtf(tree->eps2);
        int block = 256;
        int grid  = (n + block - 1) / block;
        k_load_pos_mass<<<grid, block>>>(
            n, global_eps, (float4*)tree->d_ptclPos.ptr, (float4*)tree->d_ptclAux.ptr,
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
