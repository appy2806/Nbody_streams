#pragma once

#include "cudamem.h"
#include "Particle4.h"
#include "rtc.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <sstream>
#include <vector>

#define __out

#if 1
#define QUADRUPOLE
#endif

static void kernelSuccess(const char kernel[] = "kernel")
{
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "%s launch failed: %s\n", kernel, cudaGetErrorString(err));
    assert(0);
  }
}

struct GroupData
{
  private:
    int2 packed_data;
  public:
    __host__ __device__ GroupData(const int2 data) : packed_data(data) {}
    __host__ __device__ GroupData(const int pbeg, const int np)
    {
      packed_data.x = pbeg;
      packed_data.y = np;
    }

    __host__ __device__ int pbeg() const {return packed_data.x;}
    __host__ __device__ int np  () const {return packed_data.y;}
};

struct CellData
{
  private:
    enum {NLEAF_SHIFT = 29};
    enum {NLEAF_MASK  = ~(0x7U << NLEAF_SHIFT)};
    enum {LEVEL_SHIFT = 27};
    enum {LEVEL_MASK  = ~(0x1FU << LEVEL_SHIFT)};
    uint4 packed_data;
  public:
    __host__ __device__ CellData(
        const int level,
        const unsigned int parentCell,
        const unsigned int nBeg,
        const unsigned int nEnd,
        const unsigned int first = 0xFFFFFFFF,
        const unsigned int n = 0xFFFFFFFF)
    {
      int packed_firstleaf_n = 0xFFFFFFFF;
      if (n != 0xFFFFFFFF)
        packed_firstleaf_n = first | ((unsigned int)n << NLEAF_SHIFT);
      packed_data = make_uint4(parentCell | (level << LEVEL_SHIFT), packed_firstleaf_n, nBeg, nEnd);
    }

    __host__ __device__ CellData(const uint4 data) : packed_data(data) {}

    __host__ __device__ int n()      const {return (packed_data.y >> NLEAF_SHIFT)+1;}
    __host__ __device__ int first()  const {return packed_data.y  & NLEAF_MASK;}
    __host__ __device__ int parent() const {return packed_data.x  & LEVEL_MASK;}
    __host__ __device__ int level()  const {return packed_data.x >> LEVEL_SHIFT;}
    __host__ __device__ int pbeg()   const {return packed_data.z;}
    __host__ __device__ int pend()   const {return packed_data.w;}

    __host__ __device__ bool isLeaf() const {return packed_data.y == 0xFFFFFFFF;}
    __host__ __device__ bool isNode() const {return !isLeaf();}

    __host__ __device__ void update_first(const int first)
    {
      const int _n = n()-1;
      packed_data.y = first | ((unsigned int)_n << NLEAF_SHIFT);
    }
    __host__ __device__ void update_parent(const int parent)
    {
      packed_data.x = parent | (level() << LEVEL_SHIFT);
    }
};

template<typename real_t>
struct Quadrupole
{
  private:
    typedef typename vec<4,real_t>::type real4_t;
    typedef typename vec<2,real_t>::type real2_t;
    real4_t q0;
    real2_t q1;

  public:
    __host__ __device__ Quadrupole(const real4_t _q0, const real2_t _q1) : q0(_q0), q1(_q1) {}
    __host__ __device__ Quadrupole() : q0(vec<4,real_t>::null()), q1(vec<2,real_t>::null()) {}

    __host__ __device__ real_t xx() const {return q0.x;}
    __host__ __device__ real_t yy() const {return q0.y;}
    __host__ __device__ real_t zz() const {return q0.z;}
    __host__ __device__ real_t xy() const {return q0.w;}
    __host__ __device__ real_t xz() const {return q1.x;}
    __host__ __device__ real_t yz() const {return q1.y;}

    __host__ __device__ real_t& xx() {return q0.x;}
    __host__ __device__ real_t& yy() {return q0.y;}
    __host__ __device__ real_t& zz() {return q0.z;}
    __host__ __device__ real_t& xy() {return q0.w;}
    __host__ __device__ real_t& xz() {return q1.x;}
    __host__ __device__ real_t& yz() {return q1.y;}

    __host__ __device__ real4_t get_q0() const {return q0;}
    __host__ __device__ real2_t get_q1() const {return q1;}
};


template<typename real_t>
struct Treecode
{
  typedef Particle4<real_t> Particle;

  typedef typename vec<4,real_t>::type real4_t;
  typedef typename vec<3,real_t>::type real3_t;
  typedef typename vec<2,real_t>::type real2_t;

  /* global fallback softening (used when per-particle eps not provided) */
  real_t theta, eps2;

  private:
  int nPtcl, nLevels, nCells, nLeaves, nNodes, nGroups, nCrit, nLeaf;

  public:
    int get_nPtcl() const { return nPtcl; }
    int get_nCrit() const { return nCrit; }
    int get_nLeaf() const { return nLeaf; }

  host_mem<Particle> h_ptclPos, h_ptclVel;
  std::vector<Particle> ptcl0;
  cuda_mem<Particle> d_ptclPos, d_ptclVel, d_ptclPos_tmp, d_ptclAcc;
  cuda_mem<Box<real_t> > d_domain;
  cuda_mem<Position<real_t> > d_minmax;
  cuda_mem<int2> d_level_begIdx;

  int node_max, cell_max, stack_size;
  cuda_mem<int>  d_stack_memory_pool;
  cuda_mem<CellData> d_cellDataList, d_cellDataList_tmp;
  cuda_mem<GroupData> d_groupList;
  cuda_mem<int>  d_leafList;

  cuda_mem<int>  d_key, d_value;
  cuda_mem<int>  d_origIdx;   /* buildTree-sorted-idx → original-user-idx */

  cuda_mem<real4_t> d_cellSize,  d_cellMonopole;
  cuda_mem<real4_t> d_cellQuad0;
  cuda_mem<real2_t> d_cellQuad1;

  /* per-particle softening in two orderings:
   *   d_ptclEpsTree[i] = eps for tree-sorted particle i  (source particles)
   *   d_ptclEpsGrp [i] = eps for group-sorted particle i (query particles)
   * Both are populated automatically by buildTree/makeGroups.
   * Set via treeGPU_interface tree_set_pos_mass_eps_device(); the .x field of
   * d_ptclVel carries eps through the tree sort before being extracted here. */
  cuda_mem<float> d_ptclEpsTree, d_ptclEpsGrp;

  /* per-cell max softening (max(eps) over all particles in the cell).
   * Populated by computeMultipoles() alongside monopole/quadrupole data.
   * Used by approxAcc: eps2_ij = max(eps_i^2, eps_cell_max^2).
   * This makes direct and cell-approx interactions use the same max convention,
   * consistent with nbody_streams and standard multi-species N-body practice. */
  cuda_mem<float> d_cellEpsMax;

  /* Pre-allocated scratch for buildOctreeHost — eliminates 4× cudaMalloc/cudaFree per step.
   * BUILD_MAX_WORK  : work-queue depth (must be >= max nodes at any single tree level).
   *                   65536 is safe for NLEAF=64, N<=4M balanced trees.
   *                   Increase for highly-clustered configurations.
   * BUILD_PENDING_WORK_BYTES : must be >= sizeof(PendingWork<real_t>) in buildTree.cu.
   *                   Verified at compile time via static_assert in buildOctreeHost. */
  static constexpr int BUILD_MAX_WORK          = 262144;   /* 4× original 65536 */
  static constexpr int BUILD_PENDING_WORK_BYTES = 64;      /* >= sizeof(PendingWork<float>)=44 */

  cuda_mem<int>  d_build_octCounter;    /* (8+8)        = 16 ints                           */
  cuda_mem<int>  d_build_octCounterN;   /* (8+8+8+64+8) = 96 ints                           */
  cuda_mem<char> d_build_workQueue;     /* BUILD_MAX_WORK × BUILD_PENDING_WORK_BYTES bytes   */
  cuda_mem<int>  d_build_workCount;     /* 1 int                                             */

  double grav_potential;
  int verbose;

  Treecode(const real_t _eps = 0.01, const real_t _theta = 0.75, const int _ncrit = 2*WARP_SIZE)
  {
    eps2  = _eps*_eps;
    theta = _theta;
    nCrit = _ncrit;
    verbose = 1;
    d_domain.alloc(1);
    d_minmax.alloc(2048);
    d_level_begIdx.alloc(32);  /* max 32 levels */
    CUDA_SAFE_CALL(cudaMemset(d_level_begIdx,0, 32*sizeof(int2)));
  }

  void alloc(const int nPtcl)
  {
    this->nPtcl = nPtcl;
    h_ptclPos.alloc(nPtcl);
    h_ptclVel.alloc(nPtcl);
    d_ptclPos.alloc(nPtcl);
    d_ptclVel.alloc(nPtcl);
    d_ptclPos_tmp.alloc(nPtcl);
    d_ptclAcc.alloc(nPtcl);

    /* per-particle softening (float, 4N bytes each) */
    d_ptclEpsTree.alloc(nPtcl);
    d_ptclEpsGrp .alloc(nPtcl);

    /* allocate stack memory */
    node_max = nPtcl/5;
    if (node_max < 65536) node_max = 65536;
    stack_size = (8+8+8+64+8)*node_max;
    if (verbose) fprintf(stderr, "stack_size= %g MB \n", sizeof(int)*stack_size/1024.0/1024.0);
    d_stack_memory_pool.alloc(stack_size);

    /* allocate celldata memory */
    cell_max = nPtcl;
    if (verbose) fprintf(stderr, "celldata= %g MB \n", cell_max*sizeof(CellData)/1024.0/1024.0);
    d_cellDataList.alloc(cell_max);
    d_cellDataList_tmp.alloc(cell_max);
    d_key.alloc(cell_max);
    d_value.alloc(cell_max);
    d_origIdx.alloc(nPtcl);

    /* Pre-allocate d_leafList so d_leafList.realloc(nLeaves) in buildTree is a no-op.
     * nLeaves <= nCells <= nPtcl always holds, so nPtcl is a safe upper bound. */
    d_leafList.alloc(nPtcl);

    /* Pre-allocate buildOctreeHost scratch (eliminates per-step GPU malloc/free churn). */
    d_build_octCounter .alloc(16);
    d_build_octCounterN.alloc(96);
    d_build_workQueue  .alloc(static_cast<size_t>(BUILD_MAX_WORK) * BUILD_PENDING_WORK_BYTES);
    d_build_workCount  .alloc(1);
  };

  void buildTree(const int nLeaf = 16);
  void computeMultipoles();
  void makeGroups(int levelSplit = 1, const int nCrit = 64);
  double4 computeForces(const bool INTCOUNT = true);

};
