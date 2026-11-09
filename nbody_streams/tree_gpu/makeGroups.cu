#include "Treecode.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

/* Interaction groups for the tree walk, built from tree cells.
 *
 * A group is a set of <= nCrit particles that walk the tree together, sharing
 * one opening test against the group's bounding box, so a group must be
 * spatially compact.  Groups used to be fixed chunks of nCrit particles along
 * a Peano-Hilbert ordering, split only at level-`levelSplit` key boundaries.
 * When far outliers inflate the domain, one level-5 cell holds the whole
 * system and a chunk can straddle the dense core and the sparse halo: its box
 * then covers the core, every cell is opened, and one warp sums directly over
 * all N particles while the rest of the GPU waits.  Measured: a Plummer
 * sphere with 2M (3M) particles had one group with 2.0M (3.0M) direct
 * interactions per particle against an average of ~4000, and the force pass
 * took 17x (28x) longer than at 1M.  Which N trigger it depends on where the
 * chunk boundaries land.
 *
 * Now groups are cut from "container" cells: cells with <= GROUP_CONTAINER_FACTOR
 * * nCrit particles whose parent has more (or top-level cells).  Each
 * container's contiguous particle range (tree order) is cut into
 * ceil(n/nCrit) equal chunks; buildTree sorts the particles inside each leaf
 * by original index, so the membership is reproducible.  A chunk can only
 * straddle regions inside one container, so the worst case is bounded by the container size instead of
 * N (measured max ~1e4 direct interactions per particle at 1M-3M).
 * Every particle is in exactly one container: the first cell on its path
 * from the top with <= GROUP_CONTAINER_FACTOR * nCrit particles.
 *
 * GROUP_CONTAINER_FACTOR = 32 (containers of <= 2048 particles for nCrit =
 * 64): against the Peano-Hilbert groups, median force errors are 9-14% lower
 * and 99th percentiles equal within 3% (Hernquist 1e5/1e6, tracer mixes,
 * offset cluster); smaller containers give smaller, less full groups (K = 1,
 * i.e. pure cell groups, was 2x slower), larger ones a larger worst case.
 *
 * Groups index particles in tree order (the order of d_ptclPos), so the walk
 * reads query particles and their softening from the tree-ordered arrays: no
 * Peano-Hilbert sort or particle reshuffle.
 */
#define GROUP_CONTAINER_FACTOR 32

namespace makeGroups
{
  __device__ unsigned int groupCounter = 0;

  static __device__ __forceinline__ void emit(GroupData *groupList, const int pbeg, const int np)
  {
    const int g = atomicAdd(&groupCounter, 1);
    groupList[g] = GroupData(pbeg, np);
  }

  /* One thread per cell.  A container is a cell with <= groupMax particles
   * whose parent has > groupMax (or a top-level cell), or a leaf with more
   * than groupMax particles (depth cap).  Its contiguous particle range (tree
   * order, deterministic: buildTree sorts the particles inside each leaf) is
   * cut into ceil(n/nCrit) equal chunks. */
  static __global__ void cellGroups(
      const int nCells,
      const int topBeg,
      const int topEnd,                 /* exclusive */
      const CellData *cells,
      const int nCrit,
      const int groupMax,
      GroupData *groupList)
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= nCells) return;

    const CellData cell = cells[idx];
    const int pbeg = cell.pbeg();
    const int np   = cell.pend() - pbeg;
    if (np <= 0) return;

    bool container;
    if (np <= groupMax)
    {
      if (idx >= topBeg && idx < topEnd)
        container = true;
      else
      {
        const CellData parent = cells[cell.parent()];
        container = (parent.pend() - parent.pbeg()) > groupMax;
      }
    }
    else
      container = cell.isLeaf();      /* oversized leaf (depth cap) */

    if (!container) return;

    const int nchunk = (np + nCrit - 1) / nCrit;
    for (int c = 0; c < nchunk; c++)
    {
      const int b = (int)(((long long)np * c) / nchunk);
      const int e = (int)(((long long)np * (c + 1)) / nchunk);
      emit(groupList, pbeg + b, e - b);
    }
  }

  /* GroupData is a packed int2 {pbeg, np}; sorted as int2 (Thrust needs a
   * default-constructible value type) */
  struct groupByBeg
  {
    __host__ __device__ bool operator()(const int2 a, const int2 b) const
    {
      return a.x < b.x;
    }
  };
};

  template<typename real_t>
void Treecode<real_t>::makeGroups(int /* levelSplit: unused, kept for the C API */, const int nCrit)
{
  this->nCrit = nCrit;

  d_groupList.realloc(nPtcl);

  const double t0 = verbose ? (cudaDeviceSynchronize(), rtc()) : 0.0;

  nGroups = 0;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(makeGroups::groupCounter, &nGroups, sizeof(int)));

  int2 levels[32];
  d_level_begIdx.d2h(levels, 32);
  const int topBeg = levels[1].x;
  const int topEnd = levels[1].y + 1;

  const int nthread = 256;
  const int nblock  = (nCells - 1)/nthread + 1;
  makeGroups::cellGroups<<<nblock, nthread>>>(nCells, topBeg, topEnd, d_cellDataList,
      nCrit, GROUP_CONTAINER_FACTOR * nCrit, d_groupList);
  kernelSuccess("makeGroups");

  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nGroups, makeGroups::groupCounter, sizeof(int)));

  /* walk groups in particle (tree) order: neighbouring warps then touch
   * nearby cells, which keeps the cell data in L2 */
  static_assert(sizeof(GroupData) == sizeof(int2), "GroupData must be a packed int2");
  thrust::device_ptr<int2> gbeg(reinterpret_cast<int2*>(d_groupList.ptr));
  thrust::sort(thrust::cuda::par(thrust_alloc), gbeg, gbeg + nGroups, makeGroups::groupByBeg());

  if (verbose)
  {
    cudaDeviceSynchronize();
    const double dt = rtc() - t0;
    fprintf(stderr, " makeGroups done in %g sec : %g Mptcl/sec\n", dt, nPtcl/1e6/dt);
    fprintf(stderr, "nGroup= %d <nCrit>= %g \n", nGroups, nPtcl*1.0/nGroups);
  }
}

#include "TreecodeInstances.h"
