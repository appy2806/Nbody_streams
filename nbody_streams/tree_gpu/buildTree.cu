#include "Treecode.h"

#define NWARPS_OCTREE2 3
#define NWARPS2 NWARPS_OCTREE2
#define NWARPS  (1<<NWARPS2)
#define MAX_WORK 65536 // default was 65536 (2^16), try to increase if you get "Error: pending work queue overflow, increase MAX_WORK and recompile" message
// 1048576 is 2^20, which should be enough for 10^7 particles, but you can increase it further if you have more particles
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "cuda_primitives.h"

namespace treeBuild
{

  /* Work item for host-driven tree build (replaces CDP1 recursive launches) */
  template<typename T>
  struct PendingWork {
    Box<T> box;
    int cellParentIndex;
    int cellFirstChildIndex;
    int octant_mask;
    int mempool_offset;   /* offset into memPool for octCounterNbase */
    int nCellmax;
    int nSubNodes_y;
    int level;
  };

  static __forceinline__ __device__ void computeGridAndBlockSize(dim3 &grid, dim3 &block, const int np)
  {
    const int NTHREADS = (1<<NWARPS_OCTREE2) * WARP_SIZE;
    block = dim3(NTHREADS);
    assert(np > 0);
    grid = dim3(min(max(np/(NTHREADS*4),1), 512));
  }

  __device__ unsigned int retirementCount = 0;

  __constant__ int d_node_max;
  __constant__ int d_cell_max;

  __device__ unsigned int nnodes = 0;
  __device__ unsigned int nleaves = 0;
  __device__ unsigned int nlevels = 0;
  __device__ unsigned int nbodies_leaf = 0;
  __device__ unsigned int ncells = 0;

  __device__   int *memPool;
  __device__   CellData *cellDataList;
  __device__   void *ptclVel_tmp;
  __device__   int *origIdx_buf;

  template<int NTHREAD2>
   static __device__ float2 minmax_block(float2 sum)
    {
      extern __shared__ float shdata[];
      float *shMin = shdata;
      float *shMax = shdata + (1<<NTHREAD2);

      const int tid = threadIdx.x;
      shMin[tid] = sum.x;
      shMax[tid] = sum.y;
      __syncthreads();

#pragma unroll    
      for (int i = NTHREAD2-1; i >= 5; i--) // was >= 6 — one extra step
      {
        const int offset = 1 << i;
        if (tid < offset)
        {
          shMin[tid] = sum.x = fminf(sum.x, shMin[tid + offset]);
          shMax[tid] = sum.y = fmaxf(sum.y, shMax[tid + offset]);
        }
        __syncthreads();
      }

      // warp-level reduction: race-free on Volta/Ampere/Hopper
      if (tid < 32)
      {
        // volatile float *vshMin = shMin;
        // volatile float *vshMax = shMax;
      #pragma unroll
        for (int i = 4; i >= 0; i--) // was 5..0 via volatile — now 4..0 via shfl
        {
          sum.x = fminf(sum.x, __shfl_down_sync(0xFFFFFFFFu, sum.x, 1 << i));
          sum.y = fmaxf(sum.y, __shfl_down_sync(0xFFFFFFFFu, sum.y, 1 << i));
        }
      }

      __syncthreads();

      return sum;
    }


  template<const int NTHREAD2, typename T>
    static __global__ void computeBoundingBox(
        const int n,
        __out Position<T> *minmax_ptr,
        __out Box<T>      *box_ptr,
        const Particle4<T> *ptclPos)
    {
      const int NTHREAD = 1<<NTHREAD2;
      const int NBLOCK  = NTHREAD;

      Position<T> bmin(T(+1e10)), bmax(T(-1e10));

      const int nbeg = blockIdx.x * NTHREAD + threadIdx.x;
      for (int i = nbeg; i < n; i += NBLOCK*NTHREAD)
        if (i < n)
        {
          const Particle4<T> p = ptclPos[i];
          const Position<T> pos(p.x(), p.y(), p.z());
          bmin = Position<T>::min(bmin, pos);
          bmax = Position<T>::max(bmax, pos);
        }   

      float2 res;
      res = minmax_block<NTHREAD2>(make_float2(bmin.x, bmax.x)); bmin.x = res.x; bmax.x = res.y;
      res = minmax_block<NTHREAD2>(make_float2(bmin.y, bmax.y)); bmin.y = res.x; bmax.y = res.y;
      res = minmax_block<NTHREAD2>(make_float2(bmin.z, bmax.z)); bmin.z = res.x; bmax.z = res.y;

      if (threadIdx.x == 0) 
      {
        minmax_ptr[blockIdx.x         ] = bmin;
        minmax_ptr[blockIdx.x + NBLOCK] = bmax;
      }

      __shared__ int lastBlock; /* with bool, doesn't compile in CUDA 6.0 */
      __threadfence();
      __syncthreads();

      if (threadIdx.x == 0)
      {
        const int ticket = atomicInc(&retirementCount, NBLOCK);
        lastBlock = (ticket == NBLOCK - 1);
      }

      __syncthreads();

#if 1
      if (lastBlock)
      {

        bmin = minmax_ptr[threadIdx.x];
        bmax = minmax_ptr[threadIdx.x + NBLOCK];

        float2 res;
        res = minmax_block<NTHREAD2>(make_float2(bmin.x, bmax.x)); bmin.x = res.x; bmax.x = res.y;
        res = minmax_block<NTHREAD2>(make_float2(bmin.y, bmax.y)); bmin.y = res.x; bmax.y = res.y;
        res = minmax_block<NTHREAD2>(make_float2(bmin.z, bmax.z)); bmin.z = res.x; bmax.z = res.y;

        __syncthreads();

        if (threadIdx.x == 0)
        {
#if 0
          printf("bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
          printf("bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);
#endif
          const Position<T> cvec((bmax.x+bmin.x)*T(0.5), (bmax.y+bmin.y)*T(0.5), (bmax.z+bmin.z)*T(0.5));
          const Position<T> hvec((bmax.x-bmin.x)*T(0.5), (bmax.y-bmin.y)*T(0.5), (bmax.z-bmin.z)*T(0.5));
          const T h = fmax(hvec.z, fmax(hvec.y, hvec.x));
          T hsize = T(1.0);
          while (hsize > h) hsize *= T(0.5);
          while (hsize < h) hsize *= T(2.0);

          const int NMAXLEVEL = 20;

          const T hquant = hsize / T(1<<NMAXLEVEL);
          const long long nx = (long long)(cvec.x/hquant);
          const long long ny = (long long)(cvec.y/hquant);
          const long long nz = (long long)(cvec.z/hquant);

          const Position<T> centre(hquant * T(nx), hquant * T(ny), hquant * T(nz));

          *box_ptr = Box<T>(centre, hsize);
          retirementCount = 0;
        }
      }
#endif
    }

  /*******************/

  template<int NLEAF, typename T, bool STOREIDX>
    static __global__ void
    __launch_bounds__( 256, 8)
    buildOctant(
        Box<T> box,
        const int cellParentIndex,
        const int cellIndexBase,
        const int octantMask,
        __out int *octCounterBase,
        Particle4<T> *ptcl,
        Particle4<T> *buff,
        const int level,
        PendingWork<T> *pendingWork,
        int *pendingWorkCount)
    {
      typedef typename vec<4,T>::type T4;
      /* compute laneIdx & warpIdx for each of the threads:
       *   the thread block contains only 8 warps
       *   a warp is responsible for a single octant of the cell 
       */   
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);
      const int warpIdx = threadIdx.x >> WARP_SIZE2;

      /* We launch a 2D grid:
       *   the y-corrdinate carries info about which parent cell to process
       *   the x-coordinate is just a standard approach for CUDA parallelism 
       */
      const int octant2process = (octantMask >> (3*blockIdx.y)) & 0x7;

      /* get the pointer to atomic data that for a given octant */
      int *octCounter = octCounterBase + blockIdx.y*(8+8+8+64+8);

      /* read data about the current cell */
      const int data  = octCounter[laneIdx];
      const int nBeg  = __shfl_sync(FULL_MASK, data, 1, WARP_SIZE);
      const int nEnd  = __shfl_sync(FULL_MASK, data, 2, WARP_SIZE);
      /* if we are not at the root level, compute the geometric box
       * of the cell */
      if (!STOREIDX)
        box = ChildBox(box, octant2process);


      /* countes number of particles in each octant of a child octant */
      __shared__ int nShChildrenFine[NWARPS][9][8];
      __shared__ int nShChildren[8][8];

      Box<T> *shChildBox = (Box<T>*)&nShChildren[0][0];

      int *shdata = (int*)&nShChildrenFine[0][0][0];
#pragma unroll 
      for (int i = 0; i < 8*9*NWARPS; i += NWARPS*WARP_SIZE)
        if (i + threadIdx.x < 8*9*NWARPS)
          shdata[i + threadIdx.x] = 0;

      if (laneIdx == 0 && warpIdx < 8)
        shChildBox[warpIdx] = ChildBox(box, warpIdx);

      __syncthreads();

      /* process particle array */
      const int nBeg_block = nBeg + blockIdx.x * blockDim.x;
      for (int i = nBeg_block; i < nEnd; i += gridDim.x * blockDim.x)
      {
        Particle4<T> p4 = ptcl[min(i+threadIdx.x, nEnd-1)];

        int p4octant = p4.get_oct();
        if (STOREIDX)
        {
          p4.set_idx(i + threadIdx.x);
          p4octant = Octant(box.centre, Position<T>(p4.x(), p4.y(), p4.z()));
        }

        p4octant = i+threadIdx.x < nEnd ? p4octant : 0xF; 

        /* compute suboctant of the octant into which particle will fall */
        if (p4octant < 8)
        {
          const int p4subOctant = Octant(shChildBox[p4octant].centre, Position<T>(p4.x(), p4.y(), p4.z()));
          p4.set_oct(p4subOctant);
        }

        /* compute number of particles in each of the octants that will be processed by thead block */
        int np = 0;
#pragma unroll
        for (int octant = 0; octant < 8; octant++)
        {
          const int sum = warpBinReduce(p4octant == octant);
          if (octant == laneIdx)
            np = sum;
        }

        /* increment atomic counters in a single instruction for thread-blocks to participated */
        int addrB0;
        if (laneIdx < 8)
          addrB0 = atomicAdd(&octCounter[8+8+laneIdx], np);

        /* compute addresses where to write data */
        int cntr = 32;
        int addrW = -1;
#pragma unroll
        for (int octant = 0; octant < 8; octant++)
        {
          const int sum = warpBinReduce(p4octant == octant);

          if (sum > 0)
          {
            const int offset = warpBinExclusiveScan1(p4octant == octant);
            const int addrB = __shfl_sync(FULL_MASK, addrB0, octant, WARP_SIZE);
            if (p4octant == octant)
              addrW = addrB + offset;
            cntr -= sum;
            if (cntr == 0) break;
          }
        }

        /* write the data in a single instruction */ 
        if (addrW >= 0)
          buff[addrW] = p4;

        /* count how many particles in suboctants in each of the octants */
        cntr = 32;
#pragma unroll
        for (int octant = 0; octant < 8; octant++)
        {
          if (cntr == 0) break;
          const int sum = warpBinReduce(p4octant == octant);
          if (sum > 0)
          {
            const int subOctant = p4octant == octant ? p4.get_oct() : -1;
#pragma unroll
            for (int k = 0; k < 8; k += 4)
            {
              const int4 sum = make_int4(
                  warpBinReduce(k+0 == subOctant),
                  warpBinReduce(k+1 == subOctant),
                  warpBinReduce(k+2 == subOctant),
                  warpBinReduce(k+3 == subOctant));
              if (laneIdx == 0)
              {
                int4 value = *(int4*)&nShChildrenFine[warpIdx][octant][k];
                value.x += sum.x;
                value.y += sum.y;
                value.z += sum.z;
                value.w += sum.w;
                *(int4*)&nShChildrenFine[warpIdx][octant][k] = value;
              }
            }
            cntr -= sum;
          }
        }
      }
      __syncthreads();

      if (warpIdx >= 8) return;


#pragma unroll
      for (int k = 0; k < 8; k += 4)
      {
        int4 nSubOctant = laneIdx < NWARPS ? (*(int4*)&nShChildrenFine[laneIdx][warpIdx][k]) : make_int4(0,0,0,0);
#pragma unroll
        for (int i = NWARPS2-1; i >= 0; i--)
        {
          nSubOctant.x += __shfl_xor_sync(FULL_MASK, nSubOctant.x, 1<<i, NWARPS);
          nSubOctant.y += __shfl_xor_sync(FULL_MASK, nSubOctant.y, 1<<i, NWARPS);
          nSubOctant.z += __shfl_xor_sync(FULL_MASK, nSubOctant.z, 1<<i, NWARPS);
          nSubOctant.w += __shfl_xor_sync(FULL_MASK, nSubOctant.w, 1<<i, NWARPS);
        }
        if (laneIdx == 0)
          *(int4*)&nShChildren[warpIdx][k] = nSubOctant;
      }

      __syncthreads();

      if (laneIdx < 8)
        if (nShChildren[warpIdx][laneIdx] > 0)
          atomicAdd(&octCounter[8+16+warpIdx*8 + laneIdx], nShChildren[warpIdx][laneIdx]);

      __syncthreads();  /* must be present, otherwise race conditions occurs between parent & children */


      /* detect last thread block for unique y-coordinate of the grid:
       * mind, this cannot be done on the host, because we don't detect last 
       * block on the grid, but instead the last x-block for each of the y-coordainte of the grid
       * this should increase the degree of parallelism
       */

      int *shmem = &nShChildren[0][0];
      if (warpIdx == 0)
        shmem[laneIdx] = 0;

      int &lastBlock = shmem[0];
      if (threadIdx.x == 0)
      { __threadfence();
        const int ticket = atomicAdd(octCounter, 1);
        lastBlock = (ticket == gridDim.x-1);
      }
      __syncthreads();

      if (!lastBlock) return;

      __syncthreads();

      /* okay, we are in the last thread block, do the analysis and decide what to do next */

      if (warpIdx == 0)
        shmem[laneIdx] = 0;

      if (threadIdx.x == 0)
        atomicCAS(&nlevels, level, level+1);

      __syncthreads();

      /* compute beginning and then end addresses of the sorted particles  in the child cell */

      const int nCell = __shfl_sync(FULL_MASK, data, 8+warpIdx, WARP_SIZE);
      const int nEnd1 = octCounter[8+8+warpIdx];
      const int nBeg1 = nEnd1 - nCell;

      if (laneIdx == 0)
        shmem[warpIdx] = nCell;
      __syncthreads();

      const int npCell = laneIdx < 8 ? shmem[laneIdx] : 0;

      /* compute number of children that needs to be further split, and cmopute their offsets */
      const int2 nSubNodes = warpBinExclusiveScan(npCell > NLEAF);
      const int2 nLeaves   = warpBinExclusiveScan(npCell > 0 && npCell <= NLEAF);
      if (warpIdx == 0 && laneIdx < 8)
      {
        shmem[8 +laneIdx] = nSubNodes.x;
        shmem[16+laneIdx] = nLeaves.x;
      }

      int nCellmax = npCell;
#pragma unroll
      for (int i = 2; i >= 0; i--)
        nCellmax = max(nCellmax, __shfl_xor_sync(FULL_MASK, nCellmax, 1<<i, WARP_SIZE));

      /* if there is at least one cell to split, increment nuumber of the nodes */
      if (threadIdx.x == 0 && nSubNodes.y > 0)
      {
        shmem[16+8] = atomicAdd(&nnodes,nSubNodes.y);
#if 1   /* temp solution, a better one is to use RingBuffer */
        assert(shmem[16+8] < d_node_max);
#endif
      }

      /* writing linking info, parent, child and particle's list */
      const int nChildrenCell = warpBinReduce(npCell > 0);
      if (threadIdx.x == 0 && nChildrenCell > 0)
      {
        const int cellFirstChildIndex = atomicAdd(&ncells, nChildrenCell);
#if 1
        assert(cellFirstChildIndex + nChildrenCell < d_cell_max);
#endif
        /*** keep in mind, the 0-level will be overwritten ***/
        assert(nChildrenCell > 0);
        assert(nChildrenCell <= 8);
        const CellData cellData(level,cellParentIndex, nBeg, nEnd, cellFirstChildIndex, nChildrenCell-1);
        assert(cellData.first() < ncells);
        assert(cellData.isNode());
        cellDataList[cellIndexBase + blockIdx.y] = cellData;
        shmem[16+9] = cellFirstChildIndex;
      }

      __syncthreads();
      const int cellFirstChildIndex = shmem[16+9];
      /* compute atomic data offset for cell that need to be split */
      const int next_node = shmem[16+8];
      int *octCounterNbase = &memPool[next_node*(8+8+8+64+8)];

      const int nodeOffset = shmem[8 +warpIdx];
      const int leafOffset = shmem[16+warpIdx];

      /* if cell needs to be split, populate it shared atomic data */
      if (nCell > NLEAF)
      {
        int *octCounterN = octCounterNbase + nodeOffset*(8+8+8+64+8);

        /* number of particles in each cell's subcells */
        const int nSubCell = laneIdx < 8 ? octCounter[8+16+warpIdx*8 + laneIdx] : 0;

        /* compute offsets */
        int cellOffset = nSubCell;
#pragma unroll
        for(int i = 0; i < 3; i++)  /* log2(8) steps */
          cellOffset = shfl_scan_add_step(cellOffset, 1 << i);
        cellOffset -= nSubCell;

        /* store offset in memory */

        cellOffset = __shfl_up_sync(FULL_MASK, cellOffset, 8, WARP_SIZE);
        if (laneIdx < 8) cellOffset = nSubCell;
        else            cellOffset += nBeg1;
        cellOffset = __shfl_up_sync(FULL_MASK, cellOffset, 8, WARP_SIZE);

        if (laneIdx <  8) cellOffset = 0;
        if (laneIdx == 1) cellOffset = nBeg1;
        if (laneIdx == 2) cellOffset = nEnd1;

        if (laneIdx < 24)
          octCounterN[laneIdx] = cellOffset;
      }

      /***************************/
      /*  enqueue child work     */
      /***************************/

      /* warps cooperate to build octant mask, then thread 0 writes a work item
       * to the host-driven pending work queue (replaces CDP1 child kernel launch) */
      if (nSubNodes.y > 0 && warpIdx == 0)
      {
        /* build octant mask */
        int octant_mask = npCell > NLEAF ?  (laneIdx << (3*nSubNodes.x)) : 0;
#pragma unroll
        for (int i = 4; i >= 0; i--)
          octant_mask |= __shfl_xor_sync(FULL_MASK, octant_mask, 1<<i, WARP_SIZE);

        if (threadIdx.x == 0)
        {
          const int wi = atomicAdd(pendingWorkCount, 1);
          if (wi >= MAX_WORK)
          {
            printf("Error: pending work queue overflow, increase MAX_WORK and recompile\n");
            assert(0);
          }
          pendingWork[wi].box                = box;
          pendingWork[wi].cellParentIndex     = cellIndexBase + blockIdx.y;
          pendingWork[wi].cellFirstChildIndex = cellFirstChildIndex;
          pendingWork[wi].octant_mask         = octant_mask;
          pendingWork[wi].mempool_offset      = static_cast<int>(octCounterNbase - memPool);
          pendingWork[wi].nCellmax            = nCellmax;
          pendingWork[wi].nSubNodes_y         = nSubNodes.y;
          pendingWork[wi].level               = level + 1;
        }
      }

      /******************/
      /* process leaves */
      /******************/

      if (nCell <= NLEAF && nCell > 0)
      {
        if (laneIdx == 0)
        {
          atomicAdd(&nleaves,1);
          atomicAdd(&nbodies_leaf, nEnd1-nBeg1);
          const CellData leafData(level+1, cellIndexBase+blockIdx.y, nBeg1, nEnd1);
          assert(!leafData.isNode());
          cellDataList[cellFirstChildIndex + nSubNodes.y + leafOffset] = leafData;
        }
        if (!(level&1))
        {
          for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
            if (i < nEnd1)
            {
              Particle4<T> pos = buff[i];
              const int oidx = pos.get_idx();
              Particle4<T> vel = ((Particle4<T>*)ptclVel_tmp)[oidx];
#ifdef PSHFL_SANITY_CHECK
              pos.mass() = T(oidx);
#else
              pos.mass() = vel.mass();
#endif
              ptcl[i] = pos;
              buff[i] = vel;
              origIdx_buf[i] = oidx;
            }
        }
        else
        {
          for (int i = nBeg1+laneIdx; i < nEnd1; i += WARP_SIZE)
            if (i < nEnd1)
            {
              Particle4<T> pos = buff[i];
              const int oidx = pos.get_idx();
              Particle4<T> vel = ((Particle4<T>*)ptclVel_tmp)[oidx];
#ifdef PSHFL_SANITY_CHECK
              pos.mass() = T(oidx);
#else
              pos.mass() = vel.mass();
#endif
              buff[i] = pos;
              ptcl[i] = vel;
              origIdx_buf[i] = oidx;
            }
        }
      }
    }

  template<typename T>
    static __global__ void countAtRootNode(
        const int n,
        __out int *octCounter,
        const Box<T> box,
        const Particle4<T> *ptclPos)
    {
      int np_octant[8] = {0};
      const int beg = blockIdx.x * blockDim.x + threadIdx.x;
      for (int i = beg; i < n; i += gridDim.x * blockDim.x)
        if (i < n)
        {
          const Particle4<T> p = ptclPos[i];
          const Position<T> pos(p.x(), p.y(), p.z());
          const int octant = Octant(box.centre, pos);
          np_octant[0] += (octant == 0);
          np_octant[1] += (octant == 1);
          np_octant[2] += (octant == 2);
          np_octant[3] += (octant == 3);
          np_octant[4] += (octant == 4);
          np_octant[5] += (octant == 5);
          np_octant[6] += (octant == 6);
          np_octant[7] += (octant == 7);
        };

      const int laneIdx = threadIdx.x & (WARP_SIZE-1);
#pragma unroll
      for (int k = 0; k < 8; k++)
      {
        int np = np_octant[k];
#pragma unroll
        for (int i = 4; i >= 0; i--)
          np += __shfl_xor_sync(FULL_MASK, np, 1<<i, WARP_SIZE);
        if (laneIdx == 0)
          atomicAdd(&octCounter[8+k],np);
      }
    }

  /*-----------------------------------------------------------------------
   * Host-driven tree build (replaces CDP1 buildOctree __global__ kernel).
   * The original buildOctree ran <<<1,1>>> as a serial orchestrator that
   * launched child kernels from device code.  CUDA 13 removed CDP1, so we
   * move the orchestration to host code and use a pending-work queue that
   * buildOctant writes to instead of recursively launching children.
   *---------------------------------------------------------------------*/
  /* Extract per-particle eps from the tree-sorted vel buffer (.x field) into a flat float array.
   * Called after buildOctreeHost + pointer swap so that d_ptclVel is in tree order. */
  static __global__ void extract_eps(const int n, const float4 *d_vel, float *d_epsTree)
  {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    d_epsTree[i] = d_vel[i].x;
  }

  template<int NLEAF, typename T>
    static void buildOctreeHost(
        const int n,
        const Box<T> *d_domain_ptr,
        CellData *d_cellDataList_ptr,
        int *d_stack_memory_pool,
        Particle4<T> *ptcl,
        Particle4<T> *buff,
        Particle4<T> *d_ptclVel,
        int *d_origIdx_ptr,
        /* Pre-allocated scratch (eliminates per-step cudaMalloc/cudaFree): */
        int  *d_octCounter_pre,    /* (8+8)        ints */
        int  *d_octCounterN_pre,   /* (8+8+8+64+8) ints */
        void *d_workQueue_pre,     /* max_work × sizeof(PendingWork<T>) bytes */
        int  *d_workCount_pre,     /* 1 int */
        int   max_work)            /* capacity of d_workQueue_pre in PendingWork<T> units */
    {
      static_assert(sizeof(PendingWork<T>) <= 64,
          "PendingWork<T> exceeds BUILD_PENDING_WORK_BYTES=64; increase it in Treecode.h");
      /* --- set __device__ pointers via cudaMemcpyToSymbol --- */
      {
        CellData *tmp = d_cellDataList_ptr;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(cellDataList, &tmp, sizeof(CellData*)));
      }
      {
        void *tmp = (void*)d_ptclVel;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(ptclVel_tmp, &tmp, sizeof(void*)));
      }
      {
        int *tmp = d_origIdx_ptr;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(origIdx_buf, &tmp, sizeof(int*)));
      }
      {
        int *tmp = d_stack_memory_pool;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(memPool, &tmp, sizeof(int*)));
      }

      /* --- zero counters --- */
      {
        unsigned int zero = 0;
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(nnodes,       &zero, sizeof(unsigned int)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(nleaves,      &zero, sizeof(unsigned int)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(nlevels,      &zero, sizeof(unsigned int)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(ncells,        &zero, sizeof(unsigned int)));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(nbodies_leaf, &zero, sizeof(unsigned int)));
      }

      /* --- read domain box from device --- */
      Box<T> h_domain;
      CUDA_SAFE_CALL(cudaMemcpy(&h_domain, d_domain_ptr, sizeof(Box<T>), cudaMemcpyDeviceToHost));

      /* --- use pre-allocated octCounter (no cudaMalloc per step) --- */
      int *d_octCounter = d_octCounter_pre;
      CUDA_SAFE_CALL(cudaMemset(d_octCounter, 0, (8+8)*sizeof(int)));

      countAtRootNode<T><<<256, 256>>>(n, d_octCounter, h_domain, ptcl);
      CUDA_SAFE_CALL(cudaDeviceSynchronize());

      /* --- read counts back to host --- */
      int h_octCounter[16];
      CUDA_SAFE_CALL(cudaMemcpy(h_octCounter, d_octCounter, 16*sizeof(int), cudaMemcpyDeviceToHost));
      /* no cudaFree — buffer is owned by the Treecode struct */

      {
        int total = 0;
        for (int k = 8; k < 16; k++)
        {
          //fprintf(stderr, "octCounter[%d]= %d\n", k-8, h_octCounter[k]);
          total += h_octCounter[k];
        }
        //fprintf(stderr, "total= %d  n= %d\n", total, n);
      }

      /* --- build octCounterN on host, copy to device --- */
      int h_octCounterN[8+8+8+64+8];
      memset(h_octCounterN, 0, sizeof(h_octCounterN));
      for (int k = 0; k < 8; k++)
      {
        h_octCounterN[     k] = 0;
        h_octCounterN[8+   k] = h_octCounter[8+k];
        h_octCounterN[8+8 +k] = k == 0 ? 0 : h_octCounterN[8+8+k-1] + h_octCounterN[8+k-1];
        h_octCounterN[8+16+k] = 0;
      }
      for (int k = 8; k < 64; k++)
        h_octCounterN[8+16+k] = 0;

      h_octCounterN[1] = 0;
      h_octCounterN[2] = n;

      int *d_octCounterN = d_octCounterN_pre;
      CUDA_SAFE_CALL(cudaMemcpy(d_octCounterN, h_octCounterN, (8+8+8+64+8)*sizeof(int), cudaMemcpyHostToDevice));

      /* --- use pre-allocated work queue (no cudaMalloc per step) --- */
      PendingWork<T> *d_workQueue = reinterpret_cast<PendingWork<T>*>(d_workQueue_pre);
      int            *d_workCount = d_workCount_pre;
      CUDA_SAFE_CALL(cudaMemset(d_workCount, 0, sizeof(int)));

      /* --- launch root-level buildOctant (STOREIDX=true) --- */
      {
        const int NTHREADS = (1<<NWARPS_OCTREE2) * WARP_SIZE;
        dim3 block(NTHREADS);
        dim3 grid(std::min(std::max(n/(NTHREADS*4),1), 512));

        buildOctant<NLEAF,T,true><<<grid, block>>>(
            h_domain, 0, 0, 0, d_octCounterN,
            ptcl, buff, 0, d_workQueue, d_workCount);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }

      /* --- host-driven level-by-level loop --- */
      std::vector<PendingWork<T>> h_workItems(max_work);

      for (int level = 1; level < 32; level++)
      {
        int h_workCount = 0;
        CUDA_SAFE_CALL(cudaMemcpy(&h_workCount, d_workCount, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_workCount > max_work) {
            fprintf(stderr, "FATAL: work queue overflow (h_workCount=%d, capacity=%d).\n"
                            "       Increase BUILD_MAX_WORK in Treecode.h and recompile.\n",
                            h_workCount, max_work);
            exit(1);
          }

        if (h_workCount == 0) break;

        assert(h_workCount <= max_work);
        CUDA_SAFE_CALL(cudaMemcpy(h_workItems.data(), d_workQueue,
              h_workCount * sizeof(PendingWork<T>), cudaMemcpyDeviceToHost));

        /* reset counter for next level */
        CUDA_SAFE_CALL(cudaMemset(d_workCount, 0, sizeof(int)));

        /* ptcl/buff swap: even levels use (ptcl_orig, buff_orig),
         * odd levels use (buff_orig, ptcl_orig) */
        Particle4<T> *ptcl_arg = (level & 1) ? buff : ptcl;
        Particle4<T> *buff_arg = (level & 1) ? ptcl : buff;

        const int NTHREADS = (1<<NWARPS_OCTREE2) * WARP_SIZE;

        for (int i = 0; i < h_workCount; i++)
        {
          const PendingWork<T> &w = h_workItems[i];
          dim3 block(NTHREADS);
          dim3 grid(std::min(std::max(w.nCellmax/(NTHREADS*4), 1), 512));
          grid.y = w.nSubNodes_y;

          int *octCounterNbase = d_stack_memory_pool + w.mempool_offset;

          buildOctant<NLEAF,T,false><<<grid, block>>>(
              w.box, w.cellParentIndex, w.cellFirstChildIndex,
              w.octant_mask, octCounterNbase,
              ptcl_arg, buff_arg, w.level,
              d_workQueue, d_workCount);
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
      }

      /* --- no cleanup needed: scratch buffers are owned by the Treecode struct --- */
    }


  static __global__ void
    get_cell_levels(const int n, const CellData cellList[], CellData cellListOut[], int key[], int value[])
    {
      const int idx = blockIdx.x*blockDim.x + threadIdx.x;
      if (idx >= n) return;

      const CellData cell = cellList[idx];
      key  [idx] = cell.level();
      value[idx] = idx;
      cellListOut[idx] = cell;
    }

  static __global__ void
    write_newIdx(const int n, const int value[], int moved_to_idx[])
    {
      const int newIdx = blockIdx.x*blockDim.x + threadIdx.x;
      if (newIdx >= n) return;

      const int oldIdx = value[newIdx];
      moved_to_idx[oldIdx] = newIdx;
    }


  static __global__ void
    compute_level_begIdx(const int n, const int levels[], int2 level_begendIdx[])
    {
      const int gidx = blockIdx.x*blockDim.x + threadIdx.x;
      if (gidx >= n) return;

      extern __shared__ int shLevels[];

      const int tid = threadIdx.x;
      shLevels[tid+1] = levels[gidx];

      // compute signed block range safely (avoid unsigned underflow when blockIdx.x==0)
      const int blockStart = int(blockIdx.x) * int(blockDim.x);
      const int blockEnd   = min(blockStart + int(blockDim.x) - 1, n - 1);

      int shIdx = 0;
      int gmIdx = max(blockStart - 1, 0);        // signed subtraction: safe now
      if (tid == 1)
      {
        shIdx = blockDim.x+1;
        gmIdx = min(blockEnd + 1, n-1);
      }
      if (tid < 2)
        shLevels[shIdx] = levels[gmIdx];

      __syncthreads();

      const int idx = tid+1;
      const int currLevel = shLevels[idx];
      const int prevLevel = shLevels[idx-1];
      if (currLevel != prevLevel || gidx == 0)
        level_begendIdx[currLevel].x = gidx;

      const int nextLevel = shLevels[idx+1];
      if (currLevel != nextLevel || gidx == n-1)
        level_begendIdx[currLevel].y = gidx;
    }
  
  __device__  unsigned int leafIdx_counter = 0;
  static __global__ void
    shuffle_cells(const int n, const int value[], const int moved_to_idx[], const CellData cellListIn[], CellData cellListOut[])
    {
      const int idx = blockIdx.x*blockDim.x + threadIdx.x;
      if (idx >= n) return;

      const int mapIdx = value[idx];
      CellData cell = cellListIn[mapIdx];
      if (cell.isNode())
      {
        const int firstOld = cell.first();
        const int firstNew = moved_to_idx[firstOld];
        cell.update_first(firstNew);
      }
      if (cell.parent() > 0)
        cell.update_parent(moved_to_idx[cell.parent()]);

      cellListOut[idx] = cell;

      if (threadIdx.x == 0 && blockIdx.x == 0)
        leafIdx_counter = 0;
    }

  template<int NTHREAD2>
  static __global__ 
    void collect_leaves(const int n, const CellData *cellList, int *leafList)
    {
      const int gidx = blockDim.x*blockIdx.x + threadIdx.x;

      const CellData cell = cellList[min(gidx,n-1)];

      __shared__ int shdata[1<<NTHREAD2];

      int value = gidx < n && cell.isLeaf();
      shdata[threadIdx.x] = value;
#pragma unroll
      for (int offset2 = 0; offset2 < NTHREAD2; offset2++)
      {
        const int offset = 1 << offset2;
        __syncthreads(); 
        if (threadIdx.x >= offset)
          value += shdata[threadIdx.x - offset];
        __syncthreads();
        shdata[threadIdx.x] = value;
      }

      const int nwrite  = shdata[threadIdx.x];
      const int scatter = nwrite - (gidx < n && cell.isLeaf());

      __syncthreads();

      if (threadIdx.x == blockDim.x-1 && nwrite > 0)
        shdata[0] = atomicAdd(&leafIdx_counter, nwrite);

      __syncthreads();

      if (cell.isLeaf())
        leafList[shdata[0] + scatter] = gidx;
    }
}


  template<typename real_t>
void Treecode<real_t>::buildTree(const int nLeaf)
{
  this->nLeaf = nLeaf;
  assert(nLeaf == 16 || nLeaf == 24 || nLeaf == 32 || nLeaf == 48 || nLeaf == 64);
  /* compute bounding box */

  {
    const int NTHREAD2 = 8;
    const int NTHREAD  = 1<<NTHREAD2;
    const int NBLOCK   = NTHREAD;

    assert(2*NBLOCK <= 2048);  /* see Treecode constructor for d_minmax allocation */
    cudaDeviceSynchronize();
    kernelSuccess("cudaDomainSize0");
    const double t0 = rtc();
    treeBuild::computeBoundingBox<NTHREAD2,real_t><<<NBLOCK,NTHREAD,NTHREAD*sizeof(float2)>>>
      (nPtcl, d_minmax, d_domain, d_ptclPos);
    kernelSuccess("cudaDomainSize");
    const double dt = rtc() - t0;
    if (verbose){
      fprintf(stderr, "cudaDomainSize done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);
    }
  }

  /*** build tree ***/

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(treeBuild::d_node_max, &node_max, sizeof(int), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(treeBuild::d_cell_max, &cell_max, sizeof(int), 0, cudaMemcpyHostToDevice));

//  cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount,16384);

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<16,real_t,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<16,real_t,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<24,real_t,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<24,real_t,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<32,real_t,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<32,real_t,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<48,real_t,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<48,real_t,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<64,real_t,true>,  cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&treeBuild::buildOctant<64,real_t,false>, cudaFuncCachePreferShared));

  CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));

  {
    CUDA_SAFE_CALL(cudaMemset(d_stack_memory_pool,0,stack_size*sizeof(int)));
    cudaDeviceSynchronize();
    const double t0 = rtc();
    /* Pass pre-allocated scratch to eliminate 4× cudaMalloc/cudaFree per step. */
    #define _BONSAI_SCRATCH \
        d_build_octCounter.ptr, d_build_octCounterN.ptr, \
        d_build_workQueue.ptr,  d_build_workCount.ptr,   \
        BUILD_MAX_WORK
    switch(nLeaf)
    {
      case 16:
        treeBuild::buildOctreeHost<16,real_t>(
            nPtcl, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel, d_origIdx,
            _BONSAI_SCRATCH);
        break;
      case 24:
        treeBuild::buildOctreeHost<24,real_t>(
            nPtcl, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel, d_origIdx,
            _BONSAI_SCRATCH);
        break;
      case 32:
        treeBuild::buildOctreeHost<32,real_t>(
            nPtcl, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel, d_origIdx,
            _BONSAI_SCRATCH);
        break;
      case 48:
        treeBuild::buildOctreeHost<48,real_t>(
            nPtcl, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel, d_origIdx,
            _BONSAI_SCRATCH);
        break;
      case 64:
        treeBuild::buildOctreeHost<64,real_t>(
            nPtcl, d_domain, d_cellDataList, d_stack_memory_pool, d_ptclPos, d_ptclPos_tmp, d_ptclVel, d_origIdx,
            _BONSAI_SCRATCH);
        break;
      default:
        assert(0);
    }
    #undef _BONSAI_SCRATCH
    kernelSuccess("buildOctree");
    const double t1 = rtc();
    const double dt = t1 - t0;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nLevels, treeBuild::nlevels, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nCells,  treeBuild::ncells, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nNodes,  treeBuild::nnodes, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nLeaves, treeBuild::nleaves, sizeof(int)));
    if (verbose){
      fprintf(stderr, " buildOctree done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);
    }
      std::swap(d_ptclPos_tmp.ptr, d_ptclVel.ptr);
      /* After the swap, d_ptclVel is the tree-sorted vel buffer with eps in .x.
       * Extract eps into the dedicated float array so makeGroups and computeForces can use it. */
      {
        const int nthread = 256;
        const int nblock  = (nPtcl - 1) / nthread + 1;
        treeBuild::extract_eps<<<nblock, nthread>>>(
            nPtcl,
            (const float4*)d_ptclVel.ptr,
            d_ptclEpsTree.ptr);
        kernelSuccess("extract_eps");
      }
  }

  /* sort nodes by level */
  {
    cudaDeviceSynchronize();
    const double t0 = rtc();
    const int nthread = 256;
    const int nblock  = (nCells-1)/nthread  + 1;
    treeBuild::get_cell_levels<<<nblock,nthread>>>(nCells, d_cellDataList, d_cellDataList_tmp, d_key, d_value);

    thrust::device_ptr<int> keys_beg(d_key.ptr);
    thrust::device_ptr<int> keys_end(d_key.ptr + nCells);
    thrust::device_ptr<int> vals_beg(d_value.ptr);

    thrust::stable_sort_by_key(keys_beg, keys_end, vals_beg); 

    /* compute begining & end of each level */
    treeBuild::compute_level_begIdx<<<nblock,nthread,(nthread+2)*sizeof(int)>>>(nCells, d_key, d_level_begIdx);

    treeBuild::write_newIdx <<<nblock,nthread>>>(nCells, d_value, d_key);
    treeBuild::shuffle_cells<<<nblock,nthread>>>(nCells, d_value, d_key, d_cellDataList_tmp, d_cellDataList);

    /* group leaves */

    d_leafList.realloc(nLeaves);
    const int NTHREAD2 = 8;
    const int NTHREAD  = 256;
    const int nblock1 = (nCells-1)/NTHREAD+1;
    treeBuild::collect_leaves<NTHREAD2><<<nblock1,NTHREAD>>>(nCells, d_cellDataList, d_leafList);

    kernelSuccess("shuffle");
    const double t1 = rtc();
    const double dt = t1 - t0;
    if (verbose) {
      fprintf(stderr, " shuffle done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);
    }
#if 0
    int nnn;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nnn, treeBuild::leafIdx_counter, sizeof(int)));
    printf("nnn= %d  nLeaves= %d\n", nnn , nLeaves);
    assert(nnn == nLeaves);
    std::vector<int> leaf(nLeaves);
    d_leafList.d2h(&leaf[0]);
    for (int i = 0; i < nLeaves; i++)
      printf("leaf= %d :  %d \n",i, leaf[i]);

#endif
  }



#if 0  /* tree consistency */
  {
    std::vector<char> cells_storage(sizeof(CellData)*nCells);
    CellData *cells = (CellData*)&cells_storage[0];
    d_cellDataList.d2h(&cells[0], nCells);
    int2 levels[32];
    d_level_begIdx.d2h(levels);
    std::vector<unsigned long long> keys(nPtcl);
    for (int i= 1; i < 32; i++)
    {
      const int2 lv = levels[i];
      if (lv.y == 0) break;
      int jk = 0;
      for (int j = lv.x; j <= lv.y; j++)
        keys[jk++] = ((unsigned long long)cells[j].pbeg() << 32) | cells[j].pend();
      //      thrust::sort(&keys[0], &keys[jk]);
      int np = 0;
      for (int j = 0; j < jk ;j++)
      {
        const int pbeg = keys[j] >> 32;
        const int pend = keys[j] & 0xFFFFFFFF;
        np += pend-pbeg;
        printf("  cell= %d: np= %d: pbeg= %d  pend= %d \n", j, pend-pbeg, pbeg, pend);
      }
      printf("level= %d  ncells= %d   %d %d :: np= %d\n", i, lv.y-lv.x+1, lv.x, lv.y+1,np);
    }

    fflush(stdout);
    assert(0);

  }
#endif

#if 0  /* tree consistency */
  {
    std::vector<char> cells_storage(sizeof(CellData)*nCells);
    CellData *cells = (CellData*)&cells_storage[0];
    d_cellDataList.d2h(&cells[0], nCells);
    int2 levels[32];
    d_level_begIdx.d2h(levels);
    std::vector<unsigned long long> keys(nPtcl);
    std::vector<int> currLevel, nextLevel;
    currLevel.reserve(nPtcl);
    nextLevel.reserve(nPtcl);
    for (int i = 0; i < 8; i++)
      currLevel.push_back(i);



    for (int i= 1; i < 32; i++)
    {
      const int2 lv = levels[i];
      if (lv.y == 0) break;
      int jk = 0;
      for (int j = lv.x; j <= lv.y; j++)
        keys[jk++] = ((unsigned long long)cells[j].pbeg() << 32) | cells[j].pend();
      //      thrust::sort(&keys[0], &keys[jk]);
      int np = 0;
      for (int j = 0; j < jk ;j++)
      {
        const int pbeg = keys[j] >> 32;
        const int pend = keys[j] & 0xFFFFFFFF;
        np += pend-pbeg;
        printf("  cell= %d: np= %d: pbeg= %d  pend= %d \n", j, pend-pbeg, pbeg, pend);
      }
      printf("level= %d  ncells= %d   %d %d :: np= %d\n", i, lv.y-lv.x+1, lv.x, lv.y+1,np);
    }

    fflush(stdout);
    assert(0);

  }
#endif

#if 0
  { /* print tree structure */
    fprintf(stderr, " ncells= %d nLevels= %d  nNodes= %d nLeaves= %d (%d) \n", nCells, nLevels, nNodes, nLeaves, nNodes+nLeaves);

#if 0
    std::vector<char> cells_storage(sizeof(CellData)*nCells);
    CellData *cells = (CellData*)&cells_storage[0];
    d_cellDataList.d2h(&cells[0], nCells);

    int cellL[33] = {0};
    int np=0;
    for (int i = 0; i < nCells; i++)
    {
      const CellData cell = cells[i];
      assert(cell.level() >= 0);
      assert(cell.level() < 32);

      if (cell.isNode())
        assert(cell.first() + cell.n() <= nCells);
      else
        np += cell.pend() - cell.pbeg();
    }
    fprintf(stderr, "np_leaf= %d\n", np);
    int addr = 0;
    int nlev = 0;
    for (int i= 0; i < 32; i++)
    {
      nlev++;
      printf("level= %d  ncells= %d   %d %d \n", i, cellL[i], addr, addr + cellL[i]);
      addr += cellL[i];
      if (cellL[i+1] == 0) break;
    }
#endif

    int2 levels[32];
    d_level_begIdx.d2h(levels);
    for (int i= 0; i < nLevels; i++)
    {
      const int2 lv = levels[i];
      printf("level= %d  ncells= %d   %d %d \n", i, lv.y-lv.x+1, lv.x, lv.y+1);
    }

#if 0
    for (int i = 0; i < nCells; i++)
    {
      printf("cellIdx= %d  isNode= %s: lev= %d first= %d  n= %d  pbeg= %d  pend =%d\n",
          i, cells[i].isNode() ? "true ":"false", cells[i].level(),
          cells[i].first(), cells[i].n(), cells[i].pbeg(), cells[i].pend());
    }
    fflush(stdout);
    assert(0);
#endif

  }
#endif
}

#include "TreecodeInstances.h"
