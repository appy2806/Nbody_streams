#include "Treecode.h"

#include "cuda_primitives.h"

namespace computeForces
{

#define CELL_LIST_MEM_PER_WARP (4096*32)

  template<int SHIFT>
    __forceinline__ static __device__ int ringAddr(const int i)
    {
      return (i & ((CELL_LIST_MEM_PER_WARP<<SHIFT) - 1));
    }


  /*******************************/
  /****** Opening criterion ******/
  /*******************************/

  //Improved Barnes Hut criterium
  static __device__ bool split_node_grav_impbh(
      const float4 cellSize,
      const float3 groupCenter,
      const float3 groupSize)
  {
    float3 dr = make_float3(
        fabsf(groupCenter.x - cellSize.x) - (groupSize.x),
        fabsf(groupCenter.y - cellSize.y) - (groupSize.y),
        fabsf(groupCenter.z - cellSize.z) - (groupSize.z)
        );

    dr.x += fabsf(dr.x); dr.x *= 0.5f;
    dr.y += fabsf(dr.y); dr.y *= 0.5f;
    dr.z += fabsf(dr.z); dr.z *= 0.5f;

    const float ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

    return (ds2 < fabsf(cellSize.w));
  }

  /******* force due to monopoles *********/

  template<typename real_t>
  static __device__ __forceinline__ typename vec<4,real_t>::type add_acc(
      typename vec<4,real_t>::type acc,  const float3 pos,
      const float massj, const float3 posj,
      const float eps2)
  {
    const float3 dr = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);

    const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;
    const float rinv   = rsqrtf(r2);
    const float rinv2  = rinv*rinv;
    const float mrinv  = massj * rinv;
    const float mrinv3 = mrinv * rinv2;

    acc.w -= mrinv;
    acc.x += mrinv3 * dr.x;
    acc.y += mrinv3 * dr.y;
    acc.z += mrinv3 * dr.z;

    return acc;
  }


  /******* force due to quadrupoles *********/

  template<typename real_t>
  static __device__ __forceinline__ typename vec<4,real_t>::type add_acc(
      typename vec<4,real_t>::type acc,
      const float3 pos,
      const float mass, const float3 com,
      const float4 Q0,  const float2 Q1, float eps2)
  {
    const float3 dr = make_float3(pos.x - com.x, pos.y - com.y, pos.z - com.z);
    const float  r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z + eps2;

    const float rinv  = rsqrtf(r2);
    const float rinv2 = rinv *rinv;
    const float mrinv  =  mass*rinv;
    const float mrinv3 = rinv2*mrinv;
    const float mrinv5 = rinv2*mrinv3;
    const float mrinv7 = rinv2*mrinv5;

    float  D0  =  mrinv;
    float  D1  = -mrinv3;
    float  D2  =  mrinv5*(  3.0f);
    float  D3  =  mrinv7*(-15.0f);

    const float q11 = Q0.x;
    const float q22 = Q0.y;
    const float q33 = Q0.z;
    const float q12 = Q0.w;
    const float q13 = Q1.x;
    const float q23 = Q1.y;

    const float  q  = q11 + q22 + q33;
    const float3 qR = make_float3(
        q11*dr.x + q12*dr.y + q13*dr.z,
        q12*dr.x + q22*dr.y + q23*dr.z,
        q13*dr.x + q23*dr.y + q33*dr.z);
    const float qRR = qR.x*dr.x + qR.y*dr.y + qR.z*dr.z;

    acc.w  -= D0 + 0.5f*(D1*q + D2*qRR);
    float C = D1 + 0.5f*(D2*q + D3*qRR);
    acc.x  += C*dr.x + D2*qR.x;
    acc.y  += C*dr.y + D2*qR.y;
    acc.z  += C*dr.z + D2*qR.z;

    return acc;
  }


  /******* evaluate forces from particles (direct summation) *******/
  /*
   * Multi-species softening — max convention (same as nbody_streams):
   *   eps2_ij = max(eps_i^2, eps_j^2)
   *
   * Symmetric: max(eps_i, eps_j) == max(eps_j, eps_i) → Newton's 3rd law satisfied.
   * Consistent with approxAcc which uses max(eps_i^2, eps_cell_max^2).
   *
   * iEps2[k] = eps_i^2 for query particle k in this warp group (loaded in treewalk).
   * g_ptclEps[ptclIdx] = eps_j for tree-sorted source particle (from d_ptclEpsTree).
   */
  template<int NI, bool FULL, typename real_t>
    static __device__ __forceinline__ void directAcc(
        typename vec<4,real_t>::type acc_i[NI],
        const float3 pos_i[NI],
        const float  iEps2[NI],
        const int ptclIdx,
        const float4 * __restrict__ g_ptcl,
        const float  * __restrict__ g_ptclEps)
    {
      const float4 M0  = (FULL || ptclIdx >= 0) ? __ldg(&g_ptcl[ptclIdx])    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float  Ej  = (FULL || ptclIdx >= 0) ? __ldg(&g_ptclEps[ptclIdx]) : 0.0f;

      for (int j = 0; j < WARP_SIZE; j++)
      {
        const float4 jM0 = make_float4(
            __shfl_sync(FULL_MASK, M0.x, j),
            __shfl_sync(FULL_MASK, M0.y, j),
            __shfl_sync(FULL_MASK, M0.z, j),
            __shfl_sync(FULL_MASK, M0.w, j));
        const float  jeps  = __shfl_sync(FULL_MASK, Ej, j);
        const float  jeps2 = jeps * jeps;
        const float  jmass = jM0.w;
        const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
        for (int k = 0; k < NI; k++)
        {
          const float eps2_ij = fmaxf(iEps2[k], jeps2);   /* max convention */
          acc_i[k] = add_acc<real_t>(acc_i[k], pos_i[k], jmass, jpos, eps2_ij);
        }
      }
    }

  /******* evaluate forces from cells (multipole approximation) *******/
  /*
   * Max softening convention — consistent with directAcc:
   *   eps2_ij = max(eps_i^2, eps_cell_max^2)
   *
   * eps_cell_max = max eps over all particles in the cell, computed in
   * computeMultipoles() and stored in d_cellEpsMax.  Using the cell's max
   * eps ensures that a tightly-softened particle (small eps_i) seeing a
   * distant cell whose particles are coarsely softened (large eps_j) gets
   * the same protection as in a direct interaction — avoiding the
   * inconsistency of the old eps_i^2-only approach.
   */
#ifdef  QUADRUPOLE
  template<int NI, bool FULL, typename real_t>
    static __device__ __forceinline__ void approxAcc(
        typename vec<4,real_t>::type acc_i[NI],
        const float3 pos_i[NI],
        const float  iEps2[NI],
        const int cellIdx,
        const float4 * __restrict__ g_cellMonopole,
        const float4 * __restrict__ g_cellQuad0,
        const float2 * __restrict__ g_cellQuad1,
        const float  * __restrict__ g_cellEpsMax)
    {
      float4 M0, Q0;
      float2 Q1;
      float  cellEpsMax;
      if (FULL || cellIdx >= 0)
      {
        M0 = __ldg(&g_cellMonopole[cellIdx]);
        const Quadrupole<float> Q(__ldg(&g_cellQuad0[cellIdx]), __ldg(&g_cellQuad1[cellIdx]));
        Q0 = Q.get_q0();
        Q1 = Q.get_q1();
        cellEpsMax = __ldg(&g_cellEpsMax[cellIdx]);
      }
      else
      {
        M0 = Q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        Q1 = make_float2(0.0f, 0.0f);
        cellEpsMax = 0.0f;
      }

      for (int j = 0; j < WARP_SIZE; j++)
      {
        const float4 jM0 = make_float4(
            __shfl_sync(FULL_MASK, M0.x, j),
            __shfl_sync(FULL_MASK, M0.y, j),
            __shfl_sync(FULL_MASK, M0.z, j),
            __shfl_sync(FULL_MASK, M0.w, j));
        const float4 jQ0 = make_float4(
            __shfl_sync(FULL_MASK, Q0.x, j),
            __shfl_sync(FULL_MASK, Q0.y, j),
            __shfl_sync(FULL_MASK, Q0.z, j),
            __shfl_sync(FULL_MASK, Q0.w, j));
        const float2 jQ1 = make_float2(
            __shfl_sync(FULL_MASK, Q1.x, j),
            __shfl_sync(FULL_MASK, Q1.y, j));
        const float  jCellEpsMax = __shfl_sync(FULL_MASK, cellEpsMax, j);
        const float  jCellEps2   = jCellEpsMax * jCellEpsMax;
        const float  jmass = jM0.w;
        const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
        for (int k = 0; k < NI; k++)
        {
          const float eps2_ij = fmaxf(iEps2[k], jCellEps2);  /* max convention */
          acc_i[k] = add_acc<real_t>(acc_i[k], pos_i[k], jmass, jpos, jQ0, jQ1, eps2_ij);
        }
      }
    }
#else
  template<int NI, bool FULL, typename real_t>
    static __device__ __forceinline__ void approxAcc(
        float4 acc_i[NI],
        const float3 pos_i[NI],
        const float  iEps2[NI],
        const int cellIdx,
        const float4 * __restrict__ g_cellMonopole,
        const float4 * __restrict__ g_cellQuad0,
        const float2 * __restrict__ g_cellQuad1,
        const float  * __restrict__ g_cellEpsMax)
    {
      const float4 M0 = (FULL || cellIdx >= 0) ? __ldg(&g_cellMonopole[cellIdx]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      const float  cellEpsMax = (FULL || cellIdx >= 0) ? __ldg(&g_cellEpsMax[cellIdx]) : 0.0f;

      for (int j = 0; j < WARP_SIZE; j++)
      {
        const float4 jM0 = make_float4(
            __shfl_sync(FULL_MASK, M0.x, j),
            __shfl_sync(FULL_MASK, M0.y, j),
            __shfl_sync(FULL_MASK, M0.z, j),
            __shfl_sync(FULL_MASK, M0.w, j));
        const float  jCellEpsMax = __shfl_sync(FULL_MASK, cellEpsMax, j);
        const float  jCellEps2   = jCellEpsMax * jCellEpsMax;
        const float  jmass = jM0.w;
        const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
#pragma unroll
        for (int k = 0; k < NI; k++)
        {
          const float eps2_ij = fmaxf(iEps2[k], jCellEps2);  /* max convention */
          acc_i[k] = add_acc<real_t>(acc_i[k], pos_i[k], jmass, jpos, eps2_ij);
        }
      }
    }
#endif



  template<int SHIFT, int BLOCKDIM2, int NI, typename real_t>
    static __device__
    uint2 treewalk_warp(
        typename vec<4,real_t>::type acc_i[NI],
        const float3 _pos_i[NI],
        const float  iEps2[NI],
        const float3 groupCentre,
        const float3 groupSize,
        const int2 top_cells,
        int *shmem,
        int *cellList,
        const uint4  * __restrict__ g_cellData,
        const float4 * __restrict__ g_cellSize,
        const float4 * __restrict__ g_cellMonopole,
        const float4 * __restrict__ g_cellQuad0,
        const float2 * __restrict__ g_cellQuad1,
        const float  * __restrict__ g_cellEpsMax,
        const float4 * __restrict__ g_ptcl,
        const float  * __restrict__ g_ptclEps)
    {
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);

      float3 pos_i[NI];
#pragma unroll 1
      for (int i = 0; i < NI; i++)
        pos_i[i] = _pos_i[i];

      uint2 interactionCounters = {0,0};

      volatile int *tmpList = shmem;

      int approxCellIdx, directPtclIdx;

      int directCounter = 0;
      int approxCounter = 0;


      for (int root_cell = top_cells.x; root_cell < top_cells.y; root_cell += WARP_SIZE)
        if (root_cell + laneIdx < top_cells.y)
          cellList[ringAddr<SHIFT>(root_cell - top_cells.x + laneIdx)] = root_cell + laneIdx;

      int nCells = top_cells.y - top_cells.x;

      int cellListBlock        = 0;
      int nextLevelCellCounter = 0;

      unsigned int cellListOffset = 0;

      while (nCells > 0)
      {
        const int cellListIdx = cellListBlock + laneIdx;
        const bool useCell    = cellListIdx < nCells;
        const int cellIdx     = cellList[ringAddr<SHIFT>(cellListOffset + cellListIdx)];
        cellListBlock += min(WARP_SIZE, nCells - cellListBlock);

        const float4   cellSize = __ldg(&g_cellSize[cellIdx]);
        const CellData cellData(__ldg(&g_cellData[cellIdx]));

        const bool splitCell = split_node_grav_impbh(cellSize, groupCentre, groupSize) ||
          (cellData.pend() - cellData.pbeg() < 3);

        const bool isNode = cellData.isNode();

        {
          const int firstChild = cellData.first();
          const int nChild= cellData.n();
          bool splitNode  = isNode && splitCell && useCell;

          const int2 childScatter = warpIntExclusiveScan(nChild & (-splitNode));

          if (childScatter.y + nCells - cellListBlock > (CELL_LIST_MEM_PER_WARP<<SHIFT))
            return make_uint2(0xFFFFFFFF,0xFFFFFFFF);

          int nChildren  = childScatter.y;
          int nProcessed = 0;
          int2 scanVal   = {0,0};
          const int offset = cellListOffset + nCells + nextLevelCellCounter;
          while (nChildren > 0)
          {
            tmpList[laneIdx] = 1;
            __syncwarp(FULL_MASK);
            if (splitNode && (childScatter.x - nProcessed < WARP_SIZE))
            {
              splitNode = false;
              tmpList[childScatter.x - nProcessed] = -1-firstChild;
            }
            __syncwarp(FULL_MASK);
            scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
            if (laneIdx < nChildren)
              cellList[ringAddr<SHIFT>(offset + nProcessed + laneIdx)] = scanVal.x;
            nChildren  -= WARP_SIZE;
            nProcessed += WARP_SIZE;
          }
          nextLevelCellCounter += childScatter.y;
        }

        {
          const bool approxCell    = !splitCell && useCell;
          const int2 approxScatter = warpBinExclusiveScan(approxCell);

          const int scatterIdx = approxCounter + approxScatter.x;
          tmpList[laneIdx] = approxCellIdx;
          __syncwarp(FULL_MASK);
          if (approxCell && scatterIdx < WARP_SIZE)
            tmpList[scatterIdx] = cellIdx;
          __syncwarp(FULL_MASK);

          approxCounter += approxScatter.y;

          if (approxCounter >= WARP_SIZE)
          {
            approxAcc<NI,true,real_t>(acc_i, pos_i, iEps2, tmpList[laneIdx],
                g_cellMonopole, g_cellQuad0, g_cellQuad1, g_cellEpsMax);

            approxCounter -= WARP_SIZE;
            const int scatterIdx = approxCounter + approxScatter.x - approxScatter.y;
            if (approxCell && scatterIdx >= 0)
              tmpList[scatterIdx] = cellIdx;
            __syncwarp(FULL_MASK);
            interactionCounters.x += WARP_SIZE;
          }
          approxCellIdx = tmpList[laneIdx];
        }

        {
          const bool isLeaf = !isNode;
          bool isDirect = splitCell && isLeaf && useCell;

          const int firstBody = cellData.pbeg();
          const int     nBody = cellData.pend() - cellData.pbeg();

          const int2 childScatter = warpIntExclusiveScan(nBody & (-isDirect));
          int nParticle  = childScatter.y;
          int nProcessed = 0;
          int2 scanVal   = {0,0};

          while (nParticle > 0)
          {
            tmpList[laneIdx] = 1;
            __syncwarp(FULL_MASK);
            if (isDirect && (childScatter.x - nProcessed < WARP_SIZE))
            {
              isDirect = false;
              tmpList[childScatter.x - nProcessed] = -1-firstBody;
            }
            __syncwarp(FULL_MASK);
            scanVal = inclusive_segscan_warp(tmpList[laneIdx], scanVal.y);
            const int  ptclIdx = scanVal.x;

            if (nParticle >= WARP_SIZE)
            {
              directAcc<NI,true, real_t>(acc_i, pos_i, iEps2, ptclIdx, g_ptcl, g_ptclEps);
              nParticle  -= WARP_SIZE;
              nProcessed += WARP_SIZE;
              interactionCounters.y += WARP_SIZE;
            }
            else
            {
              const int scatterIdx = directCounter + laneIdx;
              tmpList[laneIdx] = directPtclIdx;
              __syncwarp(FULL_MASK);
              if (scatterIdx < WARP_SIZE)
                tmpList[scatterIdx] = ptclIdx;
              __syncwarp(FULL_MASK);

              directCounter += nParticle;

              if (directCounter >= WARP_SIZE)
              {
                directAcc<NI,true, real_t>(acc_i, pos_i, iEps2, tmpList[laneIdx], g_ptcl, g_ptclEps);
                directCounter -= WARP_SIZE;
                const int scatterIdx = directCounter + laneIdx - nParticle;
                if (scatterIdx >= 0)
                  tmpList[scatterIdx] = ptclIdx;
                __syncwarp(FULL_MASK);
                interactionCounters.y += WARP_SIZE;
              }
              directPtclIdx = tmpList[laneIdx];

              nParticle = 0;
            }
          }
        }

        if (cellListBlock >= nCells)
        {
          cellListOffset += nCells;
          nCells = nextLevelCellCounter;
          cellListBlock = nextLevelCellCounter = 0;
        }

      }

      if (approxCounter > 0)
      {
        approxAcc<NI,false, real_t>(acc_i, pos_i, iEps2,
            laneIdx < approxCounter ? approxCellIdx : -1,
            g_cellMonopole, g_cellQuad0, g_cellQuad1, g_cellEpsMax);
        interactionCounters.x += approxCounter;
        approxCounter = 0;
      }

      if (directCounter > 0)
      {
        directAcc<NI,false,real_t>(acc_i, pos_i, iEps2,
            laneIdx < directCounter ? directPtclIdx : -1,
            g_ptcl, g_ptclEps);
        interactionCounters.y += directCounter;
        directCounter = 0;
      }

      return interactionCounters;
    }

  __device__ unsigned int retired_groupCount = 0;

  __device__ unsigned long long g_direct_sum = 0;
  __device__ unsigned int       g_direct_max = 0;

  __device__ unsigned long long g_approx_sum = 0;
  __device__ unsigned int       g_approx_max = 0;

  __device__ double grav_potential = 0.0;

  template<int NTHREAD2, bool STATS, int NI>
    __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    static __global__
    void treewalk(
        const int nGroups,
        const GroupData *groupList,
        const int start_level,
        const int2 *level_begIdx,
        const Particle4<float> *ptclPos,
        __out Particle4<float> *acc,
        __out int    *gmem_pool,
        const uint4  * __restrict__ g_cellData,
        const float4 * __restrict__ g_cellSize,
        const float4 * __restrict__ g_cellMonopole,
        const float4 * __restrict__ g_cellQuad0,
        const float2 * __restrict__ g_cellQuad1,
        const float  * __restrict__ g_cellEpsMax,    /* max eps per cell (for approx interactions) */
        const float4 * __restrict__ g_ptcl,
        const float  * __restrict__ g_ptclEpsTree,   /* eps for tree-sorted source particles */
        const float  * __restrict__ g_ptclEpsGrp)    /* eps for group-sorted query particles */
    {
      typedef float real_t;
      typedef typename vec<3,real_t>::type real3_t;
      typedef typename vec<4,real_t>::type real4_t;

      typedef float real_acc;
      typedef typename vec<4,real_acc>::type real4_acc;

      const int NTHREAD = 1<<NTHREAD2;
      const int shMemSize = NTHREAD;
      __shared__ int shmem_pool[shMemSize];

      const int laneIdx = threadIdx.x & (WARP_SIZE-1);
      const int warpIdx = threadIdx.x >> WARP_SIZE2;

      const int NWARP2 = NTHREAD2 - WARP_SIZE2;
      const int sh_offs = (shMemSize >> NWARP2) * warpIdx;
      int *shmem = shmem_pool + sh_offs;
      int *gmem  =  gmem_pool + CELL_LIST_MEM_PER_WARP*((blockIdx.x<<NWARP2) + warpIdx);

      int2 top_cells = level_begIdx[start_level];
      top_cells.y++;

      while(1)
      {
        int groupIdx = 0;
        if (laneIdx == 0)
          groupIdx = atomicAdd(&retired_groupCount, 1);
        groupIdx = __shfl_sync(FULL_MASK, groupIdx, 0, WARP_SIZE);

        if (groupIdx >= nGroups)
          return;

        const GroupData group = groupList[groupIdx];
        const int pbeg = group.pbeg();
        const int np   = group.np();

        real3_t iPos[NI];
        real_t  iMass[NI];
        float   iEps2[NI];   /* per-query-particle eps^2 */

#pragma unroll
        for (int i = 0; i < NI; i++)
        {
          const int idx = min(pbeg + i*WARP_SIZE+laneIdx, pbeg+np-1);
          const Particle4<real_t> ptcl = ptclPos[idx];
          iPos [i] = make_float3(ptcl.x(), ptcl.y(), ptcl.z());
          iMass[i] = ptcl.mass();
          const float ei = g_ptclEpsGrp[idx];
          iEps2[i] = ei * ei;
        }

        real3_t rmin = {iPos[0].x, iPos[0].y, iPos[0].z};
        real3_t rmax = rmin;

#pragma unroll
        for (int i = 0; i < NI; i++)
          addBoxSize(rmin, rmax, Position<real_t>(iPos[i].x, iPos[i].y, iPos[i].z));

        rmin.x = __shfl_sync(FULL_MASK, rmin.x, 0);
        rmin.y = __shfl_sync(FULL_MASK, rmin.y, 0);
        rmin.z = __shfl_sync(FULL_MASK, rmin.z, 0);
        rmax.x = __shfl_sync(FULL_MASK, rmax.x, 0);
        rmax.y = __shfl_sync(FULL_MASK, rmax.y, 0);
        rmax.z = __shfl_sync(FULL_MASK, rmax.z, 0);

        const real_t half = static_cast<real_t>(0.5f);
        const real3_t cvec = {half*(rmax.x+rmin.x), half*(rmax.y+rmin.y), half*(rmax.z+rmin.z)};
        const real3_t hvec = {half*(rmax.x-rmin.x), half*(rmax.y-rmin.y), half*(rmax.z-rmin.z)};

        const int SHIFT = 0;

        real4_acc iAcc[NI] = {vec<4,real_acc>::null()};

        uint2 counters;
        counters = treewalk_warp<SHIFT,NTHREAD2,NI,real_acc>
          (iAcc, iPos, iEps2, cvec, hvec, top_cells, shmem, gmem,
           g_cellData, g_cellSize, g_cellMonopole, g_cellQuad0, g_cellQuad1,
           g_cellEpsMax, g_ptcl, g_ptclEpsTree);

        assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));

        const int pidx = pbeg + laneIdx;
        if (STATS)
        {
          int direct_max = counters.y;
          int direct_sum = 0;

          int approx_max = counters.x;
          int approx_sum = 0;

          real_acc gpot = static_cast<real_acc>(0.0f);

#pragma unroll
          for (int i = 0; i < NI; i++)
            if (i*WARP_SIZE + pidx < pbeg + np)
            {
              gpot       += iAcc[i].w*iMass[i];
              approx_sum += counters.x;
              direct_sum += counters.y;
            }

#pragma unroll
          for (int i = WARP_SIZE2-1; i >= 0; i--)
          {
            direct_max  = max(direct_max, __shfl_xor_sync(FULL_MASK, direct_max, 1<<i));
            direct_sum += __shfl_xor_sync(FULL_MASK, direct_sum, 1<<i);

            approx_max  = max(approx_max, __shfl_xor_sync(FULL_MASK, approx_max, 1<<i));
            approx_sum += __shfl_xor_sync(FULL_MASK, approx_sum, 1<<i);

            gpot += shfl_xor(gpot, 1<<i);
          }

          if (laneIdx == 0)
          {
            atomicMax(&g_direct_max,                     direct_max);
            atomicAdd(&g_direct_sum, (unsigned long long)direct_sum);

            atomicMax(&g_approx_max,                     approx_max);
            atomicAdd(&g_approx_sum, (unsigned long long)approx_sum);

            atomicAdd_double(&grav_potential, static_cast<real_acc>(0.5f)*gpot);
          }
        }

#pragma unroll
        for (int i = 0; i < NI; i++)
          if (pidx + i*WARP_SIZE< pbeg + np)
          {
            const real4_t iacc = {iAcc[i].x, iAcc[i].y, iAcc[i].z, iAcc[i].w};
            acc[i*WARP_SIZE + pidx] = iacc;
          }
      }
    }
}

/* ---------- Active-group compaction (block-timestep support) ----------
 * k_mark_group_active: writes a 0/1 flag per group based on whether the
 *   first particle in that group is active (d_ptclActiveGrp).
 * k_compact_groups: atomically gathers active GroupData entries into a
 *   contiguous output array.  Order is not preserved (atomic), but the
 *   treewalk uses its own atomicAdd counter so order doesn't matter.
 * Both kernels are called inside computeForces when use_active is true.
 * --------------------------------------------------------------------- */
namespace activeCompact {
  __device__ int g_nActive;   /* reset to 0 before compaction */

  /* A group is active if ANY of its particles is active.
   * Scans up to ncrit (≤64) particles per group — cheap and guarantees
   * that all originally-active particles receive a force update,
   * even when the active mask has no spatial coherence. */
  static __global__ void k_mark_group_active(
      const int nGroups,
      const GroupData * __restrict__ groupList,
      const float * __restrict__ d_activeGrp,
      int * __restrict__ groupFlag)
  {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nGroups) return;
    const int pbeg = groupList[g].pbeg();
    const int np   = groupList[g].np();
    int any = 0;
    for (int j = 0; j < np && !any; j++)
      any = (d_activeGrp[pbeg + j] > 0.5f) ? 1 : 0;
    groupFlag[g] = any;
  }

  static __global__ void k_compact_groups(
      const int nGroups,
      const GroupData * __restrict__ groupList,
      const int * __restrict__ groupFlag,
      GroupData * __restrict__ out)
  {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= nGroups) return;
    if (groupFlag[g]) {
      int idx = atomicAdd(&g_nActive, 1);
      out[idx] = groupList[g];
    }
  }
} /* namespace activeCompact */

  template<typename real_t>
double4 Treecode<real_t>::computeForces(const bool INTCOUNT)
{
  const int NTHREAD2 = 7;
  const int NTHREAD  = 1<<NTHREAD2;
  cuda_mem<int> d_gmem_pool;

  const int nblock = 8*13;
  d_gmem_pool.alloc(CELL_LIST_MEM_PER_WARP*nblock*(NTHREAD/WARP_SIZE));

  const int starting_level = 1;
  int value = 0;
  cudaDeviceSynchronize();
  const double t0 = rtc();
  unsigned long long lzero = 0;
  unsigned int       uzero = 0;
  double              fzero = 0.0;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::retired_groupCount, &value, sizeof(int)));
  if (INTCOUNT)
  {
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_direct_sum, &lzero, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_direct_max, &uzero, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_approx_sum, &lzero, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::g_approx_max, &uzero, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(computeForces::grav_potential, &fzero, sizeof(double)));
  }

  /* --- Active-group compaction (block-timestep path) ---
   * When use_active is true, compact d_groupList into d_activeGroupListData
   * (only groups whose first particle is active).  Pass the compacted list
   * and its count to treewalk so inactive particles are skipped entirely.
   * When use_active is false, pass the full list (original behaviour).
   *
   * d_ptclAcc zero-init: treewalk only writes slots belonging to active groups;
   * inactive particle slots are never touched.  Zero-fill ensures callers get
   * (0,0,0,0) for inactive particles rather than uninitialized GPU memory,
   * which may contain NaN bit patterns from a previous allocation. */
  const GroupData * active_list_ptr = d_groupList.ptr;
  int               active_n        = nGroups;
  if (use_active)
  {
    const int nb = (nGroups + 255) / 256;

    /* Zero d_ptclAcc before the treewalk.  The treewalk only writes slots
     * belonging to active groups; inactive particle slots are never touched.
     * Without this, those slots contain uninitialized GPU memory (potentially
     * NaN bit patterns), which k_extract_unsorted would scatter into the output.
     * cudaMemset is on stream 0 and is serialized before the kernels below. */
    CUDA_SAFE_CALL(cudaMemset(d_ptclAcc.ptr, 0, (size_t)nPtcl * sizeof(real4_t)));

    /* Mark each group active/inactive by scanning all particles in the group.
     * Stream-0 ordering: memset completes before this kernel launches. */
    activeCompact::k_mark_group_active<<<nb, 256>>>(
        nGroups, d_groupList.ptr, d_ptclActiveGrp.ptr, d_groupFlag.ptr);

    /* Reset the atomic counter then compact.
     * cudaMemcpyToSymbol (host→device, stream 0) is a synchronous host call:
     * it blocks until k_mark_group_active completes (stream-0 serialization),
     * guaranteeing d_groupFlag is fully written before k_compact_groups reads it. */
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(activeCompact::g_nActive, &value, sizeof(int)));
    activeCompact::k_compact_groups<<<nb, 256>>>(
        nGroups, d_groupList.ptr, d_groupFlag.ptr, d_activeGroupListData.ptr);

    /* Drain the GPU so the host read of g_nActive sees the final value. */
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nActiveGroups, activeCompact::g_nActive, sizeof(int)));

    active_list_ptr = d_activeGroupListData.ptr;
    active_n        = nActiveGroups;
    if (verbose)
      fprintf(stderr, " active groups: %d / %d (%.1f%%)\n",
              nActiveGroups, nGroups, 100.0*nActiveGroups/nGroups);
  }

  if (INTCOUNT)
  {
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,true,1>, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,true,2>, cudaFuncCachePreferL1));
    if (nCrit <= WARP_SIZE)
      computeForces::treewalk<NTHREAD2,true,1><<<nblock,NTHREAD>>>(
          active_n, active_list_ptr, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool,
          (const uint4*)d_cellDataList.ptr, d_cellSize.ptr, d_cellMonopole.ptr,
          d_cellQuad0.ptr, d_cellQuad1.ptr,
          d_cellEpsMax.ptr,
          (const float4*)d_ptclPos.ptr,
          d_ptclEpsTree.ptr, d_ptclEpsGrp.ptr);
    else if (nCrit <= 2*WARP_SIZE)
      computeForces::treewalk<NTHREAD2,true,2><<<nblock,NTHREAD>>>(
          active_n, active_list_ptr, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool,
          (const uint4*)d_cellDataList.ptr, d_cellSize.ptr, d_cellMonopole.ptr,
          d_cellQuad0.ptr, d_cellQuad1.ptr,
          d_cellEpsMax.ptr,
          (const float4*)d_ptclPos.ptr,
          d_ptclEpsTree.ptr, d_ptclEpsGrp.ptr);
    else
      assert(0);
  }
  else
  {
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,false,1>, cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&computeForces::treewalk<NTHREAD2,false,2>, cudaFuncCachePreferL1));
    if (nCrit <= WARP_SIZE)
      computeForces::treewalk<NTHREAD2,false,1><<<nblock,NTHREAD>>>(
          active_n, active_list_ptr, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool,
          (const uint4*)d_cellDataList.ptr, d_cellSize.ptr, d_cellMonopole.ptr,
          d_cellQuad0.ptr, d_cellQuad1.ptr,
          d_cellEpsMax.ptr,
          (const float4*)d_ptclPos.ptr,
          d_ptclEpsTree.ptr, d_ptclEpsGrp.ptr);
    else if (nCrit <= 2*WARP_SIZE)
      computeForces::treewalk<NTHREAD2,false,2><<<nblock,NTHREAD>>>(
          active_n, active_list_ptr, starting_level, d_level_begIdx,
          d_ptclPos_tmp, d_ptclAcc,
          d_gmem_pool,
          (const uint4*)d_cellDataList.ptr, d_cellSize.ptr, d_cellMonopole.ptr,
          d_cellQuad0.ptr, d_cellQuad1.ptr,
          d_cellEpsMax.ptr,
          (const float4*)d_ptclPos.ptr,
          d_ptclEpsTree.ptr, d_ptclEpsGrp.ptr);
    else
      assert(0);
  }
  kernelSuccess("treewalk");
  const double t1 = rtc();
  const double dt = t1 - t0;
  if (verbose) fprintf(stderr, " treewalk done in %g sec : %g Mptcl/sec\n",  dt, nPtcl/1e6/dt);


  double4 interactions = {0.0, 0.0, 0.0, 0.0};

  if (INTCOUNT)
  {
    unsigned long long direct_sum, approx_sum;
    unsigned int direct_max, approx_max;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&direct_sum,     computeForces::g_direct_sum, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&direct_max,     computeForces::g_direct_max, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&approx_sum,     computeForces::g_approx_sum, sizeof(unsigned long long)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&approx_max,     computeForces::g_approx_max, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&grav_potential, computeForces::grav_potential, sizeof(double)));
    interactions.x = direct_sum*1.0/nPtcl;
    interactions.y = direct_max;
    interactions.z = approx_sum*1.0/nPtcl;
    interactions.w = approx_max;
    if (verbose) fprintf(stderr, " grav potential= %g \n", grav_potential);
  }

  return interactions;
}

#include "TreecodeInstances.h"
