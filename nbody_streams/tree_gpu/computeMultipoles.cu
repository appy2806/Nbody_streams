#include "Treecode.h"

#include "cuda_primitives.h"

/* Cell multipoles (monopole + quadrupole), bounding boxes and max softening.
 *
 * Bottom-up scheme:
 *   1. leafMultipoles   : one warp per leaf, reduces the leaf's particles.
 *   2. nodeMultipoles   : one thread per node, one launch per tree level from
 *                         the deepest level up, combines the (at most 8)
 *                         children with the parallel-axis theorem.
 * Total work is O(N + nCells).  The previous scheme reduced every cell
 * directly over all of its particles, i.e. O(N x depth), with the 8 top
 * cells each streaming N/8 particles through a single warp.
 *
 * Precision: leaf moments are accumulated about a reference point inside the
 * leaf (its first particle), so the quadrupole Q = <xx> - <x><x> does not
 * cancel catastrophically when a cell is small compared with its distance
 * from the origin.  Centres of mass are carried between levels in double
 * (d_cellComD) and child offsets d = com_child - com_parent are formed in
 * double, so node quadrupoles keep the same property.
 */

namespace multipoles
{

  template<typename treal, typename real_t>
    static __device__ __forceinline__
    void addMonopole(double4 &_M, const Particle4<real_t> ptcl)
    {
      const treal x = ptcl.x();
      const treal y = ptcl.y();
      const treal z = ptcl.z();
      const treal m = ptcl.mass();
      typename vec<4,treal>::type M = {m*x,m*y,m*z,m};
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        M.x += shfl_xor(M.x, 1<<i);
        M.y += shfl_xor(M.y, 1<<i);
        M.z += shfl_xor(M.z, 1<<i);
        M.w += shfl_xor(M.w, 1<<i);
      }

      _M.x += M.x;
      _M.y += M.y;
      _M.z += M.z;
      _M.w += M.w;
    }

  template<typename treal, typename real_t>
    static __device__ __forceinline__
    void addQuadrupole(Quadrupole<double> &_Q, const Particle4<real_t> ptcl)
    {
      const treal x = ptcl.x();
      const treal y = ptcl.y();
      const treal z = ptcl.z();
      const treal m = ptcl.mass();
      Quadrupole<treal> Q;
      Q.xx() = m * x*x;
      Q.yy() = m * y*y;
      Q.zz() = m * z*z;
      Q.xy() = m * x*y;
      Q.xz() = m * x*z;
      Q.yz() = m * y*z;
#pragma unroll
      for (int i = WARP_SIZE2-1; i >= 0; i--)
      {
        Q.xx() += shfl_xor(Q.xx(), 1<<i);
        Q.yy() += shfl_xor(Q.yy(), 1<<i);
        Q.zz() += shfl_xor(Q.zz(), 1<<i);
        Q.xy() += shfl_xor(Q.xy(), 1<<i);
        Q.xz() += shfl_xor(Q.xz(), 1<<i);
        Q.yz() += shfl_xor(Q.yz(), 1<<i);
      }

      _Q.xx() += Q.xx();
      _Q.yy() += Q.yy();
      _Q.zz() += Q.zz();
      _Q.xy() += Q.xy();
      _Q.xz() += Q.xz();
      _Q.yz() += Q.yz();
    }

  /* Write the per-cell outputs shared by the leaf and node passes. */
  template<typename real_t>
    static __device__ __forceinline__
    void storeCell(
        const int cellIdx,
        const double4 comD,                 /* absolute COM (xyz) and mass (w) */
        const Quadrupole<double> &Q,        /* central second moments per unit mass */
        const float3 rmin, const float3 rmax,
        const float cellEpsMax,
        const real_t inv_theta,
        typename vec<4,real_t>::type *sizeList,
        typename vec<4,real_t>::type *monopoleList,
        typename vec<4,real_t>::type *quadrpl0List,
        typename vec<2,real_t>::type *quadrpl1List,
        float  *epsMaxList,
        double4 *comDList,
        float4 *boxMinList,
        float4 *boxMaxList)
    {
      typedef typename vec<4,real_t>::type real4_t;
      typedef typename vec<2,real_t>::type real2_t;

      /* Opening radius l/theta + s: l = longest side of the tight particle
       * box, s = |box centre - COM| (Salmon-Warren offset). */
      const real_t cx = (rmax.x+rmin.x)*real_t(0.5f), hx = (rmax.x-rmin.x)*real_t(0.5f);
      const real_t cy = (rmax.y+rmin.y)*real_t(0.5f), hy = (rmax.y-rmin.y)*real_t(0.5f);
      const real_t cz = (rmax.z+rmin.z)*real_t(0.5f), hz = (rmax.z-rmin.z)*real_t(0.5f);
      const real_t dx = cx - real_t(comD.x);
      const real_t dy = cy - real_t(comD.y);
      const real_t dz = cz - real_t(comD.z);
      const real_t s  = sqrt(dx*dx + dy*dy + dz*dz);
      const real_t l  = max(real_t(2.0f)*max(hx, max(hy, hz)), real_t(1.0e-6f));
      const real_t cellOp = l*inv_theta + s;

      sizeList    [cellIdx] = make_float4(comD.x, comD.y, comD.z, cellOp*cellOp);
      monopoleList[cellIdx] = (real4_t){real_t(comD.x), real_t(comD.y), real_t(comD.z), real_t(comD.w)};
      const double4 q0 = Q.get_q0();
      const double2 q1 = Q.get_q1();
      quadrpl0List[cellIdx] = (real4_t){real_t(q0.x), real_t(q0.y), real_t(q0.z), real_t(q0.w)};
      quadrpl1List[cellIdx] = (real2_t){real_t(q1.x), real_t(q1.y)};
      epsMaxList  [cellIdx] = cellEpsMax;
      comDList    [cellIdx] = comD;
      boxMinList  [cellIdx] = make_float4(rmin.x, rmin.y, rmin.z, 0.0f);
      boxMaxList  [cellIdx] = make_float4(rmax.x, rmax.y, rmax.z, 0.0f);
    }

  /* Pass 1: one warp per leaf. */
  template<int NTHREAD2, typename real_t>
    static __global__
    __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
    void leafMultipoles(
        const int nLeaves,
        const int *leafList,
        const CellData *cells,
        const Particle4<real_t> *ptclPos,
        const float * __restrict__ ptclEps,     /* eps for tree-sorted particles (d_ptclEpsTree) */
        const real_t inv_theta,
        __out typename vec<4,real_t>::type *sizeList,
        __out typename vec<4,real_t>::type *monopoleList,
        __out typename vec<4,real_t>::type *quadrpl0List,
        __out typename vec<2,real_t>::type *quadrpl1List,
        __out float   *epsMaxList,
        __out double4 *comDList,
        __out float4  *boxMinList,
        __out float4  *boxMaxList)
    {
      const int warpIdx = threadIdx.x >> WARP_SIZE2;
      const int laneIdx = threadIdx.x & (WARP_SIZE-1);

      const int NWARP2  = NTHREAD2 - WARP_SIZE2;
      const int leafIdx = (blockIdx.x<<NWARP2) + warpIdx;
      if (leafIdx >= nLeaves) return;

      const int cellIdx = leafList[leafIdx];
      const CellData cell = cells[cellIdx];

      const real_t huge = static_cast<real_t>(1e10f);
      typename vec<3,real_t>::type rmin = {+huge,+huge,+huge};
      typename vec<3,real_t>::type rmax = {-huge,-huge,-huge};
      double4 M = {0.0};
      Quadrupole<double> Q;
      float cellEpsMax = 0.0f;

      const int firstBody = cell.pbeg();
      const int  lastBody = cell.pend();
      const Particle4<real_t> pref(ptclPos[firstBody]);
      const double refx = pref.x();
      const double refy = pref.y();
      const double refz = pref.z();

      for (int i = firstBody; i < lastBody; i += WARP_SIZE)
      {
        Particle4<real_t> ptcl(ptclPos[min(i+laneIdx,lastBody-1)]);
        if (i + laneIdx >= lastBody)
          ptcl.mass() = static_cast<real_t>(0.0f);

        addBoxSize(rmin, rmax, Position<real_t>(ptcl.x(),ptcl.y(),ptcl.z()));

        Particle4<real_t> prel = ptcl;   /* position relative to the reference point */
        prel.x() -= static_cast<real_t>(refx);
        prel.y() -= static_cast<real_t>(refy);
        prel.z() -= static_cast<real_t>(refz);
        addMonopole<float>(M, prel);
        addQuadrupole<float>(Q, prel);

        /* warp-level max reduction for eps */
        const float ei_raw = __ldg(&ptclEps[min(i+laneIdx, lastBody-1)]);
        float ei = (i + laneIdx < lastBody) ? ei_raw : 0.0f;
#pragma unroll
        for (int k = WARP_SIZE2-1; k >= 0; k--)
          ei = fmaxf(ei, __shfl_xor_sync(0xFFFFFFFF, ei, 1<<k));
        cellEpsMax = fmaxf(cellEpsMax, ei);
      }

      if (laneIdx == 0)
      {
        /* A leaf holding only massless particles (e.g. tracers) has M.w == 0:
         * give it zero mass and quadrupole and a COM at the reference point
         * (a finite point inside the cell) so it contributes nothing instead
         * of propagating NaN into the opening test and every force. */
        const double inv_mass = M.w > 0.0 ? 1.0/M.w : 0.0;
        M.x *= inv_mass;
        M.y *= inv_mass;
        M.z *= inv_mass;
        Q.xx() = Q.xx()*inv_mass - M.x*M.x;
        Q.yy() = Q.yy()*inv_mass - M.y*M.y;
        Q.zz() = Q.zz()*inv_mass - M.z*M.z;
        Q.xy() = Q.xy()*inv_mass - M.x*M.y;
        Q.xz() = Q.xz()*inv_mass - M.x*M.z;
        Q.yz() = Q.yz()*inv_mass - M.y*M.z;
        /* back to absolute coordinates (the quadrupole is translation invariant) */
        const double4 comD = make_double4(M.x + refx, M.y + refy, M.z + refz, M.w);

        storeCell<real_t>(cellIdx, comD, Q,
            make_float3(rmin.x, rmin.y, rmin.z), make_float3(rmax.x, rmax.y, rmax.z),
            cellEpsMax, inv_theta,
            sizeList, monopoleList, quadrpl0List, quadrpl1List, epsMaxList,
            comDList, boxMinList, boxMaxList);
      }
    }

  /* Pass 2: one thread per node of one level; children are already done. */
  template<typename real_t>
    static __global__
    void nodeMultipoles(
        const int cellBeg,
        const int cellEnd,                      /* exclusive */
        const CellData *cells,
        const real_t inv_theta,
        __out typename vec<4,real_t>::type *sizeList,
        __out typename vec<4,real_t>::type *monopoleList,
        __out typename vec<4,real_t>::type *quadrpl0List,
        __out typename vec<2,real_t>::type *quadrpl1List,
        __out float   *epsMaxList,
        __out double4 *comDList,
        __out float4  *boxMinList,
        __out float4  *boxMaxList)
    {
      const int cellIdx = cellBeg + blockIdx.x*blockDim.x + threadIdx.x;
      if (cellIdx >= cellEnd) return;

      const CellData cell = cells[cellIdx];
      if (cell.isLeaf()) return;

      const int first = cell.first();
      const int nChild = cell.n();

      /* COM about the first child's COM (double) */
      const double4 ref = comDList[first];
      double mtot = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
      float3 rmin = make_float3(+1e10f, +1e10f, +1e10f);
      float3 rmax = make_float3(-1e10f, -1e10f, -1e10f);
      float cellEpsMax = 0.0f;
      for (int k = 0; k < nChild; k++)
      {
        const int c = first + k;
        const double4 cc = comDList[c];
        mtot += cc.w;
        sx   += cc.w * (cc.x - ref.x);
        sy   += cc.w * (cc.y - ref.y);
        sz   += cc.w * (cc.z - ref.z);
        const float4 bmin = boxMinList[c];
        const float4 bmax = boxMaxList[c];
        rmin.x = fminf(rmin.x, bmin.x); rmax.x = fmaxf(rmax.x, bmax.x);
        rmin.y = fminf(rmin.y, bmin.y); rmax.y = fmaxf(rmax.y, bmax.y);
        rmin.z = fminf(rmin.z, bmin.z); rmax.z = fmaxf(rmax.z, bmax.z);
        cellEpsMax = fmaxf(cellEpsMax, epsMaxList[c]);
      }
      const double inv_mass = mtot > 0.0 ? 1.0/mtot : 0.0;
      const double4 comD = make_double4(ref.x + sx*inv_mass, ref.y + sy*inv_mass,
                                        ref.z + sz*inv_mass, mtot);

      /* parallel-axis theorem: Q = sum_c m_c (Q_c + d_c d_c^T) / M */
      Quadrupole<double> Q;
      for (int k = 0; k < nChild; k++)
      {
        const int c = first + k;
        const double4 cc = comDList[c];
        const Quadrupole<real_t> qc(quadrpl0List[c], quadrpl1List[c]);
        const double dx = cc.x - comD.x;
        const double dy = cc.y - comD.y;
        const double dz = cc.z - comD.z;
        Q.xx() += cc.w * (double(qc.xx()) + dx*dx);
        Q.yy() += cc.w * (double(qc.yy()) + dy*dy);
        Q.zz() += cc.w * (double(qc.zz()) + dz*dz);
        Q.xy() += cc.w * (double(qc.xy()) + dx*dy);
        Q.xz() += cc.w * (double(qc.xz()) + dx*dz);
        Q.yz() += cc.w * (double(qc.yz()) + dy*dz);
      }
      Q.xx() *= inv_mass; Q.yy() *= inv_mass; Q.zz() *= inv_mass;
      Q.xy() *= inv_mass; Q.xz() *= inv_mass; Q.yz() *= inv_mass;

      storeCell<real_t>(cellIdx, comD, Q, rmin, rmax, cellEpsMax, inv_theta,
          sizeList, monopoleList, quadrpl0List, quadrpl1List, epsMaxList,
          comDList, boxMinList, boxMaxList);
    }

};

  template<typename real_t>
void Treecode<real_t>::computeMultipoles()
{
  d_cellSize    .realloc(nCells);
  d_cellMonopole.realloc(nCells);
  d_cellQuad0   .realloc(nCells);
  d_cellQuad1   .realloc(nCells);
  d_cellEpsMax  .realloc(nCells);
  d_cellComD    .realloc(nCells);
  d_cellBoxMin  .realloc(nCells);
  d_cellBoxMax  .realloc(nCells);

  const double t0 = verbose ? (cudaDeviceSynchronize(), rtc()) : 0.0;
  const real_t inv_theta = real_t(1.0/theta);

  /* pass 1: leaves */
  {
    const int NTHREAD2 = 8;
    const int NTHREAD  = 1<<NTHREAD2;
    const int NWARP    = 1<<(NTHREAD2-WARP_SIZE2);
    const int nblock   = (nLeaves-1)/NWARP + 1;
    multipoles::leafMultipoles<NTHREAD2,real_t><<<nblock,NTHREAD>>>(
        nLeaves, d_leafList, d_cellDataList, d_ptclPos,
        d_ptclEpsTree.ptr, inv_theta,
        d_cellSize, d_cellMonopole, d_cellQuad0, d_cellQuad1, d_cellEpsMax.ptr,
        d_cellComD.ptr, d_cellBoxMin.ptr, d_cellBoxMax.ptr);
  }

  /* pass 2: nodes, deepest level first.  Cells are sorted by level and
   * d_level_begIdx[L] = (first, last) cell of level L for L = 1..nLevels
   * (entries above nLevels may be stale from an earlier, deeper tree). */
  {
    int2 levels[32];
    d_level_begIdx.d2h(levels, 32);
    const int maxLevel = std::min(nLevels, 31);
    const int NTHREAD = 128;
    for (int L = maxLevel; L >= 1; L--)
    {
      const int beg = levels[L].x;
      const int end = levels[L].y + 1;
      if (beg < 0 || end > nCells || end <= beg) continue;
      const int nblock = (end - beg - 1)/NTHREAD + 1;
      multipoles::nodeMultipoles<real_t><<<nblock,NTHREAD>>>(
          beg, end, d_cellDataList, inv_theta,
          d_cellSize, d_cellMonopole, d_cellQuad0, d_cellQuad1, d_cellEpsMax.ptr,
          d_cellComD.ptr, d_cellBoxMin.ptr, d_cellBoxMax.ptr);
    }
  }

  if (verbose)
  {
    kernelSuccess("cellMultipole");
    const double dt = rtc() - t0;
    fprintf(stderr, " cellMultipole done in %g sec : %g Mptcl/sec  %g Mcell/sec\n",
        dt, nPtcl/1e6/dt, nCells/1e6/dt);
  }
}

#include "TreecodeInstances.h"
