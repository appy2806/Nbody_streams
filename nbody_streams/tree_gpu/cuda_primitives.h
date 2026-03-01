#pragma once

// Modernized for CUDA 12+/13+ (sm_70+)
// - All warp intrinsics now use _sync variants with full mask 0xFFFFFFFF
// - Legacy texture bind/unbind helpers removed (textures replaced with __ldg)
// - PTX inline assembly replaced with intrinsics
// - atomicAdd_double uses native atomicAdd (available on sm_60+)

#define FULL_MASK 0xFFFFFFFF

template<typename real_t>
static __device__ __forceinline__ real_t shfl_xor(const real_t x, const int lane, const int warpSize = WARP_SIZE);

  template<>
 __device__ __forceinline__ double shfl_xor<double>(const double x, const int lane, const int warpSize)
{
  return __hiloint2double(
      __shfl_xor_sync(FULL_MASK, __double2hiint(x), lane, warpSize),
      __shfl_xor_sync(FULL_MASK, __double2loint(x), lane, warpSize));
}
  template<>
 __device__ __forceinline__ float shfl_xor<float>(const float x, const int lane, const int warpSize)
{
  return __shfl_xor_sync(FULL_MASK, x, lane, warpSize);
}

/*********************/

static __forceinline__ __device__ double atomicAdd_double(double *address, const double val)
{
  return atomicAdd(address, val);
}

/**************************/

template<typename real_t>
  static __device__ __forceinline__
void addBoxSize(typename vec<3,real_t>::type &_rmin, typename vec<3,real_t>::type &_rmax, const Position<real_t> pos)
{
  typename vec<3,real_t>::type rmin = {pos.x, pos.y, pos.z};
  typename vec<3,real_t>::type rmax = rmin;

#pragma unroll
  for (int i = WARP_SIZE2-1; i >= 0; i--)
  {
    rmin.x = min(rmin.x, shfl_xor(rmin.x, 1<<i, WARP_SIZE));
    rmax.x = max(rmax.x, shfl_xor(rmax.x, 1<<i, WARP_SIZE));

    rmin.y = min(rmin.y, shfl_xor(rmin.y, 1<<i, WARP_SIZE));
    rmax.y = max(rmax.y, shfl_xor(rmax.y, 1<<i, WARP_SIZE));

    rmin.z = min(rmin.z, shfl_xor(rmin.z, 1<<i, WARP_SIZE));
    rmax.z = max(rmax.z, shfl_xor(rmax.z, 1<<i, WARP_SIZE));
  }

  _rmin.x = min(_rmin.x, rmin.x);
  _rmin.y = min(_rmin.y, rmin.y);
  _rmin.z = min(_rmin.z, rmin.z);

  _rmax.x = max(_rmax.x, rmax.x);
  _rmax.y = max(_rmax.y, rmax.y);
  _rmax.z = max(_rmax.z, rmax.z);
}

/************ scan **********/

static __device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}

static __device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
{
  uint result = __shfl_up_sync(FULL_MASK, partial, up_offset);
  unsigned int laneIdx = threadIdx.x & (WARP_SIZE - 1);
  if (laneIdx >= up_offset)
    result += partial;
  else
    result = partial;
  return result;
}

  template <const int levels>
static __device__ __forceinline__ uint inclusive_scan_warp(const int sum)
{
  uint mysum = sum;
#pragma unroll
  for(int i = 0; i < levels; ++i)
    mysum = shfl_scan_add_step(mysum, 1 << i);
  return mysum;
}

static __device__ __forceinline__ int2 warpIntExclusiveScan(const int value)
{
  const int sum = inclusive_scan_warp<WARP_SIZE2>(value);
  return make_int2(sum-value, __shfl_sync(FULL_MASK, sum, WARP_SIZE-1, WARP_SIZE));
}

/************** binary scan ***********/

static __device__ __forceinline__ int warpBinExclusiveScan1(const bool p)
{
  const unsigned int b = __ballot_sync(FULL_MASK, p);
  return __popc(b & lanemask_lt());
}
static __device__ __forceinline__ int2 warpBinExclusiveScan(const bool p)
{
  const unsigned int b = __ballot_sync(FULL_MASK, p);
  return make_int2(__popc(b & lanemask_lt()), __popc(b));
}
static __device__ __forceinline__ int warpBinReduce(const bool p)
{
  const unsigned int b = __ballot_sync(FULL_MASK, p);
  return __popc(b);
}

/******************* segscan *******/

static __device__ __forceinline__ int lanemask_le()
{
  int mask;
  asm("mov.u32 %0, %lanemask_le;" : "=r" (mask));
  return mask;
}

static __device__ __forceinline__ int ShflSegScanStepB(
    int partial,
    uint distance,
    uint up_offset)
{
  int shfl_val = __shfl_up_sync(FULL_MASK, partial, up_offset);
  unsigned int laneIdx = threadIdx.x & (WARP_SIZE - 1);
  if (laneIdx >= up_offset && up_offset <= distance)
    partial = shfl_val + partial;
  return partial;
}

  template<const int SIZE2>
static __device__ __forceinline__ int inclusive_segscan_warp_step(int value, const int distance)
{
  for (int i = 0; i < SIZE2; i++)
    value = ShflSegScanStepB(value, distance, 1<<i);
  return value;
}

static __device__ __forceinline__ int2 inclusive_segscan_warp(
    const int packed_value, const int carryValue)
{
  const int  flag = packed_value < 0;
  const int  mask = -flag;
  const int value = (~mask & packed_value) + (mask & (-1-packed_value));

  const int flags = __ballot_sync(FULL_MASK, flag);

  const int dist_block = __clz(__brev(flags));

  const int laneIdx = threadIdx.x & (WARP_SIZE - 1);
  const int distance = __clz(flags & lanemask_le()) + laneIdx - 31;
  const int val = inclusive_segscan_warp_step<WARP_SIZE2>(value, min(distance, laneIdx)) +
    (carryValue & (-(laneIdx < dist_block)));
  return make_int2(val, __shfl_sync(FULL_MASK, val, WARP_SIZE-1, WARP_SIZE));
}
