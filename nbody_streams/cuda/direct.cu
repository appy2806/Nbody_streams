// Direct-sum forces and potentials. Compile with -DKERNEL_ID=<0..4> (softening,
// a compile-time constant) and -DREAL=float|double (SoA kernels). The int
// kernel_id argument of the entry points is kept for the launch signature only.
// Launch with TILE_SIZE threads per block.
#include "softening.cuh"

#ifndef KERNEL_ID
#error "define KERNEL_ID"
#endif
#ifndef REAL
#define REAL double
#endif
#ifndef TILE_SIZE
#define TILE_SIZE 128
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
constexpr int kUnroll = UNROLL;

template <typename T, bool KAHAN>
__device__ __forceinline__ void accumulate(T& sum, T& comp, T term) {
    if constexpr (KAHAN) {
        T y = term - comp;
        T t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    } else {
        sum += term;
    }
}

// Shared-memory tile of sources: float4 (x, y, z, m) + h, or SoA.
template <typename T, bool F4> struct Tile;
template <> struct Tile<float, true> {
    float4 pm[TILE_SIZE];
    float h[TILE_SIZE];
    __device__ __forceinline__ void get(int k, float& x, float& y, float& z, float& m) const {
        float4 p = pm[k]; x = p.x; y = p.y; z = p.z; m = p.w;
    }
};
template <typename T> struct Tile<T, false> {
    T x[TILE_SIZE], y[TILE_SIZE], z[TILE_SIZE], m[TILE_SIZE], h[TILE_SIZE];
    __device__ __forceinline__ void get(int k, T& x_, T& y_, T& z_, T& m_) const {
        x_ = x[k]; y_ = y[k]; z_ = z[k]; m_ = m[k];
    }
};

// Sum over one shared tile. MASK drops the self pair; only the block's own
// (diagonal) tile can contain it.
template <typename T, int K, bool POT, bool KAHAN, bool F4, bool MASK>
__device__ __forceinline__ void tile_sum(
    const Tile<T, F4>& sh, int tid, T px, T py, T pz, T ph, T eps2,
    T& s0, T& s1, T& s2, T& c0, T& c1, T& c2)
{
#pragma unroll (kUnroll)
    for (int k = 0; k < TILE_SIZE; k++) {
        T xj, yj, zj, mj;
        sh.get(k, xj, yj, zj, mj);
        T dx = xj - px;
        T dy = yj - py;
        T dz = zj - pz;
        T r2 = fma_t(dx, dx, fma_t(dy, dy, fma_t(dz, dz, eps2)));
        T h_eff = fmax_t(ph, sh.h[k]);
        if constexpr (MASK) mj = mj * (T)(k != tid);
        if constexpr (POT) {
            accumulate<T, KAHAN>(s0, c0, mj * potential<K>(r2, h_eff));
        } else {
            T f = mj * force_factor<K>(r2, h_eff);
            if constexpr (KAHAN) {
                accumulate<T, true>(s0, c0, f * dx);
                accumulate<T, true>(s1, c1, f * dy);
                accumulate<T, true>(s2, c2, f * dz);
            } else {
                s0 = fma_t(f, dx, s0);
                s1 = fma_t(f, dy, s1);
                s2 = fma_t(f, dz, s2);
            }
        }
    }
}

template <typename T, int K, bool POT, bool KAHAN, bool F4>
__device__ __forceinline__ void direct_sum(
    const float4* __restrict__ pm, const T* __restrict__ x, const T* __restrict__ y,
    const T* __restrict__ z, const T* __restrict__ mass, const T* __restrict__ h,
    T* __restrict__ o0, T* __restrict__ o1, T* __restrict__ o2, T G, T eps2, int N)
{
    __shared__ Tile<T, F4> sh;
    const int tid = threadIdx.x;
    const int i = blockIdx.x * TILE_SIZE + tid;

    T px = 0, py = 0, pz = 0, ph = 0;
    if (i < N) {
        if constexpr (F4) {
            float4 p = pm[i];
            px = p.x; py = p.y; pz = p.z;
        } else {
            px = x[i]; py = y[i]; pz = z[i];
        }
        ph = h[i];
    }

    T s0 = 0, s1 = 0, s2 = 0, c0 = 0, c1 = 0, c2 = 0;
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; tile++) {
        // Padding sources have m = 0; r2 >= eps2 > 0 keeps their terms finite (zero),
        // so the inner loop always runs the full tile.
        const int j = tile * TILE_SIZE + tid;
        if constexpr (F4) {
            sh.pm[tid] = (j < N) ? pm[j] : make_float4(0.f, 0.f, 0.f, 0.f);
        } else {
            sh.x[tid] = (j < N) ? x[j] : T(0);
            sh.y[tid] = (j < N) ? y[j] : T(0);
            sh.z[tid] = (j < N) ? z[j] : T(0);
            sh.m[tid] = (j < N) ? mass[j] : T(0);
        }
        sh.h[tid] = (j < N) ? h[j] : T(0);
        __syncthreads();
        if (tile == blockIdx.x)
            tile_sum<T, K, POT, KAHAN, F4, true>(sh, tid, px, py, pz, ph, eps2, s0, s1, s2, c0, c1, c2);
        else
            tile_sum<T, K, POT, KAHAN, F4, false>(sh, tid, px, py, pz, ph, eps2, s0, s1, s2, c0, c1, c2);
        __syncthreads();
    }

    if (i < N) {
        o0[i] = G * s0;
        if constexpr (!POT) {
            o1[i] = G * s1;
            o2[i] = G * s2;
        }
    }
}

#define F4_FORCE(NAME, KAHAN)                                                        \
    extern "C" __global__ void NAME(const float4* __restrict__ pos_mass,     \
        const float* __restrict__ h, float* __restrict__ ax, float* __restrict__ ay, \
        float* __restrict__ az, float G, float eps2, int kernel_id, int N) {         \
        direct_sum<float, KERNEL_ID, false, KAHAN, true>(                            \
            pos_mass, nullptr, nullptr, nullptr, nullptr, h, ax, ay, az, G, eps2, N);\
    }
#define F4_POT(NAME, KAHAN)                                                          \
    extern "C" __global__ void NAME(const float4* __restrict__ pos_mass,     \
        const float* __restrict__ h, float* __restrict__ pot, float G, float eps2,  \
        int kernel_id, int N) {                                                      \
        direct_sum<float, KERNEL_ID, true, KAHAN, true>(                             \
            pos_mass, nullptr, nullptr, nullptr, nullptr, h, pot, nullptr, nullptr,  \
            G, eps2, N);                                                             \
    }
#define SOA_FORCE(NAME, KAHAN)                                                       \
    extern "C" __global__ void NAME(const REAL* __restrict__ x,              \
        const REAL* __restrict__ y, const REAL* __restrict__ z,                     \
        const REAL* __restrict__ mass, const REAL* __restrict__ h,                  \
        REAL* __restrict__ ax, REAL* __restrict__ ay, REAL* __restrict__ az,        \
        REAL G, REAL eps2, int kernel_id, int N) {                                   \
        direct_sum<REAL, KERNEL_ID, false, KAHAN, false>(                            \
            nullptr, x, y, z, mass, h, ax, ay, az, G, eps2, N);                      \
    }
#define SOA_POT(NAME, KAHAN)                                                         \
    extern "C" __global__ void NAME(const REAL* __restrict__ x,              \
        const REAL* __restrict__ y, const REAL* __restrict__ z,                     \
        const REAL* __restrict__ mass, const REAL* __restrict__ h,                  \
        REAL* __restrict__ pot, REAL G, REAL eps2, int kernel_id, int N) {           \
        direct_sum<REAL, KERNEL_ID, true, KAHAN, false>(                             \
            nullptr, x, y, z, mass, h, pot, nullptr, nullptr, G, eps2, N);           \
    }

F4_FORCE(nbody_forces_kernel_float4, false)
F4_FORCE(nbody_forces_kahan_kernel_float4, true)
F4_POT(nbody_potential_kernel_float4, false)
F4_POT(nbody_potential_kahan_kernel_float4, true)
SOA_FORCE(nbody_forces_kernel, false)
SOA_FORCE(nbody_forces_kahan_kernel, true)
SOA_POT(nbody_potential_kernel, false)
SOA_POT(nbody_potential_kahan_kernel, true)
