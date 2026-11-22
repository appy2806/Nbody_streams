// Direct-sum forces and potentials. Compile with -DKERNEL_ID=<0..4> (softening,
// a compile-time constant) and -DREAL=float|double (SoA kernels). The int
// kernel_id argument of the entry points is kept for the launch signature only.
// Launch with TILE_SIZE threads per block.
//
// Source split: the *_split entry points take a 2-D grid (gridDim.x target
// blocks, gridDim.y = S slices). Slice y sums source tiles y, y+S, y+2S, ...
// and writes its raw partial sums (no G) to part[(y*NC + c)*N + i], c over the
// NC = (3 or 1) * (2 if Kahan: sums then compensations) components. The
// reduce_* kernels add the S slices in the fixed order 0..S-1 and apply G, so
// results are deterministic. The plain entry points (S = 1) write G * sum.
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

template <typename T, int K, bool POT, bool KAHAN, bool F4, bool SPLIT = false>
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
    const int tile0 = SPLIT ? (int)blockIdx.y : 0;
    const int tile_step = SPLIT ? (int)gridDim.y : 1;
    for (int tile = tile0; tile < num_tiles; tile += tile_step) {
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

    if constexpr (SPLIT) {
        // o0 is the partial-sum buffer; this slice's NC rows start at y*NC*N.
        constexpr int NC = (POT ? 1 : 3) * (KAHAN ? 2 : 1);
        if (i < N) {
            T* p = o0 + (size_t)blockIdx.y * NC * N + i;
            p[0] = s0;
            if constexpr (!POT) { p[(size_t)N] = s1; p[2 * (size_t)N] = s2; }
            if constexpr (KAHAN) {
                constexpr int C = POT ? 1 : 3;
                p[C * (size_t)N] = c0;
                if constexpr (!POT) { p[(C + 1) * (size_t)N] = c1; p[(C + 2) * (size_t)N] = c2; }
            }
        }
    } else {
        if (i < N) {
            o0[i] = G * s0;
            if constexpr (!POT) {
                o1[i] = G * s1;
                o2[i] = G * s2;
            }
        }
    }
}

// Combine S slices of partial sums in the fixed order s = 0..S-1 and apply G.
// Plain: sum in T. Kahan: each slice contributes sum - comp, accumulated in
// double (float) or with Kahan compensation (double).
template <typename T, bool KAHAN>
__device__ __forceinline__ T reduce_one(const T* __restrict__ p, size_t stride, size_t comp_off,
                                        int S, T G)
{
    if constexpr (!KAHAN) {
        T acc = p[0];
        for (int s = 1; s < S; s++) acc += p[s * stride];
        return G * acc;
    } else if constexpr (sizeof(T) == 4) {
        double acc = 0.0;
        for (int s = 0; s < S; s++)
            acc += (double)p[s * stride] - (double)p[s * stride + comp_off];
        return G * (T)acc;
    } else {
        T acc = 0, c = 0;
        for (int s = 0; s < S; s++) {
            accumulate<T, true>(acc, c, p[s * stride]);
            accumulate<T, true>(acc, c, -p[s * stride + comp_off]);
        }
        return G * (acc - c);
    }
}

template <typename T, bool POT, bool KAHAN>
__device__ __forceinline__ void reduce_split(const T* __restrict__ part, T* __restrict__ o0,
    T* __restrict__ o1, T* __restrict__ o2, T G, int S, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    constexpr int C = POT ? 1 : 3;
    constexpr int NC = C * (KAHAN ? 2 : 1);
    const size_t stride = (size_t)NC * N;
    const size_t comp_off = (size_t)C * N;
    const T* p = part + i;
    o0[i] = reduce_one<T, KAHAN>(p, stride, comp_off, S, G);
    if constexpr (!POT) {
        o1[i] = reduce_one<T, KAHAN>(p + (size_t)N, stride, comp_off, S, G);
        o2[i] = reduce_one<T, KAHAN>(p + 2 * (size_t)N, stride, comp_off, S, G);
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

// Source-split entry points (2-D grid, see top of file) and their reductions.
#define F4_SPLIT(NAME, POT, KAHAN)                                                   \
    extern "C" __global__ void NAME(const float4* __restrict__ pos_mass,             \
        const float* __restrict__ h, float* __restrict__ part, float eps2, int N) {  \
        direct_sum<float, KERNEL_ID, POT, KAHAN, true, true>(                        \
            pos_mass, nullptr, nullptr, nullptr, nullptr, h, part, nullptr, nullptr, \
            0.f, eps2, N);                                                           \
    }
#define SOA_SPLIT(NAME, POT, KAHAN)                                                  \
    extern "C" __global__ void NAME(const REAL* __restrict__ x,                      \
        const REAL* __restrict__ y, const REAL* __restrict__ z,                     \
        const REAL* __restrict__ mass, const REAL* __restrict__ h,                  \
        REAL* __restrict__ part, REAL eps2, int N) {                                 \
        direct_sum<REAL, KERNEL_ID, POT, KAHAN, false, true>(                        \
            nullptr, x, y, z, mass, h, part, nullptr, nullptr, REAL(0), eps2, N);    \
    }
#define REDUCE_FORCE(NAME, KAHAN)                                                    \
    extern "C" __global__ void NAME(const REAL* __restrict__ part,                   \
        REAL* __restrict__ ax, REAL* __restrict__ ay, REAL* __restrict__ az,        \
        REAL G, int S, int N) {                                                      \
        reduce_split<REAL, false, KAHAN>(part, ax, ay, az, G, S, N);                 \
    }
#define REDUCE_POT(NAME, KAHAN)                                                      \
    extern "C" __global__ void NAME(const REAL* __restrict__ part,                   \
        REAL* __restrict__ pot, REAL G, int S, int N) {                              \
        reduce_split<REAL, true, KAHAN>(part, pot, nullptr, nullptr, G, S, N);       \
    }

F4_SPLIT(nbody_forces_kernel_float4_split, false, false)
F4_SPLIT(nbody_forces_kahan_kernel_float4_split, false, true)
F4_SPLIT(nbody_potential_kernel_float4_split, true, false)
F4_SPLIT(nbody_potential_kahan_kernel_float4_split, true, true)
SOA_SPLIT(nbody_forces_kernel_split, false, false)
SOA_SPLIT(nbody_forces_kahan_kernel_split, false, true)
SOA_SPLIT(nbody_potential_kernel_split, true, false)
SOA_SPLIT(nbody_potential_kahan_kernel_split, true, true)
REDUCE_FORCE(nbody_reduce_forces, false)
REDUCE_FORCE(nbody_reduce_forces_kahan, true)
REDUCE_POT(nbody_reduce_potential, false)
REDUCE_POT(nbody_reduce_potential_kahan, true)
