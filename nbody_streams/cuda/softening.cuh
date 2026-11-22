// Softening kernels for the direct sum, shared by every CUDA kernel.
// force_factor<K>(r2, h) multiplies (x_j - x_i); potential<K>(r2, h) is per unit G m.
// K: 0 newtonian, 1 plummer, 2 dehnen_k1, 3 dehnen_k2, 4 spline (h = support radius).
#pragma once

__device__ __forceinline__ float rsqrt_t(float x) { return rsqrtf(x); }
__device__ __forceinline__ double rsqrt_t(double x) { return rsqrt(x); }
__device__ __forceinline__ float sqrt_t(float x) { return sqrtf(x); }
__device__ __forceinline__ double sqrt_t(double x) { return sqrt(x); }
__device__ __forceinline__ float fma_t(float a, float b, float c) { return fmaf(a, b, c); }
__device__ __forceinline__ double fma_t(double a, double b, double c) { return fma(a, b, c); }
__device__ __forceinline__ float fmax_t(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double fmax_t(double a, double b) { return fmax(a, b); }

template <int K, typename T>
__device__ __forceinline__ T force_factor(T r2, T h) {
    if constexpr (K == 0) {
        T inv_r = rsqrt_t(r2);
        return inv_r * inv_r * inv_r;
    } else if constexpr (K == 1) {
        T s = rsqrt_t(r2 + h * h);
        return s * s * s;
    } else if constexpr (K == 2) {
        T h2 = h * h;
        T s = rsqrt_t(r2 + h2);
        T inv_d = s * s;
        T inv_d32 = inv_d * s;
        T inv_d52 = inv_d32 * inv_d;
        return inv_d32 + T(1.5) * h2 * inv_d52;
    } else if constexpr (K == 3) {
        T h2 = h * h;
        T h4 = h2 * h2;
        T s = rsqrt_t(r2 + h2);
        T inv_d = s * s;
        T inv_d32 = inv_d * s;
        T inv_d52 = inv_d32 * inv_d;
        T inv_d72 = inv_d52 * inv_d;
        return inv_d32 + T(1.5) * h2 * inv_d52 + T(1.875) * h4 * inv_d72;
    } else {
        static_assert(K == 4, "unknown softening kernel");
        T r = sqrt_t(r2);
        if (r >= h) {
            T inv_r = T(1) / r;
            return inv_r * inv_r * inv_r;
        }
        T hinv = T(1) / h;
        T q = r * hinv;
        T h3inv = hinv * hinv * hinv;
        if (q < T(1e-8)) return h3inv * T(10.666666666666666);
        T q2 = q * q;
        if (q <= T(0.5))
            return h3inv * fma_t(q2, fma_t(T(32.0), q, T(-38.4)), T(10.666666666666666));
        T inv_q3 = T(1) / (q2 * q);
        return h3inv * (T(21.333333333333333)
                        + q * (T(-48.0) + q * (T(38.4) - T(10.666666666666667) * q))
                        - T(0.0666666666666667) * inv_q3);
    }
}

template <int K, typename T>
__device__ __forceinline__ T potential(T r2, T h) {
    if constexpr (K == 0) {
        T r = sqrt_t(r2);
        return (r > T(0)) ? T(-1) / r : T(0);
    } else if constexpr (K == 1) {
        return -rsqrt_t(r2 + h * h);
    } else if constexpr (K == 2) {
        T h2 = h * h;
        T s = rsqrt_t(r2 + h2);
        return -s - T(0.5) * h2 * (s * s * s);
    } else if constexpr (K == 3) {
        T h2 = h * h;
        T h4 = h2 * h2;
        T s = rsqrt_t(r2 + h2);
        T inv_d32 = s * s * s;
        T inv_d52 = inv_d32 * s * s;
        return -s - T(0.5) * h2 * inv_d32 - T(0.375) * h4 * inv_d52;
    } else {
        static_assert(K == 4, "unknown softening kernel");
        T r = sqrt_t(r2);
        if (h == T(0) || r >= h) return T(-1) / r;
        T hinv = T(1) / h;
        T q = r * hinv;
        if (q < T(1e-8)) return T(-2.8) * hinv;
        T q2 = q * q;
        if (q <= T(0.5))
            return (T(-2.8) + q2 * (T(5.33333333333333333) + q2 * (T(6.4) * q - T(9.6)))) * hinv;
        return (T(-3.2) + T(0.066666666666666666666) / q
                + q2 * (T(10.666666666666666666666)
                        + q * (T(-16.0) + q * (T(9.6) - T(2.1333333333333333333333) * q)))) * hinv;
    }
}
