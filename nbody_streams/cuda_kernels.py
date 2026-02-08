"""
nbody_streams.cuda_kernels
CUDA kernels for N-body force and potential computation.
All kernels properly handle partial tiles for momentum/energy conservation.

Author: Arpit Arora
Date: Sept 2025
"""

# ============================================================================
# RAW CUDA KERNEL (Compiled with nvcc for maximum performance)
# ============================================================================

# ============================================================================
# NEW: Float4 optimized kernel (float32 only)
# ============================================================================

_NBODY_KERNEL_TEMPLATE_FLOAT4 = r'''
#define TILE_SIZE 128

// Same branch-free kernel functions as before
extern "C" __device__ __forceinline__ 
float compute_newtonian(float r2) {{
    float inv_r = rsqrtf(r2);
    return inv_r * inv_r * inv_r;
}}

extern "C" __device__ __forceinline__ 
float compute_plummer(float r2, float h) {{
    float h2 = h * h;
    float denom = r2 + h2;
    float inv_sqrt_d = rsqrtf(denom);
    return inv_sqrt_d * inv_sqrt_d * inv_sqrt_d;
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k1(float r2, float h) {{
    float h2 = h * h;
    float denom = r2 + h2;
    float inv_sqrt_d = rsqrtf(denom);
    float inv_d = inv_sqrt_d * inv_sqrt_d;
    float inv_d32 = inv_d * inv_sqrt_d;
    float inv_d52 = inv_d32 * inv_d;
    return inv_d32 + 1.5f * h2 * inv_d52;
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k2(float r2, float h) {{
    float h2 = h * h;
    float h4 = h2 * h2;
    float denom = r2 + h2;
    float inv_sqrt_d = rsqrtf(denom);
    float inv_d = inv_sqrt_d * inv_sqrt_d;
    float inv_d32 = inv_d * inv_sqrt_d;
    float inv_d52 = inv_d32 * inv_d;
    float inv_d72 = inv_d52 * inv_d;
    return inv_d32 + 1.5f * h2 * inv_d52 + 3.75f * h4 * inv_d72;
}}

extern "C" __device__ __forceinline__ 
float compute_spline(float r2, float h) {{
    float r = sqrtf(r2);
    
    if (r >= h) {{
        float inv_r = 1.0f / r;
        return inv_r * inv_r * inv_r;
    }}
    
    float hinv = 1.0f / h;
    float q = r * hinv;
    
    if (q < 1e-8f) {{
        float h3inv = hinv * hinv * hinv;
        return h3inv * 10.666666666666666f;
    }}
    
    float h3inv = hinv * hinv * hinv;
    float q2 = q * q;
    
    if (q <= 0.5f) {{
        return h3inv * fmaf(q2, fmaf(32.0f, q, -38.4f), 10.666666666666666f);
    }}
    
    float q3 = q2 * q;
    float inv_q3 = 1.0f / q3;
    return h3inv * (21.333333333333333f + q * (-48.0f + q * (38.4f - 10.666666666666667f * q)) - 0.0666666666666667f * inv_q3);
}}

// >>>>>>> MAIN CHANGE: Float4 kernel uses vectorized loads
extern "C" __global__
void nbody_forces_kernel_float4(
    const float4* __restrict__ pos_mass,  // x,y,z,mass interleaved
    const float* __restrict__ h,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    float G,
    float eps2,
    int kernel_id,
    int N
) {{
    __shared__ float4 sh_pos_mass[TILE_SIZE];
    __shared__ float sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    bool active = (i < N);

    float4 my_pm;
    float my_h;
    if (active) {{
        my_pm = pos_mass[i];  // Single 128-bit load!
        my_h = h[i];
    }}
    
    float sum_ax = 0.0f;
    float sum_ay = 0.0f;
    float sum_az = 0.0f;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;

        if (j < N) {{
            sh_pos_mass[tid] = pos_mass[j];  // Vectorized load
            sh_h[tid] = h[j];
        }} else {{
            sh_pos_mass[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            sh_h[tid] = 0.0f;
        }}
        
        __syncthreads();

        if (active) {{ 
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                float4 other = sh_pos_mass[k];
                float dx = other.x - my_pm.x;
                float dy = other.y - my_pm.y;
                float dz = other.z - my_pm.z;
                
                float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
                float h_eff = fmaxf(my_h, sh_h[k]);
                
                float kern;
                switch(kernel_id) {{
                    case 0: kern = compute_newtonian(r2); break;
                    case 1: kern = compute_plummer(r2, h_eff); break;
                    case 2: kern = compute_dehnen_k1(r2, h_eff); break;
                    case 3: kern = compute_dehnen_k2(r2, h_eff); break;
                    case 4: kern = compute_spline(r2, h_eff); break;
                    default: kern = 0.0f; break;
                }}
                
                float not_self = (float)(i != j_global);
                float factor = other.w * kern * not_self;  // other.w = mass
                
                sum_ax = fmaf(factor, dx, sum_ax);
                sum_ay = fmaf(factor, dy, sum_ay);
                sum_az = fmaf(factor, dz, sum_az);
            }}
        }}
        
        __syncthreads();
    }}

    if (active) {{
        ax[i] = G * sum_ax;
        ay[i] = G * sum_ay;
        az[i] = G * sum_az;
    }}
}}
'''

_KAHAN_KERNEL_TEMPLATE_FLOAT4 = r'''
#define TILE_SIZE 128

// Same branch-free kernel functions (copy from float4 template)
extern "C" __device__ __forceinline__ 
float compute_newtonian(float r2) {{
    float inv_r = rsqrtf(r2);
    return inv_r * inv_r * inv_r;
}}

extern "C" __device__ __forceinline__ 
float compute_plummer(float r2, float h) {{
    float h2 = h * h;
    float denom = r2 + h2;
    float inv_sqrt_d = rsqrtf(denom);
    return inv_sqrt_d * inv_sqrt_d * inv_sqrt_d;
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k1(float r2, float h) {{
    float h2 = h * h;
    float denom = r2 + h2;
    float inv_sqrt_d = rsqrtf(denom);
    float inv_d = inv_sqrt_d * inv_sqrt_d;
    float inv_d32 = inv_d * inv_sqrt_d;
    float inv_d52 = inv_d32 * inv_d;
    return inv_d32 + 1.5f * h2 * inv_d52;
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k2(float r2, float h) {{
    float h2 = h * h;
    float h4 = h2 * h2;
    float denom = r2 + h2;
    float inv_sqrt_d = rsqrtf(denom);
    float inv_d = inv_sqrt_d * inv_sqrt_d;
    float inv_d32 = inv_d * inv_sqrt_d;
    float inv_d52 = inv_d32 * inv_d;
    float inv_d72 = inv_d52 * inv_d;
    return inv_d32 + 1.5f * h2 * inv_d52 + 3.75f * h4 * inv_d72;
}}

extern "C" __device__ __forceinline__ 
float compute_spline(float r2, float h) {{
    float r = sqrtf(r2);
    
    if (r >= h) {{
        float inv_r = 1.0f / r;
        return inv_r * inv_r * inv_r;
    }}
    
    float hinv = 1.0f / h;
    float q = r * hinv;
    
    if (q < 1e-8f) {{
        float h3inv = hinv * hinv * hinv;
        return h3inv * 10.666666666666666f;
    }}
    
    float h3inv = hinv * hinv * hinv;
    float q2 = q * q;
    
    if (q <= 0.5f) {{
        return h3inv * fmaf(q2, fmaf(32.0f, q, -38.4f), 10.666666666666666f);
    }}
    
    float q3 = q2 * q;
    float inv_q3 = 1.0f / q3;
    return h3inv * (21.333333333333333f + q * (-48.0f + q * (38.4f - 10.666666666666667f * q)) - 0.0666666666666667f * inv_q3);
}}

// >>>>>>> CHANGE: Float4 + Kahan summation
extern "C" __global__
void nbody_forces_kahan_kernel_float4(
    const float4* __restrict__ pos_mass,  // ← Float4 input
    const float* __restrict__ h,
    float* __restrict__ ax,
    float* __restrict__ ay,
    float* __restrict__ az,
    float G,
    float eps2,
    int kernel_id,
    int N
) {{
    __shared__ float4 sh_pos_mass[TILE_SIZE];  // ← Float4 shared memory
    __shared__ float sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    bool active = (i < N);

    float4 my_pm;  // ← Float4 register
    float my_h;
    if (active) {{
        my_pm = pos_mass[i];  // ← Vectorized load
        my_h = h[i];
    }}
    
    float sum_ax = 0.0f;
    float sum_ay = 0.0f;
    float sum_az = 0.0f;
    
    // >>>>>>> CHANGE: Kahan compensation terms
    float comp_ax = 0.0f;
    float comp_ay = 0.0f;
    float comp_az = 0.0f;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;

        if (j < N) {{
            sh_pos_mass[tid] = pos_mass[j];  // ← Vectorized load
            sh_h[tid] = h[j];
        }} else {{
            sh_pos_mass[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            sh_h[tid] = 0.0f;
        }}
        
        __syncthreads();

        if (active) {{ 
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                float4 other = sh_pos_mass[k];  // ← Float4 access
                float dx = other.x - my_pm.x;
                float dy = other.y - my_pm.y;
                float dz = other.z - my_pm.z;
                
                float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
                float h_eff = fmaxf(my_h, sh_h[k]);
                
                float kern;
                switch(kernel_id) {{
                    case 0: kern = compute_newtonian(r2); break;
                    case 1: kern = compute_plummer(r2, h_eff); break;
                    case 2: kern = compute_dehnen_k1(r2, h_eff); break;
                    case 3: kern = compute_dehnen_k2(r2, h_eff); break;
                    case 4: kern = compute_spline(r2, h_eff); break;
                    default: kern = 0.0f; break;
                }}
                
                float not_self = (float)(i != j_global);
                float factor = other.w * kern * not_self;  // ← other.w = mass
                
                // >>>>>>> Kahan summation (unchanged!)
                // Kahan summation for X component
                float term_x = factor * dx;
                float y_x = term_x - comp_ax;
                float t_x = sum_ax + y_x;
                comp_ax = (t_x - sum_ax) - y_x;
                sum_ax = t_x;
                
                // Kahan summation for Y component
                float term_y = factor * dy;
                float y_y = term_y - comp_ay;
                float t_y = sum_ay + y_y;
                comp_ay = (t_y - sum_ay) - y_y;
                sum_ay = t_y;
                
                // Kahan summation for Z component
                float term_z = factor * dz;
                float y_z = term_z - comp_az;
                float t_z = sum_az + y_z;
                comp_az = (t_z - sum_az) - y_z;
                sum_az = t_z;
            }}
        }}
        
        __syncthreads();
    }}

    if (active) {{
        ax[i] = G * sum_ax;
        ay[i] = G * sum_ay;
        az[i] = G * sum_az;
    }}
}}
'''

_POTENTIAL_KERNEL_TEMPLATE_FLOAT4 = r'''
#define TILE_SIZE 128

// Branch-free potential kernel functions (float only)
extern "C" __device__ __forceinline__ 
float compute_newtonian_potential(float r2) {{
    float r = sqrtf(r2);
    return (r > 0.0f) ? -1.0f / r : 0.0f;
}}

extern "C" __device__ __forceinline__ 
float compute_plummer_potential(float r2, float h) {{
    return -rsqrtf(r2 + h * h);
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k1_potential(float r2, float h) {{
    float h2 = h * h;
    float denom = r2 + h2;
    float inv_sqrt = rsqrtf(denom);
    float inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;
    return -inv_sqrt - 0.5f * h2 * inv_d32;
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k2_potential(float r2, float h) {{
    float h2 = h * h;
    float h4 = h2 * h2;
    float denom = r2 + h2;
    float inv_sqrt = rsqrtf(denom);
    float inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;
    float inv_d52 = inv_d32 * inv_sqrt * inv_sqrt;
    return -inv_sqrt - 0.5f * h2 * inv_d32 - 0.375f * h4 * inv_d52;
}}

extern "C" __device__ __forceinline__ 
float compute_spline_potential(float r2, float h) {{
    float r = sqrtf(r2);
    
    if (h == 0.0f || r >= h) {{
        return -1.0f / r;
    }}
    
    float hinv = 1.0f / h;
    float q = r * hinv;
    
    if (q < 1e-8f) {{
        return -2.8f * hinv;
    }}
    
    if (q <= 0.5f) {{
        float q2 = q * q;
        float q4 = q2 * q2;
        return (-2.8f + q2 * (5.33333333333333333f + q4 * (6.4f * q - 9.6f))) * hinv;
    }}
    
    if (q <= 1.0f) {{
        float q2 = q * q;
        float q3 = q2 * q;
        float q4 = q2 * q2;
        float q5 = q4 * q;
        return (-3.2f + 0.066666666666666666666f / q 
                + q2 * (10.666666666666666666666f 
                + q * (-16.0f + q * (9.6f - 2.1333333333333333333333f * q)))) * hinv;
    }}
    
    return -1.0f / r;
}}

// >>>>>>> Float4 vectorized potential kernel
extern "C" __global__
void nbody_potential_kernel_float4(
    const float4* __restrict__ pos_mass,
    const float* __restrict__ h,
    float* __restrict__ pot,
    float G,
    float eps2,
    int kernel_id,
    int N
) {{
    __shared__ float4 sh_pos_mass[TILE_SIZE];
    __shared__ float sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    bool active = (i < N);

    float4 my_pm;
    float my_h;
    if (active) {{
        my_pm = pos_mass[i];
        my_h = h[i];
    }}
    
    float sum_pot = 0.0f;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;
        
        if (j < N) {{
            sh_pos_mass[tid] = pos_mass[j];
            sh_h[tid] = h[j];
        }} else {{
            sh_pos_mass[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            sh_h[tid] = 0.0f;
        }}
        
        __syncthreads();
        
        if (active) {{
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                float4 other = sh_pos_mass[k];
                float dx = other.x - my_pm.x;
                float dy = other.y - my_pm.y;
                float dz = other.z - my_pm.z;
                
                float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
                float h_eff = fmaxf(my_h, sh_h[k]);
                
                float phi;
                switch(kernel_id) {{
                    case 0: phi = compute_newtonian_potential(r2); break;
                    case 1: phi = compute_plummer_potential(r2, h_eff); break;
                    case 2: phi = compute_dehnen_k1_potential(r2, h_eff); break;
                    case 3: phi = compute_dehnen_k2_potential(r2, h_eff); break;
                    case 4: phi = compute_spline_potential(r2, h_eff); break;
                    default: phi = 0.0f; break;
                }}
                
                float not_self = (float)(i != j_global);
                sum_pot += other.w * phi * not_self;
            }}
        }}
        
        __syncthreads();
    }}
    
    if (active) {{
        pot[i] = G * sum_pot;
    }}
}}
'''

_POTENTIAL_KAHAN_KERNEL_TEMPLATE_FLOAT4 = r'''
#define TILE_SIZE 128

// Same kernel functions as float4 version (copy from above)
extern "C" __device__ __forceinline__ 
float compute_newtonian_potential(float r2) {{
    float r = sqrtf(r2);
    return (r > 0.0f) ? -1.0f / r : 0.0f;
}}

extern "C" __device__ __forceinline__ 
float compute_plummer_potential(float r2, float h) {{
    return -rsqrtf(r2 + h * h);
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k1_potential(float r2, float h) {{
    float h2 = h * h;
    float denom = r2 + h2;
    float inv_sqrt = rsqrtf(denom);
    float inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;
    return -inv_sqrt - 0.5f * h2 * inv_d32;
}}

extern "C" __device__ __forceinline__ 
float compute_dehnen_k2_potential(float r2, float h) {{
    float h2 = h * h;
    float h4 = h2 * h2;
    float denom = r2 + h2;
    float inv_sqrt = rsqrtf(denom);
    float inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;
    float inv_d52 = inv_d32 * inv_sqrt * inv_sqrt;
    return -inv_sqrt - 0.5f * h2 * inv_d32 - 0.375f * h4 * inv_d52;
}}

extern "C" __device__ __forceinline__ 
float compute_spline_potential(float r2, float h) {{
    float r = sqrtf(r2);
    
    if (h == 0.0f || r >= h) {{
        return -1.0f / r;
    }}
    
    float hinv = 1.0f / h;
    float q = r * hinv;
    
    if (q < 1e-8f) {{
        return -2.8f * hinv;
    }}
    
    if (q <= 0.5f) {{
        float q2 = q * q;
        float q4 = q2 * q2;
        return (-2.8f + q2 * (5.33333333333333333f + q4 * (6.4f * q - 9.6f))) * hinv;
    }}
    
    if (q <= 1.0f) {{
        float q2 = q * q;
        float q3 = q2 * q;
        float q4 = q2 * q2;
        float q5 = q4 * q;
        return (-3.2f + 0.066666666666666666666f / q 
                + q2 * (10.666666666666666666666f 
                + q * (-16.0f + q * (9.6f - 2.1333333333333333333333f * q)))) * hinv;
    }}
    
    return -1.0f / r;
}}

// >>>>>>> Kahan + Float4 potential kernel
extern "C" __global__
void nbody_potential_kahan_kernel_float4(
    const float4* __restrict__ pos_mass,
    const float* __restrict__ h,
    float* __restrict__ pot,
    float G,
    float eps2,
    int kernel_id,
    int N
) {{
    __shared__ float4 sh_pos_mass[TILE_SIZE];
    __shared__ float sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    bool active = (i < N);

    float4 my_pm;
    float my_h;
    if (active) {{
        my_pm = pos_mass[i];
        my_h = h[i];
    }}
    
    float sum_pot = 0.0f;
    float comp_pot = 0.0f;  // >>>>>>> Kahan compensation term
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;
        
        if (j < N) {{
            sh_pos_mass[tid] = pos_mass[j];
            sh_h[tid] = h[j];
        }} else {{
            sh_pos_mass[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            sh_h[tid] = 0.0f;
        }}
        
        __syncthreads();
        
        if (active) {{
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                float4 other = sh_pos_mass[k];
                float dx = other.x - my_pm.x;
                float dy = other.y - my_pm.y;
                float dz = other.z - my_pm.z;
                
                float r2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
                float h_eff = fmaxf(my_h, sh_h[k]);
                
                float phi;
                switch(kernel_id) {{
                    case 0: phi = compute_newtonian_potential(r2); break;
                    case 1: phi = compute_plummer_potential(r2, h_eff); break;
                    case 2: phi = compute_dehnen_k1_potential(r2, h_eff); break;
                    case 3: phi = compute_dehnen_k2_potential(r2, h_eff); break;
                    case 4: phi = compute_spline_potential(r2, h_eff); break;
                    default: phi = 0.0f; break;
                }}
                
                float not_self = (float)(i != j_global);
                float term = other.w * phi * not_self;
                
                // >>>>>>> Kahan summation
                float y = term - comp_pot;
                float t = sum_pot + y;
                comp_pot = (t - sum_pot) - y;
                sum_pot = t;
            }}
        }}
        
        __syncthreads();
    }}
    
    if (active) {{
        pot[i] = G * sum_pot;
    }}
}}
'''

# ==============================================================================
# Non vectorized optimizations: N-BODY FORCES KERNEL
# ==============================================================================

_NBODY_KERNEL_TEMPLATE= r'''
#define TILE_SIZE 128  // Try this instead of 256
// >>>>>>> NEW: Branch-free kernel functions (replaces compute_kernel_factor if/else chain)
extern "C" __device__ __forceinline__ 
{T} compute_newtonian({T} r2) {{
    {T} inv_r = {RSQRT}(r2);
    return inv_r * inv_r * inv_r;
}}

extern "C" __device__ __forceinline__ 
{T} compute_plummer({T} r2, {T} h) {{
    {T} h2 = h * h;
    {T} denom = r2 + h2;
    {T} inv_sqrt_d = {RSQRT}(denom);
    return inv_sqrt_d * inv_sqrt_d * inv_sqrt_d;
}}

extern "C" __device__ __forceinline__ 
{T} compute_dehnen_k1({T} r2, {T} h) {{
    {T} h2 = h * h;
    {T} denom = r2 + h2;
    {T} inv_sqrt_d = {RSQRT}(denom);
    {T} inv_d = inv_sqrt_d * inv_sqrt_d;
    {T} inv_d32 = inv_d * inv_sqrt_d;
    {T} inv_d52 = inv_d32 * inv_d;
    return inv_d32 + 1.5 * h2 * inv_d52;
}}

extern "C" __device__ __forceinline__ 
{T} compute_dehnen_k2({T} r2, {T} h) {{
    {T} h2 = h * h;
    {T} h4 = h2 * h2;
    {T} denom = r2 + h2;
    {T} inv_sqrt_d = {RSQRT}(denom);
    {T} inv_d = inv_sqrt_d * inv_sqrt_d;
    {T} inv_d32 = inv_d * inv_sqrt_d;
    {T} inv_d52 = inv_d32 * inv_d;
    {T} inv_d72 = inv_d52 * inv_d;
    return inv_d32 + 1.5 * h2 * inv_d52 + 3.75 * h4 * inv_d72;
}}

extern "C" __device__ __forceinline__ 
{T} compute_spline({T} r2, {T} h) {{
    {T} r = {SQRT}(r2);
    
    if (r >= h) {{
        {T} inv_r = 1.0 / r;
        return inv_r * inv_r * inv_r;
    }}
    
    {T} hinv = 1.0 / h;
    {T} q = r * hinv;
    
    if (q < 1e-8) {{
        {T} h3inv = hinv * hinv * hinv;
        return h3inv * 10.666666666666666;
    }}
    
    {T} h3inv = hinv * hinv * hinv;
    {T} q2 = q * q;
    
    if (q <= 0.5) {{
        return h3inv * {FMA}(q2, {FMA}(32.0, q, -38.4), 10.666666666666666);
    }}
    
    {T} q3 = q2 * q;
    {T} inv_q3 = 1.0 / q3;
    return h3inv * (21.333333333333333 + q * (-48.0 + q * (38.4 - 10.666666666666667 * q)) - 0.0666666666666667 * inv_q3);
}}
// <<<<<

extern "C" __global__
void nbody_forces_kernel(
    const {T}* __restrict__ x,
    const {T}* __restrict__ y, 
    const {T}* __restrict__ z,
    const {T}* __restrict__ mass,
    const {T}* __restrict__ h,
    {T}* __restrict__ ax,
    {T}* __restrict__ ay,
    {T}* __restrict__ az,
    {T} G,
    {T} eps2,
    int kernel_id,
    int N
) {{
    __shared__ {T} sh_x[TILE_SIZE];
    __shared__ {T} sh_y[TILE_SIZE];
    __shared__ {T} sh_z[TILE_SIZE];
    __shared__ {T} sh_m[TILE_SIZE];
    __shared__ {T} sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Don't return early! Let all threads participate in loading
    bool active = (i < N);

    {T} my_x, my_y, my_z, my_h;
    if (active) {{
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        my_h = h[i];
    }}
    
    {T} sum_ax = 0.0;
    {T} sum_ay = 0.0;
    {T} sum_az = 0.0;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;

        // ALL threads load (or zero) shared memory
        if (j < N) {{
            sh_x[tid] = x[j];
            sh_y[tid] = y[j];
            sh_z[tid] = z[j];
            sh_m[tid] = mass[j];
            sh_h[tid] = h[j];
        }} else {{
            sh_x[tid] = 0.0;
            sh_y[tid] = 0.0;
            sh_z[tid] = 0.0;
            sh_m[tid] = 0.0;  // Zero out mass for out-of-range threads
            sh_h[tid] = 0.0;  // Zero out smoothing length for out-of-range threads
        }}
        
        __syncthreads();

        // Only active threads compute
        if (active) {{ 
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                {T} dx = sh_x[k] - my_x;
                {T} dy = sh_y[k] - my_y;
                {T} dz = sh_z[k] - my_z;
                
                {T} r2 = {FMA}(dx, dx, {FMA}(dy, dy, {FMA}(dz, dz, eps2)));
                {T} h_eff = {FMAX}(my_h, sh_h[k]);
                
                // >>>>>>> CHANGE 2: Replace big if/else with switch for branch-free
                {T} kern;
                switch(kernel_id) {{
                    case 0: kern = compute_newtonian(r2); break;
                    case 1: kern = compute_plummer(r2, h_eff); break;
                    case 2: kern = compute_dehnen_k1(r2, h_eff); break;
                    case 3: kern = compute_dehnen_k2(r2, h_eff); break;
                    case 4: kern = compute_spline(r2, h_eff); break;
                    default: kern = 0.0; break;
                }}
                // <<<<<
                
                // >>>>>>> CHANGE 3: Branch-free self-interaction (was: if (i == j_global) continue;)
                {T} not_self = ({T})(i != j_global);
                {T} factor = sh_m[k] * kern * not_self;
                // <<<<<
                
                sum_ax = {FMA}(factor, dx, sum_ax);
                sum_ay = {FMA}(factor, dy, sum_ay);
                sum_az = {FMA}(factor, dz, sum_az);
            }}
        }}
        
        __syncthreads();
    }}

    // Only active threads write results
    if (active) {{
        ax[i] = G * sum_ax;
        ay[i] = G * sum_ay;
        az[i] = G * sum_az;
    }}
}}
'''

# ==============================================================================
# REGULAR N-BODY FORCES KERNEL
# ==============================================================================

_NBODY_KERNEL_TEMPLATE_LEGACY = r'''
#define TILE_SIZE 128  // Try this instead of 256

extern "C" __device__ __forceinline__ 
{T} compute_kernel_factor({T} r2, {T} h, int kernel_id) {{
    // Compute force kernel factor: returns coefficient such that F = G * m * factor * dr
    
    if (kernel_id == 0) {{  // Newtonian: 1/r^3
        {T} r = {SQRT}(r2);
        return 1.0 / (r * r * r);
    }}
    
    if (kernel_id == 1) {{  // Plummer: 1/(r^2 + h^2)^(3/2)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        return {RSQRT}(denom * denom * denom);
    }}
    
    if (kernel_id == 2) {{  // Dehnen k=1 (C2 correction, falcON default)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        {T} sqrt_d = {SQRT}(denom);
        {T} inv_d32 = 1.0 / (denom * sqrt_d);
        {T} inv_d52 = inv_d32 / denom;
        return inv_d32 + 1.5 * h2 * inv_d52;
    }}
    
    if (kernel_id == 3) {{  // Dehnen k=2 (C4 correction)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        {T} sqrt_d = {SQRT}(denom);
        {T} inv_d32 = 1.0 / (denom * sqrt_d);
        {T} inv_d52 = inv_d32 / denom;
        {T} inv_d72 = inv_d52 / denom;
        return inv_d32 + 1.5 * h2 * inv_d52 + 3.75 * h2 * h2 * inv_d72;
    }}
    
    if (kernel_id == 4) {{  // Spline kernel (Monaghan 1992, compact support)
        {T} r = {SQRT}(r2);
        if (r >= h) {{
            return 1.0 / (r * r * r);
        }}
        
        {T} hinv = 1.0 / h;
        {T} h3inv = hinv * hinv * hinv;
        {T} q = r * hinv;
        
        if (q < 1e-8) {{
            return h3inv * 10.666666666666666;
        }}
        
        {T} q2 = q * q;
        if (q <= 0.5) {{
            return h3inv * (10.666666666666666 + q2 * (-38.4 + 32.0 * q));
        }}
        
        {T} q3 = q2 * q;
        return h3inv * (21.333333333333333 - 48.0 * q + 38.4 * q2 - 10.666666666666667 * q3 - 0.0666666666666667 / q3);
    }}
    
    return 0.0;
}}

extern "C" __global__
void nbody_forces_kernel(
    const {T}* __restrict__ x,
    const {T}* __restrict__ y, 
    const {T}* __restrict__ z,
    const {T}* __restrict__ mass,
    const {T}* __restrict__ h,
    {T}* __restrict__ ax,
    {T}* __restrict__ ay,
    {T}* __restrict__ az,
    {T} G,
    {T} eps2,
    int kernel_id,
    int N
) {{
    __shared__ {T} sh_x[TILE_SIZE];
    __shared__ {T} sh_y[TILE_SIZE];
    __shared__ {T} sh_z[TILE_SIZE];
    __shared__ {T} sh_m[TILE_SIZE];
    __shared__ {T} sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Don't return early! Let all threads participate in loading
    bool active = (i < N);

    {T} my_x, my_y, my_z, my_h;
    if (active) {{
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        my_h = h[i];
    }}
    
    {T} sum_ax = 0.0;
    {T} sum_ay = 0.0;
    {T} sum_az = 0.0;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;

        // ALL threads load (or zero) shared memory
        if (j < N) {{
            sh_x[tid] = x[j];
            sh_y[tid] = y[j];
            sh_z[tid] = z[j];
            sh_m[tid] = mass[j];
            sh_h[tid] = h[j];
        }} else {{
            sh_x[tid] = 0.0;
            sh_y[tid] = 0.0;
            sh_z[tid] = 0.0;
            sh_m[tid] = 0.0;  // Zero out mass for out-of-range threads
            sh_h[tid] = 0.0;  // Zero out smoothing length for out-of-range threads
        }}
        
        __syncthreads();

        // Only active threads compute
        if (active) {{ 
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                if (i == j_global) continue;
                
                {T} dx = sh_x[k] - my_x;
                {T} dy = sh_y[k] - my_y;
                {T} dz = sh_z[k] - my_z;
                
                {T} r2 = {FMA}(dx, dx, {FMA}(dy, dy, {FMA}(dz, dz, eps2)));
                {T} h_eff = {FMAX}(my_h, sh_h[k]);
                {T} kern = compute_kernel_factor(r2, h_eff, kernel_id);
                
                {T} mj = sh_m[k];
                {T} factor = mj * kern;
                
                sum_ax = {FMA}(factor, dx, sum_ax);
                sum_ay = {FMA}(factor, dy, sum_ay);
                sum_az = {FMA}(factor, dz, sum_az);
            }}
        }}
        
        __syncthreads();
    }}

    // Only active threads write results
    if (active) {{
        ax[i] = G * sum_ax;
        ay[i] = G * sum_ay;
        az[i] = G * sum_az;
    }}
}}
'''

# ==============================================================================
# KAHAN SUMMATION N-BODY FORCES KERNEL
# ==============================================================================

_KAHAN_KERNEL_TEMPLATE = r'''
#define TILE_SIZE 128  // Try this instead of 256

extern "C" __device__ __forceinline__ 
{T} compute_kernel_factor({T} r2, {T} h, int kernel_id) {{
    // Compute force kernel factor: returns coefficient such that F = G * m * factor * dr
    
    if (kernel_id == 0) {{  // Newtonian: 1/r^3
        {T} r = {SQRT}(r2);
        return 1.0 / (r * r * r);
    }}
    
    if (kernel_id == 1) {{  // Plummer: 1/(r^2 + h^2)^(3/2)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        return {RSQRT}(denom * denom * denom);
    }}
    
    if (kernel_id == 2) {{  // Dehnen k=1 (C2 correction, falcON default)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        {T} sqrt_d = {SQRT}(denom);
        {T} inv_d32 = 1.0 / (denom * sqrt_d);
        {T} inv_d52 = inv_d32 / denom;
        return inv_d32 + 1.5 * h2 * inv_d52;
    }}
    
    if (kernel_id == 3) {{  // Dehnen k=2 (C4 correction)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        {T} sqrt_d = {SQRT}(denom);
        {T} inv_d32 = 1.0 / (denom * sqrt_d);
        {T} inv_d52 = inv_d32 / denom;
        {T} inv_d72 = inv_d52 / denom;
        return inv_d32 + 1.5 * h2 * inv_d52 + 3.75 * h2 * h2 * inv_d72;
    }}
    
    if (kernel_id == 4) {{  // Spline kernel (Monaghan 1992, compact support)
        {T} r = {SQRT}(r2);
        if (r >= h) {{
            return 1.0 / (r * r * r);
        }}
        
        {T} hinv = 1.0 / h;
        {T} h3inv = hinv * hinv * hinv;
        {T} q = r * hinv;
        
        if (q < 1e-8) {{
            return h3inv * 10.666666666666666;
        }}
        
        {T} q2 = q * q;
        if (q <= 0.5) {{
            return h3inv * (10.666666666666666 + q2 * (-38.4 + 32.0 * q));
        }}
        
        {T} q3 = q2 * q;
        return h3inv * (21.333333333333333 - 48.0 * q + 38.4 * q2 - 10.666666666666667 * q3 - 0.0666666666666667 / q3);
    }}
    
    return 0.0;
}}

extern "C" __global__
void nbody_forces_kahan_kernel(
    const {T}* __restrict__ x,
    const {T}* __restrict__ y,
    const {T}* __restrict__ z,
    const {T}* __restrict__ mass,
    const {T}* __restrict__ h,
    {T}* __restrict__ ax,
    {T}* __restrict__ ay,
    {T}* __restrict__ az,
    {T} G,
    {T} eps2,
    int kernel_id,
    int N
) {{
    __shared__ {T} sh_x[TILE_SIZE];
    __shared__ {T} sh_y[TILE_SIZE];
    __shared__ {T} sh_z[TILE_SIZE];
    __shared__ {T} sh_m[TILE_SIZE];
    __shared__ {T} sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Don't return early! Let all threads participate in loading
    bool active = (i < N);

    {T} my_x, my_y, my_z, my_h;
    if (active) {{
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        my_h = h[i];
    }}
    
    // Kahan summation: sum and compensation for each component
    {T} sum_ax = 0.0, comp_ax = 0.0;
    {T} sum_ay = 0.0, comp_ay = 0.0;
    {T} sum_az = 0.0, comp_az = 0.0;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;

        // ALL threads load (or zero) shared memory
        if (j < N) {{
            sh_x[tid] = x[j];
            sh_y[tid] = y[j];
            sh_z[tid] = z[j];
            sh_m[tid] = mass[j];
            sh_h[tid] = h[j];
        }} else {{
            sh_x[tid] = 0.0;
            sh_y[tid] = 0.0;
            sh_z[tid] = 0.0;
            sh_m[tid] = 0.0;
            sh_h[tid] = 0.0;
        }}
        
        __syncthreads();
        
        // Only active threads compute
        if (active) {{
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                if (i == j_global) continue;
                
                {T} dx = sh_x[k] - my_x;
                {T} dy = sh_y[k] - my_y;
                {T} dz = sh_z[k] - my_z;
                
                {T} r2 = {FMA}(dx, dx, {FMA}(dy, dy, {FMA}(dz, dz, eps2)));
                {T} h_eff = {FMAX}(my_h, sh_h[k]);
                {T} kern = compute_kernel_factor(r2, h_eff, kernel_id);
                
                {T} mj = sh_m[k];
                {T} factor = mj * kern;
                
                // Kahan summation for X component
                {T} term_x = factor * dx;
                {T} y_x = term_x - comp_ax;
                {T} t_x = sum_ax + y_x;
                comp_ax = (t_x - sum_ax) - y_x;
                sum_ax = t_x;
                
                // Kahan summation for Y component
                {T} term_y = factor * dy;
                {T} y_y = term_y - comp_ay;
                {T} t_y = sum_ay + y_y;
                comp_ay = (t_y - sum_ay) - y_y;
                sum_ay = t_y;
                
                // Kahan summation for Z component
                {T} term_z = factor * dz;
                {T} y_z = term_z - comp_az;
                {T} t_z = sum_az + y_z;
                comp_az = (t_z - sum_az) - y_z;
                sum_az = t_z;
            }}
        }}
        
        __syncthreads();
    }}
    
    // Only active threads write results
    if (active) {{
        ax[i] = G * sum_ax;
        ay[i] = G * sum_ay;
        az[i] = G * sum_az;
    }}
}}
'''

# ==============================================================================
# POTENTIAL KERNEL (Regular)
# ==============================================================================

_POTENTIAL_KERNEL_TEMPLATE = r'''
#define TILE_SIZE 128

extern "C" __device__ __forceinline__ 
{T} compute_potential_kernel({T} r2, {T} h, int kernel_id) {{
    // Returns -1/r equivalent for potential Φ(r)
    // Note: This is the POTENTIAL, not the force!
    
    {T} r = {SQRT}(r2);
    
    if (kernel_id == 0) {{  // Newtonian: -1/r
        return (r > 0.0) ? -1.0 / r : 0.0;
    }}
    
    if (kernel_id == 1) {{  // Plummer: -1/sqrt(r² + h²)
        return -{RSQRT}(r2 + h * h);
    }}
    
    if (kernel_id == 2) {{  // Dehnen P1 (k=1)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        {T} inv_sqrt = {RSQRT}(denom);           // 1/sqrt(r²+h²)
        {T} inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;  // 1/(r²+h²)^(3/2)
        return -inv_sqrt - 0.5 * h2 * inv_d32;
    }}
    
    if (kernel_id == 3) {{  // Dehnen P2 (k=2)
        {T} h2 = h * h;
        {T} h4 = h2 * h2;
        {T} denom = r2 + h2;
        {T} inv_sqrt = {RSQRT}(denom);
        {T} inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;
        {T} inv_d52 = inv_d32 * inv_sqrt * inv_sqrt;
        return -inv_sqrt - 0.5 * h2 * inv_d32 - 0.375 * h4 * inv_d52;
    }}
    
    if (kernel_id == 4) {{  // Spline kernel (Monaghan 1992)
        if (h == 0.0 || r >= h) {{
            return -1.0 / r;
        }}
        
        {T} hinv = 1.0 / h;
        {T} q = r * hinv;
        
        if (q < 1e-8) {{
            // At origin: lim(q→0) = -2.8/h
            return -2.8 * hinv;
        }}
        
        if (q <= 0.5) {{
            // Inner region: q ≤ 0.5
            {T} q2 = q * q;
            {T} q4 = q2 * q2;
            return (-2.8 + q2 * (5.33333333333333333 + q4 * (6.4 * q - 9.6))) * hinv;
        }}
        
        if (q <= 1.0) {{
            // Outer region: 0.5 < q ≤ 1.0
            {T} q2 = q * q;
            {T} q3 = q2 * q;
            {T} q4 = q2 * q2;
            {T} q5 = q4 * q;
            return (-3.2 + 0.066666666666666666666 / q 
                    + q2 * (10.666666666666666666666 
                    + q * (-16.0 + q * (9.6 - 2.1333333333333333333333 * q)))) * hinv;
        }}
        
        // Beyond softening: pure Newtonian
        return -1.0 / r;
    }}
    
    return 0.0;
}}

extern "C" __global__
void nbody_potential_kernel(
    const {T}* __restrict__ x,
    const {T}* __restrict__ y, 
    const {T}* __restrict__ z,
    const {T}* __restrict__ mass,
    const {T}* __restrict__ h,
    {T}* __restrict__ pot,
    {T} G,
    {T} eps2,
    int kernel_id,
    int N
) {{
    __shared__ {T} sh_x[TILE_SIZE];
    __shared__ {T} sh_y[TILE_SIZE];
    __shared__ {T} sh_z[TILE_SIZE];
    __shared__ {T} sh_m[TILE_SIZE];
    __shared__ {T} sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Don't return early! Let all threads participate in loading
    bool active = (i < N);

    {T} my_x, my_y, my_z, my_h;
    if (active) {{
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        my_h = h[i];
    }}
    
    {T} sum_pot = 0.0;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;
        
        // ALL threads load (or zero) shared memory
        if (j < N) {{
            sh_x[tid] = x[j];
            sh_y[tid] = y[j];
            sh_z[tid] = z[j];
            sh_m[tid] = mass[j];
            sh_h[tid] = h[j];
        }} else {{
            sh_x[tid] = 0.0;
            sh_y[tid] = 0.0;
            sh_z[tid] = 0.0;
            sh_m[tid] = 0.0;
            sh_h[tid] = 0.0;
        }}
        
        __syncthreads();
        
        // Only active threads compute
        if (active) {{
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                if (i == j_global) continue;  // Skip self-potential
                
                {T} dx = sh_x[k] - my_x;
                {T} dy = sh_y[k] - my_y;
                {T} dz = sh_z[k] - my_z;
                
                {T} r2 = {FMA}(dx, dx, {FMA}(dy, dy, {FMA}(dz, dz, eps2)));
                {T} h_eff = {FMAX}(my_h, sh_h[k]);
                {T} phi = compute_potential_kernel(r2, h_eff, kernel_id);
                
                {T} mj = sh_m[k];
                sum_pot += mj * phi;
            }}
        }}
        
        __syncthreads();
    }}
    
    // Only active threads write results
    if (active) {{
        pot[i] = G * sum_pot;
    }}
}}
'''

# ==============================================================================
# POTENTIAL KERNEL (Kahan)
# ==============================================================================

_POTENTIAL_KAHAN_KERNEL_TEMPLATE = r'''
#define TILE_SIZE 128

extern "C" __device__ __forceinline__ 
{T} compute_potential_kernel({T} r2, {T} h, int kernel_id) {{
    // Returns -1/r equivalent for potential Φ(r)
    // Note: This is the POTENTIAL, not the force!
    
    {T} r = {SQRT}(r2);
    
    if (kernel_id == 0) {{  // Newtonian: -1/r
        return (r > 0.0) ? -1.0 / r : 0.0;
    }}
    
    if (kernel_id == 1) {{  // Plummer: -1/sqrt(r² + h²)
        return -{RSQRT}(r2 + h * h);
    }}
    
    if (kernel_id == 2) {{  // Dehnen P1 (k=1)
        {T} h2 = h * h;
        {T} denom = r2 + h2;
        {T} inv_sqrt = {RSQRT}(denom);
        {T} inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;
        return -inv_sqrt - 0.5 * h2 * inv_d32;
    }}
    
    if (kernel_id == 3) {{  // Dehnen P2 (k=2)
        {T} h2 = h * h;
        {T} h4 = h2 * h2;
        {T} denom = r2 + h2;
        {T} inv_sqrt = {RSQRT}(denom);
        {T} inv_d32 = inv_sqrt * inv_sqrt * inv_sqrt;
        {T} inv_d52 = inv_d32 * inv_sqrt * inv_sqrt;
        return -inv_sqrt - 0.5 * h2 * inv_d32 - 0.375 * h4 * inv_d52;
    }}
    
    if (kernel_id == 4) {{  // Spline kernel (Monaghan 1992)
        if (h == 0.0 || r >= h) {{
            return -1.0 / r;
        }}
        
        {T} hinv = 1.0 / h;
        {T} q = r * hinv;
        
        if (q < 1e-8) {{
            return -2.8 * hinv;
        }}
        
        if (q <= 0.5) {{
            {T} q2 = q * q;
            {T} q4 = q2 * q2;
            return (-2.8 + q2 * (5.33333333333333333 + q4 * (6.4 * q - 9.6))) * hinv;
        }}
        
        if (q <= 1.0) {{
            {T} q2 = q * q;
            {T} q3 = q2 * q;
            {T} q4 = q2 * q2;
            {T} q5 = q4 * q;
            return (-3.2 + 0.066666666666666666666 / q 
                    + q2 * (10.666666666666666666666 
                    + q * (-16.0 + q * (9.6 - 2.1333333333333333333333 * q)))) * hinv;
        }}
        
        return -1.0 / r;
    }}
    
    return 0.0;
}}

extern "C" __global__
void nbody_potential_kahan_kernel(
    const {T}* __restrict__ x,
    const {T}* __restrict__ y, 
    const {T}* __restrict__ z,
    const {T}* __restrict__ mass,
    const {T}* __restrict__ h,
    {T}* __restrict__ pot,
    {T} G,
    {T} eps2,
    int kernel_id,
    int N
) {{
    __shared__ {T} sh_x[TILE_SIZE];
    __shared__ {T} sh_y[TILE_SIZE];
    __shared__ {T} sh_z[TILE_SIZE];
    __shared__ {T} sh_m[TILE_SIZE];
    __shared__ {T} sh_h[TILE_SIZE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Don't return early! Let all threads participate in loading
    bool active = (i < N);

    {T} my_x, my_y, my_z, my_h;
    if (active) {{
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        my_h = h[i];
    }}
    
    // Kahan summation for potential
    {T} sum_pot = 0.0;
    {T} comp_pot = 0.0;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;
        
        // ALL threads load (or zero) shared memory
        if (j < N) {{
            sh_x[tid] = x[j];
            sh_y[tid] = y[j];
            sh_z[tid] = z[j];
            sh_m[tid] = mass[j];
            sh_h[tid] = h[j];
        }} else {{
            sh_x[tid] = 0.0;
            sh_y[tid] = 0.0;
            sh_z[tid] = 0.0;
            sh_m[tid] = 0.0;
            sh_h[tid] = 0.0;
        }}
        
        __syncthreads();
        
        // Only active threads compute
        if (active) {{
            int tile_end = min(TILE_SIZE, N - tile_start);
            
            #pragma unroll 8
            for (int k = 0; k < tile_end; k++) {{
                int j_global = tile_start + k;
                
                if (i == j_global) continue;  // Skip self-potential
                
                {T} dx = sh_x[k] - my_x;
                {T} dy = sh_y[k] - my_y;
                {T} dz = sh_z[k] - my_z;
                
                {T} r2 = {FMA}(dx, dx, {FMA}(dy, dy, {FMA}(dz, dz, eps2)));
                {T} h_eff = {FMAX}(my_h, sh_h[k]);
                {T} phi = compute_potential_kernel(r2, h_eff, kernel_id);
                
                {T} mj = sh_m[k];
                
                // Kahan summation
                {T} term = mj * phi;
                {T} y = term - comp_pot;
                {T} t = sum_pot + y;
                comp_pot = (t - sum_pot) - y;
                sum_pot = t;
            }}
        }}
        
        __syncthreads();
    }}
    
    // Only active threads write results
    if (active) {{
        pot[i] = G * sum_pot;
    }}
}}
'''


# ==============================================================================
# TESTS FOR CONSERVATION
# ==============================================================================

CONSERVATION_TESTS = """
# Conservation tests for N-body forces and potential

import numpy as np

def test_momentum_conservation(forces, masses):
    '''
    Test Newton's 3rd law: total momentum change should be zero.
    Net force = sum(m_i * a_i) should be zero for isolated system.
    '''
    net_force = np.sum(masses[:, None] * forces, axis=0)
    net_force_mag = np.linalg.norm(net_force)
    total_mass = np.sum(masses)
    
    print("MOMENTUM CONSERVATION TEST")
    print(f"  Net force vector: {net_force}")
    print(f"  Net force magnitude: {net_force_mag:.6e}")
    print(f"  Net force per unit mass: {net_force_mag/total_mass:.6e}")
    print(f"  Status: {'✓ PASS' if net_force_mag/total_mass < 1e-10 else '✗ FAIL'}")
    
    return net_force_mag/total_mass < 1e-10

def test_angular_momentum_conservation(forces, masses, positions):
    '''
    Test that net torque is zero: τ = sum(r × F) = 0
    '''
    torques = np.cross(positions, masses[:, None] * forces)
    net_torque = np.sum(torques, axis=0)
    net_torque_mag = np.linalg.norm(net_torque)
    
    print("\\nANGULAR MOMENTUM CONSERVATION TEST")
    print(f"  Net torque vector: {net_torque}")
    print(f"  Net torque magnitude: {net_torque_mag:.6e}")
    print(f"  Status: {'✓ PASS' if net_torque_mag < 1e-10 else '✗ FAIL'}")
    
    return net_torque_mag < 1e-10

def test_energy_conservation_virial(forces, masses, positions):
    '''
    Test virial theorem for equilibrium system.
    For a bound system: 2*KE + PE = 0 (virial equilibrium)
    Here we just check if PE is computed correctly via forces.
    
    Relation: F = -∇Φ, so we can check consistency.
    '''
    # Compute potential energy directly (double loop to avoid double counting)
    N = len(masses)
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            PE -= masses[i] * masses[j] / r  # Newtonian only!
    
    print("\\nPOTENTIAL ENERGY TEST (Newtonian)")
    print(f"  Total potential energy: {PE:.6e}")
    
    return PE

def test_potential_symmetry(potential_gpu, potential_cpu, tol=1e-12):
    '''
    Test that GPU and CPU potential agree.
    '''
    diff = np.abs(potential_gpu - potential_cpu)
    rel_err = diff / np.maximum(np.abs(potential_cpu), 1e-20)
    
    print("\\nPOTENTIAL SYMMETRY TEST (GPU vs CPU)")
    print(f"  Max absolute error: {diff.max():.6e}")
    print(f"  Mean absolute error: {diff.mean():.6e}")
    print(f"  Max relative error: {rel_err.max():.6e}")
    print(f"  Mean relative error: {rel_err.mean():.6e}")
    print(f"  Status: {'✓ PASS' if rel_err.max() < tol else '✗ FAIL'}")
    
    return rel_err.max() < tol

def test_total_potential_energy(potentials, masses):
    '''
    Total potential energy should be sum(m_i * Φ_i) / 2
    (divide by 2 because each pair is counted twice)
    '''
    total_PE = 0.5 * np.sum(masses * potentials)
    
    print("\\nTOTAL POTENTIAL ENERGY")
    print(f"  E_pot = 0.5 * sum(m_i * Φ_i) = {total_PE:.6e}")
    
    return total_PE

# Example usage:
'''
# After computing forces
test_momentum_conservation(forces_gpu, masses)
test_angular_momentum_conservation(forces_gpu, masses, positions)

# After computing potentials
test_potential_symmetry(potential_gpu, potential_cpu)
total_PE = test_total_potential_energy(potential_gpu, masses)
'''
"""

__all__ = [
    "_NBODY_KERNEL_TEMPLATE",
    "_NBODY_KERNEL_TEMPLATE_FLOAT4",
    "_KAHAN_KERNEL_TEMPLATE",
    "_KAHAN_KERNEL_TEMPLATE_FLOAT4",
    "_POTENTIAL_KERNEL_TEMPLATE",
    "_POTENTIAL_KERNEL_TEMPLATE_FLOAT4",
    "_POTENTIAL_KAHAN_KERNEL_TEMPLATE",
    "_POTENTIAL_KAHAN_KERNEL_TEMPLATE_FLOAT4"
]

if __name__ == "__main__":
    print("Fixed CUDA kernels for N-body computations")
    print("=" * 80)
    print("\nAvailable templates:")
    print("  1. _NBODY_KERNEL_TEMPLATE - Regular force computation")
    print("  2. _NBODY_KERNEL_TEMPLATE_FLOAT4 - Optimized float4 force computation")
    print("  3. _KAHAN_KERNEL_TEMPLATE - Kahan force computation")
    print("  4. _KAHAN_KERNEL_TEMPLATE4 - Kahan force computation")
    print("  5. _POTENTIAL_KERNEL_TEMPLATE - Regular potential computation")
    print("  6. _POTENTIAL_KERNEL_TEMPLATE_FLOAT4 - Regular potential computation")
    print("  7. _POTENTIAL_KAHAN_KERNEL_TEMPLATE - Kahan potential computation")
    print("  8. _POTENTIAL_KAHAN_KERNEL_TEMPLATE_FLOAT4 - Kahan potential computation")
    print("\nAll kernels properly handle partial tiles for conservation.")
    print("All kernels support {T} templating for float32/float64.")