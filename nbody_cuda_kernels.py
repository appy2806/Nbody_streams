"""
CUDA kernels for N-body force and potential computation.
All kernels properly handle partial tiles for momentum/energy conservation.

Author: Arpit Arora
Date: Sept 2025
"""

# ============================================================================
# RAW CUDA KERNEL (Compiled with nvcc for maximum performance)
# ============================================================================


# ==============================================================================
# REGULAR N-BODY FORCES KERNEL
# ==============================================================================

_NBODY_KERNEL_TEMPLATE = r'''
#define TILE_SIZE 256

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
# tiled test kernel only works in blocks of 256
# For N not divisible by 256, the last block will have inactive threads that do not compute forces, but they still participate in loading shared memory (with zero mass) to ensure momentum conservation.
# ==============================================================================

_NBODY_KERNEL_TEMPLATE_LEGACY = r'''
#define TILE_SIZE 256

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

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const bool active = (i < N);

    {T} my_x = 0.0, my_y = 0.0, my_z = 0.0, my_h = 0.0;
    if (active) {{
        my_x = x[i];
        my_y = y[i];
        my_z = z[i];
        my_h = h[i];
    }}

    {T} sum_ax = 0.0;
    {T} sum_ay = 0.0;
    {T} sum_az = 0.0;

    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Main tile loop: use fast path for all but last tile
    for (int tile = 0; tile < num_tiles; ++tile) {{
        int tile_start = tile * TILE_SIZE;
        int j = tile_start + tid;

        // Decide fast/slow tile once (cheap)
        bool tile_full = (tile != num_tiles - 1) || (N % TILE_SIZE == 0);

        if (tile_full) {{
            // fast path: no bound checks
            sh_x[tid] = x[tile_start + tid];
            sh_y[tid] = y[tile_start + tid];
            sh_z[tid] = z[tile_start + tid];
            sh_m[tid] = mass[tile_start + tid];
            sh_h[tid] = h[tile_start + tid];
        }} else {{
            // last tile: guard loads
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
        }}

        __syncthreads();

        if (active) {{
            int tile_end = (tile_full) ? TILE_SIZE : (N - tile_start);
            #pragma unroll 8
            for (int k = 0; k < tile_end; ++k) {{
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
#define TILE_SIZE 256

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
#define TILE_SIZE 256

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
#define TILE_SIZE 256

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
    "_KAHAN_KERNEL_TEMPLATE",
    "_POTENTIAL_KERNEL_TEMPLATE",
    "_POTENTIAL_KAHAN_KERNEL_TEMPLATE",
]


if __name__ == "__main__":
    print("Fixed CUDA kernels for N-body computations")
    print("=" * 80)
    print("\nAvailable templates:")
    print("  1. _NBODY_KERNEL_TEMPLATE - Regular force computation")
    print("  2. _KAHAN_KERNEL_TEMPLATE - Kahan force computation")
    print("  3. _POTENTIAL_KERNEL_TEMPLATE - Regular potential computation")
    print("  4. _POTENTIAL_KAHAN_KERNEL_TEMPLATE - Kahan potential computation")
    print("\nAll kernels properly handle partial tiles for conservation.")
    print("All kernels support {T} templating for float32/float64.")