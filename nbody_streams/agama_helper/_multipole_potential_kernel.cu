// =============================================================================
//  _multipole_potential_kernel.cu
//  GPU kernels for evaluating Agama Multipole BFE potentials.
//
//  Phase 1 optimizations over baseline potential_kernels.cu:
//    - __ldg() on poly reads  -> 128-byte read-only (texture) cache path
//    - fma() throughout Horner evaluation in quintic_eval
//    - Shared memory for lm_l[], lm_m[] arrays (loaded cooperatively per block)
//    - Designed to pair with Python-side radius sorting for L1 cache reuse
//
//  Replicates Agama's MultipoleInterp1d (potential_multipole.cpp) exactly:
//    - Quintic C2 splines with log-scaling of radial coefficients
//    - Legendre recurrence from src/math_sphharm.cpp (PREFACT/COEF arrays)
//    - Angular assembly:  mul = 2*sqrt(pi) for m=0, 2*sqrt(2*pi) for m!=0
//    - Flat harmonic index: c = l*(l+1)+m,  m in [-l, l]
//      cos-modes: m >= 0,  T_m = cos(m*phi)
//      sin-modes: m <  0,  T_m = sin(|m|*phi)
//
//  Log-scaling (Agama convention, when all Phi[l=0] < 0):
//    c=0:  stored = log(invPhi0 - 1/Phi_0)
//          eval:   expX = exp(stored), Phi_0 = 1/(invPhi0 - expX)
//    c>0:  stored = Phi_c / Phi_0
//          eval:   Phi_c = stored * Phi_0   (with full chain rule for derivs)
//
//  Spline polynomial layout (6 coefficients per interval per harmonic):
//    C(s) = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))   s in [0,1]
//    poly[(c * n_intervals + k) * 6 + {0..5}] = {a0..a5}
//    dC/ds      = a1 + s*(2a2 + s*(3a3 + s*(4a4 + 5a5*s)))
//    d2C/ds2    = 2a2 + s*(6a3 + s*(12a4 + 20a5*s))
//
//  Hessian output layout (matches agama.Potential.forceDeriv):
//    hess_out per particle: [Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
//    -> force derivatives = -hess = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
//
//  Performance notes:
//    - On-the-fly NORM_LM: pfact_cur recurrence, one sqrt() per (absm,l) pair
//    - Rolling trig state: one sincos() per particle, 2 FMAs per m-group step
//    - Template DO_GRAD: compiler eliminates dead branches
//    - __launch_bounds__(256,2): guides register allocation
//    - Separate x,y,z inputs: coalesced warp reads
//
//  Supported lmax <= 32.
// =============================================================================

#include <math.h>

// ---------------------------------------------------------------------------
// Normalization constants from math_sphharm.cpp lines 26-35.
// PREFACT[m] = sqrt((2m+1)/(4*pi*(2m)!))  where  P_m^m = COEF[m] * sin^m(theta)
// NORM_LM[l*(l+1)+m] = PREFACT[m] * prod_{l'=m+1}^{l} sqrt((2l'+1)/(2l'-1)*(l'-m)/(l'+m))
// is computed on-the-fly via pfact_cur recurrence (no table needed).
// ---------------------------------------------------------------------------

__constant__ double PREFACT[33] = {
    0.2820947917738782,   0.3454941494713355,   0.1287580673410632,
    0.02781492157551894,  0.004214597070904597, 0.0004911451888263050,
    4.647273819914057e-05, 3.700296470718545e-06, 2.542785532478802e-07,
    1.536743406172476e-08, 8.287860012085477e-10, 4.035298721198747e-11,
    1.790656309174350e-12, 7.299068453727266e-14, 2.751209457796109e-15,
    9.643748535232993e-17, 3.159120301003413e-18,
    // m=17..32
    9.7128523792757254e-20, 2.8133797388083950e-21, 7.7031283932527611e-23,
    1.9996982303404461e-24, 4.9350344437027067e-26, 1.1606510034403700e-27,
    2.6071087700583650e-29, 5.6045237507440526e-31, 1.1551615366899411e-32,
    2.2866979724449814e-34, 4.3542905194887202e-36, 7.9872656096230955e-38,
    1.4133029977144382e-39, 2.4153082278063579e-41, 3.9913255828701590e-43,
    6.3847411917613422e-45
};

__constant__ double COEF[33] = {
     0.2820947917738782, -0.3454941494713355,  0.3862742020231896,
    -0.4172238236327841,  0.4425326924449826,  -0.4641322034408582,
     0.4830841135800662, -0.5000395635705506,   0.5154289843972843,
    -0.5295529414924496,  0.5426302919442215,  -0.5548257538066191,
     0.5662666637421912, -0.5770536647012670,   0.5872677968601020,
    -0.5969753602424046,  0.6062313441538353,
    // m=17..32
    -0.6150819049288285,  0.6235661940609217, -0.6317177321159495,
     0.6395654582577622, -0.6471345437139063,  0.6544470305506929,
    -0.6615223392074217,  0.6683776760786229, -0.6750283640247020,
     0.6814881127780765, -0.6877692419885917,  0.6938828665927833,
    -0.6998390519465775,  0.7056469444937100, -0.7113148824901042,
     0.7168504903544893
};

// 2*sqrt(pi) and 2*sqrt(2*pi)
#define MUL0  3.5449077018110318
#define MUL1  5.0132565706694072

// ---------------------------------------------------------------------------
//  quintic_eval --- evaluate a degree-5 polynomial and its 1st/2nd derivatives.
//
//  Layout: poly[(c * n_intervals + k) * 6 + {0..5}] = {a0, a1, a2, a3, a4, a5}
//    C(s)    = a0 + s*(a1 + s*(a2 + s*(a3 + s*(a4 + s*a5))))
//    C'(s)   = a1 + s*(2a2 + s*(3a3 + s*(4a4 + 5a5*s)))
//    C''(s)  = 2a2 + s*(6a3 + s*(12a4 + 20a5*s))
//  Pass NULL to skip d/d2 outputs.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void
quintic_eval(const double* __restrict__ poly,
             int c, int k, int n_intervals, double s,
             double* val, double* dval_ds, double* d2val_ds2)
{
    // __ldg: route through 128-byte read-only (texture) cache.
    // After radius sorting, consecutive threads in a warp land in the same
    // radial interval k, so all 6 doubles for a given (c,k) entry are shared
    // across the warp -> high L1 hit rate.
    int base = (c * n_intervals + k) * 6;
    double a0=__ldg(poly+base  ), a1=__ldg(poly+base+1), a2=__ldg(poly+base+2);
    double a3=__ldg(poly+base+3), a4=__ldg(poly+base+4), a5=__ldg(poly+base+5);
    // fma() Horner form: fewer rounding operations + compiler can pipeline MAs.
    *val = fma(s, fma(s, fma(s, fma(s, fma(s, a5, a4), a3), a2), a1), a0);
    if (dval_ds)
        *dval_ds = fma(s, fma(s, fma(s, fma(s, 5.0*a5, 4.0*a4), 3.0*a3), 2.0*a2), a1);
    if (d2val_ds2)
        *d2val_ds2 = fma(s, fma(s, fma(s, 20.0*a5, 12.0*a4), 6.0*a3), 2.0*a2);
}


// ---------------------------------------------------------------------------
//  compute_Plm --- REMOVED: replaced by on-the-fly recurrence in each kernel.
//  (Keeping this comment as a tombstone so git blame is clear.)
//
//  OLD: fill normalized associated Legendre P_l^|m|(cos theta)
//  for all (l,m) with 0 <= l <= lmax, 0 <= m <= l.
//  Flat index: c = l*(l+1)+m  (positive m only)
// ---------------------------------------------------------------------------

// compute_Plm removed --- see tombstone comment above.
// All kernels now use an on-the-fly recurrence (8 registers/m-group vs 289-element arrays).


// ---------------------------------------------------------------------------
//  multipole_eval_device<DO_GRAD>
//  One thread per particle: potential + (optionally) gradient.
//  Handles quintic splines and Agama log-scaling.
// ---------------------------------------------------------------------------

template<bool DO_GRAD>
__device__ void
multipole_eval_device(
    int tid,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l,
    const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out,
    double* __restrict__ grad_out,
    int N)
{
    if (tid >= N) return;

    double px = x[tid], py = y[tid], pz = z[tid];
    double r2   = px*px + py*py + pz*pz;
    double r    = sqrt(r2);
    double Rcyl = sqrt(px*px + py*py);

    if (r < 1.0e-300) {
        // At exact origin: Phi = W (finite), force = 0
        phi_out[tid] = MUL0 * inner_W * PREFACT[0];
        if (DO_GRAD) { grad_out[3*tid]=0.; grad_out[3*tid+1]=0.; grad_out[3*tid+2]=0.; }
        return;
    }

    double logr = log(r);

    // Inner extrapolation: for r < r_min, use power-law model
    // Phi_0(r) = U * (r/r0)^s + W; higher harmonics scale as (r/r0)^l
    if (logr < logr_min) {
        double dlr = logr - logr_min;  // negative
        double r_ratio_s = (inner_s != 0.0) ? exp(inner_s * dlr) : 1.0;
        double Phi0 = inner_U * r_ratio_s + inner_W;
        phi_out[tid] = MUL0 * Phi0 * PREFACT[0];  // l=0 only (higher l vanish as r^l -> 0)
        if (DO_GRAD) {
            // dPhi/d(logr) = U * s * (r/r0)^s
            double dPhi0_dlr = inner_U * inner_s * r_ratio_s;
            double inv_r2 = 1.0 / r2;
            // gradient = dPhi/dlogr * (x/r^2) for each component
            double g = MUL0 * dPhi0_dlr * PREFACT[0];
            grad_out[3*tid+0] = g * (px * inv_r2);
            grad_out[3*tid+1] = g * (py * inv_r2);
            grad_out[3*tid+2] = g * (pz * inv_r2);
        }
        return;
    }

    int k = (int)((logr - logr_min) * inv_dlogr);
    if (k < 0) k = 0;
    if (k >= n_intervals) {
        // Outer extrapolation (l=0 only): Phi(r) = W*(r/r_max)^(-1) + U*(r/r_max)^s
        double logr_max = logr_min + n_intervals * dlogr;
        double dlr      = logr - logr_max;   // > 0
        double exp_neg  = exp(-dlr);
        double exp_s    = exp(outer_s * dlr);
        double Phi0     = outer_W * exp_neg + outer_U * exp_s;
        phi_out[tid]    = MUL0 * Phi0 * PREFACT[0];
        if (DO_GRAD) {
            double dPhi0_dlr = -outer_W * exp_neg + outer_s * outer_U * exp_s;
            double g = MUL0 * dPhi0_dlr * PREFACT[0];
            double inv_r2 = 1.0 / r2;
            grad_out[3*tid+0] = g * px * inv_r2;
            grad_out[3*tid+1] = g * py * inv_r2;
            grad_out[3*tid+2] = g * pz * inv_r2;
        }
        return;
    }
    double s = (logr - (logr_min + k * dlogr)) * inv_dlogr;
    if (s < 0.0) s = 0.0; if (s > 1.0) s = 1.0;

    double cos_theta = pz   / r;
    double sin_theta = Rcyl / r;
    double phi_az    = atan2(py, px);

    // Rolling trig state: cm = cos(absm*phi), sm = sin(absm*phi).
    // Advanced once per m-group via: cm_new = cm*cph - sm*sph; sm = sm*cph + cm*sph.
    double cph, sph;
    sincos(phi_az, &sph, &cph);
    double cm = 1.0, sm = 0.0;  // cos(0), sin(0)

    // P_0^0 = PREFACT[0] (constant), dP_0^0/dθ = 0 --- no arrays needed.
    const bool near_pole = (sin_theta < 1.0e-10);

    double Phi            = 0.0;
    double dPhi_dlogr     = 0.0;
    double dPhi_dtheta    = 0.0;
    double dPhi_dphi_os   = 0.0;  // dPhi/dphi / sin_theta (pole-safe)

    // -----------------------------------------------------------------------
    // c=0 (l=0, m=0): evaluate and apply log-scaling inverse transform
    // -----------------------------------------------------------------------
    double C0_sc, dC0_sc_ds = 0.0;
    quintic_eval(poly, 0, k, n_intervals, s,
                 &C0_sc,
                 DO_GRAD ? &dC0_sc_ds : (double*)0,
                 (double*)0);

    double C0_val  = C0_sc;
    double dC0_dlr = 0.0;

    if (log_scaling) {
        double expX   = exp(C0_sc);
        double Phi0   = 1.0 / (invPhi0 - expX);
        double dPhidX = Phi0 * Phi0 * expX;
        if (DO_GRAD) dC0_dlr = dPhidX * (dC0_sc_ds * inv_dlogr);
        C0_val = Phi0;
    } else {
        if (DO_GRAD) dC0_dlr = dC0_sc_ds * inv_dlogr;
    }

    // c=0: Ylm=PREFACT[0], dYlm=0 (constant), Tlm=1
    Phi += MUL0 * C0_val * PREFACT[0];
    if (DO_GRAD) dPhi_dlogr += MUL0 * dC0_dlr * PREFACT[0];
    // dPhi_dtheta += 0; dPhi_dphi_os += 0

    // -----------------------------------------------------------------------
    // c>0: on-the-fly Legendre recurrence (lm sorted by |m|, l in Python).
    // Replaces Plm_arr[289]/dPlm_arr[289] with ~8 registers per m-group.
    //   raw_cur/prev  = P_l^|m| / PREFACT[|m|]   (un-normalized)
    //   der_cur/prev  = d/dθ of the above
    //   pfact_cur     = NORM_LM[l*(l+1)+absm]  (on-the-fly, updated per l)
    // -----------------------------------------------------------------------
    double sin_pow = 1.0;  // sin^absm; updated at end of each m-group
    int ci = 1;            // cursor through sorted lm arrays; c=0 already done

    for (int absm = 0; absm <= lmax && ci < n_lm; absm++) {
        // Skip m-group if no active term has this |m|
        int ci_absm = (lm_m[ci] >= 0) ? lm_m[ci] : -lm_m[ci];
        if (ci_absm != absm) {
            sin_pow = (absm == 0) ? sin_theta : sin_pow * sin_theta;
            { double _t = cm*cph - sm*sph; sm = sm*cph + cm*sph; cm = _t; }
            continue;
        }

        // --- Initialize P_m^m / PREFACT[m] ---
        double pf = PREFACT[absm];
        double pfact_cur = pf;  // = NORM_LM[absm*(absm+1)+absm]; advanced per l in l-loop
        double raw_prev;  // holds P_{l}^absm/PREFACT at l=absm
        double der_prev = 0.0;
        if (absm == 0) {
            raw_prev = 1.0;
        } else if (absm == 1) {
            raw_prev = -sin_theta;
            if (DO_GRAD) der_prev = -cos_theta;
        } else {
            raw_prev = (pf != 0.0) ? COEF[absm] * sin_pow / pf : 0.0;
            if (DO_GRAD) der_prev = near_pole ? 0.0
                : (double)absm * COEF[absm] * sin_pow / (pf * sin_theta) * cos_theta;
        }
        // raw_cur = P_{m+1}^m / PREFACT[m]
        double raw_cur = raw_prev * cos_theta * (double)(2*absm + 1);
        double der_cur = 0.0;
        if (DO_GRAD)
            der_cur = der_prev*cos_theta*(double)(2*absm+1)
                    - raw_prev*sin_theta*(double)(2*absm+1);

        // Helper macro: accumulate one active term at (l_val, m_val=+-absm, ci_idx)
        // Uses raw_lm = raw_cur (l > absm) or raw_prev (l == absm).
        // pfact_cur = NORM_LM[l_val*(l_val+1)+absm] (maintained by caller).
        // cm/sm = cos(absm*phi) / sin(absm*phi) (rolling state, outer scope).
#define ACCUM_LM(ci_idx, l_val, m_val, raw_lm, der_lm)                          \
        {                                                                         \
            double _Ylm  = (raw_lm) * pfact_cur;                                 \
            double _Cc_sc, _dCc_sc_ds = 0.0;                                     \
            quintic_eval(poly, (ci_idx), k, n_intervals, s,                      \
                         &_Cc_sc, DO_GRAD ? &_dCc_sc_ds : (double*)0, (double*)0); \
            double _Cv = _Cc_sc, _dCv = 0.0;                                     \
            if (log_scaling) {                                                    \
                double _dlr = _dCc_sc_ds * inv_dlogr;                            \
                if (DO_GRAD) _dCv = _dlr*C0_val + _Cc_sc*dC0_dlr;               \
                _Cv = _Cc_sc * C0_val;                                            \
            } else {                                                              \
                if (DO_GRAD) _dCv = _dCc_sc_ds * inv_dlogr;                     \
            }                                                                     \
            double _mul = (absm == 0) ? MUL0 : MUL1;                            \
            double _Tlm = ((m_val) >= 0) ? cm : sm;                              \
            Phi += _mul * _Cv * _Ylm * _Tlm;                                     \
            if (DO_GRAD) {                                                        \
                double _dYlm = (der_lm) * pfact_cur;                             \
                double _dTlm = ((m_val) >= 0)                                    \
                    ? -(double)absm * sm                                          \
                    :  (double)absm * cm;                                         \
                dPhi_dlogr  += _mul * _dCv  * _Ylm * _Tlm;                      \
                dPhi_dtheta += _mul * _Cv   * _dYlm * _Tlm;                     \
                double _Pos = (absm == 0) ? 0.0                                  \
                    : (sin_theta > 1.0e-10) ? _Ylm / sin_theta                  \
                    : ((absm == 1) ? _dYlm : 0.0);                               \
                dPhi_dphi_os += _mul * _Cv * _Pos * _dTlm;                      \
            }                                                                     \
        }

        // --- l = absm (diagonal P_m^m) ---
        if (lm_l[ci] == absm) {
            if (lm_m[ci] == absm)                          // cos mode
                { ACCUM_LM(ci, absm, absm,  raw_prev, der_prev); ci++; }
            if (absm > 0 && ci < n_lm &&
                lm_l[ci] == absm && lm_m[ci] == -absm)    // sin mode
                { ACCUM_LM(ci, absm, -absm, raw_prev, der_prev); ci++; }
        }

        // --- l = absm+1 … lmax ---
        for (int l = absm + 1; l <= lmax; l++) {
            if (l > absm + 1) {
                double inv_lm = 1.0 / (double)(l - absm);
                double nr = fma((double)(2*l-1)*cos_theta, raw_cur,
                                -(double)(l+absm-1)*raw_prev) * inv_lm;
                double nd = 0.0;
                if (DO_GRAD)
                    nd = fma((double)(2*l-1),
                             fma(cos_theta, der_cur, -sin_theta*raw_cur),
                             -(double)(l+absm-1)*der_prev) * inv_lm;
                raw_prev = raw_cur; der_prev = der_cur;
                raw_cur  = nr;      der_cur  = nd;
            }
            // pfact_cur advances from PREFACT[absm] -> NORM_LM[l*(l+1)+absm]
            pfact_cur *= sqrt((2.0*l+1)/(2.0*l-1) * (double)(l-absm)/(double)(l+absm));
            if (ci >= n_lm) break;
            if (lm_l[ci] != l) {
                // Not active at l but next ci has higher l (same absm or different)
                int next_absm2 = (lm_m[ci] >= 0) ? lm_m[ci] : -lm_m[ci];
                if (next_absm2 != absm) break;   // done with this m-group
                continue;                         // skip to next l
            }
            if ((lm_m[ci] >= 0 ? lm_m[ci] : -lm_m[ci]) != absm) break;

            if (lm_m[ci] == absm)                          // cos mode
                { ACCUM_LM(ci, l, absm,  raw_cur, der_cur); ci++; }
            if (absm > 0 && ci < n_lm &&
                lm_l[ci] == l && lm_m[ci] == -absm)       // sin mode
                { ACCUM_LM(ci, l, -absm, raw_cur, der_cur); ci++; }

            if (ci >= n_lm ||
                ((lm_m[ci] >= 0 ? lm_m[ci] : -lm_m[ci]) != absm)) break;
        }
#undef ACCUM_LM

        sin_pow = (absm == 0) ? sin_theta : sin_pow * sin_theta;
        { double _t = cm*cph - sm*sph; sm = sm*cph + cm*sph; cm = _t; }
    }

    phi_out[tid] = Phi;

    if (DO_GRAD) {
        double inv_r  = 1.0 / r;
        double inv_r2 = inv_r * inv_r;
        double cos_phi = (Rcyl > 1.0e-300) ? px / Rcyl : 1.0;
        double sin_phi = (Rcyl > 1.0e-300) ? py / Rcyl : 0.0;

        // Gradient in Cartesian using pole-safe phi contribution:
        //   dPhi/dphi * dphi/dx_i = (dPhi/dphi / sin_theta) * (-sin_phi / r)  for x
        //                         = (dPhi/dphi / sin_theta) * ( cos_phi / r)  for y
        // Since R = r*sin_theta, this is equivalent to the standard formula
        // away from the pole, but finite at sin_theta=0.
        grad_out[3*tid+0] = dPhi_dlogr*(px*inv_r2)
                          + dPhi_dtheta*(cos_theta*cos_phi*inv_r)
                          + dPhi_dphi_os*(-sin_phi*inv_r);
        grad_out[3*tid+1] = dPhi_dlogr*(py*inv_r2)
                          + dPhi_dtheta*(cos_theta*sin_phi*inv_r)
                          + dPhi_dphi_os*(cos_phi*inv_r);
        grad_out[3*tid+2] = dPhi_dlogr*(pz*inv_r2)
                          + dPhi_dtheta*(-sin_theta*inv_r);
    }
}


// ---- C-linkage __global__ wrappers ----

extern "C" __global__ __launch_bounds__(256,2) void
multipole_potential_kernel(
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l, const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out, double* __restrict__ grad_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    multipole_eval_device<false>(tid, x,y,z,poly,logr_min,dlogr,inv_dlogr,
                                 n_intervals,n_lm,lmax,lm_l,lm_m,
                                 log_scaling,invPhi0,inner_s,inner_U,inner_W,
                                 outer_s,outer_U,outer_W,
                                 phi_out,grad_out,N);
}

extern "C" __global__ __launch_bounds__(256,2) void
multipole_force_kernel(
    const double* __restrict__ x, const double* __restrict__ y, const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l, const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out, double* __restrict__ grad_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    multipole_eval_device<true>(tid, x,y,z,poly,logr_min,dlogr,inv_dlogr,
                                n_intervals,n_lm,lmax,lm_l,lm_m,
                                log_scaling,invPhi0,inner_s,inner_U,inner_W,
                                outer_s,outer_U,outer_W,
                                phi_out,grad_out,N);
}


// ---------------------------------------------------------------------------
//  multipole_hess_kernel --- potential + gradient + Hessian (6 components)
//
//  Output layout matches agama.Potential.forceDeriv:
//    hess_out[6*i + {0..5}] = [Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
//  -> force derivatives = −H = [dFx/dx, dFy/dy, dFz/dz, dFx/dy, dFy/dz, dFz/dx]
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(256, 2) void
multipole_hess_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l,
    const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double* __restrict__ phi_out,
    double* __restrict__ grad_out,
    double* __restrict__ hess_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double px = x[tid], py = y[tid], pz = z[tid];
    double r2   = px*px + py*py + pz*pz;
    double r    = sqrt(r2);
    double Rcyl = sqrt(px*px + py*py);

    if (r < 1.0e-300) {
        phi_out[tid] = MUL0 * inner_W * PREFACT[0];
        for (int i=0;i<3;i++) grad_out[3*tid+i]=0.;
        for (int i=0;i<6;i++) hess_out[6*tid+i]=0.;
        return;
    }

    double logr = log(r);

    // Inner extrapolation (same as force kernel, plus Hessian from radial-only l=0)
    if (logr < logr_min) {
        double dlr = logr - logr_min;
        double r_ratio_s = (inner_s != 0.0) ? exp(inner_s * dlr) : 1.0;
        double Phi0 = inner_U * r_ratio_s + inner_W;
        phi_out[tid] = MUL0 * Phi0 * PREFACT[0];
        double dPhi0_dlr = inner_U * inner_s * r_ratio_s;
        double d2Phi0_dlr2 = inner_U * inner_s * inner_s * r_ratio_s;
        double g = MUL0 * dPhi0_dlr * PREFACT[0];
        double inv_r2 = 1.0 / r2;
        grad_out[3*tid+0] = g * px * inv_r2;
        grad_out[3*tid+1] = g * py * inv_r2;
        grad_out[3*tid+2] = g * pz * inv_r2;
        // Hessian: d2Phi/dxi dxj = (d2Phi/dlogr^2 - dPhi/dlogr)*xi*xj/r^4
        //                         + dPhi/dlogr * delta_ij / r^2
        double A = MUL0 * PREFACT[0] * (d2Phi0_dlr2 - 2.0*dPhi0_dlr) * inv_r2 * inv_r2;
        double B = MUL0 * PREFACT[0] * dPhi0_dlr * inv_r2;
        double coords[3] = {px, py, pz};
        int ii[6]={0,1,2,0,1,0}, jj[6]={0,1,2,1,2,2};
        for (int p=0;p<6;p++)
            hess_out[6*tid+p] = A*coords[ii[p]]*coords[jj[p]] + (ii[p]==jj[p] ? B : 0.0);
        return;
    }

    int k = (int)((logr - logr_min) * inv_dlogr);
    if (k < 0) k = 0;
    if (k >= n_intervals) {
        // Outer extrapolation (l=0 only): Phi(r) = W*(r/r_max)^(-1) + U*(r/r_max)^s
        double logr_max  = logr_min + n_intervals * dlogr;
        double dlr       = logr - logr_max;   // > 0
        double exp_neg   = exp(-dlr);
        double exp_s     = exp(outer_s * dlr);
        double Phi0      = outer_W * exp_neg + outer_U * exp_s;
        double dPhi0     = -outer_W * exp_neg + outer_s * outer_U * exp_s;
        double d2Phi0    = outer_W * exp_neg + outer_s * outer_s * outer_U * exp_s;
        phi_out[tid] = MUL0 * Phi0 * PREFACT[0];
        double inv_r2 = 1.0 / r2;
        double g = MUL0 * dPhi0 * PREFACT[0];
        grad_out[3*tid+0] = g * px * inv_r2;
        grad_out[3*tid+1] = g * py * inv_r2;
        grad_out[3*tid+2] = g * pz * inv_r2;
        double A = MUL0 * PREFACT[0] * (d2Phi0 - 2.0*dPhi0) * inv_r2 * inv_r2;
        double B = MUL0 * PREFACT[0] * dPhi0 * inv_r2;
        double coords[3] = {px, py, pz};
        int ii[6]={0,1,2,0,1,0}, jj[6]={0,1,2,1,2,2};
        for (int p=0;p<6;p++)
            hess_out[6*tid+p] = A*coords[ii[p]]*coords[jj[p]] + (ii[p]==jj[p] ? B : 0.0);
        return;
    }
    double s = (logr - (logr_min + k * dlogr)) * inv_dlogr;
    if (s < 0.0) s = 0.0; if (s > 1.0) s = 1.0;

    double cos_theta = pz   / r;
    double sin_theta = Rcyl / r;
    double phi_az    = atan2(py, px);

    double cph, sph;
    sincos(phi_az, &sph, &cph);
    double cm = 1.0, sm = 0.0;  // rolling: cos(absm*phi), sin(absm*phi)

    double Phi=0., dPhi_dlr=0., d2Phi_dlr2=0.;
    double dPhi_dth=0., d2Phi_dth2=0., d2Phi_dlr_dth=0.;
    double dPhi_dph=0., d2Phi_dph2=0., d2Phi_dlr_dph=0., d2Phi_dth_dph=0.;
    double dPhi_dph_os=0.;  // dPhi/dphi / sin_theta (pole-safe, for gradient)

    // -----------------------------------------------------------------------
    // c=0: evaluate + log-scaling inverse transform (with 2nd derivatives)
    // -----------------------------------------------------------------------
    double C0_sc, dC0_sc_ds, d2C0_sc_ds2;
    quintic_eval(poly, 0, k, n_intervals, s, &C0_sc, &dC0_sc_ds, &d2C0_sc_ds2);

    double C0_val      = C0_sc;
    double dC0_dlr     = dC0_sc_ds * inv_dlogr;
    double d2C0_dlr2   = d2C0_sc_ds2 * inv_dlogr * inv_dlogr;

    if (log_scaling) {
        double expX   = exp(C0_sc);
        double Phi0   = 1.0 / (invPhi0 - expX);
        double dPhidX = Phi0 * Phi0 * expX;
        // d2Phi/dX2 = dPhidX * (1 + 2*Phi0*expX) ... from Agama:
        // d2Phi0 = dPhidX*(d2C0_dlr2 + dC0_dlr^2 * Phi0*(invPhi0+expX))
        double d2Phi0_dlr2 = dPhidX * (d2C0_dlr2
                             + dC0_dlr * dC0_dlr * Phi0 * (invPhi0 + expX));
        double dPhi0_dlr   = dPhidX * dC0_dlr;
        C0_val    = Phi0;
        dC0_dlr   = dPhi0_dlr;
        d2C0_dlr2 = d2Phi0_dlr2;
    }

    // c=0 angular contribution (l=0, m=0):  P_0^0 = PREFACT[0] (constant), dP=0, d2P=0
    Phi              += MUL0 * C0_val    * PREFACT[0];
    dPhi_dlr         += MUL0 * dC0_dlr   * PREFACT[0];
    d2Phi_dlr2       += MUL0 * d2C0_dlr2 * PREFACT[0];
    // dPhi_dth, d2Phi_dth2, d2Phi_dlr_dth all 0 (dP_0^0/dθ = 0)

    // -----------------------------------------------------------------------
    // c>0: on-the-fly Legendre recurrence (lm sorted by |m|, l in Python).
    // Replaces Plm_arr/dPlm_arr/d2Plm_arr[289] with ~12 registers per m-group.
    //   raw_cur/prev  = P_l^|m| / PREFACT[|m|]   (un-normalized)
    //   der_cur/prev  = d(P_l^|m|)/dθ / PREFACT[|m|]
    //   d2Ylm via Legendre ODE: d2P/dθ^2 = -cot·dP/dθ - [l(l+1) - m^2/sin^2]·P
    // -----------------------------------------------------------------------
    const bool near_pole = (sin_theta < 1.0e-10);

#define ACCUM_HESS(ci_idx, l_val, m_val, raw_lm, der_lm)                            \
    {                                                                                 \
        double _Ylm   = (raw_lm) * pfact_cur;                                        \
        double _dYlm  = (der_lm) * pfact_cur;                                        \
        double _d2Ylm = near_pole ? 0.0 :                                            \
            (-cos_theta/sin_theta * _dYlm                                            \
             - ((double)((l_val)*((l_val)+1)) - (double)(absm*absm)                  \
                / (sin_theta*sin_theta)) * _Ylm);                                    \
        double _Cc_sc, _dCc_sc_ds, _d2Cc_sc_ds2;                                    \
        quintic_eval(poly,(ci_idx),k,n_intervals,s,&_Cc_sc,&_dCc_sc_ds,&_d2Cc_sc_ds2); \
        double _dCc_dlr   = _dCc_sc_ds  * inv_dlogr;                                \
        double _d2Cc_dlr2 = _d2Cc_sc_ds2 * inv_dlogr * inv_dlogr;                   \
        double _Cv = _Cc_sc, _dv = _dCc_dlr, _d2v = _d2Cc_dlr2;                    \
        if (log_scaling) {                                                            \
            _d2v = _d2Cc_dlr2*C0_val + 2.0*_dCc_dlr*dC0_dlr + _Cc_sc*d2C0_dlr2;   \
            _dv  = _dCc_dlr*C0_val + _Cc_sc*dC0_dlr;                                \
            _Cv  = _Cc_sc * C0_val;                                                  \
        }                                                                             \
        double _mul   = (absm==0) ? MUL0 : MUL1;                                    \
        double _Tlm   = ((m_val)>=0) ? cm : sm;                                      \
        double _dTlm  = ((m_val)>=0) ? -(double)absm*sm                              \
                                      :  (double)absm*cm;                             \
        double _d2Tlm = -(double)(absm*absm) * _Tlm;                                \
        double _YT   = _Ylm*_Tlm,  _dYT  = _dYlm*_Tlm,  _d2YT  = _d2Ylm*_Tlm;    \
        double _YdT  = _Ylm*_dTlm, _dYdT = _dYlm*_dTlm, _Yd2T  = _Ylm*_d2Tlm;    \
        Phi              += _mul*_Cv  *_YT;                                          \
        dPhi_dlr         += _mul*_dv  *_YT;                                          \
        d2Phi_dlr2       += _mul*_d2v *_YT;                                          \
        dPhi_dth         += _mul*_Cv  *_dYT;                                         \
        d2Phi_dth2       += _mul*_Cv  *_d2YT;                                        \
        d2Phi_dlr_dth    += _mul*_dv  *_dYT;                                         \
        dPhi_dph         += _mul*_Cv  *_YdT;                                         \
        d2Phi_dph2       += _mul*_Cv  *_Yd2T;                                        \
        d2Phi_dlr_dph    += _mul*_dv  *_YdT;                                         \
        d2Phi_dth_dph    += _mul*_Cv  *_dYdT;                                        \
        double _Plm_os;                                                              \
        if (absm == 0)       { _Plm_os = 0.0; }                                     \
        else if (!near_pole) { _Plm_os = _Ylm / sin_theta; }                        \
        else                 { _Plm_os = (absm==1) ? _dYlm : 0.0; }                 \
        dPhi_dph_os += _mul*_Cv * _Plm_os * _dTlm;                                  \
    }

    double sin_pow = 1.0;
    int ci = 1;

    for (int absm = 0; absm <= lmax && ci < n_lm; absm++) {
        int ci_absm = (lm_m[ci] >= 0) ? lm_m[ci] : -lm_m[ci];
        if (ci_absm != absm) {
            sin_pow = (absm == 0) ? sin_theta : sin_pow * sin_theta;
            { double _t = cm*cph - sm*sph; sm = sm*cph + cm*sph; cm = _t; }
            continue;
        }
        double pf = PREFACT[absm];
        double pfact_cur = pf;
        double raw_prev;
        if      (absm == 0) raw_prev = 1.0;
        else if (absm == 1) raw_prev = -sin_theta;
        else                raw_prev = (pf != 0.0) ? COEF[absm]*sin_pow/pf : 0.0;
        // d(raw_prev)/dθ: P_m^m = C_m*sin^m -> dP_m^m/dθ = m*cot(θ)*P_m^m
        double der_prev;
        if      (absm == 0) der_prev = 0.0;
        else if (absm == 1) der_prev = -cos_theta;
        else                der_prev = near_pole ? 0.0
                                       : (double)absm * (cos_theta/sin_theta) * raw_prev;
        // l=absm+1 seed: P_{m+1}^m = (2m+1)*cos(θ)*P_m^m
        double raw_cur = raw_prev * cos_theta * (double)(2*absm + 1);
        double der_cur = (double)(2*absm + 1) * (cos_theta*der_prev - sin_theta*raw_prev);

        if (lm_l[ci] == absm) {
            if (lm_m[ci] == absm)
                { ACCUM_HESS(ci, absm, absm,  raw_prev, der_prev); ci++; }
            if (absm > 0 && ci < n_lm &&
                lm_l[ci] == absm && lm_m[ci] == -absm)
                { ACCUM_HESS(ci, absm, -absm, raw_prev, der_prev); ci++; }
        }
        for (int l = absm + 1; l <= lmax; l++) {
            if (l > absm + 1) {
                double inv_lm = 1.0 / (double)(l - absm);
                double nr = fma((double)(2*l-1)*cos_theta, raw_cur,
                                -(double)(l+absm-1)*raw_prev) * inv_lm;
                double nd = ((double)(2*l-1)*(-sin_theta*raw_cur + cos_theta*der_cur)
                             - (double)(l+absm-1)*der_prev) * inv_lm;
                raw_prev = raw_cur;  der_prev = der_cur;
                raw_cur  = nr;       der_cur  = nd;
            }
            pfact_cur *= sqrt((2.0*l+1)/(2.0*l-1) * (double)(l-absm)/(double)(l+absm));
            if (ci >= n_lm) break;
            if (lm_l[ci] != l) {
                int na2 = (lm_m[ci] >= 0) ? lm_m[ci] : -lm_m[ci];
                if (na2 != absm) break;
                continue;
            }
            if ((lm_m[ci] >= 0 ? lm_m[ci] : -lm_m[ci]) != absm) break;
            if (lm_m[ci] == absm)
                { ACCUM_HESS(ci, l, absm,  raw_cur, der_cur); ci++; }
            if (absm > 0 && ci < n_lm &&
                lm_l[ci] == l && lm_m[ci] == -absm)
                { ACCUM_HESS(ci, l, -absm, raw_cur, der_cur); ci++; }
            if (ci >= n_lm ||
                ((lm_m[ci] >= 0 ? lm_m[ci] : -lm_m[ci]) != absm)) break;
        }
#undef ACCUM_HESS
        sin_pow = (absm == 0) ? sin_theta : sin_pow * sin_theta;
        { double _t = cm*cph - sm*sph; sm = sm*cph + cm*sph; cm = _t; }
    }

    phi_out[tid] = Phi;

    // Cartesian gradient (pole-safe: uses dPhi_dph_os = dPhi/dphi / sin_theta)
    double inv_r  = 1.0/r, inv_r2 = inv_r*inv_r;
    double inv_R  = (Rcyl>1.e-300) ? 1./Rcyl : 0.;
    double cos_phi= (Rcyl>1.e-300) ? px/Rcyl : 1.;
    double sin_phi= (Rcyl>1.e-300) ? py/Rcyl : 0.;

    double dlr_dx=px*inv_r2, dlr_dy=py*inv_r2, dlr_dz=pz*inv_r2;
    double dth_dx= cos_theta*cos_phi*inv_r;
    double dth_dy= cos_theta*sin_phi*inv_r;
    double dth_dz=-sin_theta*inv_r;
    // Pole-safe gradient: dPhi/dphi * dphi/dx = (dPhi_dph/sin_theta) * (-sin_phi/r)
    grad_out[3*tid+0]=dPhi_dlr*dlr_dx+dPhi_dth*dth_dx+dPhi_dph_os*(-sin_phi*inv_r);
    grad_out[3*tid+1]=dPhi_dlr*dlr_dy+dPhi_dth*dth_dy+dPhi_dph_os*(cos_phi*inv_r);
    grad_out[3*tid+2]=dPhi_dlr*dlr_dz+dPhi_dth*dth_dz;

    // Cartesian Hessian --- full chain rule
    // (uses standard inv_R formulation; at exact pole, phi-dependent terms -> 0)
    double dph_dx=-sin_phi*inv_R, dph_dy=cos_phi*inv_R;
    double inv_r3=inv_r2*inv_r, inv_r4=inv_r2*inv_r2, inv_r5=inv_r4*inv_r;
    double inv_R2=inv_R*inv_R;

    double d2lr_xx=inv_r2-2.*px*px*inv_r4, d2lr_xy=-2.*px*py*inv_r4,
           d2lr_xz=-2.*px*pz*inv_r4,       d2lr_yy=inv_r2-2.*py*py*inv_r4,
           d2lr_yz=-2.*py*pz*inv_r4,        d2lr_zz=inv_r2-2.*pz*pz*inv_r4;

    double df_dx=-pz*px*inv_r3, df_dy=-pz*py*inv_r3, df_dz=inv_r-pz*pz*inv_r3;
    double p3z=3.*pz;
    double d2f_xx=-pz*inv_r3+p3z*px*px*inv_r5, d2f_xy=p3z*px*py*inv_r5,
           d2f_xz=-px*inv_r3+p3z*px*pz*inv_r5,  d2f_yy=-pz*inv_r3+p3z*py*py*inv_r5,
           d2f_yz=-py*inv_r3+p3z*py*pz*inv_r5,   d2f_zz=-3.0*pz*inv_r3+p3z*pz*pz*inv_r5;

    double inv_sth=(sin_theta>1.e-14)?1./sin_theta:0.;
    double c_o_s3=cos_theta*inv_sth*inv_sth*inv_sth;

    double d2th_xx=-d2f_xx*inv_sth-df_dx*df_dx*c_o_s3;
    double d2th_xy=-d2f_xy*inv_sth-df_dx*df_dy*c_o_s3;
    double d2th_xz=-d2f_xz*inv_sth-df_dx*df_dz*c_o_s3;
    double d2th_yy=-d2f_yy*inv_sth-df_dy*df_dy*c_o_s3;
    double d2th_yz=-d2f_yz*inv_sth-df_dy*df_dz*c_o_s3;
    double d2th_zz=-d2f_zz*inv_sth-df_dz*df_dz*c_o_s3;

    double d2ph_xx=0.,d2ph_xy=0.,d2ph_yy=0.;
    if (Rcyl>1.e-14) {
        double inv_R4=inv_R2*inv_R2;
        d2ph_xx= 2.*px*py*inv_R4;
        d2ph_xy=(py*py-px*px)*inv_R4;
        d2ph_yy=-2.*px*py*inv_R4;
    }

    double dlr[3]={dlr_dx,dlr_dy,dlr_dz};
    double dth[3]={dth_dx,dth_dy,dth_dz};
    double dph[3]={dph_dx,dph_dy,0.};
    double d2lr[6]={d2lr_xx,d2lr_xy,d2lr_xz,d2lr_yy,d2lr_yz,d2lr_zz};
    double d2th[6]={d2th_xx,d2th_xy,d2th_xz,d2th_yy,d2th_yz,d2th_zz};
    double d2ph[6]={d2ph_xx,d2ph_xy,0.,d2ph_yy,0.,0.};

    // Agama forceDeriv layout: [Hxx, Hyy, Hzz, Hxy, Hyz, Hxz]
    // Mapping: p=0->(i=0,j=0), p=1->(i=1,j=1), p=2->(i=2,j=2)
    //          p=3->(i=0,j=1), p=4->(i=1,j=2), p=5->(i=0,j=2)
    int ii[6]={0,1,2,0,1,0};
    int jj[6]={0,1,2,1,2,2};
    // d2lr/d2th/d2ph indexed by canonical {xx,xy,xz,yy,yz,zz} order -> map ii,jj:
    // canonical slot for (i,j): 0=(0,0)=xx, 1=(0,1)=xy, 2=(0,2)=xz, 3=(1,1)=yy, 4=(1,2)=yz, 5=(2,2)=zz
    // For output p, need canonical slot of (ii[p], jj[p]):
    int can[6];
    for (int p=0;p<6;p++) {
        int i=ii[p], j=jj[p];
        // canonical index: encode upper triangle in xx,xy,xz,yy,yz,zz order
        if      (i==0&&j==0) can[p]=0;
        else if (i==0&&j==1) can[p]=1;
        else if (i==0&&j==2) can[p]=2;
        else if (i==1&&j==1) can[p]=3;
        else if (i==1&&j==2) can[p]=4;
        else                  can[p]=5;
    }

    for (int p=0;p<6;p++) {
        int i=ii[p], j=jj[p], q=can[p];
        hess_out[6*tid+p] =
            d2Phi_dlr2    *dlr[i]*dlr[j]
          + d2Phi_dlr_dth *(dlr[i]*dth[j]+dlr[j]*dth[i])
          + d2Phi_dlr_dph *(dlr[i]*dph[j]+dlr[j]*dph[i])
          + d2Phi_dth2    *dth[i]*dth[j]
          + d2Phi_dth_dph *(dth[i]*dph[j]+dth[j]*dph[i])
          + d2Phi_dph2    *dph[i]*dph[j]
          + dPhi_dlr*d2lr[q] + dPhi_dth*d2th[q] + dPhi_dph*d2ph[q];
    }
}


// ---------------------------------------------------------------------------
//  multipole_density_kernel --- rho = nabla^2 Phi / (4 pi G)
//  After unscaling via log-scaling inverse, uses standard Laplacian formula:
//    nabla^2 Phi = sum_c (d2C/dlogr^2 + dC/dlogr - l(l+1)*C) / r^2 * Y * T * mul
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(256, 2) void
multipole_density_kernel(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    const double* __restrict__ poly,
    double logr_min, double dlogr, double inv_dlogr,
    int n_intervals, int n_lm, int lmax,
    const int* __restrict__ lm_l,
    const int* __restrict__ lm_m,
    int log_scaling, double invPhi0,
    double inner_s, double inner_U, double inner_W,
    double outer_s, double outer_U, double outer_W,
    double inv_4piG,
    double* __restrict__ rho_out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double px=x[tid], py=y[tid], pz=z[tid];
    double r2=px*px+py*py+pz*pz, r=sqrt(r2), Rcyl=sqrt(px*px+py*py);

    if (r < 1.0e-300) { rho_out[tid]=0.; return; }

    double logr=log(r);

    // Inner extrapolation: density from l=0 power-law
    // nabla^2 (U*r^s + W) = U * s*(s+1) * r^(s-2) / (4*pi)
    if (logr < logr_min) {
        double dlr = logr - logr_min;
        double r_ratio_s = (inner_s != 0.0) ? exp(inner_s * dlr) : 1.0;
        // For l=0: Laplacian = (d2C/dlogr^2 + dC/dlogr) / r^2
        double dC_dlr = inner_U * inner_s * r_ratio_s;
        double d2C_dlr2 = inner_U * inner_s * inner_s * r_ratio_s;
        double laplacian = (d2C_dlr2 + dC_dlr) / (r * r);
        rho_out[tid] = MUL0 * PREFACT[0] * laplacian * inv_4piG;
        return;
    }

    int k=(int)((logr-logr_min)*inv_dlogr);
    if (k<0) k=0;
    if (k>=n_intervals) { rho_out[tid]=0.; return; }  // beyond grid: density->0 (Keplerian tail)
    double s=(logr-(logr_min+k*dlogr))*inv_dlogr;
    if (s<0.) s=0.; if (s>1.) s=1.;

    double cos_theta=pz/r, sin_theta=Rcyl/r, phi_az=atan2(py,px);

    double cph, sph;
    sincos(phi_az, &sph, &cph);
    double cm = 1.0, sm = 0.0;  // rolling: cos(absm*phi), sin(absm*phi)

    // c=0: evaluate + log-scaling inverse.  P_0^0 = PREFACT[0] (constant).
    double C0_sc, dC0_sc_ds, d2C0_sc_ds2;
    quintic_eval(poly, 0, k, n_intervals, s, &C0_sc, &dC0_sc_ds, &d2C0_sc_ds2);
    double C0_val    = C0_sc;
    double dC0_dlr   = dC0_sc_ds * inv_dlogr;
    double d2C0_dlr2 = d2C0_sc_ds2 * inv_dlogr * inv_dlogr;

    if (log_scaling) {
        double expX   = exp(C0_sc);
        double Phi0   = 1.0 / (invPhi0 - expX);
        double dPhidX = Phi0 * Phi0 * expX;
        double dC0_sc_dlr   = dC0_sc_ds  * inv_dlogr;
        double d2C0_sc_dlr2 = d2C0_sc_ds2 * inv_dlogr * inv_dlogr;
        d2C0_dlr2 = dPhidX*(d2C0_sc_dlr2 + dC0_sc_dlr*dC0_sc_dlr*Phi0*(invPhi0+expX));
        dC0_dlr   = dPhidX * dC0_sc_dlr;
        C0_val    = Phi0;
    }

    double inv_r2 = 1.0 / r2;
    double lap_sum = (d2C0_dlr2 + dC0_dlr) * inv_r2 * PREFACT[0] * MUL0;  // l(l+1)=0

    // -----------------------------------------------------------------------
    // c>0: on-the-fly Legendre (lm sorted by |m|, l).  No Plm_arr needed.
    // -----------------------------------------------------------------------
    double sin_pow = 1.0;
    int ci = 1;

    for (int absm = 0; absm <= lmax && ci < n_lm; absm++) {
        int ci_absm = (lm_m[ci] >= 0) ? lm_m[ci] : -lm_m[ci];
        if (ci_absm != absm) {
            sin_pow = (absm == 0) ? sin_theta : sin_pow * sin_theta;
            { double _t = cm*cph - sm*sph; sm = sm*cph + cm*sph; cm = _t; }
            continue;
        }
        double pf = PREFACT[absm];
        double pfact_cur = pf;
        double raw_prev;
        if      (absm == 0) raw_prev = 1.0;
        else if (absm == 1) raw_prev = -sin_theta;
        else                raw_prev = (pf != 0.0) ? COEF[absm]*sin_pow/pf : 0.0;
        double raw_cur = raw_prev * cos_theta * (double)(2*absm + 1);

#define ACCUM_DEN(ci_idx, l_val, m_val, raw_lm)                                 \
        {                                                                         \
            double _Plm = (raw_lm) * pfact_cur;                                  \
            double _Cc, _dC, _d2C;                                               \
            quintic_eval(poly,(ci_idx),k,n_intervals,s,&_Cc,&_dC,&_d2C);        \
            double _dlr = _dC*inv_dlogr, _d2lr = _d2C*inv_dlogr*inv_dlogr;      \
            double _Cv=_Cc, _dv=_dlr, _d2v=_d2lr;                               \
            if (log_scaling) {                                                    \
                _d2v = _d2lr*C0_val + 2.0*_dlr*dC0_dlr + _Cc*d2C0_dlr2;        \
                _dv  = _dlr*C0_val + _Cc*dC0_dlr;                               \
                _Cv  = _Cc*C0_val;                                                \
            }                                                                     \
            double _lap = (_d2v+_dv-(double)((l_val)*((l_val)+1))*_Cv)*inv_r2;  \
            double _mul = (absm==0) ? MUL0 : MUL1;                              \
            double _Tlm = ((m_val)>=0) ? cm : sm;                               \
            lap_sum += _mul * _lap * _Plm * _Tlm;                                \
        }

        if (lm_l[ci] == absm) {
            if (lm_m[ci] == absm)
                { ACCUM_DEN(ci, absm, absm,  raw_prev); ci++; }
            if (absm > 0 && ci < n_lm &&
                lm_l[ci] == absm && lm_m[ci] == -absm)
                { ACCUM_DEN(ci, absm, -absm, raw_prev); ci++; }
        }
        for (int l = absm + 1; l <= lmax; l++) {
            if (l > absm + 1) {
                double inv_lm = 1.0 / (double)(l - absm);
                double nr = fma((double)(2*l-1)*cos_theta, raw_cur,
                                -(double)(l+absm-1)*raw_prev) * inv_lm;
                raw_prev = raw_cur;
                raw_cur  = nr;
            }
            pfact_cur *= sqrt((2.0*l+1)/(2.0*l-1) * (double)(l-absm)/(double)(l+absm));
            if (ci >= n_lm) break;
            if (lm_l[ci] != l) {
                int na2 = (lm_m[ci] >= 0) ? lm_m[ci] : -lm_m[ci];
                if (na2 != absm) break;
                continue;
            }
            if ((lm_m[ci] >= 0 ? lm_m[ci] : -lm_m[ci]) != absm) break;
            if (lm_m[ci] == absm)
                { ACCUM_DEN(ci, l, absm,  raw_cur); ci++; }
            if (absm > 0 && ci < n_lm &&
                lm_l[ci] == l && lm_m[ci] == -absm)
                { ACCUM_DEN(ci, l, -absm, raw_cur); ci++; }
            if (ci >= n_lm ||
                ((lm_m[ci] >= 0 ? lm_m[ci] : -lm_m[ci]) != absm)) break;
        }
#undef ACCUM_DEN
        sin_pow = (absm == 0) ? sin_theta : sin_pow * sin_theta;
        { double _t = cm*cph - sm*sph; sm = sm*cph + cm*sph; cm = _t; }
    }

    rho_out[tid] = inv_4piG * lap_sum;
}
