/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/*
 * bli_gemv_t_impl.h — shared T-kernel implementation for zen (AVX2) and zen4 (AVX-512).
 *
 * Prerequisites (must be defined before including this header):
 *   ARCH_SIMD_BITS     — 256 (AVX2) or 512 (AVX-512); selects ISA in bli_gemv_common_int.h
 *   GEMV_ARCH_SUFFIX   — token appended after 'zen' or 'zen4' in generated function names
 *                        (e.g. zen_int or zen4_int, WITHOUT the leading bli_ or trailing _)
 *
 * Generated symbol naming convention with GEMV_ARCH_SUFFIX=<arch>:
 *   micro-kernel:  bli_<ch>gemv_t_block_<MR>_<NR>  (via GEMV_BLK_SUFFIX_T hook)
 *   ST caller:     bli_<ch>gemv_t_<arch>_<MR>x<NR>
 *   MT wrapper:    bli_<ch>gemv_t_<arch>_<MR>x<NR>_mt
 *   entry-point:   bli_<ch>gemv_t_<arch>
 *
 * For the real (s/d) T-kernel body (GENTFUNC_TGEMV_s):
 *   ARCH_SIMD_BITS=512 → inline AVX-512 assembly (bli_x86_asm_macros.h required)
 *   ARCH_SIMD_BITS=256 → pure C intrinsics
 */

#ifndef BLI_GEMV_T_IMPL_H
#define BLI_GEMV_T_IMPL_H

// =============================================================================
// #region T-kernel name hooks
// =============================================================================

/*
 * GEMV_BLK_SUFFIX_T: hook for T micro-kernel naming.
 * Default (AVX-512): bli_<ch>gemv_t_block_<MR>_<NR>
 * AVX2 zen shims define their own suffix before including this header.
 */
#ifndef GEMV_BLK_SUFFIX_T
  #define GEMV_BLK_SUFFIX_T(ch, MR, NR)  PASTEMAC4(ch, gemv_t_block_, MR, _, NR)
#endif

/* GENTFUNC_GEMVTS: expands to the T-direction micro-kernel function name. */
#define GENTFUNC_GEMVTS(ctype, ch, MR, NR)  GEMV_BLK_SUFFIX_T(ch, MR, NR)

// #endregion

// =============================================================================
// #region T-kernel instantiation and dispatch helpers
// =============================================================================

/* DO_GENTFUNC_TGEMV: instantiates one T-direction micro-kernel. */
#define DO_GENTFUNC_TGEMV(ctype, ch, M, i) PASTECH(GENTFUNC_TGEMV_, ch)(ctype, ch, PASTECH(CH_S_, ch), M, i);

/* DO_GENTFUNC_GEMVTS_CASE: one switch arm calling the (MR, i)-sized T-kernel.
 * References caller-scope locals: transa, conjx, m, n_rem, alpha, a_fringe,
 * rs_a, cs_a, x, incx, beta, y_fringe, incy, cntx. */
#define DO_GENTFUNC_GEMVTS_CASE(ctype, ch, MR, i)                                                 \
    case i:                                                                                       \
        GENTFUNC_GEMVTS(ctype, ch, MR, i)                                                         \
            (transa, conjx, m, n_rem, alpha, a_fringe, rs_a, cs_a, x, incx,                       \
             beta, y_fringe, incy, cntx);                                                         \
        break;

/*
 * GEN_DISPATCH_SWITCH_T: expands into switch arms case (NR-1)..case 1 for fringe dispatch.
 *
 * Unlike the N-direction which needs a 2D [m_tile][n_fringe] table, the T-direction only
 * needs n-fringe dispatch because the m-fringe is handled inside the kernel itself
 * (via mloop_full / mloop_epr / MR_left in GENTFUNC_TGEMV_s/z). So a simple switch
 * over n_rem suffices — no dispatch table is needed.
 */
#define GEN_DISPATCH_SWITCH_T_APPLY(N, ctype, ch, MR)                                             \
    RANGE_DOWN_##N(DO_GENTFUNC_GEMVTS_CASE, ctype, ch, MR)
#define GEN_DISPATCH_SWITCH_T_INDIRECT(N, ctype, ch, MR)                                          \
    GEN_DISPATCH_SWITCH_T_APPLY(N, ctype, ch, MR)
#define GEN_DISPATCH_SWITCH_T(ctype, ch, MR, NR)                                                  \
    GEN_DISPATCH_SWITCH_T_INDIRECT(DEC_##NR, ctype, ch, MR)

// #endregion

// =============================================================================
// #region GENERATE_<ch>_KERNELS_<MR>_T — T-direction kernel expansion macros
// =============================================================================

/*
 * GENERATE_T_KERNELS_FAMILY: common body — instantiates per-(MR, i) T-kernels
 * for i in [1..NR] via RANGE_DOWN(DO_GENTFUNC_TGEMV, ...).
 * Fringe dispatch is by (n%NR) switch in GENT_GEMV_CALLER; no MR-fringe variants
 * are emitted at this level.
 *
 * Named wrappers below exist so GENERATE_KERNEL can select the right one via
 * PASTECH4(GENERATE_, ch, _KERNELS_, MR, _T) at the instantiation site.
 *
 * Active configurations (used by GENERATE_KERNEL in .c files):
 *   float  / s : MR=24  (zen/AVX2),  MR=48  (zen4/AVX-512)
 *   double / d : MR=16  (zen/AVX2),  MR=32  (zen4/AVX-512)
 *   scomplex/c : MR=20  (zen/AVX2),  MR=40  (zen4/AVX-512)
 *   dcomplex/z : MR=10  (zen/AVX2),  MR=20  (zen4/AVX-512)
 */
#define GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)                                   \
    RANGE_DOWN(DO_GENTFUNC_TGEMV, ctype, ch, MR, NR)

/* float / s */
#define GENERATE_s_KERNELS_96_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_s_KERNELS_80_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_s_KERNELS_64_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_s_KERNELS_48_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_s_KERNELS_32_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_s_KERNELS_24_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_s_KERNELS_16_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)

/* double / d */
#define GENERATE_d_KERNELS_64_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_d_KERNELS_48_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_d_KERNELS_40_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_d_KERNELS_32_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_d_KERNELS_24_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_d_KERNELS_16_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_d_KERNELS_8_T(ctype, ch, MR, NR)   GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)

/* scomplex / c */
#define GENERATE_c_KERNELS_48_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_40_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_32_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_24_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_20_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_16_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_8_T(ctype, ch, MR, NR)   GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)

/* dcomplex / z */
#define GENERATE_z_KERNELS_32_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_z_KERNELS_20_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_z_KERNELS_16_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_z_KERNELS_12_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_z_KERNELS_10_T(ctype, ch, MR, NR)  GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_z_KERNELS_8_T(ctype, ch, MR, NR)   GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)
#define GENERATE_z_KERNELS_4_T(ctype, ch, MR, NR)   GENERATE_T_KERNELS_FAMILY(ctype, ch, MR, NR)

// #endregion

// ─────────────────────────────────────────────────────────────────────────────
// #region ST T kernel compile-time loop helpers (shared between ISAs)
// ─────────────────────────────────────────────────────────────────────────────

/*
 * T-direction GEMV (transposed) algorithm overview
 * =================================================
 *
 * Computes:  y = beta * y + alpha * A^T * x   (or A^H * x for conjugate-transpose)
 *   where A is (m x n), x is (m,), y is (n,), A stored column-major.
 *
 * Equivalently, each output y[j] = beta*y[j] + alpha * dot( A[:,j], x )
 * — the j-th output element is the dot product of the j-th COLUMN of A with x.
 *
 * Tile structure (T-direction = outer loop over n/NR output elements,
 *                               inner loop over m/MR input rows):
 *
 *      A  (m x n)             x  (m,)
 *   ┌───────────────────┐    ┌────┐
 *   │  NR cols = 1 tile │    │    │   ← load EPR elements of x at a time
 *   │                   │  * │    │
 *   │  ┌─────────────┐  │    │ MR │   (MR = number of x-elements processed together)
 *   │  │  MR rows    │  │    │ MR │
 *   │  ├─────────────┤  │    │ .  │
 *   │  │  MR rows    │  │    │    │
 *   │  ├─────────────┤  │    │frng│
 *   │  │ fringe rows │  │    └────┘
 *   │  └─────────────┘  │
 *   └───────────────────┘
 *         ↓
 *   yv[0..NR-1]  (NR SIMD accumulators — each accumulates a dot product)
 *         ↓ horizontal reduce
 *   y[j..j+NR-1]  (NR scalar output elements)
 *
 * For each NR-column tile (output y[i*NR .. i*NR+NR-1]):
 *   1. yv[0..NR-1] = 0                     (zero NR accumulators)
 *   2. For each MR-row block of x (j = 0..mloop_full-1):
 *        xv[0] = load EPR elements of x
 *        for each of the NR columns:
 *          av[col] = load EPR elements of A[:, col]
 *          yv[col] += av[col] * xv[0]        (FMADD)
 *      (fringe: masked load for the last m%MR elements)
 *   3. Horizontal reduce yv[0..NR-1] → scalar sums
 *   4. Apply alpha and beta, write y[i*NR .. i*NR+NR-1]
 *
 * Key difference from N-direction:
 *   - Each yv[j] accumulates a full dot product (not a row of a matrix).
 *   - At the end we must reduce (horizontal sum) each yv[j] to one scalar.
 *   - For real types, SIMD_REDUCE_ADD_P_s/d does the horizontal sum.
 *   - For complex types, alternating ± sums are needed (see HSUM_z/c in pp_common.h).
 *
 * For complex (z/c) types:
 *   - xv is split into two: xv[0] = real SIMD vector, xv[1] = permuted (imag swapped)
 *   - yv[j] accumulates A*re(x), yv[j+NR] accumulates A*im(x)
 *   - PERMUTE swaps (re,im) pairs; FMADD on both registers gives complex dot product
 *   - Horizontal sum (HSUM_z/c): alternating ±  sum of yv[j] gives temp.real,
 *                                plain sum of yv[j+NR] gives temp.imag
 */

/* Cast helpers: silence "incompatible pointer type" when complex buffers
 * are passed to SIMD intrinsics expecting float* or double*. */
#define GEMV_CAST_s(ptr) ((float*)(ptr))
#define GEMV_CAST_d(ptr) ((double*)(ptr))

/*
 * Z_GEMV_T_Y_ZERO — zero one complex accumulator pair for output slot ii (1-based).
 *
 *   yv[ii-1]      = 0   (real accumulator for column ii)
 *   yv[ii-1 + NR] = 0   (imag accumulator for column ii)
 *
 * Called by PP_LOOP(NR, Z_GEMV_T_Y_ZERO, ...) at the start of each NR-column tile.
 */
#define Z_GEMV_T_Y_ZERO(ch_s, NR, _unused, ii)                                                                                           \
    yv[((ii) - 1)] = PASTECH(SIMD_SETZERO_P_, ch_s)();                                                                                   \
    yv[((ii) - 1) + NR] = PASTECH(SIMD_SETZERO_P_, ch_s)();

/*
 * Z_GEMV_T_A_LOOP — inner FMA for one column ii (1-based) of the NR-tile, complex.
 *
 * For each of the NR output columns (fully unrolled by PP_ILOOP):
 *   av[ii-1]       = A[row_block_jj, column_ii]    (load EPR complex elements = 2*EPR floats)
 *   yv[ii-1]      += av[ii-1] * xv[0]              (real part accumulation)
 *   yv[ii-1+NR]   += av[ii-1] * xv[1]              (imag part accumulation, using permuted x)
 *
 * xv[0] = x[jj*EPR .. jj*EPR+EPR-1] (consecutive complex x elements, as packed scalars)
 * xv[1] = PERMUTE(xv[0]) — (re,im) pairs swapped within each element
 *         so that FMADD on xv[1] computes the cross-term A*im(x).
 *
 * Called by PP_ILOOP(NR, Z_GEMV_T_A_LOOP, ...) inside Z_GEMV_T_X_LOOP.
 * Uses caller-scope: abuf, xv[2], yv[NR*2], av[NR], EPR, rs_a, cs_a.
 */
#define Z_GEMV_T_A_LOOP(ch_s, jj, NR, ii)                                                                                                \
    av[((ii) - 1)] = PASTECH(SIMD_LOADU_P_, ch_s)(PASTECH(GEMV_CAST_, ch_s)(abuf + (((ii) - 1) * rs_a) + (((jj) - 1) * (EPR) * cs_a))); \
    yv[((ii) - 1)] = PASTECH(SIMD_FMADD_P_, ch_s)(av[((ii) - 1)], xv[0], yv[((ii) - 1)]);                                               \
    yv[((ii) - 1) + NR] = PASTECH(SIMD_FMADD_P_, ch_s)(av[((ii) - 1)], xv[0 + 1], yv[((ii) - 1) + NR]);

/*
 * Z_GEMV_T_X_LOOP — process one EPR-wide block of x rows (jj-th block), complex.
 *
 * Loads EPR complex x-elements and computes their contribution to all NR accumulators:
 *   xv[0] = x[jj*EPR .. jj*EPR+EPR-1]    (EPR complex elements packed as scalar pairs)
 *   xv[1] = PERMUTE(xv[0])               (re,im swapped for imaginary cross-term)
 *   PP_ILOOP(NR, Z_GEMV_T_A_LOOP, ...) — for each of the NR output columns:
 *     load A[row_block_jj, col_ii] and accumulate into yv[ii-1] and yv[ii-1+NR]
 *
 * Called by PP_LOOP(MR/EPR, Z_GEMV_T_X_LOOP, ...) inside the full-tile inner loop.
 * Uses caller-scope: abuf, xbuf, xv[2], yv[NR*2], av[NR], EPR, cs_a, rs_a.
 */
#define Z_GEMV_T_X_LOOP(ch_s, ch, NR, jj)                                                                                                \
    xv[0] = PASTECH(SIMD_LOADU_P_, ch_s)(PASTECH(GEMV_CAST_, ch_s)(xbuf + ((jj) - 1) * (EPR)));                                         \
    xv[0 + 1] = PASTECH(SIMD_PERMUTE_P_, ch_s)(xv[0], PASTECH(PERM_MASK_, ch));                                                          \
    PP_ILOOP(NR, Z_GEMV_T_A_LOOP, ch_s, jj, NR)

/*
 * Z_GEMV_T_A_FRINGE_LOOP — masked FMA for the last partial x-block, complex.
 *
 * Same as Z_GEMV_T_A_LOOP but uses a masked load for MR_left complex elements:
 *   mask = (1 << (MR_left * 2)) - 1    — lowest 2*MR_left scalar lanes
 *   av[ii-1] = A[xfull*EPR .. xfull*EPR+MR_left-1, col_ii]  (masked)
 *   yv[ii-1]     += av[ii-1] * xv[0]
 *   yv[ii-1+NR]  += av[ii-1] * xv[1]
 *
 * "xfull" = MR/EPR = number of full row-blocks already processed.
 * Called by PP_ILOOP(NR, Z_GEMV_T_A_FRINGE_LOOP, ...) inside Z_GEMV_T_FRINGE_1.
 */
#define Z_GEMV_T_A_FRINGE_LOOP(ch_s, xfull, NR, ii)                                                                                      \
    av[((ii) - 1)] = SIMD_MASKZ_LOADU_P_##ch_s((1 << (MR_left * 2)) - 1, PASTECH(GEMV_CAST_, ch_s)(abuf + (((ii) - 1) * rs_a) + (xfull) * (EPR) * cs_a)); \
    yv[((ii) - 1)] = PASTECH(SIMD_FMADD_P_, ch_s)(av[((ii) - 1)], xv[0], yv[((ii) - 1)]);                                               \
    yv[((ii) - 1) + NR] = PASTECH(SIMD_FMADD_P_, ch_s)(av[((ii) - 1)], xv[0 + 1], yv[((ii) - 1) + NR]);

/*
 * Z_GEMV_T_FRINGE_0/1 — fringe x-row handler for complex types.
 *
 * _FRINGE_0: no-op — MR divides m exactly.
 * _FRINGE_1: handles the last MR_left complex elements of x (< EPR).
 *   Masked-loads x and A, accumulates into yv[0..NR-1] and yv[NR..2*NR-1].
 *   PP_IF(IS_FRINGE_OF(MR,ch), Z_GEMV_T_FRINGE_, ...) selects at compile time.
 */
#define Z_GEMV_T_FRINGE_0(ch_s, ch, NR, xfull) /* empty */
#define Z_GEMV_T_FRINGE_1(ch_s, ch, NR, xfull)                                                                                           \
    xv[0] = SIMD_MASKZ_LOADU_P_##ch_s((1 << (MR_left * 2)) - 1, PASTECH(GEMV_CAST_, ch_s)(xbuf + (xfull) * (EPR)));                     \
    xv[0 + 1] = PASTECH(SIMD_PERMUTE_P_, ch_s)(xv[0], PASTECH(PERM_MASK_, ch));                                                          \
    PP_ILOOP(NR, Z_GEMV_T_A_FRINGE_LOOP, ch_s, xfull, NR)

/*
 * Z_GEMV_T_UPDATE — write one complex dot-product result to y[ii-1], complex types.
 *
 * After the inner loop, yv[ii-1] and yv[ii-1+NR] hold the accumulated real and
 * imaginary partial sums across all m rows.  This macro:
 *
 *   1. Reduces the SIMD accumulators to a scalar complex dot-product (temp):
 *        HSUM_z/c(NR): alternating ± lane-sum on yv[ii] gives temp.real
 *                      plain lane-sum on yv[ii+NR] gives temp.imag
 *        For conjugate-transpose (A^H): sign patterns are reversed (HSUM_z/c_CONJ).
 *
 *   2. Applies alpha (complex multiply) and beta (complex multiply-add):
 *        if beta == 0:
 *          y[ii] = alpha * temp
 *        else:
 *          y[ii] = beta * y[ii] + alpha * temp
 *
 *   3. Advances ybuf by incy to the next output element.
 *
 * Called by PP_LOOP(NR, Z_GEMV_T_UPDATE, ...) after the inner loops.
 * Uses caller-scope: yv[NR*2], ybuf, incy, alpha, beta, temp.
 */
#define Z_GEMV_T_UPDATE(ch, NR, _unused, ii_loop)                                                \
    {                                                                                            \
        dim_t ii = ((ii_loop) - 1);                                                              \
        if ( transa == BLIS_CONJ_TRANSPOSE )                                                     \
        {                                                                                        \
            PASTECH2(HSUM_, ch, _CONJ)(NR)                                                       \
        }                                                                                        \
        else                                                                                     \
        {                                                                                        \
            PASTECH(HSUM_, ch)(NR);                                                              \
        }                                                                                        \
        if ( PASTEMAC( ch, eq0 )( *beta ) )                                                      \
        {                                                                                        \
            ybuf->real = (temp.real * alpha->real) - (temp.imag * alpha->imag);                  \
            ybuf->imag = (temp.real * alpha->imag) + (temp.imag * alpha->real);                  \
        }                                                                                        \
        else                                                                                     \
        {                                                                                        \
            y_local = *ybuf;                                                                     \
            ybuf->real =                                                                         \
                (ybuf->real * beta->real) - (ybuf->imag * beta->imag) +                          \
                (temp.real * alpha->real) - (temp.imag * alpha->imag);                           \
            ybuf->imag =                                                                         \
                (y_local.real * beta->imag) + (ybuf->imag * beta->real) +                        \
                (temp.real * alpha->imag) + (temp.imag * alpha->real);                           \
        }                                                                                        \
        ybuf += incy;                                                                            \
    }

/*
 * S_GEMV_T_UPDATE_REDUCE — reduce one SIMD accumulator to a scalar, real types.
 *
 *   yreduce[ii-1] = horizontal_sum( yv[kk_off + ii - 1] )
 *
 * kk_off is the base index within yv[] for this EPR-wide group.
 * Called by PP_ILOOP(EPR, S_GEMV_T_UPDATE_REDUCE, ...) inside S_GEMV_T_UPDATE_VEC.
 */
#define S_GEMV_T_UPDATE_REDUCE(ch, kk_off, _unused, ii)                                          \
    yreduce[((ii) - 1)] = PASTECH(SIMD_REDUCE_ADD_P_, ch)(yv[(kk_off) + ((ii) - 1)]);

/*
 * S_GEMV_T_UPDATE_VEC — write EPR scalar dot-product results to y at once, real types.
 *
 * This "vectorized output" path is used when incy==1 (contiguous y):
 *   1. Reduce EPR adjacent yv accumulators to EPR scalars: yreduce[0..EPR-1].
 *      PP_ILOOP(EPR, S_GEMV_T_UPDATE_REDUCE, ...) does this in one unrolled sweep.
 *   2. Load those EPR scalars as a SIMD vector yred_v.
 *   3. Multiply by alpha: yred_v *= alpha.
 *   4. If beta != 0: ycur = beta * y[ybuf..] + alpha * yred_v  (FMA).
 *      If beta == 0: ycur = alpha * yred_v.
 *   5. Store EPR results back to ybuf; advance ybuf by EPR*incy.
 *
 * "kk" is the 1-based group index (1..NR/EPR); processes yv[(kk-1)*EPR .. kk*EPR-1].
 * Used when NR is a multiple of EPR (S_GEMV_T_UPDATE_DISPATCH_0).
 * Uses caller-scope: yv[NR], ybuf, alpha, beta, incy.
 */
#define S_GEMV_T_UPDATE_VEC(ch, ctype, _unused2, kk)                                             \
    {                                                                                            \
        ctype yreduce[EPR_OF(ch)];                                                               \
        PP_ILOOP(EPR_OF(ch), S_GEMV_T_UPDATE_REDUCE, ch, ((kk) - 1) * EPR_OF(ch), 0)             \
        PASTECH(SIMD_VEC_, ch) yred_v  = PASTECH(SIMD_LOADU_P_,  ch)(yreduce);                   \
        PASTECH(SIMD_VEC_, ch) alpha_v = PASTECH(SIMD_SET1_P_,   ch)(*alpha);                    \
        PASTECH(SIMD_VEC_, ch) ycur_v;                                                           \
        if ( PASTEMAC( ch, eq0 )( *beta ) )                                                      \
        {                                                                                        \
            ycur_v = PASTECH(SIMD_MUL_P_, ch)(yred_v, alpha_v);                                  \
        }                                                                                        \
        else                                                                                     \
        {                                                                                        \
            PASTECH(SIMD_VEC_, ch) beta_v = PASTECH(SIMD_SET1_P_, ch)(*beta);                    \
            ycur_v = PASTECH(SIMD_LOADU_P_, ch)(ybuf);                                           \
            ycur_v = PASTECH(SIMD_MUL_P_,   ch)(ycur_v, beta_v);                                 \
            ycur_v = PASTECH(SIMD_FMADD_P_, ch)(yred_v, alpha_v, ycur_v);                        \
        }                                                                                        \
        PASTECH(SIMD_STOREU_P_, ch)(ybuf, ycur_v);                                               \
        ybuf += EPR_OF(ch) * incy;                                                               \
    }

/*
 * S_GEMV_T_UPDATE — write one scalar dot-product result to y[ii-1], real types.
 *
 * Scalar fallback (used when incy != 1 or NR is not a multiple of EPR):
 *   yred_s = horizontal_sum(yv[ii-1]) * alpha
 *   y[ii-1] = beta == 0 ? yred_s : y[ii-1] * beta + yred_s
 *   ybuf += incy
 *
 * Uses caller-scope: yv[NR], ybuf, alpha, beta, incy.
 */
#define S_GEMV_T_UPDATE(ch, NR, ctype, ii)                                                       \
    {                                                                                            \
        ctype yred_s = PASTECH(SIMD_REDUCE_ADD_P_, ch)(yv[((ii) - 1)]) * (*alpha);               \
        (*ybuf) = PASTEMAC( ch, eq0 )( *beta ) ? yred_s : ((*ybuf) * (*beta)) + yred_s;          \
        ybuf += incy;                                                                            \
    }

/*
 * S_GEMV_T_UPDATE_DISPATCH_0/1 — choose vectorized vs scalar output path.
 *
 * _DISPATCH_0: NR IS a multiple of EPR → use vectorized path when incy==1,
 *              scalar path otherwise.
 *   - incy==1: PP_LOOP(NR/EPR, S_GEMV_T_UPDATE_VEC, ...) writes EPR outputs at once.
 *   - incy!=1: PP_LOOP(NR, S_GEMV_T_UPDATE, ...) writes one scalar at a time.
 *
 * _DISPATCH_1: NR is NOT a multiple of EPR → always use scalar path.
 *
 * PP_IF(IS_FRINGE_OF(NR,ch), S_GEMV_T_UPDATE_DISPATCH_, ...) selects at compile time.
 */
#define S_GEMV_T_UPDATE_DISPATCH_0(ch, ctype, NR, _unused3)                                      \
    if (incy == 1) {                                                                             \
        PP_LOOP(PP_DIV(NR, EPR_OF(ch)), S_GEMV_T_UPDATE_VEC, ch, ctype, 0)                       \
    } else {                                                                                     \
        PP_LOOP(NR, S_GEMV_T_UPDATE, ch, NR, ctype)                                              \
    }

/* NR is NOT a multiple of EPR → fall back to scalar path for all NR outputs. */
#define S_GEMV_T_UPDATE_DISPATCH_1(ch, ctype, NR, _unused3)                                      \
    PP_LOOP(NR, S_GEMV_T_UPDATE, ch, NR, ctype)

// #endregion

// ─────────────────────────────────────────────────────────────────────────────
// #region ISA-specific real T-kernel inner-loop helpers
// ─────────────────────────────────────────────────────────────────────────────

#if ARCH_SIMD_BITS == 512

/* AVX-512 (zen4): inline assembly using BLIS ASM macros.
 * bli_x86_asm_macros.h must be included before this header.
 *
 * The AVX-512 path uses inline assembly instead of C intrinsics for the real T-kernel.
 * This is done because AVX-512 has 32 ZMM registers — enough to hold NR accumulator
 * registers (ZMM(1)..ZMM(NR)) plus one broadcast register (ZMM(0)) simultaneously.
 * Keeping all of them in registers across the inner loop avoids spill/reload and lets
 * the CPU execute the VFMADD231 instructions back-to-back without memory latency.
 *
 * Register assignment in the assembly block:
 *   ZMM(0)           = x[j] broadcast (loaded fresh each inner iteration)
 *   ZMM(1)..ZMM(NR)  = yv[0..NR-1] accumulators (zeroed before loop, spilled after)
 *   RAX              = current A row pointer (a + ii*rs_a, incremented by rs_a each row)
 *   RBX              = current x pointer (incremented by epr_bytes each iteration)
 *   R9               = rs_a in bytes
 *   R10              = epr_bytes  (EPR * sizeof(elem))
 *   R11              = cs_a * EPR in bytes  (advance A by one EPR-wide column slice)
 *   R12              = base of current A column group (incremented by cs_a*EPR outer)
 *   K1               = fringe mask (set once outside the loop)
 *   RSI              = yv[] base address (for spilling ZMM accumulators after ASM block)
 */

/*
 * S_GEMV_T_Y_ZERO (AVX-512) — zero one yv accumulator register.
 *   VXORPS/PD ZMM(ii), ZMM(ii), ZMM(ii)  →  ZMM(ii) = 0
 * Called by PP_LOOP(NR, S_GEMV_T_Y_ZERO, ...) before the inner ASM loop.
 */
#define S_GEMV_T_Y_ZERO(ch, _u1, _u2, ii)                                                        \
    PASTECH(ASM_VXOR_, ch)(ZMM(ii), ZMM(ii), ZMM(ii))

/*
 * S_GEMV_T_SPILL_YV (AVX-512) — store one ZMM accumulator to the yv[] C array.
 *   VMOVUPS/PD [RSI + (ii-1)*64], ZMM(ii)   →  yv[ii-1] = ZMM(ii)
 *
 * After the ASM block, yv[] is read by S_GEMV_T_UPDATE_DISPATCH to produce scalar outputs.
 * RSI is bound to yv (the C array pointer) via the "S" register constraint.
 */
#define S_GEMV_T_SPILL_YV(ch, _u1, _u2, ii)                                                      \
    PASTECH(ASM_VMOVUP_, ch)(MEM(RSI, ((ii) - 1) * 64), ZMM(ii))

/* S_GEMV_T_YV_CLOBBERS — ZMM register clobber list for the inline ASM block.
 * ZMM(0) is the x-broadcast scratch; ZMM(1)..ZMM(31) are the NR accumulators + spares.
 * All must be declared so the compiler knows the ASM modifies them. */
#define S_GEMV_T_YV_CLOBBERS \
    "zmm1",  "zmm2",  "zmm3",  "zmm4",  "zmm5",  "zmm6",  "zmm7",  "zmm8",                      \
    "zmm9",  "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16",                     \
    "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24",                     \
    "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",

/*
 * S_GEMV_T_A_LOOP (AVX-512) — one FMA step: accumulate one row of A into yv[ii-1].
 *
 *   VFMADD231PS/PD ZMM(ii), ZMM(0), [RAX]
 *     ZMM(ii) += ZMM(0) * A[row_ii, current_x_block]
 *   ADD RAX, R9     — advance RAX by rs_a bytes to next row
 *
 * ZMM(0) = broadcast(x[j]), set once per S_GEMV_T_X_LOOP iteration.
 * PP_ILOOP(NR, S_GEMV_T_A_LOOP, ...) unrolls this for all NR accumulators.
 */
#define S_GEMV_T_A_LOOP(ch, _u1, _u2, ii)                                                        \
    PASTECH(ASM_VFMADD231P_, ch)(ZMM(ii), ZMM(0), MEM(RAX))                                      \
    ADD(RAX, R9)

/*
 * S_GEMV_T_X_LOOP (AVX-512) — process one EPR-wide block of x (jj-th block), real types.
 *
 *   MOV RAX, R12             — reset A pointer to start of this x-block's row group
 *   VMOVUPS ZMM(0), [RBX]   — load x[j..j+EPR-1] into ZMM(0) (broadcast register)
 *   ADD RBX, R10             — advance x pointer by epr_bytes
 *   PP_ILOOP(NR, S_GEMV_T_A_LOOP, ...) — for each of NR columns:
 *     ZMM(ii) += ZMM(0) * A[row_ii, col]
 *     RAX += R9  (move to next row of A)
 *   ADD R12, R11             — advance A column base by cs_a*EPR bytes
 *   MOV RAX, R12             — restore RAX for next iteration
 *
 * Called by PP_LOOP(MR/EPR, S_GEMV_T_X_LOOP, ...) inside the ASM full-tile loop.
 */
#define S_GEMV_T_X_LOOP(ch, _u1, NR, jj)                                                          \
    MOV(RAX, R12)                                                                                 \
    PASTECH(ASM_VMOVUP_, ch)(ZMM(0), MEM(RBX))                                                    \
    ADD(RBX, R10)                                                                                 \
    PP_ILOOP(NR, S_GEMV_T_A_LOOP, ch, 0, 0)                                                       \
    ADD(R12, R11)                                                                                 \
    MOV(RAX, R12)

/*
 * S_GEMV_T_A_FRINGE_LOOP (AVX-512) — masked FMA for the partial row register.
 *
 *   VFMADD231PS/PD ZMM(ii){k1}, ZMM(0), [RAX]   — masked FMA, only k1 lanes
 *   ADD RAX, R9   — advance to next row
 *
 * K1 is set to (1 << MR_left) - 1 before the fringe block via KMOVW.
 * Called by PP_ILOOP(NR, S_GEMV_T_A_FRINGE_LOOP, ...) inside S_GEMV_T_FRINGE_1.
 */
#define S_GEMV_T_A_FRINGE_LOOP(ch, _u1, _u2, ii)                                                 \
    PASTECH(ASM_VFMADD231P_, ch)(ZMM(ii) MASK_K(1), ZMM(0), MEM(RAX))                            \
    ADD(RAX, R9)

/*
 * S_GEMV_T_FRINGE_0/1 (AVX-512) — fringe x-row handler for real types.
 *
 * _FRINGE_0: no-op — MR divides m exactly.
 * _FRINGE_1: handles the last MR_left elements of x.
 *   MOV RAX, R12             — reset A pointer
 *   VMOVUPS ZMM(0){k1z}, [RBX]  — masked load of x (k1 has MR_left bits set)
 *   PP_ILOOP(NR, S_GEMV_T_A_FRINGE_LOOP, ...) — masked FMA for all NR columns
 */
#define S_GEMV_T_FRINGE_0(ch, _u1, NR, xfull) /* empty */
#define S_GEMV_T_FRINGE_1(ch, _u1, NR, xfull)                                                    \
    MOV(RAX, R12)                                                                                \
    PASTECH(ASM_VMOVUP_, ch)(ZMM(0) MASK_KZ(1), MEM(RBX))                                        \
    PP_ILOOP(NR, S_GEMV_T_A_FRINGE_LOOP, ch, 0, 0)

/* GENTFUNC_TGEMV_s: real T-kernel, AVX-512 inline assembly path. */
#define GENTFUNC_TGEMV_s(ctype, ch, ch_s, MR, NR)                                                   \
void GENTFUNC_GEMVTS(ctype, ch, MR, NR)                                                             \
     (                                                                                              \
       trans_t transa, conj_t conjx, dim_t m, dim_t n, ctype * alpha,                               \
       ctype * a, inc_t cs_a, inc_t rs_a, ctype * x, inc_t incx, ctype * beta,                      \
       ctype * y, inc_t incy, cntx_t * cntx                                                         \
     )                                                                                              \
{                                                                                                   \
    const dim_t ELEM_SIZE = sizeof(ctype);                                                          \
    const dim_t EPR = PASTECH(ELEM_PER_REG_, ch_s);                                                 \
    ctype *restrict abuf = a;                                                                       \
    ctype *restrict xbuf = x;                                                                       \
    ctype *restrict ybuf = y;                                                                       \
                                                                                                    \
    /* The asm block reads these with 64-bit MOV/TEST, so they must be 64-bit    \
       even when dim_t is 32-bit (BLIS_INT_TYPE_SIZE=32). */                     \
    const int64_t mloop_full = m / MR;                                                              \
    const int64_t mloop_epr  = (m % MR) / EPR;                                                      \
    const int64_t MR_left    = m % EPR;                                                             \
                                                                                                    \
    const int64_t rs_a_bytes     = (int64_t)rs_a * ELEM_SIZE;                                       \
    const int64_t cs_a_epr_bytes = (int64_t)cs_a * EPR * ELEM_SIZE;                                 \
    const int64_t epr_bytes      = (int64_t)EPR  * ELEM_SIZE;                                       \
                                                                                                    \
    PASTECH(SIMD_VEC_, ch) yv[NR];                                                                  \
                                                                                                    \
    for (dim_t i = 0; i < (n / NR); ++i)                                                            \
    {                                                                                               \
        ybuf = y + (NR * i * incy);                                                                 \
        xbuf = x;                                                                                   \
        abuf = a + (NR * i * rs_a);                                                                 \
                                                                                                    \
        BEGIN_ASM()                                                                                 \
        MOV(RAX, VAR(abuf))                                                                         \
        MOV(RBX, VAR(xbuf))                                                                         \
        MOV(R9,  VAR(rs_a_bytes))                                                                   \
        MOV(R10, VAR(epr_bytes))                                                                    \
        MOV(R11, VAR(cs_a_epr_bytes))                                                               \
        MOV(R12, RAX)                                                                               \
                                                                                                    \
        PP_LOOP(NR, S_GEMV_T_Y_ZERO, ch, 0, 0)                                                      \
                                                                                                    \
        MOV(RCX, VAR(mloop_full))                                                                   \
        TEST(RCX, RCX)                                                                              \
        JZ(EPR_LOOP_ENTRY)                                                                          \
        LABEL(MLOOP_START)                                                                          \
            PP_LOOP(PP_DIV(MR, EPR_OF(ch)), S_GEMV_T_X_LOOP, ch, 0, NR)                             \
            DEC(RCX)                                                                                \
            JNZ(MLOOP_START)                                                                        \
                                                                                                    \
        LABEL(EPR_LOOP_ENTRY)                                                                       \
        MOV(RCX, VAR(mloop_epr))                                                                    \
        TEST(RCX, RCX)                                                                              \
        JZ(TAIL_ENTRY)                                                                              \
        LABEL(EPR_LOOP_START)                                                                       \
            S_GEMV_T_X_LOOP(ch, 0, NR, 1)                                                           \
            DEC(RCX)                                                                                \
            JNZ(EPR_LOOP_START)                                                                     \
                                                                                                    \
        LABEL(TAIL_ENTRY)                                                                           \
        MOV(RCX, VAR(MR_left))                                                                      \
        TEST(RCX, RCX)                                                                              \
        JZ(MEND)                                                                                    \
        MOV(RDX, IMM(1))                                                                            \
        SAL(RDX, CL)                                                                                \
        DEC(RDX)                                                                                    \
        KMOVW(K(1), EDX)                                                                            \
        S_GEMV_T_FRINGE_1(ch, 0, NR, 0)                                                             \
        LABEL(MEND)                                                                                 \
                                                                                                    \
        PP_LOOP(NR, S_GEMV_T_SPILL_YV, ch, 0, 0)                                                    \
                                                                                                    \
        END_ASM(                                                                                    \
            : [abuf] "+r"(abuf), [xbuf] "+r"(xbuf)                                                  \
            : [rs_a_bytes]      "m"(rs_a_bytes),                                                    \
              [cs_a_epr_bytes]  "m"(cs_a_epr_bytes),                                                \
              [epr_bytes]       "m"(epr_bytes),                                                     \
              [mloop_full]      "m"(mloop_full),                                                    \
              [mloop_epr]       "m"(mloop_epr),                                                     \
              [MR_left]         "m"(MR_left),                                                       \
              [yv_ptr]          "S"(yv)                                                             \
            : "rax", "rbx", "rcx", "rdx",                                                           \
              "r9", "r10", "r11", "r12",                                                            \
              "k1",                                                                                 \
              "zmm0",                                                                               \
              S_GEMV_T_YV_CLOBBERS                                                                  \
              "memory"                                                                              \
        )                                                                                           \
                                                                                                    \
        PP_IF(IS_FRINGE_OF(NR, ch), S_GEMV_T_UPDATE_DISPATCH_, ch, ctype, NR, 0)                    \
    }                                                                                               \
} /* End of GENTFUNC_TGEMV_s (AVX-512) */

#elif ARCH_SIMD_BITS == 256

/* AVX2 (zen): pure C intrinsics — no k-registers or ZMM.
 *
 * Unlike the AVX-512 path, we cannot hold all NR accumulator registers
 * simultaneously in named registers across a loop body.  Instead:
 *   - yv[] is a C array of YMM-wide SIMD vectors.
 *   - avbuf advances by EPR*cs_a each inner iteration (pointer arithmetic).
 *   - Fringe uses an integer mask (number of valid lanes, not a bitmask) since
 *     AVX2 masked loads take a lane-count argument, not a k-register.
 */

/*
 * S_GEMV_T_Y_ZERO (AVX2) — zero one yv accumulator slot.
 *   yv[ii-1] = 0   (YMM zero)
 * Called by PP_LOOP(NR, S_GEMV_T_Y_ZERO, ...) before the inner loop.
 */
#define S_GEMV_T_Y_ZERO(ch, _u1, _u2, ii)                                                        \
    yv[(ii) - 1] = PASTECH(SIMD_SETZERO_P_, ch)();

/*
 * S_GEMV_T_A_LOOP_INTR (AVX2) — one FMA step for column ii (1-based), real types.
 *
 *   yv[ii-1] += A[current_row_block, col_ii] * xv0
 *
 * avbuf points to A[current_row_block, col_1]; each column is rs_a elements apart.
 * xv0 is the broadcast x-vector for the current row block, loaded by S_GEMV_T_X_LOOP_INTR.
 * Called by PP_ILOOP(NR, S_GEMV_T_A_LOOP_INTR, ...) inside S_GEMV_T_X_LOOP_INTR.
 */
#define S_GEMV_T_A_LOOP_INTR(ch, _u1, _u2, ii)                                                   \
    yv[(ii) - 1] = PASTECH(SIMD_FMADD_P_, ch)(                                                   \
        PASTECH(SIMD_LOADU_P_, ch)(avbuf + ((ii) - 1) * rs_a),                                   \
        xv0, yv[(ii) - 1]);

/*
 * S_GEMV_T_A_FRINGE_LOOP_INTR (AVX2) — masked FMA step for column ii on the
 * fringe row block (real types). Mirrors the AVX-512 S_GEMV_T_A_FRINGE_LOOP path,
 * which uses MASK_K(1) for both the X load and the A FMA. Without masking the
 * A load, the compiler folds (loadu, fma) into vfmadd231p* with a memory
 * operand that reads a full SIMD register past the column tail, faulting on
 * the ProtectedBuffer redzone.
 */
#define S_GEMV_T_A_FRINGE_LOOP_INTR(ch, _u1, _u2, ii)                                            \
    yv[(ii) - 1] = PASTECH(SIMD_FMADD_P_, ch)(                                                   \
        SIMD_MASKZ_LOADU_P_##ch((1 << MR_left) - 1, avbuf + ((ii) - 1) * rs_a),                  \
        xv0, yv[(ii) - 1]);

/*
 * S_GEMV_T_X_LOOP_INTR (AVX2) — process one EPR-wide block of x rows, real types.
 *
 *   xv0   = x[xbuf .. xbuf+EPR-1]    (load EPR x-elements)
 *   xbuf += EPR                       (advance x pointer)
 *   PP_ILOOP(NR, S_GEMV_T_A_LOOP_INTR, ...) — for each of NR columns:
 *     yv[ii-1] += A[row_block, col_ii] * xv0
 *   avbuf += EPR * cs_a               (advance A pointer to next row block)
 *
 * Differs from AVX-512 in that xv0 is a local YMM variable (not a ZMM register alias),
 * and avbuf is advanced via pointer arithmetic rather than a dedicated ADD instruction.
 * Called by PP_LOOP(MR/EPR, S_GEMV_T_X_LOOP_INTR, ...) for full tiles,
 * or directly in the mloop_epr loop for partial-MR residual blocks.
 */
#define S_GEMV_T_X_LOOP_INTR(ch, _u1, NR, jj)                                                    \
    {                                                                                            \
        PASTECH(SIMD_VEC_, ch) xv0 = PASTECH(SIMD_LOADU_P_, ch)(xbuf);                           \
        xbuf += EPR;                                                                             \
        PP_ILOOP(NR, S_GEMV_T_A_LOOP_INTR, ch, 0, 0)                                             \
        avbuf += EPR * cs_a;                                                                     \
    }

/*
 * S_GEMV_T_FRINGE_0/1 (AVX2) — fringe x-row handler for real types.
 *
 * _FRINGE_0: no-op — MR divides m exactly, nothing left over.
 * _FRINGE_1: handles the last MR_left elements of x (< EPR).
 *   xv0 = masked load of MR_left elements from xbuf (mask is a bitmask of MR_left low bits).
 *   PP_ILOOP(NR, S_GEMV_T_A_LOOP_INTR, ...) — partial FMA for all NR columns.
 *   Note: avbuf is NOT advanced after the fringe since it is the last block.
 */
#define S_GEMV_T_FRINGE_0(ch, _u1, NR, xfull) /* empty */
#define S_GEMV_T_FRINGE_1(ch, _u1, NR, xfull)                                                    \
    {                                                                                            \
        PASTECH(SIMD_VEC_, ch) xv0 = SIMD_MASKZ_LOADU_P_##ch((1 << MR_left) - 1, xbuf);          \
        PP_ILOOP(NR, S_GEMV_T_A_FRINGE_LOOP_INTR, ch, 0, 0)                                      \
    }

/* GENTFUNC_TGEMV_s: real T-kernel, AVX2 pure intrinsics path. rs_a and cs_a are swapped for transpose*/
#define GENTFUNC_TGEMV_s(ctype, ch, ch_s, MR, NR)                                                   \
void GENTFUNC_GEMVTS(ctype, ch, MR, NR)                                                             \
     (                                                                                              \
       trans_t transa, conj_t conjx, dim_t m, dim_t n, ctype * alpha,                               \
       ctype * a, inc_t cs_a, inc_t rs_a, ctype * x, inc_t incx, ctype * beta,                      \
       ctype * y, inc_t incy, cntx_t * cntx                                                         \
     )                                                                                              \
{                                                                                                   \
    const dim_t EPR = PASTECH(ELEM_PER_REG_, ch_s);                                                 \
    ctype *restrict xbuf;                                                                           \
    ctype *restrict ybuf;                                                                           \
    ctype *restrict avbuf;                                                                          \
                                                                                                    \
    const dim_t mloop_full = m / MR;                                                                \
    const dim_t mloop_epr  = (m % MR) / EPR;                                                        \
    const dim_t MR_left    = m % EPR;                                                               \
                                                                                                    \
    PASTECH(SIMD_VEC_, ch) yv[NR];                                                                  \
                                                                                                    \
    for (dim_t i = 0; i < (n / NR); ++i)                                                            \
    {                                                                                               \
        ybuf  = y + (NR * i * incy);                                                                \
        xbuf  = x;                                                                                  \
        avbuf = a + (NR * i * rs_a);                                                                \
                                                                                                    \
        PP_LOOP(NR, S_GEMV_T_Y_ZERO, ch, 0, 0)                                                      \
                                                                                                    \
        for (dim_t j = 0; j < mloop_full; ++j)                                                      \
        {                                                                                           \
            PP_LOOP(PP_DIV(MR, EPR_OF(ch)), S_GEMV_T_X_LOOP_INTR, ch, 0, NR)                        \
        }                                                                                           \
                                                                                                    \
        for (dim_t j = 0; j < mloop_epr; ++j)                                                       \
        {                                                                                           \
            S_GEMV_T_X_LOOP_INTR(ch, 0, NR, 1)                                                      \
        }                                                                                           \
                                                                                                    \
        if (MR_left > 0)                                                                            \
        {                                                                                           \
            S_GEMV_T_FRINGE_1(ch, 0, NR, 0)                                                         \
        }                                                                                           \
                                                                                                    \
        PP_IF(IS_FRINGE_OF(NR, ch), S_GEMV_T_UPDATE_DISPATCH_, ch, ctype, NR, 0)                    \
    }                                                                                               \
} /* End of GENTFUNC_TGEMV_s (AVX2) */

#else
#error "ARCH_SIMD_BITS must be 256 (AVX2) or 512 (AVX-512)"
#endif

// #endregion ISA-specific real T-kernel inner-loop helpers

// ─────────────────────────────────────────────────────────────────────────────
// #region Complex and double T-kernel aliases
// ─────────────────────────────────────────────────────────────────────────────

/*
 * GENTFUNC_TGEMV_z: complex T-kernel using intrinsics (both ISAs).
 * Achieves complex multiply via PERMUTE + two FMAs.
 * rs_a and cs_a are swapped for transpose
 */
#define GENTFUNC_TGEMV_z(ctype, ch, ch_s, MR, NR)                                                      \
void GENTFUNC_GEMVTS(ctype, ch, MR, NR)                                                                \
     (                                                                                                 \
       trans_t transa,                                                                                 \
       conj_t conjx,                                                                                   \
       dim_t m,                                                                                        \
       dim_t n,                                                                                        \
       ctype * alpha,                                                                                  \
       ctype * a,                                                                                      \
       inc_t cs_a,                                                                                     \
       inc_t rs_a,                                                                                     \
       ctype * x,                                                                                      \
       inc_t incx,                                                                                     \
       ctype * beta,                                                                                   \
       ctype * y,                                                                                      \
       inc_t incy,                                                                                     \
       cntx_t * cntx                                                                                   \
     )                                                                                                 \
{                                                                                                      \
    const dim_t EPR = EPR_OF(ch);  /* complex elements per SIMD register (not scalar elements) */      \
    ctype *restrict abuf = a;                                                                          \
    ctype *restrict xbuf = x;                                                                          \
    ctype *restrict ybuf = y;                                                                          \
                                                                                                       \
    const dim_t mloop_full = m / MR;                                                                   \
    const dim_t mloop_epr  = (m % MR) / EPR;                                                           \
    const dim_t MR_left    = m % EPR;                                                                  \
                                                                                                       \
    PASTECH(SIMD_VEC_, ch_s) xv[1 * 2];                                                                \
    PASTECH(SIMD_VEC_, ch_s) yv[NR * 2];                                                               \
    PASTECH(SIMD_VEC_, ch_s) av[NR];                                                                   \
                                                                                                       \
    ctype temp;                                                                                        \
    ctype y_local;                                                                                     \
                                                                                                       \
    for (dim_t i = 0; i < (n / NR); ++i)                                                               \
    {                                                                                                  \
        PP_LOOP(NR, Z_GEMV_T_Y_ZERO, ch_s, NR, 0)                                                      \
        ybuf = y + (NR * i * incy);                                                                    \
        xbuf = x;                                                                                      \
        abuf = a + (NR * i * rs_a);                                                                    \
                                                                                                       \
        for (dim_t j = 0; j < mloop_full; ++j)                                                         \
        {                                                                                              \
            PP_LOOP(PP_DIV(MR, EPR_OF(ch)), Z_GEMV_T_X_LOOP, ch_s, ch, NR)                             \
            xbuf += MR * incx;                                                                         \
            abuf += MR * cs_a;                                                                         \
        }                                                                                              \
        for (dim_t j = 0; j < mloop_epr; ++j)                                                          \
        {                                                                                              \
            Z_GEMV_T_X_LOOP(ch_s, ch, NR, 1)                                                           \
            xbuf += EPR * incx;                                                                        \
            abuf += EPR * cs_a;                                                                        \
        }                                                                                              \
        if (MR_left > 0)                                                                               \
        {                                                                                              \
            Z_GEMV_T_FRINGE_1(ch_s, ch, NR, 0)                                                         \
        }                                                                                              \
        PP_LOOP(NR, Z_GEMV_T_UPDATE, ch, NR, 0)                                                        \
    }                                                                                                  \
} // End of GENTFUNC_TGEMV_z

/* scomplex uses the same implementation as dcomplex. */
#define GENTFUNC_TGEMV_c(ctype, ch, ch_s, MR, NR)  GENTFUNC_TGEMV_z(ctype, ch, ch_s, MR, NR)
/* double uses the same real implementation as float. */
#define GENTFUNC_TGEMV_d(ctype, ch, ch_s, MR, NR)  GENTFUNC_TGEMV_s(ctype, ch, ch_s, MR, NR)

// #endregion Complex and double T-kernel aliases

// ─────────────────────────────────────────────────────────────────────────────
// #region ST dispatcher, MT wrapper, entry-point, and master GENERATE_KERNEL
// ─────────────────────────────────────────────────────────────────────────────

/*
 * Helper token-paste macros for GEMV_ARCH_SUFFIX in generated names.
 *
 * Examples with GEMV_ARCH_SUFFIX=zen_int:
 *   GEMV_T_FNAME(d,20,4)         → bli_dgemv_t_zen_int_20x4
 *   GEMV_T_FNAME_MT(d,20,4)      → bli_dgemv_t_zen_int_20x4_mt
 *   GEMV_T_ENTRY(d)              → bli_dgemv_t_zen_int
 */
#define GEMV_T_FNAME(ch, MR, NR) \
    PASTEMAC2(ch, gemv_t_, PASTECH4(GEMV_ARCH_SUFFIX, _, MR, x, NR))
#define GEMV_T_FNAME_MT(ch, MR, NR) \
    PASTEMAC3(ch, gemv_t_, PASTECH4(GEMV_ARCH_SUFFIX, _, MR, x, NR), _mt)
#define GEMV_T_ENTRY(ch) \
    PASTEMAC(ch, PASTECH(gemv_t_, GEMV_ARCH_SUFFIX))

/*
 * GENT_GEMV_CALLER: ST dispatcher for transposed GEMV.
 * Generated function: bli_<ch>gemv_t_<GEMV_ARCH_SUFFIX>_<MR>x<NR>
 *
 * Splits n into full NR-column tiles and a ≤NR-1 fringe:
 *
 *        n/NR full tiles    n%NR leftover
 *       ┌──────────────────┬──────────────┐
 *  m    │ full kernel      │ fringe-NR    │   (all rows processed by each kernel)
 *       │ (MR row blocks)  │ (NR fringe)  │
 *       └──────────────────┴──────────────┘
 *
 * For the T-direction there is no M-fringe at this level: the micro-kernel
 * itself handles the m%MR rows via its internal fringe path.
 * Only the n-dimension is split here.
 *
 * Full tiles: call GENTFUNC_GEMVTS(ctype, ch, MR, NR) with n rounded down to NR.
 * Fringe:     switch(n_rem) dispatches to GEN_DISPATCH_SWITCH_T which selects
 *             the kernel of width (1..NR-1) matching n_rem exactly.
 */
#define GENT_GEMV_CALLER(ctype, ch, MR, NR)                                                         \
void GEMV_T_FNAME(ch, MR, NR)                                                                       \
       (                                                                                            \
         trans_t transa,                                                                            \
         conj_t conjx,                                                                              \
         dim_t m,                                                                                   \
         dim_t n,                                                                                   \
         ctype * alpha,                                                                             \
         ctype * a,                                                                                 \
         inc_t rs_a,                                                                                \
         inc_t cs_a,                                                                                \
         ctype * x,                                                                                 \
         inc_t incx,                                                                                \
         ctype * beta,                                                                              \
         ctype * y,                                                                                 \
         inc_t incy,                                                                                \
         cntx_t * cntx                                                                              \
       )                                                                                            \
{                                                                                                   \
    /* Full-width tiles: process floor(n/NR)*NR output elements at once. */                         \
    if (n >= NR)                                                                                    \
    {                                                                                               \
        GENTFUNC_GEMVTS(ctype, ch, MR, NR)                                                          \
            (transa, conjx, m, ((dim_t)(n / NR)) * NR,                                              \
             alpha, a, rs_a, cs_a, x, incx, beta, y, incy, cntx);                                   \
    }                                                                                               \
                                                                                                    \
    /* Fringe: up to NR-1 remaining output elements. */                                             \
    dim_t n_rem = n % NR;                                                                           \
    if (n_rem)                                                                                      \
    {                                                                                               \
        ctype * a_fringe = a + (((dim_t)(n / NR)) * NR * cs_a);  /* past the full tiles */          \
        ctype * y_fringe = y + (((dim_t)(n / NR)) * NR * incy);                                     \
        /* GEN_DISPATCH_SWITCH_T expands case (NR-1)..case 1, selecting kernel of width n_rem. */   \
        switch (n_rem)                                                                              \
        {                                                                                           \
            GEN_DISPATCH_SWITCH_T(ctype, ch, MR, NR)                                                \
        }                                                                                           \
    }                                                                                               \
}

/*
 * GENT_T_GEMV_N_DIM: multi-threaded GEMV-T wrapper.
 * Generated function: bli_<ch>gemv_t_<GEMV_ARCH_SUFFIX>_<MR>x<NR>_mt
 *
 * Threading strategy for T-direction GEMV (A^T * x → y):
 *   Each output element y[j] = dot(A[:,j], x) depends on ALL m rows of x
 *   but only column j of A.  Columns are fully independent of each other,
 *   so we split the n output elements across threads (N-dim split).
 *
 *   Thread tid owns y[thread_start .. thread_start+job_per_thread-1]:
 *     a_slice = a + thread_start * cs_a   (columns thread_start..thread_start+job-1)
 *     y_slice = y + thread_start * incy   (corresponding output elements)
 *     x is shared read-only across all threads (no write conflict).
 *
 *   No reduction step is needed because each thread writes a disjoint slice of y.
 *   beta is applied by each thread independently to its own y slice.
 *
 * nt is chosen by bli_nthreads_l2 based on (m, n) and the BLIS_TRANSPOSE workload.
 */
#define GENT_T_GEMV_N_DIM(ctype, ch, MR, NR)                                                 \
void GEMV_T_FNAME_MT(ch, MR, NR)                                                             \
     (                                                                                       \
       trans_t transa,                                                                       \
       conj_t  conjx,                                                                        \
       dim_t   m,                                                                            \
       dim_t   n,                                                                            \
       ctype* alpha,                                                                         \
       ctype* a, inc_t rs_a, inc_t cs_a,                                                     \
       ctype* x, inc_t incx,                                                                 \
       ctype* beta,                                                                          \
       ctype* y, inc_t incy,                                                                 \
       cntx_t* cntx                                                                          \
     )                                                                                       \
{                                                                                            \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)                                             \
                                                                                             \
    /* Query recommended thread count for this problem size. */                              \
    dim_t nt = 1;                                                                            \
    bli_nthreads_l2                                                                          \
    (                                                                                        \
        BLIS_GEMV_KER,                                                                       \
        PASTEMAC(ch,type),                                                                   \
        BLIS_TRANSPOSE,                                                                      \
        bli_arch_query_id_internal(),                                                        \
        m,                                                                                   \
        n,                                                                                   \
        &nt                                                                                  \
    );                                                                                       \
                                                                                             \
    _Pragma("omp parallel num_threads(nt)")                                                  \
    {                                                                                        \
        dim_t job_per_thread = n;                                                            \
        dim_t thread_start   = 0;                                                            \
                                                                                             \
        const dim_t tid     = omp_get_thread_num();                                          \
        const dim_t nt_real = omp_get_num_threads();                                         \
                                                                                             \
        /* Divide n output elements evenly; each thread gets a contiguous slice. */          \
        bli_thread_vector_partition( n, nt_real, &thread_start, &job_per_thread, tid );      \
        GEMV_T_FNAME(ch, MR, NR)                                                             \
        (                                                                                    \
            transa,                                                                          \
            conjx,                                                                           \
            m,                                                                               \
            job_per_thread,                                                                  \
            alpha,                                                                           \
            a +  thread_start * cs_a, rs_a, cs_a,  /* thread's column slice of A */          \
            x , incx,                               /* x is shared read-only */              \
            beta,                                                                            \
            y + thread_start * incy, incy,          /* thread's output slice of y */         \
            cntx                                                                             \
        );                                                                                   \
    }                                                                                        \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4);                                             \
}

/*
 * ST_KERNEL / MT_KERNEL: resolve to the correct function for a given data type.
 *
 * ST_KERNEL always points to the single-threaded GENT_GEMV_CALLER function.
 * MT_KERNEL points to the OpenMP GENT_T_GEMV_N_DIM wrapper when OpenMP is enabled,
 * or falls back to ST_KERNEL when it is not.
 *
 * SHOULD_CALL_ST(ch): evaluates to 1 when the problem is too small for multithreading.
 *   - With OpenMP: size = m*n; SHOULD_CALL_ST_<ch> compares size against a threshold.
 *     (Threshold is ~1800 elements for real; complex types may differ.)
 *   - Without OpenMP: always 1 — MT_KERNEL is aliased to ST_KERNEL.
 *
 * CALC_SIZE: emits `dim_t size = m * n;` so SHOULD_CALL_ST_<ch> can reference it.
 *   No-op without OpenMP (threshold logic is compiled away).
 *
 * MT_KERNEL_SIGNATURE: instantiates the MT wrapper function body.
 *   Expands to GENT_T_GEMV_N_DIM(ctype, ch, MR, NR) with OpenMP, empty otherwise.
 */
#define ST_KERNEL(ch, MR, NR) GEMV_T_FNAME(ch, MR, NR)

#ifdef BLIS_ENABLE_OPENMP
    #define CALC_SIZE dim_t size = m * n;
    #define SHOULD_CALL_ST(ch) PASTECH(SHOULD_CALL_ST_, ch)
    #define MT_KERNEL(ch, MR, NR) GEMV_T_FNAME_MT(ch, MR, NR)
    #define MT_KERNEL_SIGNATURE(ctype, ch, MR, NR) GENT_T_GEMV_N_DIM(ctype, ch, MR, NR)
#else
    #define CALC_SIZE
    #define SHOULD_CALL_ST_s 1
    #define SHOULD_CALL_ST_d 1
    #define SHOULD_CALL_ST_c 1
    #define SHOULD_CALL_ST_z 1
    #define SHOULD_CALL_ST(ch) PASTECH(SHOULD_CALL_ST_, ch)
    #define MT_KERNEL(ch, MR, NR) ST_KERNEL(ch, MR, NR)
    #define MT_KERNEL_SIGNATURE(ctype, ch, MR, NR)
#endif

/*
 * GENERATE_ROOT_KERNEL: public GEMV-T entry-point.
 * Generated function: bli_<ch>gemv_t_<GEMV_ARCH_SUFFIX>
 *
 * This is the function registered with the BLIS kernel dispatch table.
 * It handles two preprocessing steps before calling the tiled kernel:
 *
 *   Step 1 — x packing / conjugation (if needed):
 *     The inner T-kernels require x to be unit-stride (incx==1) and unconjugated,
 *     because they load x with SIMD unit-stride loads.
 *     If incx != 1 OR conjx == BLIS_CONJUGATE, we allocate a temporary buffer,
 *     copy x into it (applying conjugation if needed via copyv), and set
 *     temp_incx = 1 so the inner kernel sees unit-stride x.
 *     If allocation fails, we fall through and pass original x (best-effort).
 *
 *   Step 2 — ST vs MT dispatch:
 *     CALC_SIZE computes size = m * n.
 *     SHOULD_CALL_ST(ch) compares size to a threshold (≈1800):
 *       small problem → ST_KERNEL (no OpenMP overhead)
 *       large problem → MT_KERNEL (parallel N-dim split)
 *     The selected kernel pointer is called with BLIS_NO_CONJUGATE since
 *     conjugation was already applied during packing in Step 1.
 *
 *   Step 3 — release x buffer:
 *     If a temporary buffer was allocated, release it back to the pool.
 */
#define GENERATE_ROOT_KERNEL(ctype, ch, MR, NR)                                             \
void GEMV_T_ENTRY(ch)                                                                       \
       (                                                                                    \
         trans_t transa,                                                                    \
         conj_t conjx,                                                                      \
         dim_t m,                                                                           \
         dim_t n,                                                                           \
         ctype * alpha,                                                                     \
         ctype * a,                                                                         \
         inc_t rs_a,                                                                        \
         inc_t cs_a,                                                                        \
         ctype * x,                                                                         \
         inc_t incx,                                                                        \
         ctype * beta,                                                                      \
         ctype * y,                                                                         \
         inc_t incy,                                                                        \
         cntx_t * cntx                                                                      \
        )                                                                                   \
{                                                                                           \
    void (*ker_ft)(trans_t, conj_t, dim_t, dim_t, ctype *, ctype *, inc_t, inc_t,           \
                   ctype *, inc_t, ctype *, ctype *, inc_t, cntx_t *) = NULL;               \
    rntm_t  rntm;                                                                           \
    mem_t   mem_bufX;                                                                       \
    inc_t   temp_incx = incx;                                                               \
    ctype*  x_temp = x;                                                                     \
    PASTECH(ch,copyv_ker_ft)   copyv_kr_ptr = NULL;                                         \
                                                                                            \
    /* Step 1: pack x to a unit-stride buffer and/or conjugate it. */                       \
    const bool need_conj = bli_is_conj( conjx );                                            \
    if (incx != 1 || need_conj)                                                             \
    {                                                                                       \
        mem_bufX.pblk.buf = NULL;   mem_bufX.pblk.block_size = 0;                           \
        mem_bufX.buf_type = 0;      mem_bufX.size = 0;                                      \
        mem_bufX.pool = NULL;                                                               \
        bli_rntm_init_from_global( &rntm );                                                 \
        bli_rntm_set_num_threads_only( 1, &rntm );                                          \
        bli_pba_rntm_set_pba( &rntm );                                                      \
        size_t buffer_size = m * sizeof(ctype);                                             \
                                                                                            \
        bli_pba_acquire_m                                                                   \
        (                                                                                   \
          &rntm,                                                                            \
          buffer_size,                                                                      \
          BLIS_BUFFER_FOR_B_PANEL,                                                          \
          &mem_bufX                                                                         \
        );                                                                                  \
        if ( bli_mem_is_alloc( &mem_bufX ) )                                                \
        {                                                                                   \
            x_temp = bli_mem_buffer(&mem_bufX);                                             \
            temp_incx = 1;                                                                  \
            if(cntx == NULL) cntx = bli_gks_query_cntx();                                   \
            copyv_kr_ptr = bli_cntx_get_l1v_ker_dt(PASTEMAC(ch,type), BLIS_COPYV_KER, cntx);\
            copyv_kr_ptr                                                                    \
            (                                                                               \
              need_conj ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,                               \
              m,                                                                            \
              x, incx,                                                                      \
              x_temp, temp_incx,                                                            \
              cntx                                                                          \
            );                                                                              \
        }                                                                                   \
        else                                                                                \
        {                                                                                   \
            if ( need_conj )                                                                \
            {                                                                               \
                if(cntx == NULL) cntx = bli_gks_query_cntx();                               \
                /* Call non fused code path which internally uses dot kernel*/              \
                PASTEMAC(ch, gemv_unb_var2)( transa, conjx, m, n, alpha, a, rs_a, cs_a,     \
                                             x, incx, beta, y, incy, cntx );                \
            }                                                                               \
            else                                                                            \
            {                                                                               \
                PASTEMAC(ch, gemv_zen_ref)( transa, m, n, alpha, a, rs_a, cs_a,             \
                                            x, incx, beta, y, incy, NULL );                 \
            }                                                                               \
            return;                                                                         \
        }                                                                                   \
    }                                                                                       \
                                                                                            \
    /* Step 2: choose ST or MT kernel based on problem size. */                             \
    CALC_SIZE;                                                                              \
    if (SHOULD_CALL_ST(ch)) {                                                               \
        ker_ft = ST_KERNEL(ch, MR, NR);                                                     \
    }                                                                                       \
    else                                                                                    \
    {                                                                                       \
        ker_ft = MT_KERNEL(ch, MR, NR);                                                     \
    }                                                                                       \
    /* Conjugation already applied during packing; pass BLIS_NO_CONJUGATE. */               \
    ker_ft(transa, BLIS_NO_CONJUGATE, m, n, alpha, a, rs_a, cs_a, x_temp, temp_incx,        \
           beta, y, incy, cntx);                                                            \
                                                                                            \
    /* Step 3: release the x pack buffer if we allocated one. */                            \
    if ( x_temp != x )                                                                      \
    {                                                                                       \
        bli_pba_release(&rntm , &mem_bufX);                                                 \
    }                                                                                       \
}

/*
 * GENERATE_KERNEL: master macro — instantiates all T-kernel functions for (ctype, ch, MR, NR).
 *   1. GENERATE_<ch>_KERNELS_<MR>_T  — micro-kernels + dispatch table
 *   2. GENT_GEMV_CALLER              — tiled ST dispatcher
 *   3. MT_KERNEL_SIGNATURE           — MT wrapper (if OpenMP enabled)
 *   4. GENERATE_ROOT_KERNEL          — public entry-point
 */
#define GENERATE_KERNEL(ctype, ch, MR, NR)                                                  \
    PASTECH4(GENERATE_, ch, _KERNELS_, MR, _T)(ctype, ch, MR, NR);                          \
    GENT_GEMV_CALLER(ctype, ch, MR, NR);                                                    \
    MT_KERNEL_SIGNATURE(ctype, ch, MR, NR);                                                 \
    GENERATE_ROOT_KERNEL(ctype, ch, MR, NR);

// #endregion

#endif /* BLI_GEMV_T_IMPL_H */
