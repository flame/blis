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
 * bli_gemv_n_impl.h — shared N-kernel implementation for zen (AVX2) and zen4 (AVX-512).
 *
 * Prerequisites (must be defined before including this header):
 *   ARCH_SIMD_BITS     — 256 (AVX2) or 512 (AVX-512); selects ISA in bli_gemv_common_int.h
 *   GEMV_ARCH_SUFFIX   — arch token used in generated function names
 *                        (e.g. zen_int or zen4_int)
 *
 * Optional overrides (already defaulted in bli_gemv_common_int.h):
 *   GEMV_BLK_SUFFIX_N(ch,MR,NR) — micro-kernel name suffix (zen sets _avx2, zen4 leaves empty)
 *
 * Generated symbol naming convention with GEMV_ARCH_SUFFIX=<arch>:
 *   micro-kernel (N): bli_<ch>gemv_n_block_<MR>_<NR>  (or ..._avx2 via GEMV_BLK_SUFFIX_N)
 *   micro-kernel (M): bli_<ch>gemv_m_block_<MR>_<NR>  (or ..._avx2 via GEMV_BLK_SUFFIX_N)
 *   ST caller:        bli_<ch>gemv_n_<arch>_<MR>x<NR> / bli_<ch>gemv_m_<arch>_<MR>x<NR>
 *   MT wrapper:       bli_<ch>gemv_n_<arch>_<MR>x<NR>_mt
 *   entry-point:      bli_<ch>gemv_n_<arch>
 */

#ifndef BLI_GEMV_N_IMPL_H
#define BLI_GEMV_N_IMPL_H

// =============================================================================
// #region GEMV-N control thresholds
// =============================================================================
//
// Tuning constants consumed by the per-type control functions in the shim files
// (bli_gemv_n_zen_int.c / bli_gemv_n_zen4_int.c).

// Byte-count threshold used by the ST dispatcher to choose the M-direction tiled
// caller (small problems) vs the N-direction tiled caller (large ones). Byte-
// based so it auto-scales per datatype (double: m*n < 95000).
#ifndef GEMV_N_CTRL_THRESH_BYTES
  #define GEMV_N_CTRL_THRESH_BYTES  ((dim_t)(9500 * (dim_t)sizeof(double)))
#endif

// OpenMP split-direction thresholds (element counts, dtype-independent), used by
// the public entry to choose N-split (Ndiv) vs row-split (Mdiv).
#ifndef GEMV_N_CTRL_MT_THRESH_M
  #define GEMV_N_CTRL_MT_THRESH_M     1250LL
#endif
#ifndef GEMV_N_CTRL_MT_THRESH_SIZE
  #define GEMV_N_CTRL_MT_THRESH_SIZE  ((dim_t)(700000 * 128))
#endif

// =============================================================================
// #region N-kernel name hooks
// =============================================================================

/*
 * GEMV_BLK_SUFFIX_N: hook for N micro-kernel naming.
 * Default (AVX-512): bli_<ch>gemv_n_block_<MR>_<NR>
 * AVX2 zen shims define their own suffix before including this header.
 */
#ifndef GEMV_BLK_SUFFIX_N
  #define GEMV_BLK_SUFFIX_N(ch, MR, NR)  PASTEMAC4(ch, gemv_n_block_, MR, _, NR)
#endif

/* GENTFUNC_GEMVNS: expands to the N-direction micro-kernel function name. */
#define GENTFUNC_GEMVNS(ctype, ch, MR, NR)  GEMV_BLK_SUFFIX_N(ch, MR, NR)

// #endregion

// =============================================================================
// #region N-kernel instantiation and dispatch-table helpers
// =============================================================================

// Each ST/MT caller requires a 2D table of function pointer of micro kenrels

/* DO_GENTFUNC_NGEMV: instantiates one N-direction micro-kernel. */
#define DO_GENTFUNC_NGEMV(ctype, ch, M, i)   PASTECH(GENTFUNC_NGEMV_,ch)(ctype, ch, PASTECH(CH_S_,ch), M, i);

/* DO_GENTFUNC_NMGEMV: instantiates one M-direction micro-kernel. */
#define DO_GENTFUNC_NMGEMV(ctype, ch, M, i)   GENTFUNC_GEMV(ctype, ch, M, i);

/*
 * DO_GENTFUNC_GEMVNS/NMS: emit a function-pointer entry (with comma)
 * for N/NM dispatch tables. (one entry in 2D table)
 */
#define DO_GENTFUNC_GEMVNS(ctype, ch, M, i)  GENTFUNC_GEMVNS(ctype, ch, M, i),
#define DO_GENTFUNC_GEMVNMS(ctype, ch, M, i) GENTFUNC_GEMVNMS(ctype, ch, M, i),

/* GEN_FUNC_PAIR_N/NM: instantiate kernels for full-tile MR and MR-1 fringe, NR cols. */
#define GEN_FUNC_PAIR_N(ctype, ch, NR, M)                                                        \
    RANGE_DOWN(DO_GENTFUNC_NGEMV, ctype, ch, M, NR)                                              \
    RANGE_DOWN(DO_GENTFUNC_NGEMV, ctype, ch, DEC_##M, NR)

#define GEN_FUNC_PAIR_NM(ctype, ch, NR, M)                                                       \
    RANGE_DOWN(DO_GENTFUNC_NMGEMV, ctype, ch, M, NR)                                             \
    RANGE_DOWN(DO_GENTFUNC_NMGEMV, ctype, ch, DEC_##M, NR)

// GEN_ARRAY_ROW_N/NM: generate one row of a static N/NM function-pointer dispatch table.
#define GEN_ARRAY_ROW_N(ctype, ch, NR, M)                                                        \
    { RANGE_DOWN(DO_GENTFUNC_GEMVNS, ctype, ch, M, NR) },

#define GEN_ARRAY_ROW_NM(ctype, ch, NR, M)                                                       \
    { RANGE_DOWN(DO_GENTFUNC_GEMVNMS, ctype, ch, M, NR) },

// #endregion

// =============================================================================
// #region ch_gemv_ker typedef — used in static N/NM dispatch tables
// =============================================================================

/* GENT_GEMV_FPTR: declares the ch_gemv_ker typedef for static dispatch tables. */
#define GENT_GEMV_FPTR(ctype, ch)                                                                \
    typedef void (*PASTECH(ch, gemv_ker))                                                        \
    (                                                                                            \
        trans_t transa,                                                                          \
        conj_t conjx,                                                                            \
        dim_t m,                                                                                 \
        dim_t n,                                                                                 \
        ctype *alpha,                                                                            \
        ctype *a, inc_t rs_a, inc_t cs_a,                                                        \
        ctype *x, inc_t incx,                                                                    \
        ctype *beta,                                                                             \
        ctype *y, inc_t incy,                                                                    \
        cntx_t * cntx                                                                            \
    );

GENT_GEMV_FPTR(float,    s);
GENT_GEMV_FPTR(double,   d);
GENT_GEMV_FPTR(scomplex, c);
GENT_GEMV_FPTR(dcomplex, z);

// #endregion

// =============================================================================
// #region GENERATE_<ch>_KERNELS_<MR>_N/NM — N-direction kernel expansion macros
// =============================================================================

// These macros call the function instantiaters(GEN_FUNC_PAIR_N) and kernel table
// generator for given MR and NR.

#define GENERATE_s_KERNELS_80_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, 80, 64, 48, 32, 16)                       \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 80, 16, 32, 48, 64)                   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 15, 31, 47, 63, 79)                   \
    };

#define GENERATE_s_KERNELS_64_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 48, 32, 16)                           \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[8][NR] = {    \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 16, 32, 48)                       \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 15, 31, 47, 63)                       \
    };

#define GENERATE_s_KERNELS_48_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 32, 16)                               \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[6][NR] = {    \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 16, 32)                           \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 15, 31, 47)                           \
    };

#define GENERATE_s_KERNELS_32_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 16)                                   \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[4][NR] = {    \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 16)                               \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 15, 31)                               \
    };

#define GENERATE_s_KERNELS_16_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR)                                       \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[2][NR] = {    \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR)                                   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 15)                                   \
    };

#define GENERATE_d_KERNELS_8_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, 8)                                       \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[2][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR)                                  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7 )                                  \
    };

#define GENERATE_d_KERNELS_16_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 8)                                   \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[4][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 8)                               \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7, 15)                               \
    };

#define GENERATE_d_KERNELS_24_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 16, 8)                               \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[6][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 8, 16)                           \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7, 15, 23)                           \
    };

#define GENERATE_d_KERNELS_32_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 24, 16, 8)                           \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[8][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 8, 16, 24)                       \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7, 15, 23, 31)                       \
    };

#define GENERATE_d_KERNELS_40_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, 40, 32, 24, 16, 8)                       \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 40, 8, 16, 24, 32)                   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7, 15, 23, 31, 39)                   \
    };

#define GENERATE_d_KERNELS_64_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, 64, 56, 48, 40, 32, 24, 16, 8)           \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[16][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 64, 8, 16, 24, 32, 40, 48, 56)       \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7, 15, 23, 31, 39, 47, 55, 63)       \
    };

#define GENERATE_c_KERNELS_8_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR)                                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[2][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR)                                  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7)                                   \
    };

#define GENERATE_z_KERNELS_32_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 28, 24, 20, 16, 12, 8, 4)            \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[16][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 4, 8, 12, 16, 20, 24, 28)        \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 3, 7, 11, 15, 19, 23, 27, 31)        \
    };

#define GENERATE_z_KERNELS_20_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 16, 12, 8, 4)                        \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 4, 8, 12, 16)                    \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 3, 7, 11, 15, 19)                    \
    };

#define GENERATE_z_KERNELS_16_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 12, 8, 4)                            \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[8][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 4, 8, 12)                        \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 3, 7, 11, 15)                        \
    };

#define GENERATE_z_KERNELS_12_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 8, 4)                                \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[6][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 4, 8)                            \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 3, 7, 11)                            \
    };

/* AVX2 z N-kernel: MR=10, EPR=2.
 * Table[10][NR] layout (m_idx = (m%10)/2 + is_m_left*5):
 *   rows 0-4: full-register sizes MR=10, 2, 4, 6, 8
 *   rows 5-9: fringe sizes 1, 3, 5, 7, 9 */
#define GENERATE_z_KERNELS_10_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 8, 6, 4, 2)                          \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 2, 4, 6, 8)                      \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 1, 3, 5, 7, 9)                       \
    };

#define GENERATE_z_KERNELS_8_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 4)                                   \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[4][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 4)                               \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 3, 7)                                \
    };

/* AVX2 z N-kernel: MR=2 */
#define GENERATE_z_KERNELS_2_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR)                                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[2][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR)                                  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 1)                                   \
    };

#define GENERATE_z_KERNELS_4_N(ctype, ch, MR, NR)                                     \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR)                                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[2][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR)                                  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 3)                                   \
    };

/* double / d : MR=20 (zen/AVX2), MR=40 (zen4/AVX-512).
 * AVX2 d uses EPR=4 → m_idx ranges over [0..2*MR/EPR-1] = [0..9].
 * Table layout matches the m_idx = (m%MR)/EPR + is_m_left*(MR/EPR) formula. */
#define GENERATE_d_KERNELS_20_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, MR, 16, 12, 8, 4)                        \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, MR, 4, 8, 12, 16)                    \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 3, 7, 11, 15, 19)                    \
    };

#define GENERATE_d_KERNELS_40_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, 40, 32, 24, 16, 8)                       \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 40, 8, 16, 24, 32)                   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7, 15, 23, 31, 39)                   \
    };

/* double / d NM variants.
 * AVX2 d uses EPR=4 → 10-row table (see GENERATE_d_KERNELS_20_N). */
#define GENERATE_d_KERNELS_20_NM(ctype, ch, MR, NR)                                   \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR, 16, 12, 8, 4)                       \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR, 4, 8, 12, 16)                   \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 3, 7, 11, 15, 19)                   \
    };

#define GENERATE_d_KERNELS_40_NM(ctype, ch, MR, NR)                                   \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR, 32, 24, 16, 8)                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR, 8, 16, 24, 32)                  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7, 15, 23, 31, 39)                  \
    };

/* float / s : MR=40 (zen/AVX2), MR=80 (zen4/AVX-512) */
#define GENERATE_s_KERNELS_40_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, 40, 32, 24, 16, 8)                       \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 40, 8, 16, 24, 32)                   \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 7, 15, 23, 31, 39)                   \
    };

#define GENERATE_s_KERNELS_80_N(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_N, ctype, ch, NR, 80, 64, 48, 32, 16)                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_n_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 80, 16, 32, 48, 64)                  \
        FOR_EACH(GEN_ARRAY_ROW_N, ctype, ch, NR, 15, 31, 47, 63, 79)                  \
    };

/* float / s NM variants */
#define GENERATE_s_KERNELS_40_NM(ctype, ch, MR, NR)                                   \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, 40, 32, 24, 16, 8)                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 40, 8, 16, 24, 32)                  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7, 15, 23, 31, 39)                  \
    };

#define GENERATE_c_KERNELS_20_N(ctype, ch, MR, NR) GENERATE_z_KERNELS_20_N(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_40_N(ctype, ch, MR, NR) GENERATE_d_KERNELS_40_N(ctype, ch, MR, NR)

#define GENERATE_d_KERNELS_8_NM(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR)                                     \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[2][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR)                                 \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7)                                  \
    };

#define GENERATE_d_KERNELS_16_NM(ctype, ch, MR, NR)                                   \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR, 8)                                  \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[4][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR, 8)                              \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7, 15)                              \
    };

#define GENERATE_d_KERNELS_24_NM(ctype, ch, MR, NR)                                   \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR, 16, 8)                              \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[6][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR, 8, 16)                          \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7, 15, 23)                          \
    };

#define GENERATE_d_KERNELS_32_NM(ctype, ch, MR, NR)                                   \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR, 24, 16, 8)                          \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[8][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR, 8, 16, 24)                      \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7, 15, 23, 31)                      \
    };

#define GENERATE_d_KERNELS_40_NM(ctype, ch, MR, NR)                                   \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR, 32, 24, 16, 8)                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[10][NR] = {  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR, 8, 16, 24, 32)                  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7, 15, 23, 31, 39)                  \
    };

#define GENERATE_d_KERNELS_64_NM(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, MR, 56, 48, 40, 32, 24, 16, 8)           \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[16][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, MR, 8, 16, 24, 32, 40, 48, 56)       \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 7, 15, 23, 31, 39, 47, 55)           \
    };

#define GENERATE_s_KERNELS_80_NM(ctype, ch, MR, NR)                                    \
    FOR_EACH(GEN_FUNC_PAIR_NM, ctype, ch, NR, 80, 64, 48, 32, 16)                      \
    static PASTECH(ch, gemv_ker) PASTECH4(ch, gemv_m_ker_fp_, MR, _, NR)[10][NR] = {   \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 80, 16, 32, 48, 64)                  \
        FOR_EACH(GEN_ARRAY_ROW_NM, ctype, ch, NR, 15, 31, 47, 63, 79)                  \
    };

/* scomplex / c — aliases into the corresponding double-precision tables
 * (same MR, same structure; only the base type differs). */
#define GENERATE_c_KERNELS_20_N(ctype, ch, MR, NR) GENERATE_z_KERNELS_20_N(ctype, ch, MR, NR)
#define GENERATE_c_KERNELS_40_N(ctype, ch, MR, NR) GENERATE_d_KERNELS_40_N(ctype, ch, MR, NR)

// #endregion

// ─────────────────────────────────────────────────────────────────────────────
// #region Complex-pointer cast helpers
// ─────────────────────────────────────────────────────────────────────────────

/* AVX2 intrinsics (e.g. _mm256_loadu_ps) require float* or double*, so complex
 * buffers need an explicit cast.  AVX-512 intrinsics accept void* — no cast. */
#if ARCH_SIMD_BITS == 256
#define GEMV_CAST_s(ptr) ((float*)(ptr))
#define GEMV_CAST_d(ptr) ((double*)(ptr))
#else
#define GEMV_CAST_s(ptr) (ptr)
#define GEMV_CAST_d(ptr) (ptr)
#endif

// #endregion

// ─────────────────────────────────────────────────────────────────────────────
// #region ST N kernel compile-time loop helpers
// ─────────────────────────────────────────────────────────────────────────────

/*
 * Z_GEMV_N_ALPHA_SCALE — complex alpha*x pre-scale for one column j1 (1-based).
 *
 * For complex GEMV the scaled x vector must be split into real and imaginary
 * broadcast registers to enable the FMADDSUB accumulation pattern:
 *
 *   (alpha * x[j]).real = alpha.real * x[j].real - alpha.imag * x[j].imag
 *   (alpha * x[j]).imag = alpha.real * x[j].imag + alpha.imag * x[j].real
 *
 * Writes:
 *   xv[j1-1]    = broadcast( (alpha*x[j1]).real )   — NR real broadcast registers
 *   xv[j1-1+NR] = broadcast( (alpha*x[j1]).imag )   — NR imag broadcast registers
 *
 * Called by PP_LOOP(NR, Z_GEMV_N_ALPHA_SCALE, ...) to fill all NR column slots.
 * Uses caller-scope: xbuf, incx, alpha, xv[NR*2].
 */
#define Z_GEMV_N_ALPHA_SCALE(ch_s, NR, _unused2, j1)                                                                        \
    {                                                                                                                       \
        temp = *(xbuf + ((j1) - 1) * incx);                                                                                 \
        xv[((j1) - 1)     ] = PASTECH(SIMD_SET1_P_, ch_s)( temp.real * (*alpha).real - temp.imag * (*alpha).imag );        \
        xv[((j1) - 1) + NR] = PASTECH(SIMD_SET1_P_, ch_s)( temp.real * (*alpha).imag + temp.imag * (*alpha).real );        \
    }

/*
 * Z_GEMV_N_A_LOOP — inner FMA step for one (row-register k1, column l1) tile pair (complex).
 *
 * Loads EPR packed scalar values from column l1, row-register k1 of A, then
 * accumulates the real and imaginary partial products separately:
 *
 *   av[0]          = A[row_reg_k1, col_l1]          (load MR/EPR contiguous elements)
 *   yv[0]         += av[0] * xv[l1-1]               (real part:  A * (alpha*x).real)
 *   yv[0 + yregs] += av[0] * xv[l1-1 + NR]          (imag part:  A * (alpha*x).imag)
 *
 * These two accumulators are later combined with PERMUTE + FMADDSUB in Z_GEMV_N_Y_LOOP.
 * Uses caller-scope: abuf, xv[NR*2], yv[2], yregs, EPR, rs_a, cs_a.
 * Called by PP_ILOOP(NR, Z_GEMV_N_A_LOOP, ...) inside Z_GEMV_N_Y_LOOP.
 */
#define Z_GEMV_N_A_LOOP(ch_s, NR, k1, l1)                                                                                   \
    av[0] = PASTECH(SIMD_LOADU_P_, ch_s)(PASTECH(GEMV_CAST_, ch_s)(abuf + ((k1) - 1)*(EPR)*rs_a + ((l1) - 1)*cs_a));      \
    yv[0        ] = PASTECH(SIMD_FMADD_P_, ch_s)(av[0], xv[((l1) - 1)], yv[0]);                                            \
    yv[0 + yregs] = PASTECH(SIMD_FMADD_P_, ch_s)(av[0], xv[((l1) - 1) + NR], yv[0 + yregs]);

/*
 * Z_GEMV_N_Y_LOOP — full MR-row tile update for complex types, row-register k1 (1-based).
 *
 * Implements one row-register's worth of the complex GEMV update:
 *   y[k1_rows] = beta * y[k1_rows] + A[k1_rows, 0..NR-1] * (alpha * x[0..NR-1])
 *
 * Step-by-step:
 *   1. Load beta*y (or zero if beta==0) into the real accumulator yv[0].
 *      Load beta_iv*y into the imaginary accumulator yv[0+yregs].
 *      (beta_rv and beta_iv are broadcasted beta.real and beta.imag scalars.)
 *   2. PP_ILOOP(NR, Z_GEMV_N_A_LOOP, ...) — for each of the NR columns:
 *        yv[0]         += A_col * xv[col].real
 *        yv[0+yregs]   += A_col * xv[col+NR].imag
 *   3. Permute the imaginary accumulator: swap (real,imag) pairs within each element
 *      so that FMADDSUB can combine them correctly.
 *   4. FMADDSUB: yv[0] = yv[0] +/- yv[0+yregs]
 *        even lanes: yv[0].real - yv_imag.real  (subtract cross term)
 *        odd  lanes: yv[0].imag + yv_imag.imag  (add   cross term)
 *   5. Store the finished row back to ybuf.
 *
 * Uses caller-scope: ybuf, beta_rv, beta_iv, beta_is_zero, xv[NR*2], yv[2], yregs,
 *                    EPR, incy, rs_a, cs_a, abuf.
 */
#define Z_GEMV_N_Y_LOOP(ch_s, ch, NR, k1)                                                                                   \
    yv[0     ] = beta_is_zero ? PASTECH(SIMD_SETZERO_P_, ch_s)()                                                           \
                              : PASTECH(SIMD_MUL_P_, ch_s)(beta_rv, PASTECH(SIMD_LOADU_P_, ch_s)(PASTECH(GEMV_CAST_, ch_s)(ybuf + ((k1) - 1) * (EPR)))); \
    yv[0 + yregs] = beta_is_zero ? PASTECH(SIMD_SETZERO_P_, ch_s)()                                                        \
                                 : PASTECH(SIMD_MUL_P_, ch_s)(beta_iv, PASTECH(SIMD_LOADU_P_, ch_s)(PASTECH(GEMV_CAST_, ch_s)(ybuf + ((k1) - 1) * (EPR)))); \
    PP_ILOOP(NR, Z_GEMV_N_A_LOOP, ch_s, NR, k1)                                                                             \
    yv[0 + yregs] = PASTECH(SIMD_PERMUTE_P_, ch_s)(yv[0 + yregs], PASTECH(PERM_MASK_, ch));                                \
    yv[0] = PASTECH(SIMD_FMADDSUB_P_, ch_s)(PASTECH(SIMD_SET1_P_, ch_s)(1.0), yv[0], yv[0 + yregs]);                      \
    PASTECH(SIMD_STOREU_P_, ch_s)( PASTECH(GEMV_CAST_, ch_s)(ybuf + ((k1) - 1)*(EPR)*incy), yv[0] );

/*
 * Z_GEMV_N_A_FRINGE_LOOP — inner FMA for the partial (fringe) row-register, complex.
 *
 * Same as Z_GEMV_N_A_LOOP but uses a masked load so only MR_left complex elements
 * (i.e. 2*MR_left scalar lanes) are loaded from A.  The mask is:
 *   mask = (1 << (MR_left * 2)) - 1   — sets the lowest 2*MR_left bits
 *
 * Mathematical effect (for the yfull-th row register, column l1):
 *   av[0]         = A[yfull*EPR .. yfull*EPR + MR_left - 1, col_l1]  (masked)
 *   yv[0]        += av[0] * xv[l1-1]          (real accumulator)
 *   yv[0+yregs]  += av[0] * xv[l1-1+NR]       (imag accumulator)
 *
 * Called by PP_ILOOP(NR, Z_GEMV_N_A_FRINGE_LOOP, ...) inside Z_GEMV_N_FRINGE_1.
 * Uses caller-scope: abuf, xv[NR*2], yv[2], yregs, MR_left, EPR, cs_a.
 */
#define Z_GEMV_N_A_FRINGE_LOOP(ch_s, NR, yfull, l1)                                                                         \
    av[0] = SIMD_MASKZ_LOADU_P_##ch_s((1 << (MR_left * 2))-1, PASTECH(GEMV_CAST_, ch_s)(abuf + (yfull) * (EPR) * rs_a + ((l1) - 1) * cs_a)); \
    yv[0] = PASTECH(SIMD_FMADD_P_, ch_s)(av[0], xv[((l1) - 1)], yv[0]);                                                    \
    yv[0+yregs] = PASTECH(SIMD_FMADD_P_, ch_s)(av[0], xv[((l1) - 1)+NR], yv[0+yregs]);

/*
 * Z_GEMV_N_FRINGE_0/1 — fringe row handler for complex types.
 *
 * _FRINGE_0: no-op — MR divides m exactly, nothing left over.
 * _FRINGE_1: handles the final partial tile of MR_left complex elements (< EPR).
 *
 * Z_GEMV_N_FRINGE_1 does the same as Z_GEMV_N_Y_LOOP but with masked loads/stores:
 *   1. Masked-load beta*y (or zero) for only MR_left complex lanes.
 *   2. PP_ILOOP(NR, Z_GEMV_N_A_FRINGE_LOOP, ...) — masked FMA across all NR columns.
 *   3. PERMUTE + FMADDSUB — same complex combine step as the full tile.
 *   4. Masked-store the MR_left results back to ybuf.
 *
 * "yfull" = MR/EPR = number of full row-registers already handled;
 * the fringe starts at row index yfull*EPR.
 *
 * Uses caller-scope: ybuf, beta_rv, beta_iv, beta_is_zero, xv[NR*2], yv[2], yregs,
 *                    MR_left, EPR, incy, abuf.
 */
#define Z_GEMV_N_FRINGE_0(ch_s, ch, NR, yfull) /* empty */
#define Z_GEMV_N_FRINGE_1(ch_s, ch, NR, yfull)                                                                             \
    yv[0] = beta_is_zero ? PASTECH(SIMD_SETZERO_P_, ch_s)()                                                                \
                         : PASTECH(SIMD_MUL_P_, ch_s)(beta_rv, SIMD_MASKZ_LOADU_P_##ch_s((1 << (MR_left * 2)) - 1, PASTECH(GEMV_CAST_, ch_s)(ybuf + (yfull) * (EPR)))); \
    yv[0 + yregs] = beta_is_zero ? PASTECH(SIMD_SETZERO_P_, ch_s)()                                                        \
                                 : PASTECH(SIMD_MUL_P_, ch_s)(beta_iv, SIMD_MASKZ_LOADU_P_##ch_s((1 << (MR_left * 2)) - 1, PASTECH(GEMV_CAST_, ch_s)(ybuf + (yfull) * (EPR)))); \
    PP_ILOOP(NR, Z_GEMV_N_A_FRINGE_LOOP, ch_s, NR, yfull)                                                                   \
    yv[0 + yregs] = PASTECH(SIMD_PERMUTE_P_, ch_s)(yv[0 + yregs], PASTECH(PERM_MASK_, ch));                                \
    yv[0] = PASTECH(SIMD_FMADDSUB_P_, ch_s)(PASTECH(SIMD_SET1_P_, ch_s)(1.0), yv[0], yv[0 + yregs]);                      \
    SIMD_MASK_STOREU_P_##ch_s( PASTECH(GEMV_CAST_, ch_s)(ybuf + (yfull) * (EPR) * incy), (1 << (MR_left * 2)) - 1, yv[0]);

/*
 * S_GEMV_N_ALPHA_SCALE — real alpha*x pre-scale for one column j1 (1-based).
 *
 * For real types (s/d), alpha*x is a single scalar broadcast:
 *   xv[j1-1] = broadcast( alpha * x[j1] )
 *
 * Called by PP_LOOP(NR, S_GEMV_N_ALPHA_SCALE, ...) at the start of each NR-column strip.
 * Uses caller-scope: xbuf, incx, alpha, xv[NR].
 */
#define S_GEMV_N_ALPHA_SCALE(ch, _unused1, _unused2, j1)                                                                    \
    xv[((j1) - 1)] = PASTECH(SIMD_SET1_P_, ch)( (*alpha) * (*(xbuf + ((j1) - 1) * incx)) );

/*
 * S_GEMV_N_A_LOOP — inner FMA for one (row-register k1, column l1) tile pair, real.
 *
 *   av[l1-1] = A[row_reg_k1, col_l1]          (load EPR contiguous elements)
 *   yv[k1-1] += av[l1-1] * xv[l1-1]           (accumulate into row register k1)
 *
 * Note: av[l1-1] keeps each column's A-slice in a separate register — the compiler
 * can reuse them since NR is small and unrolled.  yv[k1-1] is the single accumulator
 * for row-register k1.
 *
 * Called by PP_ILOOP(NR, S_GEMV_N_A_LOOP, ...) inside S_GEMV_N_Y_LOOP.
 * Uses caller-scope: abuf, xv[NR], yv[MR/EPR+1], av[NR], EPR, rs_a, cs_a.
 */
#define S_GEMV_N_A_LOOP(ch, _unused, k1, l1)                                                                                \
    av[((l1) - 1)] = PASTECH(SIMD_LOADU_P_, ch)(abuf + ((k1) - 1)*(EPR)*rs_a + ((l1) - 1)*cs_a);                           \
    yv[((k1) - 1)] = PASTECH(SIMD_FMADD_P_, ch)(av[((l1) - 1)], xv[((l1) - 1)], yv[((k1) - 1)]);

/*
 * S_GEMV_N_Y_LOOP — full MR-row tile update for real types, row-register k1 (1-based).
 *
 * Implements one full EPR-wide row register's GEMV update:
 *   y[rows_k1] = beta * y[rows_k1] + A[rows_k1, 0..NR-1] * (alpha * x[0..NR-1])
 *
 * Step-by-step:
 *   1. Load beta*y (or zero if beta==0) into yv[k1-1].
 *   2. PP_ILOOP(NR, S_GEMV_N_A_LOOP, ...) — for each of the NR columns:
 *        yv[k1-1] += A_col * xv[col]
 *   3. Store the finished row register back to ybuf.
 *
 * Called by PP_LOOP(MR/EPR, S_GEMV_N_Y_LOOP, ...) inside the kernel body to handle
 * all full row-registers in one MR-block.
 * Uses caller-scope: ybuf, betav, beta_is_zero, xv[NR], yv[MR/EPR+1], EPR, incy.
 */
#define S_GEMV_N_Y_LOOP(ch, NR, _unused, k1)                                                                                \
    yv[((k1) - 1)] = beta_is_zero ? PASTECH(SIMD_SETZERO_P_, ch)()                                                         \
                                  : PASTECH(SIMD_MUL_P_, ch)(betav, PASTECH(SIMD_LOADU_P_, ch)(ybuf + ((k1) - 1) * (EPR))); \
    PP_ILOOP(NR, S_GEMV_N_A_LOOP, ch, 0, k1)                                                                                \
    PASTECH(SIMD_STOREU_P_, ch)( ybuf + ((k1) - 1)*(EPR)*incy, yv[((k1) - 1)] );

/*
 * S_GEMV_N_A_FRINGE_LOOP — inner FMA for the partial (fringe) row-register, real.
 *
 * Same as S_GEMV_N_A_LOOP but uses a masked load for only MR_left elements:
 *   mask = (1 << MR_left) - 1   — lowest MR_left bits
 *   av[l1-1] = A[yfull*EPR .. yfull*EPR+MR_left-1, col_l1]  (masked load)
 *   yv[yfull] += av[l1-1] * xv[l1-1]
 *
 * Uses caller-scope: abuf, xv[NR], yv[MR/EPR+1], MR_left, EPR, rs_a, cs_a.
 * Called by PP_ILOOP(NR, S_GEMV_N_A_FRINGE_LOOP, ...) inside S_GEMV_N_FRINGE_1.
 */
#define S_GEMV_N_A_FRINGE_LOOP(ch, yfull, _unused, l1)                                                                      \
    av[((l1) - 1)] = SIMD_MASKZ_LOADU_P_##ch((1 << (MR_left))-1, abuf + (yfull) * (EPR) * rs_a + ((l1) - 1) * cs_a); \
    yv[yfull] = PASTECH(SIMD_FMADD_P_, ch)(av[((l1) - 1)], xv[((l1) - 1)], yv[yfull]);

/*
 * S_GEMV_N_FRINGE_0/1 — fringe row handler for real types.
 *
 * _FRINGE_0: no-op — MR divides m exactly, no leftover rows.
 * _FRINGE_1: handles the final partial register of MR_left elements (< EPR).
 *
 *   1. Masked-load beta*y (or zero) for only MR_left lanes.
 *   2. PP_ILOOP(NR, S_GEMV_N_A_FRINGE_LOOP, ...) — masked FMA across all NR columns.
 *   3. Masked-store the MR_left results back to ybuf.
 *
 * PP_IF(IS_FRINGE_OF(MR, ch), S_GEMV_N_FRINGE_, ...) selects _0 or _1 at compile time
 * based on whether MR % EPR leaves a remainder.
 *
 * "yfull" = MR/EPR = number of full row-registers already handled.
 * Uses caller-scope: ybuf, betav, beta_is_zero, xv[NR], yv[MR/EPR+1], MR_left, EPR, incy.
 */
#define S_GEMV_N_FRINGE_0(ch, NR, _unused, yfull) /* empty */
#define S_GEMV_N_FRINGE_1(ch, NR, _unused, yfull)                                                                           \
    yv[yfull] = beta_is_zero ? PASTECH(SIMD_SETZERO_P_, ch)()                                                               \
                             : PASTECH(SIMD_MUL_P_, ch)(betav, SIMD_MASKZ_LOADU_P_##ch((1 << (MR_left)) - 1, ybuf + (yfull) * (EPR))); \
    PP_ILOOP(NR, S_GEMV_N_A_FRINGE_LOOP, ch, yfull, 0)                                                                      \
    SIMD_MASK_STOREU_P_##ch( ybuf + (yfull) * (EPR) * incy, (1 << (MR_left)) - 1, yv[yfull]);

// #endregion

// ─────────────────────────────────────────────────────────────────────────────
// #region ST N kernel (complex and real micro-kernels)
// ─────────────────────────────────────────────────────────────────────────────

/*
 * N-direction GEMV micro-kernel algorithm overview
 * =================================================
 *
 * Computes:  y = beta * y + alpha * A * x
 *   where A is (m x n), x is (n,), y is (m,), A stored column-major (rs_a=1).
 *
 * Tile structure (N-direction = outer loop over columns, inner over rows):
 *
 *      A  (m x n)           x  (n,)
 *   ┌──────┬──────┬─┐     ┌────┐
 *   │ NR   │ NR   │.│     │ NR │  ← xv[0..NR-1] = alpha*x[0..NR-1]
 *   │ cols │ cols │.│  *  │ NR │
 *   ├──────┼──────┼─┤     │ .  │
 *   │  MR  │  MR  │.│     │    │
 *   │ rows │ rows │.│     └────┘
 *   ├──────┼──────┼─┤
 *   │  MR  │  MR  │.│     y  (m,)
 *   │ rows │ rows │.│  → ┌────┐
 *   ├──────┼──────┼─┤    │ MR │ ← yv[0] = EPR elements updated at once
 *   │fringe│fringe│.│    │ MR │
 *   └──────┴──────┴─┘    │ .  │
 *                         │frng│
 *                         └────┘
 *
 * Outer loop: strips of NR columns (i = 0..n/NR-1)
 *   ybuf resets to y start for each strip (GEMV is y += A_strip * x_strip)
 *   abuf = A + i*NR*cs_a
 *   xv[0..NR-1] = alpha * x[i*NR .. i*NR+NR-1]   (pre-scaled; complex: split re/im)
 *
 * Inner loop: tiles of MR rows (j = 0..m/MR-1)
 *   PP_LOOP(MR/EPR, Y_LOOP, ...)  — for each row-register k (1..MR/EPR):
 *     yv[0] = beta * y[row_k]  (load current y, or zero if beta==0)
 *     for each column l = 1..NR:
 *       yv[0] += A[row_k, col_l] * xv[l-1]     (FMADD)
 *     store yv[0] to y[row_k]
 *   PP_IF(IS_FRINGE_OF(MR,ch), FRINGE_): same for the last partial register
 *
 * Note on beta: after the first NR-column strip, beta becomes 1 (or {1,0} complex)
 * so subsequent strips accumulate rather than overwrite.
 *
 * Complex (z/c) types additionally:
 *   - xv is split into NR real-broadcast registers + NR imaginary-broadcast registers
 *   - accumulation uses two FMAs (real part and imaginary part separately)
 *   - PERMUTE + FMADDSEB combines the two at the end:
 *       result.re = yv_re.re - yv_im.im,  result.im = yv_re.im + yv_im.re
 *       (FMADDSUB: alternating +/- across adjacent SIMD lanes)
 */

#define GENTFUNC_NGEMV_z(ctype, ch, ch_s, MR, NR)                                                \
void GENTFUNC_GEMVNS(ctype, ch, MR, NR)                                                          \
     (                                                                                           \
       trans_t transa,                                                                           \
       conj_t conjx,                                                                             \
       dim_t m,                                                                                  \
       dim_t n,                                                                                  \
       ctype * alpha,                                                                            \
       ctype * a,                                                                                \
       inc_t rs_a,                                                                               \
       inc_t cs_a,                                                                               \
       ctype * x,                                                                                \
       inc_t incx,                                                                               \
       ctype * beta,                                                                             \
       ctype * y,                                                                                \
       inc_t incy,                                                                               \
       cntx_t * cntx                                                                             \
    )                                                                                            \
{                                                                                                \
    const dim_t EPR       = EPR_OF(ch);    /* complex elements per SIMD reg (matches MR units) */\
    ctype* restrict abuf = a;                              /* moving pointer into A          */  \
    ctype* restrict xbuf = x;                              /* moving pointer into x          */  \
    ctype* restrict ybuf = y;                              /* moving pointer into y          */  \
                                                                                                 \
    const dim_t yfull = MR / EPR;          /* # full SIMD registers to cover MR rows        */   \
    const dim_t MR_left    = m % EPR;      /* # leftover rows in fringe (< EPR)             */   \
    const dim_t yregs = 1;                 /* # y-accumulators per row-register (complex: 2)*/   \
    PASTECH(SIMD_VEC_, ch_s) xv[NR*2];    /* [0..NR-1]: alpha*x.re,  [NR..2NR-1]: alpha*x.im */  \
    PASTECH(SIMD_VEC_, ch_s) yv[1*2];     /* [0]: real acc,  [1]: imag acc (complex FMADDSUB)*/  \
    PASTECH(SIMD_VEC_, ch_s) av[1];        /* A tile scratch register                       */   \
    bool beta_is_zero = PASTEMAC(ch, eq0)( *beta );                                              \
    PASTECH(SIMD_VEC_, ch_s) beta_rv = PASTECH(SIMD_SET1_P_, ch_s)((*beta).real); /* broadcast beta.re */ \
    PASTECH(SIMD_VEC_, ch_s) beta_iv = PASTECH(SIMD_SET1_P_, ch_s)((*beta).imag); /* broadcast beta.im */ \
    ctype temp;                            /* scalar temp for complex horizontal sums         */ \
    /* When m < MR the caller still passes a full MR-wide buffer; pretend m == MR
     * so the inner macros compute at least one tile (they reference m/MR). */                   \
    if( (m/MR) < 1)                                                                              \
    {                                                                                            \
        m = MR;                                                                                  \
    }                                                                                            \
                                                                                                 \
    /* Outer loop: process NR columns at a time.
     * Each iteration: pre-scale x[i*NR..i*NR+NR-1] by alpha into xv,
     * then walk down all MR-row tiles of y, accumulating A*xv. */                               \
    for(dim_t i = 0; i < (n / NR); ++i)                                                          \
    {                                                                                            \
        ybuf = y;                          /* reset y pointer for each column strip            */\
        abuf = a + (NR * i * cs_a);        /* A pointer: skip i*NR columns                    */ \
        PP_LOOP(NR, Z_GEMV_N_ALPHA_SCALE, ch_s, NR, 0) /* fill xv[0..NR*2-1]                 */  \
                                                                                                 \
        /* Inner loop: MR-row tiles of y.
         * PP_LOOP(MR/EPR, ...): fully unrolled over row-registers (compile time).
         * PP_IF(IS_FRINGE_OF(MR,ch), ...): fringe handler if MR%EPR != 0. */                    \
        for( dim_t j = 0; j < (m / MR); ++j)                                                     \
        {                                                                                        \
            PP_LOOP(PP_DIV(MR, EPR_OF(ch)), Z_GEMV_N_Y_LOOP, ch_s, ch, NR)                       \
            PP_IF(IS_FRINGE_OF(MR, ch), Z_GEMV_N_FRINGE_, ch_s, ch, NR, PP_DIV(MR, EPR_OF(ch)))  \
            ybuf += (yfull * EPR + MR_left)*incy;                                                \
            abuf += (yfull * EPR + MR_left)*rs_a;                                                \
        }                                                                                        \
        xbuf += NR*incx;                                                                         \
        /* After the first column strip y already has the correct beta scaling applied.
         * Subsequent strips must accumulate (not overwrite), so beta becomes 1+0i. */           \
        beta_is_zero = false;                                                                    \
        beta_rv = PASTECH(SIMD_SET1_P_, ch_s)(1.0);                                              \
        beta_iv = PASTECH(SIMD_SETZERO_P_, ch_s)();                                              \
    }                                                                                            \
} // End of GENTFUNC_NGEMV_z

#define GENTFUNC_NGEMV_s(ctype, ch, ch_s, MR, NR)                                               \
void GENTFUNC_GEMVNS(ctype, ch, MR, NR)                                                         \
     (                                                                                          \
       trans_t transa, conj_t conjx, dim_t m, dim_t n, ctype * alpha,                           \
       ctype * a, inc_t rs_a, inc_t cs_a, ctype * x, inc_t incx, ctype * beta,                  \
       ctype * y, inc_t incy, cntx_t * cntx                                                     \
     )                                                                                          \
{                                                                                               \
    const dim_t EPR       = PASTECH(ELEM_PER_REG_, ch_s); /* elements per SIMD register      */ \
    ctype *restrict abuf = a;                              /* moving pointer into A          */ \
    ctype *restrict xbuf = x;                              /* moving pointer into x          */ \
    ctype *restrict ybuf = y;                              /* moving pointer into y          */ \
                                                                                                \
                                                                                                \
    const dim_t yfull_ct   = MR / EPR;             /* full row-registers in one MR tile      */ \
    const dim_t use_full   = (yfull_ct > 0) && ((MR) % EPR == 0); /* compile-time constant   */ \
    const dim_t mloop_full = use_full ? (m / MR) : 0;          /* # full MR-row tiles        */ \
    const dim_t mloop_epr  = use_full ? ((m % MR) / EPR) : (m / EPR); /* partial regs        */ \
    const dim_t MR_left    = m % EPR;              /* leftover elements below one full reg   */ \
    (void)MR_left;                                 /* silenced when IS_FRINGE_OF(MR,ch) == 0 */ \
                                                                                                \
    PASTECH(SIMD_VEC_, ch) xv[NR];                /* xv[j] = broadcast(alpha * x[i*NR+j])    */ \
    PASTECH(SIMD_VEC_, ch) yv[PP_DIV(MR, EPR_OF(ch)) + 1]; /* y accumulators (MR/EPR + fringe)*/\
    PASTECH(SIMD_VEC_, ch) av[NR];                /* A column scratch (one per NR column)     */\
    bool beta_is_zero = PASTEMAC(ch, eq0)( *beta );                                             \
    PASTECH(SIMD_VEC_, ch) betav = PASTECH(SIMD_SET1_P_, ch)(*beta); /* broadcast beta        */\
                                                                                                \
    /* Outer loop: NR-column strips of A.
     * Reset ybuf/abuf/xbuf to beginning of each strip, pre-scale x into xv. */                 \
    for (dim_t i = 0; i < (n / NR); ++i)                                                        \
    {                                                                                           \
        ybuf = y;                                                                               \
        abuf = a + (dim_t)(NR * i) * cs_a;         /* A column offset for strip i             */\
        xbuf = x + (dim_t)(NR * i) * incx;         /* x offset for strip i                   */ \
        PP_LOOP(NR, S_GEMV_N_ALPHA_SCALE, ch, 0, 0) /* fill xv[0..NR-1]                      */ \
                                                                                                \
        /* Full MR-row tiles: each iteration processes MR rows (PP_LOOP inside Y_LOOP
         * further unrolls MR/EPR row-registers at compile time). */                            \
        for (dim_t j = 0; j < mloop_full; ++j)                                                  \
        {                                                                                       \
            PP_LOOP(PP_DIV(MR, EPR_OF(ch)), S_GEMV_N_Y_LOOP, ch, NR, 0)                         \
            ybuf += MR * incy;                                                                  \
            abuf += MR * rs_a;                                                                  \
        }                                                                                       \
        /* EPR-row sub-tiles: handles (m%MR)/EPR full registers when MR < EPR or as spill. */   \
        for (dim_t j = 0; j < mloop_epr; ++j)                                                   \
        {                                                                                       \
            S_GEMV_N_Y_LOOP(ch, NR, 0, 1)                                                       \
            ybuf += EPR * incy;                                                                 \
            abuf += EPR * rs_a;                                                                 \
        }                                                                                       \
        /* Fringe: m%EPR leftover elements, handled with a masked load/store. */                \
        PP_IF(IS_FRINGE_OF(MR, ch), S_GEMV_N_FRINGE_, ch, NR, 0, 0)                             \
        /* Beta transitions to 1 after the first strip so later strips accumulate. */           \
        beta_is_zero = false;                                                                   \
        betav = PASTECH(SIMD_SET1_P_, ch)(1.0f);                                                \
    }                                                                                           \
} /* End of GENTFUNC_NGEMV_s */

/* scomplex (c) uses the same complex algorithm as dcomplex (z). */
#define GENTFUNC_NGEMV_c(ctype, ch, ch_s, MR, NR)  GENTFUNC_NGEMV_z(ctype, ch, ch_s, MR, NR)
/* double (d) uses the same real algorithm as float (s). */
#define GENTFUNC_NGEMV_d(ctype, ch, ch_s, MR, NR)  GENTFUNC_NGEMV_s(ctype, ch, ch_s, MR, NR)

// #endregion ST N kernel

// ─────────────────────────────────────────────────────────────────────────────
// #region ST M kernels (GEMV_M_* helpers and GENTFUNC_GEMV)
// ─────────────────────────────────────────────────────────────────────────────

/*
 * M-direction micro-kernel ("GEMV_M"):
 *
 * The M-direction kernel transposes the loop order compared to the N-direction kernel.
 * Instead of iterating over rows in the outer loop and columns in the inner loop,
 * here we iterate over rows (m) on the outside and process ALL columns in the inner loop.
 *
 * Algorithm for one MR-row tile:
 *   sv[0..MR/EPR] = 0                              (zero output accumulators)
 *   for col j = 0..n-1 (step NR, fully unrolled by PP_LOOP(NR, GEMV_M_COL_ACT, ...)):
 *     xv[col-j] = broadcast(alpha * x[j])
 *     for row_reg r = 1..MR/EPR (unrolled by PP_ILOOP(MR/EPR, GEMV_M_ROW_ACT, ...)):
 *       av[r-1] = A[row_reg_r, col_j]             (load EPR elements)
 *       sv[r-1] += av[r-1] * xv[col-j]            (accumulate)
 *     (fringe row if MR%EPR != 0: masked load+accumulate)
 *   if beta != 0: sv[r] = beta*y[r] + sv[r]        (scale and add existing y)
 *   else:         store sv[r] directly to y[r]
 *
 * This layout is used when the caller prefers to process full n-dimension
 * rows at once (all columns for a given row band), keeping A in cache.
 *
 * Variable naming:
 *   a_local  = pointer to current row-band of A  (a + i*rs_a)
 *   x_local  = pointer to current x element
 *   y_local  = pointer to current y row-band     (y + i*incy)
 *   j        = column index (loop variable in caller)
 *   xv[col]  = broadcast( alpha * x[col] )
 *   sv[r]    = SIMD accumulator for row-register r
 *   av[r]    = A row-register r scratch (overwritten each column)
 */

/* M-direction micro-kernel naming: bli_<ch>gemv_m_block_<MR>_<NR>. */
#define GENTFUNC_GEMVNMS(ctype, ch, MR, NR) \
    PASTEMAC4(ch, gemv_m_block_, MR, _, NR)

/*
 * GEMV_M_ROW_ACT — load one A row-register and accumulate into sv, for column col1.
 *
 *   av[row_reg1-1] = A[row_reg_{row_reg1}, col_{col1}]
 *               = load EPR elements starting at:
 *                   a_local + (j + col1 - 1)*cs_a + (row_reg1-1)*EPR*rs_a
 *   sv[row_reg1-1] += xv[col1-1] * av[row_reg1-1]     (FMA)
 *
 * "j" is the column base (runtime loop variable), "col1" is the unrolled offset (1..NR).
 * "row_reg1" ranges 1..MR/EPR (fully unrolled by PP_ILOOP).
 * Uses caller-scope: a_local, xv[NR], sv[MR/EPR+1], av[MR/EPR+1], j, EPR, rs_a, cs_a.
 */
#define GEMV_M_ROW_ACT(ch, col1, EPR, row_reg1)                                              \
    av[(row_reg1) - 1] =                                                                     \
        PASTECH(SIMD_LOADU_P_, ch)(                                                          \
            a_local + (j + (col1) - 1) * cs_a                                                \
                    + (((row_reg1) - 1) * (EPR)) * rs_a);                                    \
    sv[(row_reg1) - 1] =                                                                     \
        PASTECH(SIMD_FMADD_P_, ch)(                                                          \
            xv[(col1) - 1], av[(row_reg1) - 1], sv[(row_reg1) - 1]);

/*
 * GEMV_M_FRINGE_0/1 — partial (fringe) row-register handler for GEMV_M, per column col1.
 *
 * _FRINGE_0: no-op — MR is an exact multiple of EPR, no leftover rows.
 * _FRINGE_1: handles the final partial register when m % EPR != 0.
 *
 *   av[num_loads] = masked load of (m % EPR) elements from A:
 *       A[num_loads*EPR .. num_loads*EPR + (m%EPR) - 1, col_{col1}]
 *   mask = (1 << (m % EPR)) - 1    — lowest (m%EPR) bits
 *   sv[num_loads] += xv[col1-1] * av[num_loads]  (accumulate partial row)
 *
 * "num_loads" = MR/EPR = index of the fringe register (just past the full registers).
 * PP_IF(IS_FRINGE_##MR##_##ch, GEMV_M_FRINGE_, ...) selects _0 or _1 at compile time.
 * Uses caller-scope: a_local, xv[NR], sv[MR/EPR+1], av[MR/EPR+1], j, m,
 *                    ELEM_PER_REG, EPR, rs_a, cs_a.
 */
#define GEMV_M_FRINGE_0(ch, col1, EPR, num_loads)   /* no fringe register */
#define GEMV_M_FRINGE_1(ch, col1, EPR, num_loads)                                            \
    av[num_loads] =                                                                          \
        SIMD_MASKZ_LOADU_P_##ch(                                                             \
            (1 << ((  m  % ELEM_PER_REG ))) - 1,                                             \
            a_local + (j + (col1) - 1) * cs_a                                                \
                    + ((num_loads) * (EPR)) * rs_a);                                         \
    sv[num_loads] =                                                                          \
        PASTECH(SIMD_FMADD_P_, ch)(                                                          \
            xv[(col1) - 1], av[num_loads], sv[num_loads]);

/*
 * GEMV_M_COL_ACT — process one column (col1, 1-based) of A for the current MR-row tile.
 *
 * Steps:
 *   1. xv[col1-1] = broadcast( alpha * x[col1] )   — scale and broadcast this column's x
 *   2. x_local advances by incx                      — move to the next x element
 *   3. PP_ILOOP(MR/EPR, GEMV_M_ROW_ACT, ...)         — for each full row-register r:
 *        sv[r-1] += xv[col1-1] * A[row_reg_r, col1]
 *   4. PP_IF(IS_FRINGE_##MR##_##ch, GEMV_M_FRINGE_,...) — fringe row if needed
 *
 * Called by PP_LOOP(NR, GEMV_M_COL_ACT, ...) to process all NR columns per j-step.
 * "col1" is 1-based within the current NR-wide window; "j" is the window base.
 * Uses caller-scope: x_local, incx, alpha, xv[NR], sv[MR/EPR+1], a_local, j, m,
 *                    EPR, rs_a, cs_a, ELEM_PER_REG.
 */
#define GEMV_M_COL_ACT(ch, MR, EPR, col1)                                                    \
    xv[(col1) - 1] = PASTECH(SIMD_SET1_P_, ch)((*alpha) * (*x_local));                       \
    x_local += incx;                                                                         \
    PP_ILOOP(PP_DIV(MR, EPR), GEMV_M_ROW_ACT, ch, col1, EPR)                                 \
    PP_IF(IS_FRINGE_##MR##_##ch, GEMV_M_FRINGE_,                                             \
          ch, col1, EPR, PP_DIV(MR, EPR))

/*
 * GEMV_M_SV_ZERO / GEMV_M_SV_ZERO_FRINGE — zero out the sum-vector accumulators.
 *
 * Called once at the top of each MR-row tile before any column FMAs.
 *
 * GEMV_M_SV_ZERO: zeroes sv[reg1-1] for each full row-register reg1 = 1..MR/EPR.
 *   PP_LOOP(MR/EPR, GEMV_M_SV_ZERO, ...) zeroes all full registers.
 *
 * GEMV_M_SV_ZERO_FRINGE_0: no-op when MR%EPR == 0.
 * GEMV_M_SV_ZERO_FRINGE_1: zeroes sv[num_loads] = sv[MR/EPR] when MR%EPR != 0.
 *   PP_IF(IS_FRINGE_OF(MR,ch), GEMV_M_SV_ZERO_FRINGE_, ...) selects at compile time.
 */
#define GEMV_M_SV_ZERO(ch, MR, _unused, reg1)                                               \
    sv[(reg1) - 1] = PASTECH(SIMD_SETZERO_P_, ch)();

#define GEMV_M_SV_ZERO_FRINGE_0(ch, MR, num_loads)   /* no fringe register */
#define GEMV_M_SV_ZERO_FRINGE_1(ch, MR, num_loads)                                          \
    sv[num_loads] = PASTECH(SIMD_SETZERO_P_, ch)();

/*
 * GEMV_M_BETA_ACT — apply beta scaling and store one full row-register (reg1, 1-based).
 *
 * After all columns have been accumulated into sv[reg1-1], this macro writes
 * the result to y:
 *
 *   if beta != 0:
 *     y_old  = load y_local[(reg1-1)*EPR .. *EPR+EPR-1]
 *     sv[r]  = beta * y_old + sv[r]          (fmadd: beta * y + accumulated_sum)
 *   store sv[reg1-1] to y_local[(reg1-1)*incy*EPR]
 *
 * Note: incy is already verified == 1 at the caller level for the vectorized path,
 * so y_local[(reg1-1)*incy*EPR] == y_local[(reg1-1)*EPR].
 * Uses caller-scope: y_local, sv[MR/EPR+1], xv[NR+MR], beta_, beta, EPR, incy.
 */
#define GEMV_M_BETA_ACT(ch, MR, EPR, reg1)                                                   \
    if ( !bli_deq0( *beta ) )                                                                \
    {                                                                                        \
        xv[(reg1) - 1] =                                                                     \
            PASTECH(SIMD_LOADU_P_, ch)(y_local + ((reg1) - 1) * incy * (EPR));               \
        sv[(reg1) - 1] =                                                                     \
            PASTECH(SIMD_FMADD_P_, ch)(xv[(reg1) - 1], beta_, sv[(reg1) - 1]);               \
    }                                                                                        \
    PASTECH(SIMD_STOREU_P_, ch)(                                                             \
        y_local + ((reg1) - 1) * incy * (EPR), sv[(reg1) - 1]);

/*
 * GEMV_M_BETA_FRINGE_0/1 — beta scaling and masked store for the fringe row-register.
 *
 * _FRINGE_0: no-op when MR%EPR == 0.
 * _FRINGE_1: handles the final partial register of (m%EPR) elements.
 *
 *   if beta != 0:
 *     y_old  = masked-load (m%EPR) elements from y_local[num_loads*incy*EPR]
 *     sv[num_loads] = beta * y_old + sv[num_loads]
 *   masked-store sv[num_loads] back (only m%EPR lanes written).
 *
 * PP_IF(IS_FRINGE_OF(MR,ch), GEMV_M_BETA_FRINGE_, ...) selects at compile time.
 * Uses caller-scope: y_local, sv[MR/EPR+1], xv[NR+MR], beta_, beta, m,
 *                    ELEM_PER_REG, EPR, incy.
 */
#define GEMV_M_BETA_FRINGE_0(ch, MR, EPR, num_loads)   /* no fringe register */
#define GEMV_M_BETA_FRINGE_1(ch, MR, EPR, num_loads)                                         \
    if ( !bli_deq0( *beta ) )                                                                \
    {                                                                                        \
        xv[num_loads] =                                                                      \
            SIMD_MASKZ_LOADU_P_##ch(                                                         \
                (1 << ((  m  % ELEM_PER_REG ))) - 1,                                         \
                y_local + (num_loads) * incy * (EPR));                                       \
        sv[num_loads] =                                                                      \
            PASTECH(SIMD_FMADD_P_, ch)(xv[num_loads], beta_, sv[num_loads]);                 \
    }                                                                                        \
    SIMD_MASK_STOREU_P_##ch(                                                                 \
        y_local + (num_loads) * incy * (EPR),                                                \
        (1 << ((  m  % ELEM_PER_REG ))) - 1, sv[num_loads]);

/*
 * GEMV_M_BETA_EMIT — emit the full beta-scale-and-store sequence for one MR-row tile.
 *
 * Expands to:
 *   beta_ = broadcast(*beta)
 *   PP_LOOP(MR/EPR, GEMV_M_BETA_ACT, ...)       — full registers
 *   PP_IF(IS_FRINGE_OF(MR,ch), GEMV_M_BETA_FRINGE_, ...) — fringe if needed
 *
 * This is used directly in GENTFUNC_GEMV after the column loop finishes.
 */
#define GEMV_M_BETA_EMIT(ch, MR)                                                             \
    PASTECH(SIMD_VEC_, ch) beta_ = PASTECH(SIMD_SET1_P_, ch)( *beta );                       \
    PP_LOOP(PP_DIV(MR, EPR_OF(ch)), GEMV_M_BETA_ACT, ch, MR, EPR_OF(ch))                     \
    PP_IF(IS_FRINGE_OF(MR, ch), GEMV_M_BETA_FRINGE_,                                         \
          ch, MR, EPR_OF(ch), PP_DIV(MR, EPR_OF(ch)))

/*
 * GENTFUNC_GEMV — M-direction micro-kernel body.
 *
 * Generated function name: bli_<ch>gemv_m_block_<MR>_<NR>
 *
 * Computes: y[0..m-1] += alpha * A[0..m-1, 0..n-1] * x[0..n-1]  (beta handled per-row)
 *
 * Memory layout assumed: A is column-major (rs_a=1), x and y are stride-incx/incy vectors.
 *
 * Algorithm (outer loop over m in steps of MR):
 *
 *   for i = 0, MR, 2*MR, ... (m rows):
 *     a_local = A + i*rs_a              (pointer to this row band)
 *     y_local = y + i*incy
 *     sv[0..MR/EPR] = 0                 (zero accumulators)
 *     for j = 0, NR, 2*NR, ..., (n-1) (column window, step NR):
 *       PP_LOOP(NR, GEMV_M_COL_ACT, ...) — process NR columns:
 *         for col = 1..NR:
 *           xv[col-1] = broadcast(alpha * x[j+col-1])
 *           for row_reg = 1..MR/EPR:
 *             av[row_reg-1] = A[row_reg, j+col-1]   (EPR elements)
 *             sv[row_reg-1] += xv[col-1] * av[row_reg-1]
 *           (fringe: masked accumulate for leftover rows)
 *     beta-scale sv and store to y_local
 *
 * Note: the inner j-loop over columns is the same code path for every tile (n is
 * a runtime variable, loop not unrolled), but the inner PP_LOOP(NR, ...) over the
 * NR-wide window IS fully unrolled at compile time.
 */
#define GENTFUNC_GEMV(ctype, ch, MR, NR)                                                         \
static void GENTFUNC_GEMVNMS(ctype, ch, MR, NR)                                                  \
     (                                                                                           \
       trans_t transa,                                                                           \
       conj_t  conjx,                                                                            \
       dim_t   m,                                                                                \
       dim_t   n,                                                                                \
       ctype* alpha,                                                                             \
       ctype* a, inc_t rs_a, inc_t cs_a,                                                         \
       ctype* x, inc_t incx,                                                                     \
       ctype* beta,                                                                              \
       ctype* y, inc_t incy,                                                                     \
       cntx_t* cntx                                                                              \
     )                                                                                           \
{                                                                                                \
    const dim_t ELEM_PER_REG      = PASTECH(ELEM_PER_REG_, ch);                                  \
    const dim_t num_loads_per_MR  = (  MR / ELEM_PER_REG );                                      \
                                                                                                 \
    ctype *x_temp = x;                                                                           \
                                                                                                 \
    PASTECH(SIMD_VEC_, ch) av[num_loads_per_MR + 1];                                             \
    PASTECH(SIMD_VEC_, ch) sv[num_loads_per_MR + 1];                                             \
    PASTECH(SIMD_VEC_, ch) xv[NR + MR];                                                          \
                                                                                                 \
    for(dim_t i = 0; i < m; i += MR)                                                             \
    {                                                                                            \
        ctype* a_local = a + (0 * cs_a) + i * rs_a;                                              \
        ctype* y_local = y + i * incy;                                                           \
        ctype* x_local = x_temp;                                                                 \
        PP_LOOP(PP_DIV(MR, EPR_OF(ch)), GEMV_M_SV_ZERO, ch, MR, 0)                               \
        PP_IF(IS_FRINGE_OF(MR, ch), GEMV_M_SV_ZERO_FRINGE_,                                      \
              ch, MR, PP_DIV(MR, EPR_OF(ch)))                                                    \
        UNROLL_LOOP_FULL(NR)                                                                     \
        for(dim_t j = 0; j < n; j += NR)                                                         \
        {                                                                                        \
            PP_LOOP(NR, GEMV_M_COL_ACT, ch, MR, EPR_OF(ch))                                      \
        }                                                                                        \
        PASTECH(SIMD_VEC_, ch) beta_ = PASTECH(SIMD_SET1_P_, ch)( *beta );                       \
        PP_LOOP(PP_DIV(MR, EPR_OF(ch)), GEMV_M_BETA_ACT, ch, MR, EPR_OF(ch))                     \
        PP_IF(IS_FRINGE_OF(MR, ch), GEMV_M_BETA_FRINGE_,                                         \
              ch, MR, EPR_OF(ch), PP_DIV(MR, EPR_OF(ch)))                                        \
    }                                                                                            \
} // End of GENTFUNC_GEMV

// #endregion ST M kernels

// ─────────────────────────────────────────────────────────────────────────────
// #region MT M kernels / SCALE_BETA helpers
// ─────────────────────────────────────────────────────────────────────────────

/*
 * SCALE_BETA_n / SCALE_BETA_m — optionally pre-scale y by beta before parallel GEMV.
 *
 * When the M-dimension is split across threads (N-direction kernel), each thread
 * writes to a disjoint row range of y, so beta can be applied serially before
 * the parallel section using a fast scalar-vector multiply (scalv).
 *
 * SCALE_BETA_n(ch): calls the architecture-tuned scalv to compute y = beta * y.
 *                  Used by the N-direction MT wrapper before spawning threads.
 * SCALE_BETA_m(ch): empty — M-direction kernels handle beta internally per tile.
 */
#define SCALE_BETA_n(ch) \
    PASTEMAC(ch, scalv_zen4_int)                                                               \
    (                                                                                          \
        BLIS_NO_CONJUGATE,                                                                     \
        m,                                                                                     \
        beta,                                                                                  \
        y, incy,                                                                               \
        cntx                                                                                   \
    );
#define SCALE_BETA_m(ch)

// #endregion

// ─────────────────────────────────────────────────────────────────────────────
// #region Helper token-paste macros for generated function names
// ─────────────────────────────────────────────────────────────────────────────

/*
 * GEMV_N_FNAME / GEMV_M_FNAME: resolve ST caller name.
 * Examples with GEMV_ARCH_SUFFIX=zen_int:
 *   GEMV_N_FNAME(d,20,4) → bli_dgemv_n_zen_int_20x4
 *   GEMV_M_FNAME(d,20,4) → bli_dgemv_m_zen_int_20x4
 */
#define GEMV_N_FNAME(ch, MR, NR) \
    PASTEMAC2(ch, gemv_n_, PASTECH4(GEMV_ARCH_SUFFIX, _, MR, x, NR))
#define GEMV_M_FNAME(ch, MR, NR) \
    PASTEMAC2(ch, gemv_m_, PASTECH4(GEMV_ARCH_SUFFIX, _, MR, x, NR))
#define GEMV_N_FNAME_MT(ch, MR, NR) \
    PASTEMAC3(ch, gemv_n_, PASTECH4(GEMV_ARCH_SUFFIX, _, MR, x, NR), _mt)
#define GEMV_N_ENTRY(ch) \
    PASTEMAC(ch, PASTECH(gemv_n_, GEMV_ARCH_SUFFIX))

// #endregion

// ─────────────────────────────────────────────────────────────────────────────
// #region ST caller, MT wrapper, entry-point, and GENERATE_KERNEL
// ─────────────────────────────────────────────────────────────────────────────

/*
 * GENT_GEMV_CALLER — single-threaded GEMV dispatcher (N or M direction).
 *
 * Generated function: bli_<ch>gemv_<direction>_<GEMV_ARCH_SUFFIX>_<MR>x<NR>
 * Example: bli_dgemv_n_zen4_int_40x8
 *
 * This function splits the (m x n) problem into up to four regions based on
 * how evenly m and n divide by MR and NR:
 *
 *        n/NR full tiles    n%NR leftover
 *       ┌──────────────────┬──────────────┐
 *  m/MR │ tile [0][0]      │ tile [0][n%] │
 * full  │ full kernel      │ fringe-NR    │
 * tiles ├──────────────────┼──────────────┤
 *  m%MR │ tile [m%][0]     │ tile [m%][n%]│
 * left  │ fringe-MR        │ both fringe  │
 *       └──────────────────┴──────────────┘
 *
 * The kernel table ker_fp_[m_idx][n_idx] maps each region to the correctly-sized
 * pre-instantiated micro-kernel.  m_idx and n_idx are computed at runtime:
 *   n_idx = (NR - n%NR) % NR          (0 means full NR tile)
 *   m_idx = (m%MR)/EPR + is_m_left*(MR/EPR)  (encodes how many EPR-wide fringe regs)
 *
 * The beta applied to adjacent regions:
 *   - First region: uses caller's beta.
 *   - Later regions that share the same y rows: use beta=1 to accumulate, not overwrite.
 *
 * Falls back to the reference GEMV (bli_<ch>gemv_zen_ref) for non-unit strides or
 * transposed inputs that the optimized kernel does not handle.
 */
#define GENT_GEMV_CALLER(ctype, ch, MR, NR, direction)                                           \
void PASTEMAC2(ch, gemv_, PASTECH(PASTECH(direction, _), PASTECH4(GEMV_ARCH_SUFFIX, _, MR, x, NR))) \
     (                                                                                           \
       trans_t transa,                                                                           \
       conj_t  conjx,                                                                            \
       dim_t   m,                                                                                \
       dim_t   n,                                                                                \
       ctype* alpha,                                                                             \
       ctype* a, inc_t rs_a, inc_t cs_a,                                                         \
       ctype* x, inc_t incx,                                                                     \
       ctype* beta,                                                                              \
       ctype* y, inc_t incy,                                                                     \
       cntx_t* cntx                                                                              \
     )                                                                                           \
{                                                                                                \
     AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)                                                \
    if ( (( rs_a != 1 ) && ( cs_a != 1 )) || transa != BLIS_NO_TRANSPOSE || incy != 1)           \
    {                                                                                            \
        PASTEMAC(ch, gemv_zen_ref)                                                               \
        (                                                                                        \
          transa,                                                                                \
          m,                                                                                     \
          n,                                                                                     \
          alpha,                                                                                 \
          a, rs_a, cs_a,                                                                         \
          x, incx,                                                                               \
          beta,                                                                                  \
          y, incy,                                                                               \
          NULL                                                                                   \
        );                                                                                       \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)                                              \
        return;                                                                                  \
    }                                                                                            \
                                                                                                 \
    const dim_t elem_per_reg = PASTECH(ELEM_PER_REG_, ch);                                       \
                                                                                                 \
    bool is_m_left = ((m % MR) % elem_per_reg) >= 1 ? true : false;                              \
    dim_t m_idx = (m % MR) / elem_per_reg +                                                      \
                  (is_m_left * MR/elem_per_reg);                                                 \
    dim_t n_idx = (NR - (n % NR)) % NR;                                                          \
    ctype one   = PASTECH(ONE_,ch);                                                              \
                                                                                                 \
    if(m >= MR && n >= NR)                                                                       \
    {                                                                                            \
        PASTECH4(ch, PASTECH2(gemv_, direction, _ker_fp_), MR, _, NR)[0][0]                      \
        (                                                                                        \
            transa, conjx,                                                                       \
            ((dim_t)( m / MR )) * MR, ((dim_t)( n / NR )) * NR,                                  \
            alpha, a, rs_a, cs_a, x, incx, beta, y, incy, cntx                                   \
        );                                                                                       \
    }                                                                                            \
                                                                                                 \
    if (m >= MR && n % NR)                                                                       \
    {                                                                                            \
        PASTECH4(ch, PASTECH2(gemv_, direction, _ker_fp_), MR, _, NR)[0][n_idx]                  \
        (                                                                                        \
            transa, conjx,                                                                       \
            ((dim_t)( m / MR )) * MR, n % NR,                                                    \
            alpha,                                                                               \
            a + (((dim_t)( n / NR )) * NR * cs_a), rs_a, cs_a,                                   \
            x + (((dim_t)( n / NR )) * NR * incx), incx,                                         \
            n > NR ? &one : beta,                                                                \
            y, incy, cntx                                                                        \
        );                                                                                       \
    }                                                                                            \
                                                                                                 \
    if (n >= NR && m % MR)                                                                       \
    {                                                                                            \
        PASTECH4(ch, PASTECH2(gemv_, direction, _ker_fp_), MR, _, NR)[m_idx][0]                  \
        (                                                                                        \
            transa, conjx,                                                                       \
            m % MR, ((dim_t)( n / NR )) * NR,                                                    \
            alpha,                                                                               \
            a + (((dim_t)( m / MR )) * MR * rs_a), rs_a, cs_a,                                   \
            x, incx, beta,                                                                       \
            y + ((dim_t)( m / MR )) * MR * incy, incy, cntx                                      \
        );                                                                                       \
    }                                                                                            \
                                                                                                 \
    if (m % MR && n % NR)                                                                        \
    {                                                                                            \
        PASTECH4(ch, PASTECH2(gemv_, direction, _ker_fp_), MR, _, NR)[m_idx][n_idx]              \
        (                                                                                        \
            transa, conjx,                                                                       \
            m % MR, n % NR,                                                                      \
            alpha,                                                                               \
            a + (((dim_t)( m / MR )) * MR * rs_a) + ((dim_t)( n / NR )) * NR * cs_a, rs_a, cs_a, \
            x + (((dim_t)( n / NR )) * NR * incx), incx,                                         \
            n > NR ? &one : beta,                                                                \
            y + ((dim_t)( m / MR )) * MR * incy, incy, cntx                                      \
        );                                                                                       \
    }                                                                                            \
}

/*
 * GENT_N_GEMV_M_DIM — multi-threaded GEMV-N wrapper (partitions along M).
 *
 * Generated function: bli_<ch>gemv_n_<GEMV_ARCH_SUFFIX>_<MR>x<NR>_mt
 * Example: bli_dgemv_n_zen4_int_40x8_mt
 *
 * Threading strategy: split the M dimension (output rows) across threads.
 * Each thread gets a contiguous slice of rows [thread_start .. thread_start+job):
 *   - Thread t gets: a[thread_start*rs_a], y[thread_start*incy]
 *   - All threads share the same x (read-only)
 *   - Threads write to disjoint y rows → no synchronization needed
 *
 * This is safe because y[i] += sum_j(A[i,j] * x[j]) — row i of y only depends
 * on row i of A and the full x, so rows are fully independent.
 *
 * Falls back to the reference kernel (like the ST version) for non-unit strides.
 * Falls back to single-threaded if nt==1 (avoids OMP overhead for small problems).
 *
 * Thread count nt is queried from bli_nthreads_l2() which reads the BLIS runtime.
 */
#define GENT_N_GEMV_M_DIM(ctype, ch, MR, NR)                                                 \
void GEMV_N_FNAME_MT(ch, MR, NR)                                                             \
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
    if ( (( rs_a != 1 ) && ( cs_a != 1 )) || transa != BLIS_NO_TRANSPOSE || incy != 1)       \
    {                                                                                        \
        PASTEMAC(ch, gemv_zen_ref)                                                           \
        (                                                                                    \
          transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL                   \
        );                                                                                   \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)                                          \
        return;                                                                              \
    }                                                                                        \
    dim_t nt = 1;                                                                            \
    bli_nthreads_l2                                                                          \
    (                                                                                        \
        BLIS_GEMV_KER, PASTEMAC(ch,type), BLIS_NO_TRANSPOSE,                                 \
        bli_arch_query_id_internal(), m, n, &nt                                              \
    );                                                                                       \
                                                                                             \
    if (nt == 1)                                                                             \
    {                                                                                        \
        GEMV_N_FNAME(ch, MR, NR)                                                             \
        ( transa, conjx, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, cntx );         \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)                                          \
        return;                                                                              \
    }                                                                                        \
    _Pragma("omp parallel num_threads(nt)")                                                  \
    {                                                                                        \
        dim_t job_per_thread = m;                                                            \
        dim_t thread_start   = 0;                                                            \
        const dim_t tid     = omp_get_thread_num();                                          \
        const dim_t nt_real = omp_get_num_threads();                                         \
        bli_thread_vector_partition( m, nt_real, &thread_start, &job_per_thread, tid );      \
        GEMV_N_FNAME(ch, MR, NR)                                                             \
        (                                                                                    \
            transa, conjx,                                                                   \
            job_per_thread, n,                                                               \
            alpha,                                                                           \
            a +  thread_start * rs_a, rs_a, cs_a,                                            \
            x , incx, beta,                                                                  \
            y + thread_start * incy, incy, cntx                                              \
        );                                                                                   \
    }                                                                                        \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4);                                             \
}

/*
 * ST_KERNEL / MT_KERNEL / SHOULD_CALL_ST — kernel selection helpers.
 *
 * ST_KERNEL: resolves to the single-threaded N caller (GEMV_N_FNAME).
 * MT_KERNEL: resolves to the multi-threaded wrapper (GEMV_N_FNAME_MT) when
 *            OpenMP is enabled, or falls back to ST_KERNEL otherwise.
 *
 * SHOULD_CALL_ST(ch): true when the problem is small enough that multi-threading
 *   overhead would outweigh the benefit.  The threshold (m*n < 1800) was tuned
 *   empirically; below this, the OMP fork/join cost dominates.
 *   When OpenMP is disabled, always returns 1 (always single-threaded).
 *
 * CALC_SIZE: declares the `size = m*n` variable needed for SHOULD_CALL_ST.
 *   Empty when OpenMP is off (size is not needed).
 *
 * MT_KERNEL_SIGNATURE: expands to the full MT wrapper function body
 *   (GENT_N_GEMV_M_DIM) when OpenMP is enabled; empty otherwise.
 */
#define ST_KERNEL(ch, MR, NR) GEMV_N_FNAME(ch, MR, NR)

#ifdef BLIS_ENABLE_OPENMP
    #define CALC_SIZE dim_t size = m * n;
    #define SHOULD_CALL_ST_s (size < 1800)
    #define SHOULD_CALL_ST_d (size < 1800)
    #define SHOULD_CALL_ST_c (size < 1800)
    #define SHOULD_CALL_ST_z (size < 1800)
    #define SHOULD_CALL_ST(ch) PASTECH(SHOULD_CALL_ST_, ch)
    #define MT_KERNEL(ch, MR, NR) GEMV_N_FNAME_MT(ch, MR, NR)
    #define MT_KERNEL_SIGNATURE(ctype, ch, MR, NR) GENT_N_GEMV_M_DIM(ctype, ch, MR, NR)
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
 * GENERATE_ROOT_KERNEL — public GEMV-N entry-point.
 *
 * Generated function: bli_<ch>gemv_n_<GEMV_ARCH_SUFFIX>
 * Example: bli_dgemv_n_zen4_int
 *
 * This is the function registered in the BLIS context (cntx) and called by
 * the BLIS framework.  It handles all the "messy" pre-processing before
 * dispatching to the fast ST or MT kernel:
 *
 *  1. alpha == 0 fast path:
 *       y = beta * y  (just scale y, skip the matrix multiply entirely)
 *
 *  2. conjx handling (complex types only):
 *       If x needs conjugation, copy-and-conjugate x into a temporary contiguous
 *       buffer (x_temp, incx=1) via the BLIS copyv kernel.
 *       This lets the kernel always assume incx=1 and no conjugation.
 *
 *  3. incy != 1 handling:
 *       The vectorized N-kernel requires incy=1 (contiguous y).
 *       If incy != 1, create a temporary y buffer (y_temp, incy=1), then:
 *         a. If beta != 0: pack y into y_temp (scaled by beta via copyv).
 *         b. After the kernel runs: unpack y_temp back to y (via copyv).
 *
 *  4. ST vs MT dispatch:
 *       CALC_SIZE computes size = m*n.
 *       SHOULD_CALL_ST(ch) is true when the problem is small → use ST kernel.
 *       Otherwise → use MT kernel (multi-threaded, if OpenMP enabled).
 *
 *  5. Call ker_ft (the chosen kernel) and release any temporary buffers.
 */
#define GENERATE_ROOT_KERNEL(ctype, ch, MR, NR)                                              \
void GEMV_N_ENTRY(ch) (trans_t transa, conj_t conjx, dim_t m, dim_t n,                       \
                       ctype *alpha, ctype *a, inc_t rs_a, inc_t cs_a,                       \
                       ctype *x, inc_t incx, ctype *beta, ctype *y,                          \
                       inc_t incy, cntx_t *cntx) {                                           \
    void (*ker_ft)(trans_t, conj_t, dim_t, dim_t, ctype *, ctype *, inc_t, inc_t,            \
                 ctype *, inc_t, ctype *, ctype *, inc_t, cntx_t *) = NULL;                  \
    rntm_t  rntm;                                                                            \
    mem_t   mem_bufY;                                                                        \
    mem_t   mem_bufX;                                                                        \
    inc_t   temp_incy = incy;                                                                \
    inc_t   temp_incx = incx;                                                                \
    ctype*  y_temp = y;                                                                      \
    ctype*  x_temp = x;                                                                      \
    bool is_y_temp_buf_created = false;                                                      \
    PASTECH(ch,copyv_ker_ft)   copyv_kr_ptr = NULL;                                          \
                                                                                             \
    if (PASTEMAC(ch,eq0)(*alpha))                                                            \
    {                                                                                        \
        PASTECH(ch, scalv_ker_ft)   scalv_kr_ptr = NULL;                                     \
        scalv_kr_ptr = bli_cntx_get_l1v_ker_dt(PASTEMAC(ch,type), BLIS_SCALV_KER, cntx);     \
        scalv_kr_ptr                                                                         \
        (                                                                                    \
          BLIS_NO_CONJUGATE,                                                                 \
          m,                                                                                 \
          beta,                                                                              \
          y_temp, temp_incy,                                                                 \
          cntx                                                                               \
        );                                                                                   \
        return;                                                                              \
    }                                                                                        \
    const bool need_conj = bli_is_conj( conjx ) &&                                           \
                           bli_is_complex( PASTEMAC(ch,type) );                              \
    if ( need_conj )                                                                         \
    {                                                                                        \
        mem_bufX.pblk.buf = NULL;   mem_bufX.pblk.block_size = 0;                            \
        mem_bufX.buf_type = 0;      mem_bufX.size = 0;                                       \
        mem_bufX.pool = NULL;                                                                \
                                                                                             \
        bli_rntm_init_from_global( &rntm );                                                  \
        bli_rntm_set_num_threads_only( 1, &rntm );                                           \
        bli_pba_rntm_set_pba( &rntm );                                                       \
                                                                                             \
        size_t buffer_size_x = n * sizeof(ctype);                                            \
        bli_pba_acquire_m                                                                    \
        (                                                                                    \
          &rntm, buffer_size_x, BLIS_BUFFER_FOR_B_PANEL, &mem_bufX                           \
        );                                                                                   \
                                                                                             \
        if ( bli_mem_is_alloc( &mem_bufX ) )                                                 \
        {                                                                                    \
            x_temp = bli_mem_buffer(&mem_bufX);                                              \
            temp_incx = 1;                                                                   \
            if(cntx == NULL) cntx = bli_gks_query_cntx();                                    \
                                                                                             \
            copyv_kr_ptr = bli_cntx_get_l1v_ker_dt(PASTEMAC(ch,type), BLIS_COPYV_KER, cntx); \
            copyv_kr_ptr                                                                     \
            (                                                                                \
              BLIS_CONJUGATE, n, x, incx, x_temp, temp_incx, cntx                            \
            );                                                                               \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            if(cntx == NULL) cntx = bli_gks_query_cntx();                                    \
            /* Call non fused code path which internally uses axpy kernel*/                  \
            PASTEMAC(ch, gemv_unb_var2)( transa, conjx, m, n, alpha, a, rs_a, cs_a,          \
                                         x, incx, beta, y, incy, cntx );                     \
            return;                                                                          \
        }                                                                                    \
    }                                                                                        \
                                                                                             \
    if (incy != 1)                                                                           \
    {                                                                                        \
        mem_bufY.pblk.buf = NULL;   mem_bufY.pblk.block_size = 0;                            \
        mem_bufY.buf_type = 0;      mem_bufY.size = 0;                                       \
        mem_bufY.pool = NULL;                                                                \
        if ( !need_conj )                                                                    \
        {                                                                                    \
            bli_rntm_init_from_global( &rntm );                                              \
            bli_rntm_set_num_threads_only( 1, &rntm );                                       \
            bli_pba_rntm_set_pba( &rntm );                                                   \
        }                                                                                    \
        size_t buffer_size = m * sizeof(ctype);                                              \
        bli_pba_acquire_m                                                                    \
        (                                                                                    \
          &rntm, buffer_size, BLIS_BUFFER_FOR_B_PANEL, &mem_bufY                             \
        );                                                                                   \
        if ( bli_mem_is_alloc( &mem_bufY ) )                                                 \
        {                                                                                    \
            y_temp = bli_mem_buffer(&mem_bufY);                                              \
            temp_incy = 1;                                                                   \
            if(cntx == NULL) cntx = bli_gks_query_cntx();                                    \
            if( copyv_kr_ptr == NULL )                                                       \
                copyv_kr_ptr = bli_cntx_get_l1v_ker_dt(PASTEMAC(ch,type), BLIS_COPYV_KER, cntx); \
            if ( !PASTEMAC(ch, eq0)(*beta) )                                                 \
            {                                                                                \
                copyv_kr_ptr                                                                 \
                (                                                                            \
                  BLIS_NO_CONJUGATE, m, y, incy, y_temp, temp_incy, cntx                     \
                );                                                                           \
            }                                                                                \
            is_y_temp_buf_created = TRUE;                                                    \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            PASTEMAC(ch, gemv_zen_ref)( transa, m, n, alpha, a, rs_a, cs_a,                  \
                                        x_temp, temp_incx, beta, y, incy, NULL );             \
            if ( x_temp != x ) bli_pba_release(&rntm , &mem_bufX);                           \
            return;                                                                          \
        }                                                                                    \
    }                                                                                        \
                                                                                             \
    CALC_SIZE                                                                                \
    if (SHOULD_CALL_ST(ch)) {                                                                \
        ker_ft = ST_KERNEL(ch, MR, NR);                                                      \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        ker_ft = MT_KERNEL(ch, MR, NR);                                                      \
    }                                                                                        \
    ker_ft                                                                                   \
    (                                                                                        \
      transa, BLIS_NO_CONJUGATE,                                                             \
      m, n, alpha, a, rs_a, cs_a, x_temp, temp_incx, beta, y_temp, temp_incy, cntx           \
    );                                                                                       \
    if (is_y_temp_buf_created)                                                               \
    {                                                                                        \
        copyv_kr_ptr                                                                         \
        (                                                                                    \
          BLIS_NO_CONJUGATE, m, y_temp, temp_incy, y, incy, cntx                             \
        );                                                                                   \
        bli_pba_release(&rntm , &mem_bufY);                                                  \
    }                                                                                        \
    if ( x_temp != x )                                                                       \
    {                                                                                        \
        bli_pba_release(&rntm , &mem_bufX);                                                  \
    }                                                                                        \
}

/*
 * GENERATE_KERNEL — master macro that instantiates the full N-direction kernel family.
 *
 * One call to GENERATE_KERNEL(ctype, ch, MR_N, NR_N, MR_M, NR_M) emits four things:
 *
 *   1. GENERATE_<ch>_KERNELS_<MR_N>_N(...)
 *        → All micro-kernel function bodies (one per (MR_tile, NR_tile) combination)
 *          plus the static 2D function-pointer dispatch table ker_fp_[m_idx][n_idx].
 *
 *   2. GENT_GEMV_CALLER(ctype, ch, MR_N, NR_N, n)
 *        → The tiled single-threaded dispatcher (4-region tile dispatch described above).
 *
 *   3. MT_KERNEL_SIGNATURE(ctype, ch, MR_N, NR_N)
 *        → The multi-threaded wrapper (OpenMP parallel region, M-dimension split).
 *          Empty when OpenMP is not enabled.
 *
 *   4. GENERATE_ROOT_KERNEL(ctype, ch, MR_N, NR_N)
 *        → The public entry-point (conjugation packing, incy buffering, ST/MT dispatch).
 *
 * MR_M and NR_M specify the M-direction kernel tile sizes (reserved for future use;
 * complex types currently use N-kernel only and do not generate an M-direction kernel).
 *
 * Typical call from a .c shim:
 *   GENERATE_KERNEL(dcomplex, z, 10, 5, 10, 5)
 */
#define GENERATE_KERNEL(ctype, ch, MR_N, NR_N, MR_M, NR_M)                                  \
    PASTECH4(GENERATE_,ch,_KERNELS_,MR_N,_N)(ctype, ch, MR_N, NR_N);                        \
    GENT_GEMV_CALLER(ctype, ch, MR_N, NR_N, n);                                              \
    MT_KERNEL_SIGNATURE(ctype, ch, MR_N, NR_N);                                              \
    GENERATE_ROOT_KERNEL(ctype, ch, MR_N, NR_N);

// #endregion

// =============================================================================
// #region Real-type (s/d) control layer  —  MOVED to plain C
// =============================================================================
//
// The single-thread size dispatch (_st), the multi-threaded row/column split
// wrappers (_mt_Mdiv / _mt_Ndiv), and the public no-transpose entry-point for
// the REAL types (s/d) used to be generated here by four function-like macros
// (GENERATE_gemv_n_int_st / _mt_Mdiv / _mt_Ndiv / _int).
//
// That control logic is now written out as ordinary, debuggable, type-specific
// C directly in each arch shim (bli_gemv_n_zen{,4}_int.c) — no macros, no vtable,
// no shared void* core. The GEMV_N_CTRL_* thresholds above are the only shared
// tuning knobs. The public symbol names are unchanged.
//
// The COMPLEX types (c/z) are unaffected: they still use the GENERATE_KERNEL /
// GENERATE_ROOT_KERNEL / GENT_N_GEMV_M_DIM macros defined above.
//
// #endregion

#endif /* BLI_GEMV_N_IMPL_H */
