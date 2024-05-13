/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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


#include "blis.h"
#include "immintrin.h"

#if defined __clang__
    #define UNROLL_LOOP()      _Pragma("clang loop unroll_count(4)")
    /*
    *   in clang, unroll_count(4) generates inefficient
    *   code compared to unroll(full) when loopCount = 4.
    */
    #define UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
#elif defined __GNUC__
    #define UNROLL_LOOP()      _Pragma("GCC unroll 4")
    #define UNROLL_LOOP_FULL() _Pragma("GCC unroll 8")
#else
    #define UNROLL_LOOP()
    #define UNROLL_LOOP_FULL()
#endif

#define ZERO_REGISTERS() \
    c_reg[0] = _mm512_setzero_pd(); \
    c_reg[1] = _mm512_setzero_pd(); \
    c_reg[2] = _mm512_setzero_pd(); \
    c_reg[3] = _mm512_setzero_pd(); \
    c_reg[4] = _mm512_setzero_pd(); \
    c_reg[5] = _mm512_setzero_pd(); \
    c_reg[6] = _mm512_setzero_pd(); \
    c_reg[7] = _mm512_setzero_pd(); \

#define TRANSPOSE_8x8() \
    a_reg[0] = _mm512_unpacklo_pd(c_reg[0], c_reg[1]); \
    a_reg[1] = _mm512_unpacklo_pd(c_reg[2], c_reg[3]); \
    a_reg[2] = _mm512_unpacklo_pd(c_reg[4], c_reg[5]); \
    a_reg[3] = _mm512_unpacklo_pd(c_reg[6], c_reg[7]); \
    a_reg[4] = _mm512_unpackhi_pd(c_reg[0], c_reg[1]); \
    a_reg[5] = _mm512_unpackhi_pd(c_reg[2], c_reg[3]); \
    a_reg[6] = _mm512_unpackhi_pd(c_reg[4], c_reg[5]); \
    a_reg[7] = _mm512_unpackhi_pd(c_reg[6], c_reg[7]); \
    /*Stage2*/ \
    b_reg[0] = _mm512_shuffle_f64x2(a_reg[0], a_reg[1], 0b10001000); \
    b_reg[1] = _mm512_shuffle_f64x2(a_reg[2], a_reg[3], 0b10001000); \
    /*Stage3  1,5*/ \
    c_reg[0] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b10001000); \
    c_reg[4] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b11011101); \
    /*Stage2*/ \
    b_reg[0] = _mm512_shuffle_f64x2(a_reg[0], a_reg[1], 0b11011101); \
    b_reg[1] = _mm512_shuffle_f64x2(a_reg[2], a_reg[3], 0b11011101); \
    /*Stage3  3,7*/ \
    c_reg[2] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b10001000); \
    c_reg[6] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b11011101); \
    /*Stage2*/ \
    b_reg[0] = _mm512_shuffle_f64x2(a_reg[4], a_reg[5], 0b10001000); \
    b_reg[1] = _mm512_shuffle_f64x2(a_reg[6], a_reg[7], 0b10001000); \
    /*Stage3  2,6*/ \
    c_reg[1] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b10001000); \
    c_reg[5] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b11011101); \
    /*Stage2*/ \
    b_reg[0] = _mm512_shuffle_f64x2(a_reg[4], a_reg[5], 0b11011101); \
    b_reg[1] = _mm512_shuffle_f64x2(a_reg[6], a_reg[7], 0b11011101); \
    /*Stage3  4,8*/ \
    c_reg[3] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b10001000); \
    c_reg[7] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b11011101);

#define GEMM_MxN(M, N) \
    UNROLL_LOOP() \
    for (dim_t j = 0; j < k; ++j) \
    { \
        b_reg[0] = _mm512_mask_loadu_pd(c_reg[0], mask_n, b_curr); \
        b_curr += rs_b; \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            a_reg[ii] = _mm512_set1_pd(*( a_curr + (rs_a * ii) )); \
            c_reg[ii] = _mm512_fmadd_pd(a_reg[ii] , b_reg[0], c_reg[ii]); \
        } \
        a_curr += cs_a; \
    } \


#define STORE_COL(M, N) \
    if ((*beta) == 0) { STORE_COL_BZ(M, N) } \
    else \
    { \
        TRANSPOSE_8x8() \
        b_reg[0] = _mm512_set1_pd(*(alpha)); \
        b_reg[1] = _mm512_set1_pd(*(beta)); \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < N; ++ii) \
        { \
            c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
            a_reg[ii] = _mm512_mask_loadu_pd(c_reg[ii], (1 << (M)) - 1, c + cs_c * ii); \
            c_reg[ii] = _mm512_fmadd_pd(b_reg[1], a_reg[ii], c_reg[ii]); \
            _mm512_mask_storeu_pd(c + cs_c * ii, (1 << (M)) - 1, c_reg[ii]); \
        } \
    } \

#define STORE_COL_BZ(M, N) \
    TRANSPOSE_8x8() \
    b_reg[0] = _mm512_set1_pd(*(alpha)); \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < N; ++ii) \
    { \
        c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
        _mm512_mask_storeu_pd(c + cs_c * ii, (1 << (M)) - 1, c_reg[ii]); \
    } \

#define STORE_COL_LOWER(M, N) \
    if ((*beta) == 0) { STORE_COL_LOWER_BZ(M, N) } \
    else \
    { \
        TRANSPOSE_8x8() \
        b_reg[0] = _mm512_set1_pd(*(alpha)); \
        b_reg[1] = _mm512_set1_pd(*(beta)); \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < N; ++ii) \
        { \
            c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
            a_reg[ii] = _mm512_mask_loadu_pd(c_reg[ii], ((1 << (n_rem - ii)) -1) << ii, c + cs_c * ii); \
            c_reg[ii] = _mm512_fmadd_pd(b_reg[1], a_reg[ii], c_reg[ii]); \
            _mm512_mask_storeu_pd(c + cs_c * ii, ((1 << (n_rem - ii)) -1) << ii, c_reg[ii]); \
        } \
    } \

#define STORE_COL_LOWER_BZ(M, N) \
    TRANSPOSE_8x8() \
    b_reg[0] = _mm512_set1_pd(*(alpha)); \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < N; ++ii) \
    { \
        c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
        _mm512_mask_storeu_pd(c + cs_c * ii, ((1 << (n_rem - ii)) -1) << ii, c_reg[ii]); \
    } \

#define STORE_COL_UPPER(M, N) \
    if ((*beta) == 0) { STORE_COL_UPPER_BZ(M, N) } \
    else \
    { \
        TRANSPOSE_8x8() \
        b_reg[0] = _mm512_set1_pd(*(alpha)); \
        b_reg[1] = _mm512_set1_pd(*(beta)); \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < N; ++ii) \
        { \
            c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
            a_reg[ii] = _mm512_mask_loadu_pd(c_reg[ii], (1 << (ii+1)) - 1, c + cs_c * ii); \
            c_reg[ii] = _mm512_fmadd_pd(b_reg[1], a_reg[ii], c_reg[ii]); \
            _mm512_mask_storeu_pd(c + cs_c * ii, (1 << (ii+1)) - 1, c_reg[ii]); \
        } \
    } \

#define STORE_COL_UPPER_BZ(M, N) \
    TRANSPOSE_8x8() \
    b_reg[0] = _mm512_set1_pd(*(alpha)); \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < N; ++ii) \
    { \
        c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
        _mm512_mask_storeu_pd(c + cs_c * ii, (1 << (ii+1)) - 1, c_reg[ii]); \
    } \


#define STORE_ROW(M, N) \
    if ((*beta) == 0) { STORE_ROW_BZ(M, N) } \
    else \
    { \
        b_reg[0] = _mm512_set1_pd(*(alpha)); \
        b_reg[1] = _mm512_set1_pd(*(beta)); \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
            a_reg[ii] = _mm512_mask_loadu_pd(c_reg[ii], mask_n, c + (rs_c * ii)); \
            c_reg[ii] = _mm512_fmadd_pd(b_reg[1], a_reg[ii], c_reg[ii]); \
            _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
        } \
    } \

#define STORE_ROW_BZ(M, N) \
    b_reg[0] = _mm512_set1_pd(*(alpha)); \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
        c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
        _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
    } \

#define STORE_ROW_LOWER(M, N) \
    if ((*beta) == 0) { STORE_ROW_LOWER_BZ(M, N) } \
    else \
    { \
        b_reg[0] = _mm512_set1_pd(*(alpha)); \
        b_reg[1] = _mm512_set1_pd(*(beta)); \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
            a_reg[ii] = _mm512_mask_loadu_pd(c_reg[ii], (1 << (ii+1)) - 1, c + (rs_c * ii)); \
            c_reg[ii] = _mm512_fmadd_pd(b_reg[1], a_reg[ii], c_reg[ii]); \
            _mm512_mask_storeu_pd(c + (rs_c * ii), (1 << (ii+1)) - 1, c_reg[ii]); \
        } \
    } \

#define STORE_ROW_LOWER_BZ(M, N) \
    b_reg[0] = _mm512_set1_pd(*(alpha)); \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
        c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
        _mm512_mask_storeu_pd(c + (rs_c * ii), (1 << (ii+1)) - 1, c_reg[ii]); \
    } \

#define STORE_ROW_UPPER(M, N) \
    if ((*beta) == 0) { STORE_ROW_UPPER_BZ(M, N) } \
    else \
    { \
        b_reg[0] = _mm512_set1_pd(*(alpha)); \
        b_reg[1] = _mm512_set1_pd(*(beta)); \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
            a_reg[ii] = _mm512_mask_loadu_pd(c_reg[ii], ((1 << (n_rem - ii)) - 1) << ii, c + (rs_c * ii)); \
            c_reg[ii] = _mm512_fmadd_pd(b_reg[1], a_reg[ii], c_reg[ii]); \
            _mm512_mask_storeu_pd(c + (rs_c * ii), ((1 << (n_rem - ii)) - 1) << ii, c_reg[ii]); \
        } \
    } \

#define STORE_ROW_UPPER_BZ(M, N) \
    b_reg[0] = _mm512_set1_pd(*(alpha)); \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
        c_reg[ii] = _mm512_mul_pd(c_reg[ii], b_reg[0]); \
        _mm512_mask_storeu_pd(c + (rs_c * ii), ((1 << (n_rem - ii)) - 1) << ii, c_reg[ii]); \
    } \

#define MAIN_LOOP(M) \
    n_rem = n % 8; \
    if (n_rem == 0) n_rem = 8; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem)) - 1; \
    GEMM_MxN(M, n_rem) \
    if (cs_c == 1) { STORE_ROW(M, n_rem) } \
    else           { STORE_COL(M, n_rem) } \
    c += 8 * rs_c; \

#define MAIN_LOOP_LOWER_DIAG(M) \
    n_rem = n % 8; \
    if (n_rem == 0) n_rem = 8; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem)) - 1; \
    GEMM_MxN(M, n_rem) \
    if (cs_c == 1) { STORE_ROW_LOWER(M, n_rem) } \
    else           { STORE_COL_LOWER(M, n_rem) } \
    c += 8 * rs_c; \

#define MAIN_LOOP_UPPER_DIAG(M) \
    n_rem = n % 8; \
    if (n_rem == 0) n_rem = 8; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem)) - 1; \
    GEMM_MxN(M, n_rem) \
    if (cs_c == 1) { STORE_ROW_UPPER(M, n_rem) } \
    else           { STORE_COL_UPPER(M, n_rem) } \
    c += 8 * rs_c; \

void bli_dgemmsup_rv_zen4_asm_8x8m
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        double*    restrict alpha,
        double*    restrict a, inc_t rs_a, inc_t cs_a,
        double*    restrict b, inc_t rs_b, inc_t cs_b,
        double*    restrict beta,
        double*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[8];
    __m512d a_reg[8];
    __m512d b_reg[2];
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 8;
    dim_t m_rem = m % 8;
    double *a_curr = a, *b_curr, *c = c_;
    dim_t i =0;
    for (i = 0; i < m_main; i++)
    {
       MAIN_LOOP(8);
    }
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP(1); break;
        case 2:
            MAIN_LOOP(2); break;
        case 3:
            MAIN_LOOP(3); break;
        case 4:
            MAIN_LOOP(4); break;
        case 5:
            MAIN_LOOP(5); break;
        case 6:
            MAIN_LOOP(6); break;
        case 7:
            MAIN_LOOP(7); break;
    }
}

void bli_dgemmsup_rv_zen4_asm_8x8m_lower
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        double*    restrict alpha,
        double*    restrict a, inc_t rs_a, inc_t cs_a,
        double*    restrict b, inc_t rs_b, inc_t cs_b,
        double*    restrict beta,
        double*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[8];
    __m512d a_reg[8];
    __m512d b_reg[2];
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 8;
    dim_t m_rem = m % 8;
    double *a_curr = a, *b_curr, *c = c_;
    dim_t i = 0;
    for (i = 0; i < m_main; i++)
    {
        MAIN_LOOP_LOWER_DIAG(8);
    }
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_LOWER_DIAG(1); break;
        case 2:
            MAIN_LOOP_LOWER_DIAG(2); break;
        case 3:
            MAIN_LOOP_LOWER_DIAG(3); break;
        case 4:
            MAIN_LOOP_LOWER_DIAG(4); break;
        case 5:
            MAIN_LOOP_LOWER_DIAG(5); break;
        case 6:
            MAIN_LOOP_LOWER_DIAG(6); break;
        case 7:
            MAIN_LOOP_LOWER_DIAG(7); break;
    }
}

void bli_dgemmsup_rv_zen4_asm_8x8m_upper
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        double*    restrict alpha,
        double*    restrict a, inc_t rs_a, inc_t cs_a,
        double*    restrict b, inc_t rs_b, inc_t cs_b,
        double*    restrict beta,
        double*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[8];
    __m512d a_reg[8];
    __m512d b_reg[2];
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 8;
    dim_t m_rem = m % 8;
    double *a_curr = a, *b_curr, *c = c_;
    dim_t i = 0;
    for (i = 0; i < m_main; i++)
    {
        MAIN_LOOP_UPPER_DIAG(8);
    }
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_UPPER_DIAG(1); break;
        case 2:
            MAIN_LOOP_UPPER_DIAG(2); break;
        case 3:
            MAIN_LOOP_UPPER_DIAG(3); break;
        case 4:
            MAIN_LOOP_UPPER_DIAG(4); break;
        case 5:
            MAIN_LOOP_UPPER_DIAG(5); break;
        case 6:
            MAIN_LOOP_UPPER_DIAG(6); break;
        case 7:
            MAIN_LOOP_UPPER_DIAG(7); break;
    }
}