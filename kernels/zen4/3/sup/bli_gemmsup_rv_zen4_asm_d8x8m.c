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

/*
    8x8 lower triangular DGEMMT kernel
    This kernels expects M <= 8;

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________
    |*-------|
    |**------|
    |***-----|
    |****----|
    |*****---|
    |******--|
    |*******-|
    |********|
     ________
*/
void bli_dgemmsup_rv_zen4_asm_8x8m_lower_mle8
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
    dim_t m_rem = m % 8;
    double *a_curr = a, *b_curr, *c = c_;
    dim_t i = 0;
    if (m == 8)
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

/*
    8x8 Upper triangular DGEMMT kernel
    This kernels expects M <= 8;

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________
    |********|
    |-*******|
    |--******|
    |---*****|
    |----****|
    |-----***|
    |------**|
    |-------*|
     ________
*/
void bli_dgemmsup_rv_zen4_asm_8x8m_upper_mle8
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
    // dim_t m_main = m / 8;
    dim_t m_rem = m % 8;
    double *a_curr = a, *b_curr, *c = c_;
    dim_t i = 0;
    // for (i = 0; i < m_main; i++)
    if (m == 8)
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

/*
    The diagonal pattern repeats after every block of
    size 24x24, therefore three 24x8 kernels are added to
    make sure that entire 24x24 block gets covered.

    Diagram for Lower traingular 24x24 block

     lower_0   lower_1  lower_2
     ________ ________ ________
    |*-------|--------|--------|
    |**------|--------|--------|
    |***-----|--------|--------|
    |****----|--------|--------|
    |*****---|--------|--------|
    |******--|--------|--------|
    |*******-|--------|--------|
    |********|--------|--------|
     ________ ________ ________
    |********|*-------|--------|
    |********|**------|--------|
    |********|***-----|--------|
    |********|****----|--------|
    |********|*****---|--------|
    |********|******--|--------|
    |********|*******-|--------|
    |********|********|--------|
     ________ ________ ________
    |********|********|*-------|
    |********|********|**------|
    |********|********|***-----|
    |********|********|****----|
    |********|********|*****---|
    |********|********|******--|
    |********|********|*******-|
    |********|********|********|
     ________ ________ ________
*/

/*
    24x8 Lower traingular kernel, which computes the
    first 24x8 micro panel of the 24x24 repeating block

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________
    |*-------|          <
    |**------|          |
    |***-----|          |
    |****----|  intial 8x8 triangular panel
    |*****---|          |
    |******--|          |
    |*******-|          >
     ________
    |********|         <
    |********|         |
    |********|         |
    |********|         |
    |********|
    |********|
    |********|   16x8 full GEMM panel
    |********|
    |********|
    |********|
    |********|
    |********|         |
    |********|         |
    |********|         |
    |********|         >
     ________
*/
void bli_dgemmsup_rv_zen4_asm_24x8m_lower_0
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
    dim_t m_diag; // m for traingular kernel
    dim_t m_full; // m for full GEMM kernel
    // if m <= 8 then only diagonal region needs to be
    // computed, therefor set m_full to 0.
    if (m <= 8)
    {
        // if m <= 8, m_diag = 8 , m_full = 0
        m_diag = m;
        m_full = 0;
    }
    // if m > 8, then full diagonal(m=8) needs to be computed
    // and remaning m (m - 8) will be computed by DGEMM SUP kernel.
    else
    {
        m_diag = 8;
        m_full = m - 8;
    }

    // since the 8x8m kernel is row major,
    // call row major 8x8m upper diagonal kernel after
    // inducing transpose to solve column major lower
    // triangular GEMM
    bli_dgemmsup_rv_zen4_asm_8x8m_upper_mle8
    (
        conjb,
        conja,
        n,
        m_diag,
        k,
        alpha,
        b, cs_b, rs_b,
        a, cs_a, rs_a,
        beta,
        c_, cs_c, rs_c,
        data,
        cntx
    );

    // call full GEMM kernel for remaning parts of matrix
    bli_dgemmsup_rv_zen4_asm_24x8m
    (
        conja,
        conjb,
        m_full,
        n,
        k,
        alpha,
        a + (rs_a * m_diag), rs_a, cs_a,
        b, rs_b, cs_b,
        beta,
        c_ + (rs_c * m_diag), rs_c, cs_c,
        data,
        cntx
    );
}

/*
    24x8 Lower traingular kernel, which computes the
    second 24x8 micro panel of the 24x24 repeating block

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________
    |--------|          <
    |--------|          |
    |--------|          |
    |--------|  intial empty 8x8 panel
    |--------|          |
    |--------|          |
    |--------|          >
     ________
    |*-------|          <
    |**------|          |
    |***-----|          |
    |****----|   8x8 triangular panel
    |*****---|          |
    |******--|          |
    |*******-|          >
     ________
    |********|         <
    |********|         |
    |********|         |
    |********|         |
    |********|   8x8 full GEMM panel
    |********|         |
    |********|         |
    |********|         >
     ________
*/
void bli_dgemmsup_rv_zen4_asm_24x8m_lower_1
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
    dim_t m_diag; // m for traingular kernel
    dim_t m_full; // m for full GEMM kenrel
    
    // if m is less than 8, then only empty region is computed
    // therefore set m_diag and m_full to 0.
    if (m <= 8)
    {
        m_diag = 0;
        m_full = 0;
    }
    // if m_diag is less than 16, then only empty region and triangular
    // region needs to be computed, therefor set m_full to 0.
    else if ( m <= 16)
    {
        m_diag = m - 8;
        m_full = 0;
    }
    else
    {
        m_diag = 8;
        m_full = m - 16;
    }

    // since the 8x8m kernel is row major,
    // call row major 8x8m upper diagonal kernel after
    // inducing transpose to solve column major lower
    // triangular GEMM
    bli_dgemmsup_rv_zen4_asm_8x8m_upper_mle8
    (
        conjb,
        conja,
        n,
        m_diag,
        k,
        alpha,
        b, cs_b, rs_b,
        a + (rs_a * 8), cs_a, rs_a,
        beta,
        c_ + (rs_c * 8), cs_c, rs_c,
        data,
        cntx
    );

    // call full GEMM kernel for remaning parts of matrix
    bli_dgemmsup_rv_zen4_asm_24x8m
    (
        conja,
        conjb,
        m_full,
        n,
        k,
        alpha,
        a + (rs_a*(8+m_diag)), rs_a, cs_a,
        b, rs_b, cs_b,
        beta,
        c_ + (rs_c * (8+m_diag)), rs_c, cs_c,
        data,
        cntx
    );
}

/*
    24x8 Lower traingular kernel, which computes the
    third 24x8 micro panel of the 24x24 repeating block

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________
    |--------|          <
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|  intial empty 16x8 panel
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          |
    |--------|          >
     ________
    |*-------|          <
    |**------|          |
    |***-----|          |
    |****----|   8x8 triangular panel
    |*****---|          |
    |******--|          |
    |*******-|          >
     ________
*/
void bli_dgemmsup_rv_zen4_asm_24x8m_lower_2
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
    dim_t m_diag; // m for traingular kernel
    dim_t m_full; // m for full GEMM kernel

    // if m <= 16, only empty region needs to be computed.
    if (m <= 16)
    {
        m_diag = 0;
        m_full = 0;
    }

    // if m <= 24, initial 16 rows are empty and there is no full
    // gemm region, therefore m_diag = 0
    else if (m <= 24)
    {
        m_diag = m - 16;
        m_full = 0; 
    }
    else
    {
        m_diag = 8;
        m_full = m - 24; // m - (16(empty) + 8(diagonal))
    }

    // since the 8x8m kernel is row major,
    // call row major 8x8m upper diagonal kernel after
    // inducing transpose to solve column major lower
    // triangular GEMM
    bli_dgemmsup_rv_zen4_asm_8x8m_upper_mle8
    (
        conjb,
        conja,
        n,
        m_diag,
        k,
        alpha,
        b, cs_b, rs_b,
        a + (rs_a * 16), cs_a, rs_a,
        beta,
        c_ + (rs_c * 16), cs_c, rs_c,
        data,
        cntx
    );

    // call full GEMM kernel for remaning parts of matrix
    bli_dgemmsup_rv_zen4_asm_24x8m
    (
        conja,
        conjb,
        m_full,
        n,
        k,
        alpha,
        a + (rs_a*(16+m_diag)), rs_a, cs_a,
        b, rs_b, cs_b,
        beta,
        c_ + (rs_c * (16+m_diag)), rs_c, cs_c,
        data,
        cntx
    );
}

/*
    The diagonal pattern repeats after every block of
    size 24x24, therefore three 24x8 kernels are added to
    make sure that entire 24x24 block gets covered.

    Diagram for Upper traingular 24x24 block

     upper_0   upper_1  upper_2
     ________ ________ ________
    |********|********|********|
    |-*******|********|********|
    |--******|********|********|
    |---*****|********|********|
    |----****|********|********|
    |-----***|********|********|
    |------**|********|********|
    |-------*|********|********|
     ________ ________ ________
    |--------|********|********|
    |--------|-*******|********|
    |--------|--******|********|
    |--------|---*****|********|
    |--------|----****|********|
    |--------|-----***|********|
    |--------|------**|********|
    |--------|-------*|********|
     ________ ________ ________
    |--------|--------|********|
    |--------|--------|-*******|
    |--------|--------|--******|
    |--------|--------|---*****|
    |--------|--------|----****|
    |--------|--------|-----***|
    |--------|--------|------**|
    |--------|--------|-------*|
     ________ ________ ________

*/

/*
    24x8 Upper traingular kernel, which computes the
    first 24x8 micro panel of the 24x24 repeating block

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________ 
    |********|          <
    |-*******|          |
    |--******|          |
    |---*****| intial 8x8 triangular block
    |----****|          |
    |-----***|          |
    |------**|          |
    |-------*|          >
     ________ 
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
     ________ 
*/
void bli_dgemmsup_rv_zen4_asm_24x8m_upper_0
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
    dim_t m_diag; // m for traingular kernel
    dim_t m_full; // m for full GEMM kenrel
    
    // if m <= 8, then only diagonal region exists
    // therefore m_full = 0
    if (m <= 8)
    {
        m_diag = m;
        m_full = 0;
    }

    // if m >= 8, then initial 8 rows are computed
    // by DGEMM SUP kernel, and last 16 rows are empty
    else if (m <= 24)
    {
        m_diag = 8;
        m_full = 0;
    }
    // if m > 24, then compute inital 24 rows with existing
    // logic and use DGEMM SUP kernel for remainder.
    else
    {
        m_diag = 8;
        m_full = m - 24; // m - (16(empty) + 8(diagonal))
    }

    // call full GEMM kernel for intial part of matrix
    bli_dgemmsup_rv_zen4_asm_24x8m
    (
        conja,
        conjb,
        m_full,
        n,
        k,
        alpha,
        a, rs_a, cs_a,
        b, rs_b, cs_b,
        beta,
        c_, rs_c, cs_c,
        data,
        cntx
    );

    // since the 8x8m kernel is row major,
    // call row major 8x8m lower diagonal kernel after
    // inducing transpose to solve column major upper
    // triangular GEMM
    bli_dgemmsup_rv_zen4_asm_8x8m_lower_mle8
    (
        conjb,
        conja,
        n,
        m_diag,
        k,
        alpha,
        b, cs_b, rs_b,
        a + (rs_a*m_full), cs_a, rs_a,
        beta,
        c_ + (rs_c * m_full), cs_c, rs_c,
        data,
        cntx
    );
}

/*
    24x8 Upper traingular kernel, which computes the
    second 24x8 micro panel of the 24x24 repeating block

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________ 
    |********|          <
    |********|          |
    |********|          |
    |********|    8x8 full GEMM block
    |********|          |
    |********|          |
    |********|          |
    |********|          >
     ________ 
    |********|          <
    |-*******|          |
    |--******|          |
    |---*****|   8x8 triangular block
    |----****|          |
    |-----***|          |
    |------**|          |
    |-------*|          >
     ________ 
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
    |--------|
     ________ 
*/
void bli_dgemmsup_rv_zen4_asm_24x8m_upper_1
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
    dim_t m_diag, m_full;
    if (m <= 8)
    {
        m_diag = m;
        m_full = 0;
    }
    else if (m <= 16)
    {
        m_diag = 8;
        m_full = 0;
    }
    else
    {
        m_diag = 8;
        m_full = m - 16;
    }

    // call full GEMM kernel for intial part of matrix
    bli_dgemmsup_rv_zen4_asm_24x8m
    (
        conja,
        conjb,
        m_full,
        n,
        k,
        alpha,
        a, rs_a, cs_a,
        b, rs_b, cs_b,
        beta,
        c_, rs_c, cs_c,
        data,
        cntx
    );

    // since the 8x8m kernel is row major,
    // call row major 8x8m lower diagonal kernel after
    // inducing transpose to solve column major upper
    // triangular GEMM
    bli_dgemmsup_rv_zen4_asm_8x8m_lower_mle8
    (
        conjb,
        conja,
        n,
        m_diag,
        k,
        alpha,
        b, cs_b, rs_b,
        a + (rs_a*m_full), cs_a, rs_a,
        beta,
        c_ + (rs_c * m_full), cs_c, rs_c,
        data,
        cntx
    );
}

/*
    24x8 Upper traingular kernel, which computes the
    second 24x8 micro panel of the 24x24 repeating block

    Region marked by '*' is computed by this kernel
    Region marked by '-' is not computed.
     ________ 
    |********|          <
    |********|          |
    |********|          |
    |********|          |
    |********|          |
    |********|          |
    |********|          |
    |********|    16x8 full GEMM block
    |********|          |
    |********|          |
    |********|          |
    |********|          |
    |********|          |
    |********|          |
    |********|          |
    |********|          >
     ________
    |********|          <
    |-*******|          |
    |--******|          |
    |---*****|   8x8 triangular block
    |----****|          |
    |-----***|          |
    |------**|          |
    |-------*|          >
     ________
*/
void bli_dgemmsup_rv_zen4_asm_24x8m_upper_2
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
    dim_t m_diag, m_full;
    if (m <= 8)
    {
        m_diag = m;
        m_full = 0;
    }
    else
    {
        m_diag = 8;
        m_full = m - 8;
    }

    // call full GEMM kernel for intial part of matrix
    bli_dgemmsup_rv_zen4_asm_24x8m
    (
        conja,
        conjb,
        m_full,
        n,
        k,
        alpha,
        a, rs_a, cs_a,
        b, rs_b, cs_b,
        beta,
        c_, rs_c, cs_c,
        data,
        cntx
    );

    // since the 8x8m kernel is row major,
    // call row major 8x8m lower diagonal kernel after
    // inducing transpose to solve column major upper
    // triangular GEMM
    bli_dgemmsup_rv_zen4_asm_8x8m_lower_mle8
    (
        conjb,
        conja,
        n,
        m_diag,
        k,
        alpha,
        b, cs_b, rs_b,
        a + (rs_a*m_full), cs_a, rs_a,
        beta,
        c_ + (rs_c * m_full), cs_c, rs_c,
        data,
        cntx
    );
}
