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

#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
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
    #define UNROLL_LOOP_FULL() _Pragma("GCC unroll 4")
#else
    #define UNROLL_LOOP()
    #define UNROLL_LOOP_FULL()
#endif

/*
* Multiply dcomplex vector with a dcomplex scalar(S)
* reg_a      ->      input dcomplex vector
* reg_r      ->      vector with S->real broadcasted
* reg_i      ->      vector with S->imag broadcasted
* output     ->      vector where output is stored
*
* t_reg[5] contains [1, 1, 1, 1, 1, 1, 1, 1]
*
* (a + ib) (c + id) = (ac - bd) + i(ad + bc)
* here reg_a = [a1, b1, a2, b2, a3, b3, a4, b4]
*      reg_r = [c,  c,  c,  c,  c,  c,  c,  c ]
*      reg_i = [d,  d,  d,  d,  d,  d,  d,  d ]
*/
#define MULTIPLY_COMPLEX( reg_a, reg_r, reg_i, output )  \
    t_reg[3]    = _mm512_permute_pd(reg_a, 0x55);        \
    /* t_reg[3] = [b1, a1, b2, a2, b3, a3, b4, a4] */    \
    output      = _mm512_mul_pd(reg_a, reg_r);           \
    /* output   = c * [a1, b1, a2, b2, a3, b3, a4, b4]*/ \
    t_reg[3]    = _mm512_mul_pd(t_reg[3], reg_i);        \
    /* t_reg[3] = d * [b1, a1, b2, a2, b3, a3, b4, a4]*/ \
    output      = _mm512_fmaddsub_pd(t_reg[5], output, t_reg[3]); \
    /* output   = [a1c-b1d, a1d+b1c, a2c-b2d, a2d+b2c, ......]*/  \

/*
* Divide dcomplex vector with a dcomplex scalar(S)
* reg_a      ->      input dcomplex vector
* addr       ->      address of scalar
* output is stored in reg_a
*
* t_teg[4] contains [-1, -1, -1, -1, -1, -1, -1, -1]
* t_reg[5] contains [ 1,  1,  1,  1,  1,  1,  1,  1]
*
* (xr + i xi)/(ar + i ai) = 
*          (xrar + xiai)/(ar^2 + ai^2) +
*         i(xiar - xrai)/(ar^2 + ai^2)
*
* instead if dividing by ar^2 + ai^2, we divide
* by ar/maxabs(ar, ai) * ar + ai / maxabs(ar, ai) * ai
* in order to reduce the possibility of underflow
* when c or d are very small
* 
* here reg_a = [a1, b1, a2, b2, a3, b3, a4, b4]
*/
#define DIVIDE_COMPLEX( reg_a, addr )               \
    g_double[2] = bli_fmaxabs(addr->real, addr->imag);/*s*/    \
    g_double[0] = addr->real / g_double[2];/*ar/s*/            \
    g_double[1] = addr->imag / g_double[2];/*ai/s*/            \
    t_reg[0]    = _mm512_set1_pd(g_double[0]);/*ar/s*/         \
    t_reg[1]    = _mm512_set1_pd(g_double[1]);/*ai/s*/         \
    g_double[2] = (g_double[0] * addr->real) +                 \
                  (g_double[1] * addr->imag);                  \
                   /*(ar/s * ar) +(ai/s * ai)*/                \
    t_reg[3]   = _mm512_permute_pd(reg_a, 0x55);               \
    /*t_reg[3] = [xi,xr,xi,xr....] */                          \
    reg_a      = _mm512_mul_pd(reg_a, t_reg[0]);               \
    /* reg_a   = ar/s * [xr, xi, xr, xi ....]*/                \
    t_reg[3]   = _mm512_mul_pd(t_reg[3], t_reg[1]);            \
    /*t_reg[3] = ai/s * [xi,xr,xi,xr........] */               \
    t_reg[3]   = _mm512_mul_pd(t_reg[4], t_reg[3]);            \
    /*t_reg[3] = -ai/s * [xi,xr,xi,xr........] */              \
    t_reg[1]   = _mm512_set1_pd(g_double[2]);                  \
    /*t_reg[1] = [(ar/s * ar) +(ai/s * ai), ...] */            \
    reg_a      = _mm512_fmaddsub_pd(t_reg[5], reg_a, t_reg[3]);\
    /*reg_a    = [a1c+b1d, b1c-a1d, a2c+b2d, b2c-a2d, ....]*/  \
    reg_a      = _mm512_div_pd(reg_a, t_reg[1]);               \

// Zero the registors used for gemm accumulation
#define ZERO_REGISTERS() \
    c_reg[0] = _mm512_setzero_pd(); \
    c_reg[1] = _mm512_setzero_pd(); \
    c_reg[2] = _mm512_setzero_pd(); \
    c_reg[3] = _mm512_setzero_pd(); \
    c_reg[4] = _mm512_setzero_pd(); \
    c_reg[5] = _mm512_setzero_pd(); \
    c_reg[6] = _mm512_setzero_pd(); \
    c_reg[7] = _mm512_setzero_pd(); \
    t_reg[5] = _mm512_setzero_pd(); \
    b_reg[0] = _mm512_setzero_pd(); \
    b_reg[1] = _mm512_setzero_pd(); \
    b_reg[2] = _mm512_setzero_pd(); \
    b_reg[3] = _mm512_setzero_pd(); \

/* Initialize variable which are
*  common across all kernels.
*/
#define INIT() \
    __m512d t_reg[6]; /*temporary registers*/                \
    __m512d c_reg[8]; /*registors to hold GEMM accumulation*/\
    __m512d b_reg[4]; /*registors to hold B matrix*/         \
    t_reg[5] = _mm512_set1_pd( 1.0 ); /*(constant) used for fmaddsub*/\
    \
    double g_double[3]; \
    __mmask8 mask_m; /*registor to hold mask for laod/store*/\
    \
    dim_t m = bli_obj_length( b );        \
    dim_t n = bli_obj_width( b );         \
    dim_t cs_a = bli_obj_col_stride( a ); \
    dim_t rs_a = bli_obj_row_stride( a ); \
    dim_t cs_b = bli_obj_col_stride( b ); \
    \
    bool transa = bli_obj_has_trans( a );              \
    bool is_unitdiag = bli_obj_has_unit_diag( a );     \
    dcomplex AlphaVal = *(dcomplex *)AlphaObj->buffer; \
    \
    dim_t d_mr = 4; \
    dim_t d_nr = 4; \
    dim_t i, j;     \
    dim_t k_iter;   \
    \
    dcomplex* restrict L = bli_obj_buffer_at_off( a ); \
    dcomplex* restrict B = bli_obj_buffer_at_off( b ); \

/*
*  Perform GEMM with given value of M, N, K
*  K is always a multiple of 4
*  N is compile time constant.
*  M <= 4 and N <= 4.
*  Output is stored in registor c_reg[0] to c_reg[N-1]
*/
#define GEMM_MxN( a01_, b10_, rs_a_, cs_a_, cs_b_, k_iter_, M_, N_ ) \
    \
    UNROLL_LOOP()                           \
    for( dim_t ii = 0; ii < k_iter_; ++ii ) \
    { \
        b_reg[0] = _mm512_mask_loadu_pd(c_reg[0], mask_m, b10_);            \
        UNROLL_LOOP_FULL()                  \
        for( dim_t jj = 0; jj < N_; ++jj )  \
        { \
            t_reg[0] = _mm512_set1_pd((a01_ + cs_a_*jj)->real);             \
            t_reg[1] = _mm512_set1_pd((a01_ + cs_a_*jj)->imag);             \
            c_reg[jj]   = _mm512_fmadd_pd(t_reg[0], b_reg[0], c_reg[jj]);   \
            c_reg[jj+4] = _mm512_fmadd_pd(t_reg[1], b_reg[0], c_reg[jj+4]); \
        } \
        a01_ += rs_a_; \
        b10_ += cs_b_; \
    } \
    t_reg[5] = _mm512_set1_pd(1.0);     \
    UNROLL_LOOP_FULL()                  \
    for ( dim_t jj = 0; jj < N_; ++jj ) \
    { \
        c_reg[jj+4] = _mm512_permute_pd(c_reg[jj+4], 0x55);               \
        c_reg[jj] = _mm512_fmaddsub_pd(t_reg[5], c_reg[jj], c_reg[jj+4]); \
    } \

/*
*  Performs alpha*B - gemm_output
*  N is compile time constant.
*  M <= 4 and N <= 4.
*/
#define PRE_TRSM_NxM(AlphaVal, b11, cs_b, M, N) \
    \
    if(AlphaVal.real == 1 && AlphaVal.imag == 0) \
    { \
        UNROLL_LOOP_FULL() \
        for(int ii=0; ii<N; ++ii) { \
            b_reg[0] = _mm512_mask_loadu_pd(c_reg[ii], mask_m, b11 + (cs_b*ii)); /*load B*/ \
            c_reg[ii] = _mm512_sub_pd(b_reg[0], c_reg[ii]); /*subtract GEMM output from B*/\
        } \
    } \
    else \
    { \
        t_reg[0] = _mm512_set1_pd(AlphaVal.real); \
        t_reg[1] = _mm512_set1_pd(AlphaVal.imag); \
        UNROLL_LOOP_FULL() \
        for(int ii=0; ii<N; ++ii) { \
            b_reg[0] = _mm512_mask_loadu_pd(c_reg[ii], mask_m, b11 + (cs_b*ii)); /*load B*/ \
            MULTIPLY_COMPLEX(b_reg[0], t_reg[0], t_reg[1], b_reg[0]) /*scale B by alpha*/  \
            c_reg[ii] = _mm512_sub_pd(b_reg[0], c_reg[ii]); /*subtract GEMM output from B*/\
        } \
    } \

/*
*  Permform TRSM computation for Right Upper
*  NonTranpose variant.
*  n is compile time constant.
*  M <= 4 and N <= 4.
*
*  c_reg array contains alpha*B11 - A01*B10
*  let  alpha*B11 - A01*B10 = C
*/
#define TRSM_MAIN_RUN_NxM(M) \
    \
    UNROLL_LOOP_FULL() \
    for ( dim_t ii = 0; ii < M; ++ii ) \
    { \
        if( !is_unitdiag ) /*if not unit diag, then divide C by A diagonal*/ \
        { \
            DIVIDE_COMPLEX(c_reg[ii], (a11 + ii*cs_a)) /* C / A11(diagonal)*/\
        } \
        UNROLL_LOOP_FULL() \
        for( dim_t jj = ii+1; jj < M; ++jj ) /* C[next_col] -= C[curr_col] * A11 */\
        { \
            t_reg[0] = _mm512_set1_pd((a11 + jj*cs_a)->real);         \
            t_reg[1] = _mm512_set1_pd((a11 + jj*cs_a)->imag);         \
            MULTIPLY_COMPLEX(c_reg[ii], t_reg[0], t_reg[1], t_reg[2]) \
            c_reg[jj] = _mm512_sub_pd(c_reg[jj], t_reg[2]);           \
        } \
        a11 += rs_a; \
    } \

/*
*  Perform TRSM computation for Right Lower
*  NonTranpose variant.
*  N is compile time constant.
*/
#define TRSM_MAIN_RLNN_NXM(N) \
    \
    a11 += rs_a * (N-1); \
    UNROLL_LOOP_FULL() \
    for( dim_t ii = (N-1); ii >= 0; --ii ) \
    { \
        if( !is_unitdiag ) \
        { \
            DIVIDE_COMPLEX(c_reg[ii], (a11 + ii*cs_a)) \
        } \
        UNROLL_LOOP_FULL() \
        for( dim_t jj = (ii-1); jj >= 0; --jj ) \
        { \
            t_reg[0] = _mm512_set1_pd((a11 + jj*cs_a)->real); \
            t_reg[1] = _mm512_set1_pd((a11 + jj*cs_a)->imag); \
            MULTIPLY_COMPLEX(c_reg[ii], t_reg[0], t_reg[1], t_reg[2]) \
            c_reg[jj] = _mm512_sub_pd(c_reg[jj], t_reg[2]); \
        } \
        a11 -= rs_a; \
    } \

/*
*  Stores output from registors(c_reg) to memory(B)
*  n is a compile time constant.
*/
#define STORE_RIGHT_C( n ) \
    UNROLL_LOOP_FULL() \
    for ( dim_t ii=0; ii < n; ++ii ) \
    { \
        _mm512_mask_storeu_pd((b11 + (ii * cs_b)), mask_m, c_reg[ii]); \
    } \

/*
* Perform GEMM + TRSM computation for Right Upper NonTranpose
*
*
* Left shift 1 by M times will set (M+1)th least significant bit
* subtracting 1 from that will unset (M+1)th LSB and set last M lSBs
*
* Example:         1 << 4 = 0b00010000
*          ( 1 << 4 ) - 1 = 0b00001111       
*/
#define RUNN_FRINGE( M, N )    \
    mask_m = (1 << (M*2)) - 1; \
    \
    a01 = L + j*cs_a;          \
    a11 = L + j*cs_a + j*rs_a; \
    b10 = B + i;               \
    b11 = B + i + j*cs_b;      \
    k_iter = j;                \
    \
    ZERO_REGISTERS() \
    \
    GEMM_MxN( a01, b10, rs_a, cs_a, cs_b, k_iter, M, N ) \
    PRE_TRSM_NxM( AlphaVal, b11, cs_b, M, N )            \
    \
    t_reg[4] = _mm512_set1_pd(-1.0); \
    TRSM_MAIN_RUN_NxM( N )           \
    STORE_RIGHT_C( N )               \

/*
* Perform GEMM + TRSM computation for Right Lower NonTranpose
*/
#define RLNN_FRINGE( M, N ) \
    mask_m = (1 << (M*2)) - 1; \
    \
    a01 = L + ((j - N + d_nr) * cs_a) + (j + d_nr) * rs_a;   \
    a11 = L + (j - N + d_nr) * rs_a + (j - N + d_nr) * cs_a; \
    b10 = B + (i - M + d_mr) + (j + d_nr) * cs_b;            \
    b11 = B + (i - M + d_mr) + (j - N + d_nr) * cs_b;        \
    k_iter = (n - j - d_nr); \
    \
    ZERO_REGISTERS()         \
    GEMM_MxN( a01, b10, rs_a, cs_a, cs_b, k_iter, M, N ) \
    PRE_TRSM_NxM( AlphaVal, b11, cs_b, M, N )            \
    \
    t_reg[4] = _mm512_set1_pd(-1.0);                     \
    TRSM_MAIN_RLNN_NXM( N )                              \
    STORE_RIGHT_C( N )                                   \

;

/*
* Solves Right Upper NonTranspose TRSM when N < 4
*/
BLIS_INLINE void runn_n_rem
                 (
                   dim_t i,
                   dim_t j,
                   dim_t cs_a,
                   dim_t rs_a,
                   dim_t cs_b,
                   dim_t m,
                   dim_t n,
                   dcomplex* L,
                   dcomplex* B,
                   dim_t k_iter,
                   bool transa,
                   dcomplex AlphaVal,
                   bool is_unitdiag
                )
{
    __m512d t_reg[6];
    __m512d c_reg[8];
    __m512d b_reg[4];

    double g_double[3];
    __mmask8 mask_m;

    t_reg[5] = _mm512_set1_pd(1.0);

    dim_t d_mr = 4;
    dcomplex *a01, *a11, *b10, *b11;
    dim_t m_rem;
    dim_t n_rem = n - j;

    /*
    * Switch statements used here to make sure that
    * N is a constant and compiler can unroll the loop
    * at compile time.
    */
    switch( n_rem )
    {
    case 1:
        for( i = 0; (i+d_mr-1) < m; i += d_mr )
        {
            RUNN_FRINGE( 4, 1 )
        }
        m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, 1 )
        }
        break;
    case 2:
        for( i = 0; (i+d_mr-1) < m; i += d_mr )
        {
            RUNN_FRINGE( 4, 2 )
        }
        m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, 2 )
        }
        break;
    case 3:
        for( i = 0; (i+d_mr-1) < m; i += d_mr )
        {
            RUNN_FRINGE( 4, 3 )
        }
        m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, 3 )
        }
        break;
    default:
        break;
    }
}

// RUNN - RLTN
err_t bli_ztrsm_small_XAltB_XAuB_AVX512
      (
        obj_t* AlphaObj,
        obj_t* a,
        obj_t* b,
        cntx_t* cntx,
        cntl_t* cntl
      )
{
    INIT()
    if( transa )
    {
        /*
        * If variants being solved is RLTN
        * then after swapping rs_a and cs_a,
        * problem will become same as RUNN
        */
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    dcomplex *a01, *a11, *b10, *b11;
    for( j = 0; (j+d_nr-1) < n; j += d_nr )
    {
        for( i = 0; (i+d_mr-1) < m; i += d_mr )
        {
            RUNN_FRINGE( 4, 4 )
        }
        dim_t m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, 4 )
        }
    }
    dim_t n_rem = n - j;
    if( n_rem > 0 )
    {
        /*
        * A hack:
        *      clang/aocc generate inefficient code when
        *      all M and N are handled in one function.
        *      (AOCC tries to make sure that each of the gemm call is
        *      using independent set of registors, which causes many
        *      read/writes in stack.)
        *      So part of code is moved to a seperate function.
        */
        runn_n_rem
        (
          i, j,
          cs_a, rs_a,
          cs_b,
          m, n,
          L, B,
          k_iter,
          transa,
          AlphaVal,
          is_unitdiag
        );
    }
    return BLIS_SUCCESS;
}

/*
* Solves Right Upper NonTranspose TRSM when N < 4
*/
BLIS_INLINE void rlnn_n_rem
                 (
                   dim_t i, dim_t j,
                   dim_t cs_a, dim_t rs_a,
                   dim_t cs_b,
                   dim_t m, dim_t n,
                   dcomplex* L,
                   dcomplex* B,
                   dim_t k_iter,
                   bool transa,
                   dcomplex AlphaVal,
                   bool is_unitdiag
                 )
{
    __m512d t_reg[6];
    __m512d c_reg[8];
    __m512d b_reg[4];

    double g_double[3];
    __mmask8 mask_m;

    t_reg[5] = _mm512_set1_pd(1.0);
    dim_t d_mr = 4;
    dim_t d_nr = 4;

    dcomplex *a01, *a11, *b10, *b11;
    dim_t m_rem;
    dim_t n_rem = j + d_nr;

    switch( n_rem )
    {
    case 1:
        for( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
        {
            RLNN_FRINGE( 4, 1 )
        }
        m_rem = i + d_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, 1 )
        }
        break;
    case 2:
        for( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
        {
            RLNN_FRINGE( 4, 2 )
        }
        m_rem = i + d_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, 2 )
        }
        break;
    case 3:
        for( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
        {
            RLNN_FRINGE( 4, 3 )
        }
        m_rem = i + d_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, 3 )
        }
        break;
    default:
        break;
    }
}

// RLNN - RUTNs
err_t bli_ztrsm_small_XAutB_XAlB_AVX512
      (
        obj_t* AlphaObj,
        obj_t* a,
        obj_t* b,
        cntx_t* cntx,
        cntl_t* cntl
      )
{
    INIT()
    if( transa )
    {
        /*
        * If variants being solved is RUTN
        * then after swapping rs_a and cs_a,
        * problem will become same as RLNN
        */
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    dcomplex *a01, *a11, *b10, *b11;

    for ( j = (n - d_nr); j > -1; j -= d_nr )
    {
        for ( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
        {
            RLNN_FRINGE( 4, 4 )
        }
        dim_t m_rem = i + d_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, 4 )
        }
    }
    dim_t n_rem = j + d_nr;
    if( n_rem > 0 )
    {
        rlnn_n_rem
        (
          i, j,
          cs_a, rs_a,
          cs_b,
          m, n,
          L, B,
          k_iter,
          transa,
          AlphaVal,
          is_unitdiag
        );
    }
    return BLIS_SUCCESS;
}

/*
*  Perform a 4x4 Transpose
*  Data is read from c_reg[0] to c[4]
*  and stored back to same registors after transpose
*/
#define TRANSPOSE4x4() \
    t_reg[0] = _mm512_shuffle_f64x2(c_reg[0], c_reg[1], 0b10001000); \
    t_reg[1] = _mm512_shuffle_f64x2(c_reg[2], c_reg[3], 0b10001000); \
    t_reg[2] = _mm512_shuffle_f64x2(c_reg[0], c_reg[1], 0b11011101); \
    t_reg[3] = _mm512_shuffle_f64x2(c_reg[2], c_reg[3], 0b11011101); \
    \
    c_reg[0] = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b10001000); \
    c_reg[2] = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b11011101); \
    c_reg[1] = _mm512_shuffle_f64x2(t_reg[2], t_reg[3], 0b10001000); \
    c_reg[3] = _mm512_shuffle_f64x2(t_reg[2], t_reg[3], 0b11011101); \


/*
* Perform GEMM when B is stored in row major order,
* k_iter is a multiple of 4
*/
#define GEMM_MxN_LEFT_TRANSPOSE( a01_, b10_, rs_a_, cs_a_, rs_b_, k_iter_, M_, N_ ) \
    \
    for( dim_t ii=0; ii < k_iter_/4; ++ii ) \
    { \
        /* load 4x4 B */ \
        for( dim_t jj=0; jj < M_; ++jj ) \
        { \
            b_reg[jj] = _mm512_loadu_pd(b10_ + (jj*rs_b_)); \
        } \
        /* Transpose 4x4 B*/ \
        t_reg[0] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b10001000); \
        t_reg[1] = _mm512_shuffle_f64x2(b_reg[2], b_reg[3], 0b10001000); \
        t_reg[2] = _mm512_shuffle_f64x2(b_reg[0], b_reg[1], 0b11011101); \
        t_reg[3] = _mm512_shuffle_f64x2(b_reg[2], b_reg[3], 0b11011101); \
        b_reg[0] = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b10001000); \
        b_reg[2] = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b11011101); \
        b_reg[1] = _mm512_shuffle_f64x2(t_reg[2], t_reg[3], 0b10001000); \
        b_reg[3] = _mm512_shuffle_f64x2(t_reg[2], t_reg[3], 0b11011101); \
        \
        /*Iter 1*/ \
        UNROLL_LOOP_FULL() \
        for( dim_t jj=0; jj < N_; ++jj ) \
        { \
            t_reg[0] = _mm512_set1_pd((a01_ + cs_a_*jj)->real); \
            t_reg[1] = _mm512_set1_pd((a01_ + cs_a_*jj)->imag); \
            c_reg[jj] = _mm512_fmadd_pd(t_reg[0], b_reg[0], c_reg[jj]); \
            c_reg[jj+4] = _mm512_fmadd_pd(t_reg[1], b_reg[0], c_reg[jj+4]); \
        } \
        a01_ += rs_a_; \
        /*Iter 2*/ \
        UNROLL_LOOP_FULL() \
        for( dim_t jj=0; jj < N_; ++jj ) \
        { \
            t_reg[0] = _mm512_set1_pd((a01_ + cs_a_*jj)->real); \
            t_reg[1] = _mm512_set1_pd((a01_ + cs_a_*jj)->imag); \
            c_reg[jj] = _mm512_fmadd_pd(t_reg[0], b_reg[1], c_reg[jj]); \
            c_reg[jj+4] = _mm512_fmadd_pd(t_reg[1], b_reg[1], c_reg[jj+4]); \
        } \
        a01_ += rs_a_; \
        /*Iter 3*/ \
        UNROLL_LOOP_FULL() \
        for( dim_t jj=0; jj < N_; ++jj ) \
        { \
            t_reg[0] = _mm512_set1_pd((a01_ + cs_a_*jj)->real); \
            t_reg[1] = _mm512_set1_pd((a01_ + cs_a_*jj)->imag); \
            c_reg[jj] = _mm512_fmadd_pd(t_reg[0], b_reg[2], c_reg[jj]); \
            c_reg[jj+4] = _mm512_fmadd_pd(t_reg[1], b_reg[2], c_reg[jj+4]); \
        } \
        a01_ += rs_a_; \
        /*Iter 4*/ \
        UNROLL_LOOP_FULL() \
        for( dim_t jj=0; jj < N_; ++jj ) \
        { \
            t_reg[0] = _mm512_set1_pd((a01_ + cs_a_*jj)->real); \
            t_reg[1] = _mm512_set1_pd((a01_ + cs_a_*jj)->imag); \
            c_reg[jj] = _mm512_fmadd_pd(t_reg[0], b_reg[3], c_reg[jj]); \
            c_reg[jj+4] = _mm512_fmadd_pd(t_reg[1], b_reg[3], c_reg[jj+4]); \
        } \
        a01_ += rs_a_; \
        b10_ += 4; \
    } \
    t_reg[5] = _mm512_set1_pd(1.0); \
    UNROLL_LOOP_FULL() \
    for ( dim_t jj=0; jj < N_; ++jj ) \
    { \
        c_reg[jj+4] = _mm512_permute_pd(c_reg[jj+4], 0x55); \
        c_reg[jj] = _mm512_fmaddsub_pd(t_reg[5], c_reg[jj], c_reg[jj+4]); \
    } \

/*
* Perform GEMM + TRSM computation for Left Lower NonTranpose
* When Problem is LLNN, after a induced transpose problem
* becomes RUNN
*/
#define LLNN_FRINGE( M, N )            \
    a10 = L + (i * cs_a);              \
    a11 = L + (i * rs_a) + (i * cs_a); \
    b01 = B + j * cs_b;                \
    b11 = B + i + j * cs_b;            \
    \
    k_iter = i;                        \
    mask_m = (1 << (M*2)) - 1;         \
    \
    ZERO_REGISTERS() \
    if (!transa) { \
        /*A and B are swapped are induced transpose*/ \
        GEMM_MxN( b01, a10, 1, cs_b, rs_a, k_iter, _, N ) \
    } else { \
        GEMM_MxN_LEFT_TRANSPOSE( b01, a10, 1, cs_b, cs_a, k_iter, M, N ) \
    } \
    PRE_TRSM_NxM( AlphaVal, b11, cs_b, _, N ) \
    /*
    * RUNN kernel requires GEMM output to
    * be in column major order
    */ \
    TRANSPOSE4x4()                   \
    t_reg[4] = _mm512_set1_pd(-1.0); \
    TRSM_MAIN_RUN_NxM(M)             \
    TRANSPOSE4x4()                   \
    STORE_RIGHT_C(N)                 \

/*
* Perform GEMM + TRSM computation for Left Upper NonTranpose
*/
#define LUNN_FRINGE( M, N ) \
    mask_m = (1 << (M*2)) - 1; \
    \
    a10 = L + ((i - M + d_mr) * cs_a) + (i + d_nr) * rs_a;   \
    a11 = L + (i - M + d_mr) * rs_a + (i - M + d_nr) * cs_a; \
    b01 = B + (i + d_mr) + (j - N + d_nr) * cs_b;            \
    b11 = B + (i - M + d_mr) + (j - N + d_nr) * cs_b;        \
    k_iter = ( m - i - d_mr ); \
    \
    ZERO_REGISTERS() \
    if (!transa) { \
        GEMM_MxN( b01, a10, 1, cs_b, rs_a, k_iter, _, N ) \
    } else { \
        GEMM_MxN_LEFT_TRANSPOSE( b01, a10, 1, cs_b, cs_a, k_iter, M, N ) \
    } \
    \
    PRE_TRSM_NxM( AlphaVal, b11, cs_b, _, N ) \
    TRANSPOSE4x4()                            \
    t_reg[4] = _mm512_set1_pd(-1.0);          \
    TRSM_MAIN_RLNN_NXM( M )                   \
    TRANSPOSE4x4()                            \
    STORE_RIGHT_C( N )                        \

/*
* Solves Left Lower NonTranspose TRSM when M < 4
*/
BLIS_INLINE void llnn_m_rem
                 (
                   dim_t i, dim_t j,
                   dim_t cs_a, dim_t rs_a,
                   dim_t cs_b,
                   dim_t m, dim_t n,
                   dcomplex* L,
                   dcomplex* B,
                   dim_t k_iter,
                   bool transa,
                   dcomplex AlphaVal,
                   bool is_unitdiag
                 )
{
    __m512d t_reg[6];
    __m512d c_reg[8];
    __m512d b_reg[4];
    double g_double[3];

    __mmask8 mask_m;
    t_reg[5] = _mm512_set1_pd(1.0);

    dim_t d_nr = 4;
    dcomplex *a10, *a11, *b01, *b11;
    dim_t m_rem = m - i;
    dim_t n_rem;

    switch( m_rem )
    {
    case 1:
        for( j = 0; (j + d_nr - 1) < n; j += d_nr )
        {
            LLNN_FRINGE( 1, 4 )
        }
        n_rem = n - j;
        switch( n_rem )
        {
        case 1:
            LLNN_FRINGE( 1, 1 ); break;
        case 2:
            LLNN_FRINGE( 1, 2 ); break;
        case 3:
            LLNN_FRINGE( 1, 3 ); break;
        default:
            break;
        }
        break;
    case 2:
        for( j = 0; (j + d_nr - 1) < n; j += d_nr )
        {
            LLNN_FRINGE( 2, 4 )
        }
        n_rem = n - j;
        switch( n_rem )
        {
        case 1:
            LLNN_FRINGE( 2, 1 ); break;
        case 2:
            LLNN_FRINGE( 2, 2 ); break;
        case 3:
            LLNN_FRINGE( 2, 3 ); break;
        default:
            break;
        }
        break;
    case 3:
        for( j = 0; (j + d_nr - 1) < n; j += d_nr )
        {
            LLNN_FRINGE( 3, 4 )
        }
        n_rem = n - j;
        switch( n_rem )
        {
        case 1:
            LLNN_FRINGE( 3, 1 ); break;
        case 2:
            LLNN_FRINGE( 3, 2 ); break;
        case 3:
            LLNN_FRINGE( 3, 3 ); break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

// LLNN - LUTN
err_t bli_ztrsm_small_AutXB_AlXB_AVX512
      (
        obj_t*   AlphaObj,
        obj_t*   a,
        obj_t*   b,
        cntx_t*  cntx,
        cntl_t*  cntl
      )
{
    INIT()
    if( !transa )
    {
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    dcomplex *a10, *a11, *b01, *b11;
    for( i = 0; (i + d_mr - 1) < m; i += d_mr )
    {
        for( j = 0; j < n - d_nr + 1; j += d_nr )
        {
            LLNN_FRINGE( 4, 4 )
        }
        dim_t n_rem = n - j;
        if( n_rem > 0 )
        {
            switch( n_rem )
            {
            case 1:
                LLNN_FRINGE( 4, 1 ); break;
            case 2:
                LLNN_FRINGE( 4, 2 ); break;
            case 3:
                LLNN_FRINGE( 4, 3 ); break;
            default:
                break;
            }
        }
    }
    dim_t m_rem = m - i;
    if( m_rem > 0 )
    {
        llnn_m_rem
        (
          i, j,
          cs_a, rs_a,
          cs_b,
          m, n,
          L, B,
          k_iter,
          transa,
          AlphaVal,
          is_unitdiag
        );
    }
    return BLIS_SUCCESS;
}

/*
* Solves Left Upper NonTranspose TRSM when M < 4
*/
BLIS_INLINE void lunn_m_rem
                 (
                   dim_t i, dim_t j,
                   dim_t cs_a, dim_t rs_a,
                   dim_t cs_b,
                   dim_t m, dim_t n,
                   dcomplex* L,
                   dcomplex* B,
                   dim_t k_iter,
                   bool transa,
                   dcomplex AlphaVal,
                   bool is_unitdiag
                 )
{
    __m512d t_reg[6];
    __m512d c_reg[8];
    __m512d b_reg[4];

    double g_double[3];
    __mmask8 mask_m;

    t_reg[5] = _mm512_set1_pd(1.0);
    dim_t d_mr = 4;
    dim_t d_nr = 4;
    dcomplex *a10, *a11, *b01, *b11;
    dim_t m_rem = i + d_mr;
    dim_t n_rem;

    switch( m_rem )
    {
    case 1:
        for( j = (n - d_nr); (j + 1) > 0; j -= d_nr )
        {
            LUNN_FRINGE( 1, 4 )
        }
        n_rem = j + d_nr;
        switch( n_rem )
        {
        case 1:
            LUNN_FRINGE( 1, 1 ); break;
        case 2:
            LUNN_FRINGE( 1, 2 ); break;
        case 3:
            LUNN_FRINGE( 1, 3 ); break;
        default:
            break;
        }
        break;
    case 2:
        for( j = (n - d_nr); (j + 1) > 0; j -= d_nr )
        {
            LUNN_FRINGE( 2, 4 )
        }
        n_rem = j + d_nr;
        switch( n_rem )
        {
        case 1:
            LUNN_FRINGE( 2, 1 ); break;
        case 2:
            LUNN_FRINGE( 2, 2 ); break;
        case 3:
            LUNN_FRINGE( 2, 3 ); break;
        default:
            break;
        }
        break;
    case 3:
        for( j = (n - d_nr); (j + 1) > 0; j -= d_nr )
        {
            LUNN_FRINGE( 3, 4 )
        }
        n_rem = j + d_nr;
        switch( n_rem )
        {
        case 1:
            LUNN_FRINGE( 3, 1 ); break;
        case 2:
            LUNN_FRINGE( 3, 2 ); break;
        case 3:
            LUNN_FRINGE( 3, 3 ); break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

// LUNN - LLTN
err_t bli_ztrsm_small_AltXB_AuXB_AVX512
      (
        obj_t*   AlphaObj,
        obj_t*   a,
        obj_t*   b,
        cntx_t*  cntx,
        cntl_t*  cntl
      )
{
    INIT()
    if( !transa )
    {
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    dcomplex *a10, *a11, *b01, *b11;
    for( i = (m - d_mr); (i + 1) > 0; i -= d_mr )
    {
        for( j = (n - d_nr); (j + 1) > 0; j -= d_nr )
        {
            LUNN_FRINGE( 4, 4 )
        }
        dim_t n_rem = j + d_nr;
        if( n_rem > 0 )
        {
            switch( n_rem )
            {
            case 1:
                LUNN_FRINGE( 4, 1 ); break;
            case 2:
                LUNN_FRINGE( 4, 2 ); break;
            case 3:
                LUNN_FRINGE( 4, 3 ); break;
            default:
                break;
            }
        }
    }
    dim_t m_rem = i + d_mr;
    if( m_rem > 0 )
    {
        lunn_m_rem
        (
          i, j,
          cs_a, rs_a,
          cs_b,
          m, n,
          L, B,
          k_iter,
          transa,
          AlphaVal,
          is_unitdiag
        );
    }
    return BLIS_SUCCESS;
}

#endif //BLIS_ENABLE_SMALL_MATRIX_TRSM

