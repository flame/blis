/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#define Z_MR_ 12
#define Z_NR_ 4

#if defined __clang__
    #define UNROLL_LOOP()      _Pragma("clang loop unroll_count(4)")
    /*
    *   in clang, unroll_count(4) generates inefficient
    *   code compared to unroll(full) when loopCount = 4.
    */
    #define __UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
    #define UNROLL_LOOP_N(n)   __UNROLL_LOOP_FULL() // for clang,
    //full unroll is always more performant
#elif defined __GNUC__
    #define UNROLL_LOOP()      _Pragma("GCC unroll 4")
    #define __UNROLL_LOOP_FULL() _Pragma("GCC unroll 12")

    #define STRINGIFY(x) #x
    #define TOSTRING(x) STRINGIFY(x)
    #define UNROLL_LOOP_N(n) _Pragma(TOSTRING(GCC unroll n))
#else // unknown compiler
    #define UNROLL_LOOP()      // no unroll if compiler is not known
    #define __UNROLL_LOOP_FULL()
    #define UNROLL_LOOP_N(n)
#endif

#define ENABLE_PACK_A                    // enable pack for A, comment out this line to disable packing
                                         // removing pack A will remove support for Left variants.

#define ENABLE_ALT_N_REM                 // clear stack frame for N remainder code, this removes false
                                         // dependencies in gcc.

#define ENABLE_PACK_A_FOR_UPPER false    // enable pack A for upper variants

// #define FAST_DIV_BROADCAST            // use faster dvision method (less accurate for denormal numbers)


// Broadcast a dcomplex number from addr for division into t_reg[0] and t_reg[1]
// Lets assume we are dividing z1(a + ib) with z2 (c + id)
// Division can we done using formula z1/z2 = (bc - ad) / (c^2 + d^2)
// In order to avoid underflow (with c^2 and d^2), c and d are normalized
// c^2 + d^2 is stored in g_double
// to range [0, 1] before squaring.
#define BROADCAST_FOR_DIVIDE(addr)                                          \
    g_double[2] = bli_fmaxabs(addr->real,addr->imag);/*s=max(|real|,|imag|)*/\
    g_double[0] = addr->real / g_double[2];/*ar/s*/                         \
    g_double[1] = addr->imag / g_double[2];/*ai/s*/                         \
    t_reg[0]    = _mm512_set1_pd(g_double[0]);/*ar/s*/                      \
    t_reg[1]    = _mm512_set1_pd(g_double[1]);/*ai/s*/                      \
    g_double[2] = (g_double[0] * addr->real) +                              \
                  (g_double[1] * addr->imag);                               \
                   /*(ar/s * ar) +(ai/s * ai)*/                             \


// Broadcast dcomplex number from addr to t_reg[0] and t_reg[1]
#ifdef FAST_DIV_BROADCAST
// Fast division algo avoids normalization of divisor, this may cause underflow
// for denormals numbers but give ~5% better performance on average
#define BROADCAST_FOR_DIV_INV(addr)                                         \
    g_double[0] = (addr->real * addr->real) + (addr->imag * addr->imag);    \
    t_reg[0]    = _mm512_set1_pd(addr->real / g_double[0]);                 \
    t_reg[1]    = _mm512_set1_pd((-1 * addr->imag) / g_double[0]);          \

#else


// Find resiprocal of a complex number
// Used if TRSM preinversion is enabled
// Lets say we want to divide z1 by z2, z1/z2. If perinversion is enabled,
// instead of z1/z2, z1 is multiplied by resiprocal of z2 i.e. z1 * (1/z2)
//
// This macro finds the resiprocal (of z2) and stores the result in
// t_reg[0] and t_reg[1]
#define BROADCAST_FOR_DIV_INV(addr) \
    g_double[2] = bli_fmaxabs(addr->real, addr->imag);/*s*/                 \
    g_double[0] = addr->real / g_double[2];/*ar/s*/                         \
    g_double[1] = addr->imag / g_double[2];/*ai/s*/                         \
    g_double[2] = (g_double[0] * addr->real) +                              \
                  (g_double[1] * addr->imag);                               \
                   /*(ar/s * ar) +(ai/s * ai)*/                             \
    t_reg[0]    = _mm512_set1_pd( g_double[0] / g_double[2]);               \
    t_reg[1]    = _mm512_set1_pd(( -1 * g_double[1]) / g_double[2]);        \

#endif

// Divide reg_a with the data broadcasted by BROADCAST_FOR_DIVIDE macro
#define DIVIDE_COMPLEX(reg_a)                                               \
    t_reg[3]   = _mm512_permute_pd(reg_a, 0x55);                            \
    /*t_reg[3] = [xi,xr,xi,xr....] */                                       \
    reg_a      = _mm512_mul_pd(reg_a, t_reg[0]);                            \
    /* reg_a   = ar/s * [xr, xi, xr, xi ....]*/                             \
    t_reg[3]   = _mm512_mul_pd(t_reg[3], t_reg[1]);                         \
    /*t_reg[3] = ai/s * [xi,xr,xi,xr........] */                            \
    t_reg[3]   = _mm512_mul_pd(t_reg[4], t_reg[3]);                         \
    /*t_reg[3] = -ai/s * [xi,xr,xi,xr........] */                           \
    t_reg[2]   = _mm512_set1_pd(g_double[2]);                               \
    /*t_reg[2] = [(ar/s * ar) +(ai/s * ai), ...] */                         \
    reg_a      = _mm512_fmaddsub_pd(t_reg[5], reg_a, t_reg[3]);             \
    /*reg_a    = [a1c+b1d, b1c-a1d, a2c+b2d, b2c-a2d, ....]*/               \
    reg_a      = _mm512_div_pd(reg_a, t_reg[2]);                            \

// Scale a complex vector(reg_a) with a scaler(reg_r, reg_i)
#define MULTIPLY_COMPLEX( reg_a, reg_r, reg_i, output )                     \
    t_reg[3]    = _mm512_permute_pd(reg_a, 0x55);                           \
    /* t_reg[3] = [b1, a1, b2, a2, b3, a3, b4, a4] */                       \
    output      = _mm512_mul_pd(reg_a, reg_r);                              \
    /* output   = c * [a1, b1, a2, b2, a3, b3, a4, b4]*/                    \
    t_reg[3]    = _mm512_mul_pd(t_reg[3], reg_i);                           \
    /* t_reg[3] = d * [b1, a1, b2, a2, b3, a3, b4, a4]*/                    \
    output      = _mm512_fmaddsub_pd(t_reg[5], output, t_reg[3]);           \
    /* output   = [a1c-b1d, a1d+b1c, a2c-b2d, a2d+b2c, ......]*/            \


#ifdef BLIS_ENABLE_TRSM_PREINVERSION
    #define BROADCAST_DIV(addr)     \
        BROADCAST_FOR_DIV_INV(addr)
    #define DIV_OR_MUL_DIAG(reg) \
        MULTIPLY_COMPLEX(reg, t_reg[0], t_reg[1], reg)
#endif

#ifdef BLIS_DISABLE_TRSM_PREINVERSION
        #define BROADCAST_DIV(addr)     \
            BROADCAST_FOR_DIVIDE(addr)
        #define DIV_OR_MUL_DIAG(reg) \
            DIVIDE_COMPLEX(reg)
#endif

// initialize common variables used among all right kernels
#define INIT_R()                                                            \
    dcomplex minus_one = {-1, 0}; /* used as alpha in gemm kernel */        \
    auxinfo_t auxinfo;     /* needed for gemm kernel      */                \
    double g_double[3];                                                     \
    __m512d t_reg[6];      /*temporary registers*/                          \
    t_reg[5] = _mm512_set1_pd( 1.0 ); /*(constant) used for fmaddsub*/      \
    t_reg[4] = _mm512_set1_pd(-1.0);  /*(constant) used for mul*/           \
    __m512d c_reg[Z_MR_]; /*registers to hold GEMM accumulation*/           \
                                                                            \
    /*We have 3 load registers(MR*sizeof(dcmplex)/sizeof(register) */       \
    /* = 12*128/512). 1 mask register is need to partially load 1 register*/\
    /* 3 mask registers are needed to partially load 3 registers */         \
    __mmask8 mask_m_0 = 0b11111111; /*register to hold mask for load/store*/\
    __mmask8 mask_m_1 = 0b11111111; /*register to hold mask for load/store*/\
    __mmask8 mask_m_2 = 0b11111111; /*register to hold mask for load/store*/\
                                                                            \
    dim_t m     = bli_obj_length( b );                                      \
    dim_t n     = bli_obj_width( b );                                       \
    dim_t cs_a  = bli_obj_col_stride( a );                                  \
    dim_t rs_a  = bli_obj_row_stride( a );                                  \
    dim_t cs_b  = bli_obj_col_stride( b );                                  \
    dim_t cs_a_ = cs_a;                                                     \
    dim_t rs_a_ = rs_a;                                                     \
                                                                            \
    bool transa       = bli_obj_has_trans( a );                             \
    bool is_unitdiag  = bli_obj_has_unit_diag( a );                         \
    dcomplex AlphaVal = *(dcomplex *)AlphaObj->buffer;                      \
                                                                            \
    dim_t z_mr  = Z_MR_;                                                    \
    dim_t z_nr  = Z_NR_;                                                    \
    dim_t i, j;                                                             \
    dim_t k_iter;                                                           \
                                                                            \
    dcomplex* restrict L = bli_obj_buffer_at_off( a );                      \
    dcomplex* restrict B = bli_obj_buffer_at_off( b );                      \

// Generate load/store masks
#define GENERATE_MASK(M)                                                    \
    if(M > 8) /* if M > 8, mask0 = mask1 = [1]*8, mask2 = [1]*(M*2) */      \
    {                                                                       \
        /* if M is range (8, 12], load first 8 elements fully, and load  */ \
        /* 9th to 12th elements based on mask. In case of dcomplex, each */ \
        /* element is represented by 2 bits in mask                      */ \
        mask_m_0 = 0b11111111;     /* fully load/store elements [0, 4)   */ \
        mask_m_1 = 0b11111111;     /* fully load/store elements [4, 8)   */ \
        /* for elements indexed [8, 11), load only M-8 elements,         */ \
        /* beacuase first 8 elements are already loaded by mask_m_0 and  */ \
        /* mask_m_1. To load M-8 elements, we need a mask with least     */ \
        /* significant (M-8)*2 bits set. This can be achieved by         */ \
        /* (1 << ((M-8)*2)) - 1, where (1 << ((M-8)*2)) will generate a  */ \
        /* mask with (((M-8)*2)+1)th LSB set, followed by (((M-8)*2)+1)  */ \
        /* zeros, subtracting 1, will give us (M-8)*2 LSBs set           */ \
        mask_m_2 = (__mmask8)(1 << ((M-8)*2)) - 1;                          \
    }                                                                       \
    else if(M > 4)   /* if M is in range (4, 8] */                          \
    {                                                                       \
        mask_m_0 = 0b11111111;   /* load first 4 elements fully*/           \
        /*load M-4 elements in range [4, 8) based on mask*/                 \
        mask_m_1 = (__mmask8)(1 << ((M-4)*2)) - 1;                          \
        mask_m_2 = 0;         /* do not load last 4 elements*/              \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        mask_m_0 = (__mmask8)(1 << (M*2)) - 1;                              \
        mask_m_1 = 0;                                                       \
        mask_m_2 = 0;                                                       \
    }                                                                       \



/*
*  Perform TRSM computation for Right Upper
*  NonTranpose variant.
*  N is compile time constant.
*  N <= 4
*
*  c_reg array contains alpha*b_trsm - a_gemm*b_gemm
*  let  alpha*b_trsm - a_gemm*b_gemm = C
*/
#define TRSM_MAIN_RUN_NxN(N)                                               \
                                                                           \
    __UNROLL_LOOP_FULL()                                                   \
    for ( dim_t ii = 0; ii < N; ++ii )                                     \
    {                                                                      \
        if( !is_unitdiag )                                                 \
        {                                                                  \
            BROADCAST_DIV((a_trsm + ii*cs_a));                             \
            DIV_OR_MUL_DIAG(c_reg[ii+0]);                                  \
            DIV_OR_MUL_DIAG(c_reg[ii+4]);                                  \
            DIV_OR_MUL_DIAG(c_reg[ii+8]);                                  \
        }                                                                  \
        __UNROLL_LOOP_FULL()                                               \
        for( dim_t jj = ii+1; jj < N; ++jj )                               \
        {     /* C[next_col] -= C[curr_col] * a_trsm */                    \
            t_reg[0] = _mm512_set1_pd((a_trsm + jj*cs_a)->real);           \
            t_reg[1] = _mm512_set1_pd((a_trsm + jj*cs_a)->imag);           \
            MULTIPLY_COMPLEX(c_reg[ii+0], t_reg[0], t_reg[1], t_reg[2]);   \
            c_reg[jj+0] = _mm512_sub_pd(c_reg[jj+0], t_reg[2]);            \
            MULTIPLY_COMPLEX(c_reg[ii+4], t_reg[0], t_reg[1], t_reg[2]);   \
            c_reg[jj+4] = _mm512_sub_pd(c_reg[jj+4], t_reg[2]);            \
            MULTIPLY_COMPLEX(c_reg[ii+8], t_reg[0], t_reg[1], t_reg[2]);   \
            c_reg[jj+8] = _mm512_sub_pd(c_reg[jj+8], t_reg[2]);            \
        }                                                                  \
        a_trsm += rs_a;                                                    \
    }                                                                      \

/*
*  Perform TRSM computation for Right Lower
*  NonTranpose variant.
*  N is compile time constant.
*/
#define TRSM_MAIN_RLN_NxN(N)                                               \
                                                                           \
    a_trsm += rs_a * (N-1);                                                \
    __UNROLL_LOOP_FULL()                                                   \
    for( dim_t ii = (N-1); ii >= 0; --ii )                                 \
    {                                                                      \
        if( !is_unitdiag )                                                 \
        {                                                                  \
            BROADCAST_DIV((a_trsm + ii*cs_a));                             \
            DIV_OR_MUL_DIAG(c_reg[ii+0]);                                  \
            DIV_OR_MUL_DIAG(c_reg[ii+4]);                                  \
            DIV_OR_MUL_DIAG(c_reg[ii+8]);                                  \
        }                                                                  \
        __UNROLL_LOOP_FULL()                                               \
        for( dim_t jj = (ii-1); jj >= 0; --jj )                            \
        {                                                                  \
            t_reg[0] = _mm512_set1_pd((a_trsm + jj*cs_a)->real);           \
            t_reg[1] = _mm512_set1_pd((a_trsm + jj*cs_a)->imag);           \
            MULTIPLY_COMPLEX(c_reg[ii+0], t_reg[0], t_reg[1], t_reg[2]);   \
            c_reg[jj+0] = _mm512_sub_pd(c_reg[jj+0], t_reg[2]);            \
            MULTIPLY_COMPLEX(c_reg[ii+4], t_reg[0], t_reg[1], t_reg[2]);   \
            c_reg[jj+4] = _mm512_sub_pd(c_reg[jj+4], t_reg[2]);            \
            MULTIPLY_COMPLEX(c_reg[ii+8], t_reg[0], t_reg[1], t_reg[2]);   \
            c_reg[jj+8] = _mm512_sub_pd(c_reg[jj+8], t_reg[2]);            \
        }                                                                  \
        a_trsm -= rs_a;                                                    \
    }                                                                      \

/*
* load N columns of C (12xN) into registers
*  n is a compile time constant.
*/
#define LOAD_C( N )                                                        \
    __UNROLL_LOOP_FULL()                                                   \
    for ( dim_t ii=0; ii < N; ++ii )                                       \
    {                                                                      \
        c_reg[ii+0] = _mm512_maskz_loadu_pd(mask_m_0, b_trsm+(cs_b*ii+0)); \
        c_reg[ii+4] = _mm512_maskz_loadu_pd(mask_m_1, b_trsm+(cs_b*ii+4)); \
        c_reg[ii+8] = _mm512_maskz_loadu_pd(mask_m_2, b_trsm+(cs_b*ii+8)); \
    }                                                                      \

/*
*  Stores output from registers(c_reg) to memory(B)
*  n is a compile time constant.
*/
#define STORE_RIGHT_C( N )                                                 \
    __UNROLL_LOOP_FULL()                                                   \
    for ( dim_t ii=0; ii < N; ++ii )                                       \
    {                                                                      \
        _mm512_mask_storeu_pd((b_trsm + (ii * cs_b) +0),                   \
                                        mask_m_0, c_reg[ii+0]);            \
        _mm512_mask_storeu_pd((b_trsm + (ii * cs_b) +4),                   \
                                        mask_m_1, c_reg[ii+4]);            \
        _mm512_mask_storeu_pd((b_trsm + (ii * cs_b) +8),                   \
                                        mask_m_2, c_reg[ii+8]);            \
    }                                                                      \

/*
* Perform GEMM + TRSM computation for Right Upper NonTranpose
* b_trsm := alpha*b_trsm - a_gemm*b_gemm
* B (b_trsm) computed by previous iterations in b_gemm for current iterations
*/
#define RUNN_FRINGE( M, N )                                                \
    GENERATE_MASK(M)                                                       \
    a_gemm = L_;                                                           \
    a_trsm = L + j*cs_a + j*rs_a;                                          \
    b_gemm = B + i;                                                        \
    b_trsm = B + i + j*cs_b;                                               \
    k_iter = j;                                                            \
    bli_zgemmsup_cv_zen4_asm_12x4m                                         \
    (                                                                      \
        BLIS_NO_CONJUGATE,                                                 \
        BLIS_NO_CONJUGATE,                                                 \
        M,                                                                 \
        N,                                                                 \
        k_iter,                                                            \
        &minus_one,                                                        \
        b_gemm,                                                            \
        1,                                                                 \
        cs_b,                                                              \
        a_gemm,                                                            \
        rs_a_,                                                             \
        cs_a_,                                                             \
        &AlphaVal,                                                         \
        b_trsm, 1, cs_b,                                                   \
        &auxinfo,                                                          \
        NULL                                                               \
    );                                                                     \
    LOAD_C( N )   /*load b_trsm stored by gemm kernel*/                    \
    TRSM_MAIN_RUN_NxN( N )  /* b_trsm * a_trsm = b_trsm */                 \
    STORE_RIGHT_C( N )     /*store b_trsm back to memory*/                 \

/*
* Perform GEMM + TRSM computation for Right Lower NonTranpose
*/
#define RLNN_FRINGE( M, N )                                                \
    GENERATE_MASK(M)                                                       \
    a_gemm = L_;                                                           \
    a_trsm = L + (j - N + z_nr) * rs_a + (j - N + z_nr) * cs_a;            \
    b_gemm = B + (i - M + z_mr) + (j + z_nr) * cs_b;                       \
    b_trsm = B + (i - M + z_mr) + (j - N + z_nr) * cs_b;                   \
    k_iter = (n - j - z_nr);                                               \
    bli_zgemmsup_cv_zen4_asm_12x4m                                         \
    (                                                                      \
        BLIS_NO_CONJUGATE,                                                 \
        BLIS_NO_CONJUGATE,                                                 \
        M,                                                                 \
        N,                                                                 \
        k_iter,                                                            \
        &minus_one,                                                        \
        b_gemm,                                                            \
        1,                                                                 \
        cs_b,                                                              \
        a_gemm,                                                            \
        rs_a_,                                                             \
        cs_a_,                                                             \
        &AlphaVal,                                                         \
        b_trsm, 1, cs_b,                                                   \
        &auxinfo,                                                          \
        NULL                                                               \
    );                                                                     \
    LOAD_C( N )                                                            \
    TRSM_MAIN_RLN_NxN( N )                                                 \
    STORE_RIGHT_C( N )                                                     \



/*
* Solve Right Upper NonTranspose TRSM when N < 4
*/
BLIS_INLINE void runn_n_rem
(
    dim_t i,             dim_t j,
    dim_t cs_a,          dim_t rs_a,
    dim_t cs_a_,         dim_t rs_a_,
    dim_t cs_b,
    dim_t m,             dim_t n,
    dcomplex* L,         dcomplex* L_,
    dcomplex* p,         dcomplex* B,
    dim_t k_iter,
    bool transa,         bool bPackedA,
    dcomplex AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p; // avoid warning if pack not enabled

    dim_t z_mr = Z_MR_;
    dcomplex minus_one = {-1, 0};      // alpha for gemmsup kernel

#ifdef ENABLE_PACK_A
    dcomplex one = {1, 0};             // kappa for pack kernel
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[6];                  // temporary registers
    t_reg[5] = _mm512_set1_pd( 1.0 );  // (constant) used for fmaddsub
    t_reg[4] = _mm512_set1_pd(-1.0);   // (constatnt) used in complex multiplicaton
    double g_double[3];                // temporary registers for complex divison
    __m512d c_reg[Z_MR_];              // registers to hold GEMM accumulation

    for(dim_t i = 0; i < Z_MR_; ++i)
    {
        c_reg[i] = _mm512_setzero_pd(); // initialize c_reg to zero
    }

    __mmask8 mask_m_0, mask_m_1, mask_m_2; // masks for load/store
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dim_t m_rem;
    dim_t n_rem = n - j;
    L_ = L + j*cs_a;

#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_4xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                n_rem,
                j,
                j,
                &one,
                L + j*cs_a,
                cs_a,
                rs_a,
                p,
                Z_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = Z_NR_;
            L_ = p;
        }
#endif

    switch (n_rem)
    {
    case 3:
        for( i = 0; (i+z_mr-1) < m; i += z_mr )
        {
            RUNN_FRINGE(Z_MR_, 3 );
        }
        m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, 3 );
        }
        break;

    case 2:
        for( i = 0; (i+z_mr-1) < m; i += z_mr )
        {
            RUNN_FRINGE(Z_MR_, 2 );
        }
        m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, 2 );
        }
        break;

    case 1:
        for( i = 0; (i+z_mr-1) < m; i += z_mr )
        {
            RUNN_FRINGE(Z_MR_, 1 );
        }
        m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, 1 );
        }
        break;

    default:
        break;
    }

}

/*
* Solve Right Upper NonTranspose TRSM when N < 4
*/
BLIS_INLINE void rlnn_n_rem
(
    dim_t i,             dim_t j,
    dim_t cs_a,          dim_t rs_a,
    dim_t cs_a_,         dim_t rs_a_,
    dim_t cs_b,
    dim_t m,             dim_t n,
    dcomplex* L,         dcomplex* L_,
    dcomplex* p,         dcomplex* B,
    dim_t k_iter,
    bool transa,         bool bPackedA,
    dcomplex AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p;
    dim_t z_mr = Z_MR_;
    dim_t z_nr = Z_NR_;
    dcomplex minus_one = {-1, 0};
#ifdef ENABLE_PACK_A
    dcomplex one = {1, 0};
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[6]; /*temporary registers*/
    __m512d c_reg[Z_MR_]; /*registers to hold GEMM accumulation*/
    double g_double[3];

    t_reg[5] = _mm512_set1_pd( 1.0 ); /*(constant) used for fmaddsub*/
    t_reg[4] = _mm512_set1_pd(-1.0);

    __mmask8 mask_m_0, mask_m_1, mask_m_2;
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dim_t m_rem;
    dim_t n_rem = j + z_nr;
    L_ = L + ((j - n_rem + z_nr) * cs_a) + (j + z_nr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_4xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                n_rem,
                (n - j - z_nr),
                (n - j - z_nr),
                &one,
                L + ((j - n_rem + z_nr) * cs_a) + (j + z_nr) * rs_a,
                cs_a,
                rs_a,
                p,
                Z_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = Z_NR_;
            L_ = p;
        }
#endif
    switch (n_rem)
    {
    case 3:
        for( i = (m - z_mr); (i + 1) > 0; i -= z_mr )
        {
            RLNN_FRINGE(Z_MR_, 3);
        }
        m_rem = i + z_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, 3);
        }
        break;
    case 2:
        for( i = (m - z_mr); (i + 1) > 0; i -= z_mr )
        {
            RLNN_FRINGE(Z_MR_, 2);
        }
        m_rem = i + z_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, 2);
        }
        break;
    case 1:
        for( i = (m - z_mr); (i + 1) > 0; i -= z_mr )
        {
            RLNN_FRINGE(Z_MR_, 1);
        }
        m_rem = i + z_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, 1);
        }
        break;

    default:
        break;
    }
}


// RUNN - RLTN
err_t bli_ztrsm_small_XAltB_XAuB_ZEN5
      (
        obj_t* AlphaObj,
        obj_t* a,
        obj_t* b,
        cntx_t* cntx,
        cntl_t* cntl
      )
{
    INIT_R();
    if( transa )
    {
        /*
        * If variants being solved is RLTN
        * then after swapping rs_a and cs_a,
        * problem will become same as RUNN
        */
        i     = cs_a;
        cs_a  = rs_a;
        rs_a  = i;
        cs_a_ = cs_a;
        rs_a_ = rs_a;
    }
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dcomplex* restrict L_ = L;

#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;
    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    dcomplex* p = NULL;
    dcomplex one = {1, 0};
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (Z_NR_*n*sizeof(dcomplex)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
#endif

    for( j = 0; (j+z_nr-1) < n; j += z_nr )
    {
        // Block B matrix into blocks of M x NR
        // Block A matrix into blocks of N X NR
        L_ = L + j*cs_a; // A Buffer

#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_4xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                Z_NR_,
                j,
                j,
                &one,
                L + j*cs_a,
                cs_a,
                rs_a,
                p,
                Z_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = Z_NR_;
            L_ = p;
        }
#endif
        for( i = 0; (i+z_mr-1) < m; i += z_mr )
        {
            // Block X matrix into blocks of MR x N
            // Block B matrix into blocks of MR x NR
            RUNN_FRINGE( Z_MR_, Z_NR_ );
        }
        dim_t m_rem = m - i;
        if ( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem , Z_NR_ );
        }
    }

    dim_t n_rem = n - j;
    if( n_rem > 0 )
    {
#ifdef ENABLE_ALT_N_REM
#ifndef ENABLE_PACK_A
        dcomplex* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        runn_n_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
#else //ENABLE_ALT_N_REM
        L_ = L + j*cs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_4xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                n_rem,
                j,
                j,
                &one,
                L + j*cs_a,
                cs_a,
                rs_a,
                p,
                Z_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = Z_NR_;
            L_ = p;
        }
#endif //ENABLE_PACK_A
        for( i = 0; (i+z_mr-1) < m; i += z_mr )
        {
            RUNN_FRINGE(Z_MR_, n_rem);
        }
        dim_t m_rem = m - i;
        if( m_rem > 0 )
        {
            RUNN_FRINGE( m_rem, n_rem );
        }
#endif //ENABLE_ALT_N_REM
    }

#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
    return BLIS_SUCCESS;
}

// RLNN - RUTN
err_t bli_ztrsm_small_XAutB_XAlB_ZEN5
      (
        obj_t* AlphaObj,
        obj_t* a,
        obj_t* b,
        cntx_t* cntx,
        cntl_t* cntl
      )
{
    INIT_R();
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
        cs_a_ = cs_a;
        rs_a_ = rs_a;
    }
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dcomplex* restrict L_ = L;

#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;

    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    dcomplex* p = NULL;
    dcomplex one = {1, 0};
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (Z_NR_*n*sizeof(dcomplex)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
#endif

    for ( j = (n - z_nr); j > -1; j -= z_nr )
    {
        L_ = L + ((j - Z_NR_ + z_nr) * cs_a) + (j + z_nr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_4xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_NULL_POINTER,
                Z_NR_,
                (n - j - z_nr),
                (n - j - z_nr),
                &one,
                L_ = L + ((j - Z_NR_ + z_nr) * cs_a) + (j + z_nr) * rs_a,
                cs_a,
                rs_a,
                p,
                Z_NR_,
                cntx
            );
            cs_a_ = 1;
            rs_a_ = Z_NR_;
            L_ = p;
        }
#endif
        for ( i = (m - z_mr); (i + 1) > 0; i -= z_mr )
        {
            RLNN_FRINGE( Z_MR_, Z_NR_ );
        }
        dim_t m_rem = i + z_mr;
        if( m_rem > 0 )
        {
            RLNN_FRINGE( m_rem, Z_NR_ );
        }
    }
    dim_t n_rem = j + z_nr;
    if( n_rem > 0 )
    {
#ifndef ENABLE_PACK_A
        dcomplex* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        rlnn_n_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
    }

#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
    return BLIS_SUCCESS;
}


// transpose 4x4 matrix, input is taken from
// c_reg[0+OFFSET] to c_reg[3+OFFSET] and output is stored
// back into same registers
#define TRANSPOSE_4x4(OFFSET)                                                      \
    t_reg[0] = _mm512_shuffle_f64x2(c_reg[0+OFFSET], c_reg[1+OFFSET], 0b10001000); \
    t_reg[1] = _mm512_shuffle_f64x2(c_reg[2+OFFSET], c_reg[3+OFFSET], 0b10001000); \
    t_reg[2] = _mm512_shuffle_f64x2(c_reg[0+OFFSET], c_reg[1+OFFSET], 0b11011101); \
    t_reg[3] = _mm512_shuffle_f64x2(c_reg[2+OFFSET], c_reg[3+OFFSET], 0b11011101); \
                                                                                   \
    c_reg[0+OFFSET] = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b10001000);        \
    c_reg[2+OFFSET] = _mm512_shuffle_f64x2(t_reg[0], t_reg[1], 0b11011101);        \
    c_reg[1+OFFSET] = _mm512_shuffle_f64x2(t_reg[2], t_reg[3], 0b10001000);        \
    c_reg[3+OFFSET] = _mm512_shuffle_f64x2(t_reg[2], t_reg[3], 0b11011101);        \
                                                                                   \

// transpose 12x4 matrix stored in
// c_reg[0] to c_reg[11]
#define TRANSPOSE_12x4()                                                           \
    TRANSPOSE_4x4(0)                                                               \
    TRANSPOSE_4x4(4)                                                               \
    TRANSPOSE_4x4(8)                                                               \

// initialize common variables used among left N left kernels
#define INIT_L_N_LEFT()                                                            \
    dcomplex minus_one = {-1, 0}; /* used as alpha in gemm kernel */               \
    auxinfo_t auxinfo;     /* for dgemm kernel*/                                   \
    __m512d t_reg[10];     /*temporary registers*/                                 \
    t_reg[5] = _mm512_set1_pd( 1.0 ); /*(constant) used for fmaddsub*/             \
    t_reg[4] = _mm512_set1_pd(-1.0);                                               \
    double g_double[3];                                                            \
    __m512d c_reg[Z_MR_]; /*registers to hold GEMM accumulation*/                  \
    for(dim_t i = 0; i < Z_MR_; ++i)                                               \
    {                                                                              \
        c_reg[i] = _mm512_setzero_pd(); /*initialize c_reg to zero*/               \
    }                                                                              \
                                                                                   \
    __mmask8 mask_m_0 = 0b11111111; /*register to hold mask for load/store*/       \
    __mmask8 mask_m_1 = 0b11111111; /*register to hold mask for load/store*/       \
    __mmask8 mask_m_2 = 0b11111111; /*register to hold mask for load/store*/       \

// initialize common variables used among all left kernels
#define INIT_L()                                                                   \
    INIT_L_N_LEFT()                                                                \
    dim_t m = bli_obj_length( b );                                                 \
    dim_t n = bli_obj_width( b );                                                  \
    dim_t cs_a = bli_obj_col_stride( a );                                          \
    dim_t rs_a = bli_obj_row_stride( a );                                          \
    dim_t cs_b = bli_obj_col_stride( b );                                          \
    dim_t cs_a_ = cs_a;                                                            \
    dim_t rs_a_ = rs_a;                                                            \
                                                                                   \
    bool transa = bli_obj_has_trans( a );                                          \
    bool is_unitdiag = bli_obj_has_unit_diag( a );                                 \
    dcomplex AlphaVal = *(dcomplex *)AlphaObj->buffer;                             \
                                                                                   \
    dim_t z_mr =Z_MR_;                                                             \
    dim_t z_nr = Z_NR_;                                                            \
    dim_t i, j;                                                                    \
    dim_t k_iter;                                                                  \
                                                                                   \
    dcomplex* restrict L = bli_obj_buffer_at_off( a );                             \
    dcomplex* restrict B = bli_obj_buffer_at_off( b );                             \

/*
*  Perform TRSM computation for Left Lower
*  NonTranpose variant.
*  n is compile time constant.
*  M <= 12 and N <= 4
*
*  c_reg array contains alpha*b_trsm - a_gemm*b_gemm
*  let  alpha*b_trsm - a_gemm*b_gemm = C
*/
#define TRSM_MAIN_LLN_NxM(M)                                                       \
                                                                                   \
    __UNROLL_LOOP_FULL()                                                           \
    for ( dim_t ii = 0; ii < M; ++ii )                                             \
    {                                                                              \
        if( !is_unitdiag )                                                         \
        {                                                                          \
            BROADCAST_DIV((a_trsm + ii*cs_a));                                     \
            DIV_OR_MUL_DIAG(c_reg[ii+0]);                                          \
        }                                                                          \
        __UNROLL_LOOP_FULL()                                                       \
        for( dim_t jj = ii+1; jj < M; ++jj )/* C[next_col] -= C[curr_col]*a_trsm */\
        {                                                                          \
            t_reg[0] = _mm512_set1_pd((a_trsm + jj*cs_a)->real);                   \
            t_reg[1] = _mm512_set1_pd((a_trsm + jj*cs_a)->imag);                   \
            MULTIPLY_COMPLEX(c_reg[ii+0], t_reg[0], t_reg[1], t_reg[2]);           \
            c_reg[jj+0] = _mm512_sub_pd(c_reg[jj+0], t_reg[2]);                    \
        }                                                                          \
        a_trsm += rs_a;                                                            \
    }                                                                              \


/*
*  Perform TRSM computation for Left Upper
*  NonTranpose variant.
*  n is compile time constant.
*  M <= 12 and N <= 4
*
*  c_reg array contains alpha*b_trsm - a_gemm*b_gemm
*  let  alpha*b_trsm - a_gemm*b_gemm = C
*/
#define TRSM_MAIN_LUN_NxM(M)                                                       \
                                                                                   \
    a_trsm += rs_a * (M-1);                                                        \
    __UNROLL_LOOP_FULL()                                                           \
    for( dim_t ii = (M-1); ii >= 0; --ii )                                         \
    {                                                                              \
        if( !is_unitdiag )                                                         \
        {                                                                          \
            BROADCAST_DIV((a_trsm + ii*cs_a));                                     \
            DIV_OR_MUL_DIAG(c_reg[ii+0]);                                          \
        }                                                                          \
        UNROLL_LOOP_N(11) /*unroll loop 12 is generating warning in gcc*/          \
        for( dim_t jj = (ii-1); jj >= 0; --jj )                                    \
        {                                                                          \
            t_reg[0] = _mm512_set1_pd((a_trsm + jj*cs_a)->real);                   \
            t_reg[1] = _mm512_set1_pd((a_trsm + jj*cs_a)->imag);                   \
            MULTIPLY_COMPLEX(c_reg[ii+0], t_reg[0], t_reg[1], t_reg[2]);           \
            c_reg[jj+0] = _mm512_sub_pd(c_reg[jj+0], t_reg[2]);                    \
        }                                                                          \
        a_trsm -= rs_a;                                                            \
    }                                                                              \

/*
* Perform GEMM + TRSM computation for Left Lower NonTranpose
*/
#define LLNN_FRINGE( M, N )                                                        \
    GENERATE_MASK(M)                                                               \
    a_gemm = L_;                                                                   \
    a_trsm = L + (i * rs_a) + (i * cs_a);                                          \
    b_gemm = B + j * cs_b;                                                         \
    b_trsm = B + i + j * cs_b;                                                     \
    k_iter = i;                                                                    \
    bli_zgemmsup_cv_zen4_asm_12x4m                                                 \
    (                                                                              \
        BLIS_NO_CONJUGATE,                                                         \
        BLIS_NO_CONJUGATE,                                                         \
        M,                                                                         \
        N,                                                                         \
        k_iter,                                                                    \
        &minus_one,                                                                \
        a_gemm,                                                                    \
        rs_a_,                                                                     \
        cs_a_,                                                                     \
        b_gemm,                                                                    \
        1,                                                                         \
        cs_b,                                                                      \
        &AlphaVal,                                                                 \
        b_trsm, 1, cs_b,                                                           \
        &auxinfo,                                                                  \
        NULL                                                                       \
    );                                                                             \
    LOAD_C( N )                                                                    \
    TRANSPOSE_12x4()                                                               \
    TRSM_MAIN_LLN_NxM( M )                                                         \
    TRANSPOSE_12x4()                                                               \
    STORE_RIGHT_C( N )                                                             \

/*
* Perform GEMM + TRSM computation for Left Upper NonTranpose
*/
#define LUNN_FRINGE( M, N )                                                        \
    GENERATE_MASK(M)                                                               \
    a_gemm = L_;                                                                   \
    a_trsm = L + (i - M + z_mr) * rs_a + (i - M + z_mr) * cs_a;                    \
    b_gemm = B + (i + z_mr) + (j - N + z_nr) * cs_b;                               \
    b_trsm = B + (i - M + z_mr) + (j - N + z_nr) * cs_b;                           \
    k_iter = ( m - i - z_mr );                                                     \
    bli_zgemmsup_cv_zen4_asm_12x4m                                                 \
    (                                                                              \
        BLIS_NO_CONJUGATE,                                                         \
        BLIS_NO_CONJUGATE,                                                         \
        M,                                                                         \
        N,                                                                         \
        k_iter,                                                                    \
        &minus_one,                                                                \
        a_gemm,                                                                    \
        rs_a_,                                                                     \
        cs_a_,                                                                     \
        b_gemm,                                                                    \
        1,                                                                         \
        cs_b,                                                                      \
        &AlphaVal,                                                                 \
        b_trsm, 1, cs_b,                                                           \
        &auxinfo,                                                                  \
        NULL                                                                       \
    );                                                                             \
    LOAD_C( N )                                                                    \
    TRANSPOSE_12x4()                                                               \
    TRSM_MAIN_LUN_NxM( M )                                                         \
    TRANSPOSE_12x4()                                                               \
    STORE_RIGHT_C( N )                                                             \

/*
* Solve Left Lower NonTranspose TRSM when m < 12
*/
BLIS_INLINE void llnn_m_rem
(
    dim_t i,            dim_t j,
    dim_t cs_a,         dim_t rs_a,
    dim_t cs_a_,        dim_t rs_a_,
    dim_t cs_b,
    dim_t m,            dim_t n,
    dcomplex* L,        dcomplex* L_,
    dcomplex* p,        dcomplex* B,
    dim_t k_iter,
    bool transa,        bool bPackedA,
    dcomplex AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p;
    // dim_t z_mr = Z_MR_;
    dim_t z_nr = Z_NR_;
    dcomplex minus_one = {-1, 0};
#ifdef ENABLE_PACK_A
    dcomplex one = {1, 0};
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[10]; /*temporary registers*/
    __m512d c_reg[Z_MR_]; /*registers to hold GEMM accumulation*/
    double g_double[3];
    t_reg[5] = _mm512_set1_pd( 1.0 ); /*(constant) used for fmaddsub*/
    t_reg[4] = _mm512_set1_pd(-1.0);
    __mmask8 mask_m_0, mask_m_1, mask_m_2;
    dcomplex *a_gemm, *a_trsm, *b_trsm, *b_gemm;
    dim_t m_rem = m - i;
    dim_t n_rem;
    L_ = L + (i * cs_a);
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_12xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                m_rem,
                i,
                i,
                &one,
                L + (i*cs_a),
                cs_a,
                rs_a,
                p,
                Z_MR_,
                cntx
            );
            cs_a_ = Z_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
    for( j = 0; (j + z_nr - 1) < n; j += z_nr )
    {
        LLNN_FRINGE(m_rem, Z_NR_);
    }
    n_rem = n - j;
    if( n_rem > 0 )
    {
        LLNN_FRINGE( m_rem, n_rem );
    }
}

/*
* Solve Left Upper NonTranspose TRSM when m < 12
*/
BLIS_INLINE void lunn_m_rem
(
    dim_t i,            dim_t j,
    dim_t cs_a,         dim_t rs_a,
    dim_t cs_a_,        dim_t rs_a_,
    dim_t cs_b,
    dim_t m,            dim_t n,
    dcomplex* L,        dcomplex* L_,
    dcomplex* p,        dcomplex* B,
    dim_t k_iter,
    bool transa,        bool bPackedA,
    dcomplex AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    (void) p;
    dim_t z_mr = Z_MR_;
    dim_t z_nr = Z_NR_;
    dcomplex minus_one = {-1, 0};
#ifdef ENABLE_PACK_A
    dcomplex one = {1, 0};
#endif
    auxinfo_t auxinfo;
    __m512d t_reg[10]; /*temporary registers*/
    __m512d c_reg[Z_MR_]; /*registers to hold GEMM accumulation*/
    double g_double[3];
    t_reg[5] = _mm512_set1_pd( 1.0 ); /*(constant) used for fmaddsub*/
    t_reg[4] = _mm512_set1_pd(-1.0);
    for(dim_t i = 0; i < Z_MR_; ++i)
    {
        c_reg[i] = _mm512_setzero_pd(); /*initialize c_reg to zero*/
    }

    __mmask8 mask_m_0, mask_m_1, mask_m_2;
    dcomplex *a_gemm, *a_trsm, *b_trsm, *b_gemm;
    dim_t m_rem = i + z_mr;
    dim_t n_rem;
    L_ = L + ((i - m_rem + z_mr) * cs_a) + (i + z_mr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_12xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                m_rem,
                ( m - i - z_mr ),
                ( m - i - z_mr ),
                &one,
                L + ((i - m_rem + z_mr) * cs_a) + (i + z_mr) * rs_a,
                cs_a,
                rs_a,
                p,
                Z_MR_,
                cntx
            );
            cs_a_ = Z_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
    for( j = (n - z_nr); (j + 1) > 0; j -= z_nr )
    {
        LUNN_FRINGE(m_rem, Z_NR_);
    }
    n_rem = j + z_nr;
    if( n_rem > 0 )
    {
        LUNN_FRINGE( m_rem, n_rem );
    }
}


/*
* Solve Left Lower NonTranspose TRSM when N < 4
*/
BLIS_INLINE void llnn_n_rem
(
    dim_t i,            dim_t j,
    dim_t cs_a,         dim_t rs_a,
    dim_t cs_a_,        dim_t rs_a_,
    dim_t cs_b,
    dim_t m,            dim_t n,
    dcomplex* L,        dcomplex* L_,
    dcomplex* B,
    dim_t k_iter,
    dcomplex AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    INIT_L_N_LEFT()
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dim_t n_rem = n - j;

    LLNN_FRINGE( Z_MR_, n_rem );
}


/*
* Solve Left Lower NonTranspose TRSM when N < 4
*/
BLIS_INLINE void lunn_n_rem
(
    dim_t i,            dim_t j,
    dim_t cs_a,         dim_t rs_a,
    dim_t cs_a_,        dim_t rs_a_,
    dim_t cs_b,
    dim_t m,            dim_t n,
    dcomplex* L,        dcomplex* L_,
    dcomplex* B,
    dim_t k_iter,
    dcomplex AlphaVal,
    bool is_unitdiag,
    cntx_t* cntx
)
{
    INIT_L_N_LEFT()
    c_reg[6] = _mm512_setzero_pd(); // zerod to avoid warning in GCC
    c_reg[7] = _mm512_setzero_pd();
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dim_t z_mr = Z_MR_;
    dim_t z_nr = Z_NR_;
    dim_t n_rem = j + z_nr;
    LUNN_FRINGE( Z_MR_, n_rem );
}


// LLNN - LUTN
err_t bli_ztrsm_small_AutXB_AlXB_ZEN5
      (
        obj_t*   AlphaObj,
        obj_t*   a,
        obj_t*   b,
        cntx_t*  cntx,
        cntl_t*  cntl
      )
{
    INIT_L()
    if( !transa )
    {
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dcomplex* restrict L_ = L;
#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;
    if ( transa )
    {
        bPackedA = true;
    }
    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    dcomplex* p = NULL;
    dcomplex one = {1, 0};
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (Z_MR_*m*sizeof(dcomplex)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
    if (!bPackedA)
#endif
    {
        if (transa)
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
    }
    for( i = 0; (i + z_mr - 1) < m; i += z_mr )
    {
        L_ = L + i*cs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_12xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                Z_MR_,
                i,
                i,
                &one,
                L + i*cs_a,
                cs_a,
                rs_a,
                p,
                Z_MR_,
                cntx
            );
            cs_a_ = Z_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
        for( j = 0; j < n - z_nr + 1; j += z_nr )
        {
            LLNN_FRINGE( Z_MR_, Z_NR_ );
        }
        dim_t n_rem = n - j;
        if( n_rem > 0 )
        {
            llnn_n_rem
            (
                i, j,
                cs_a, rs_a,
                cs_a_, rs_a_, cs_b,
                m, n, L, L_, B, k_iter, AlphaVal,
                is_unitdiag, cntx
            );
        }
    }
    dim_t m_rem = m - i;
    if( m_rem > 0 )
    {
#ifndef ENABLE_PACK_A
        dcomplex* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        llnn_m_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
    }
#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
    return BLIS_SUCCESS;
}

// LUNN - LLTN
err_t bli_ztrsm_small_AltXB_AuXB_ZEN5
      (
        obj_t*   AlphaObj,
        obj_t*   a,
        obj_t*   b,
        cntx_t*  cntx,
        cntl_t*  cntl
      )
{
    INIT_L()
    if( !transa )
    {
        i = cs_a;
        cs_a = rs_a;
        rs_a = i;
    }
    dcomplex *a_gemm, *a_trsm, *b_gemm, *b_trsm;
    dcomplex* restrict L_ = L;
#ifdef ENABLE_PACK_A
    bool bPackedA = ENABLE_PACK_A_FOR_UPPER;
    if ( transa )
    {
        bPackedA = true;
    }
    rntm_t rntm;
    mem_t local_mem_buf_A_s = {0};
    dcomplex* p = NULL;
    dcomplex one = {1, 0};
    if(bPackedA)
    {
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );
        siz_t buffer_size =
            bli_pool_block_size
            (
                bli_pba_pool
                (
                    bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                    bli_rntm_pba(&rntm)
                )
            );
            bli_pba_acquire_m
            (
                &rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf_A_s
            );
        if ( !bli_mem_is_alloc(&local_mem_buf_A_s) )
            return BLIS_NULL_POINTER;
        p = bli_mem_buffer(&local_mem_buf_A_s);
        if( p == NULL )
            bPackedA = false;
        if (local_mem_buf_A_s.size < (Z_MR_*m*sizeof(dcomplex)))
        {
            bPackedA = false;
            if (bli_mem_is_alloc( &local_mem_buf_A_s ))
            {
                bli_pba_release(&rntm, &local_mem_buf_A_s);
            }
        }
    }
    if (!bPackedA)
#endif
    {
        if (transa)
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
    }
    for( i = (m - z_mr); (i + 1) > 0; i -= z_mr )
    {
        L_ = L + ((i - Z_MR_ + z_mr) * cs_a) + (i + z_mr) * rs_a;
#ifdef ENABLE_PACK_A
        if(bPackedA)
        {
            bli_zpackm_zen4_asm_12xk
            (
                BLIS_NO_CONJUGATE,
                BLIS_PACKED_COL_PANELS,
                Z_MR_,
                ( m - i - z_mr ),
                ( m - i - z_mr ),
                &one,
                L + ((i - Z_MR_ + z_mr) * cs_a) + (i + z_mr) * rs_a,
                cs_a,
                rs_a,
                p,
                Z_MR_,
                cntx
            );
            cs_a_ = Z_MR_;
            rs_a_ = 1;
            L_ = p;
        }
#endif
        for( j = (n - z_nr); (j + 1) > 0; j -= z_nr )
        {
            LUNN_FRINGE( Z_MR_, Z_NR_ );
        }
        dim_t n_rem = j + z_nr;
        if( n_rem > 0 )
        {
            lunn_n_rem
            (
                i, j,
                cs_a, rs_a,
                cs_a_, rs_a_, cs_b,
                m, n, L, L_, B, k_iter, AlphaVal,
                is_unitdiag, cntx
            );
        }
    }
    dim_t m_rem = i + z_mr;
    if( m_rem > 0 )
    {
#ifndef ENABLE_PACK_A
         dcomplex* p = NULL;
        bool bPackedA = false;
#endif //ENABLE_PACK_A
        lunn_m_rem
        (
            i, j,
            cs_a, rs_a, cs_a_, rs_a_,
            cs_b,
            m, n,
            L, L_, p, B,
            k_iter,
            transa,
            bPackedA,
            AlphaVal,
            is_unitdiag,
            cntx
        );
    }
#ifdef ENABLE_PACK_A
    if (( bPackedA ) && bli_mem_is_alloc( &local_mem_buf_A_s ))
    {
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
#endif
return BLIS_SUCCESS;
}
